# CUT_FRACTIONAL_BITS_OPT: skip MSB-adder work for redundant top slices

Flag: `CUT_FRACTIONAL_BITS_OPT` (config.h, default 0).

Under `TRUNC_DELAYED=0`, the conv/FC layer feeding a ReLU has already truncated its output by
`FRACTIONAL` bits, so the wire's **reconstructed value** is provably bounded within
`BITLENGTH - FRACTIONAL` signed bits - regardless of `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING`
(the bound follows from the truncation, not from any mask construction). The DRELU adder's top
`FRACTIONAL` bit-slices are then redundant, and everything about them can be skipped: the A2B
sharing (P0's per-slice send), the reshares, and the adder gates.

Eligibility is compile-time and comes from one of two macros, disjoint by `ROT_PREPROCESSING_OPT`:
`CUT_FRAC_ELIGIBLE` (protocols/beaver_triples.hpp) for the 2PC ROT/beaver circuits in
`programs/functions/adders/zero_add_adders/`, and `CUT_FRAC_ELIGIBLE_GENERIC` (config.h) for the
generic adders the 3PC and 4PC protocols use, where beaver_triples.hpp is not even included. Both
require `TRUNC_DELAYED == 0`, `BITLENGTH == 32` and `1 <= FRACTIONAL <= 29`.

Whether a given adder instance applies the cut is the runtime flag `g_cut_frac_active`: RELU sets
it, and the max/min comparison adders run the full circuit - except under the `RESHARE_OPT_SIM`
bake, whose stride and slot mapping are compile-time cut-aware, where they must run cut as well
(max_min.hpp).

## Slice roles (protocols/beaver_triples.hpp)

With the cut active (slice 0 = numeric MSB, `F = FRACTIONAL`):

* slices `[0, F)` - **vacant**: never prepared, shared, reshared, or read.
* slice `F` - **boundary**: its raw wire pair is masked + sent in the A2B prepare (taking over
  slice 0's original role), because the adder's output tap reads it.
* slices `(F, k)` - unchanged.

The A2B prepare/complete loops (online, PRE, and INIT - all three must skip identically) route
slices via `cut_frac_prep_vacant` / `cut_frac_prep_boundary`. Vacant slices are written as the
public constants `a := 1, b := 0` (so `g = a & b = 0` and `p = a ^ b = 1`, the identity of the
prefix operator) rather than left unprepared: RCA and PPA never read them, PPA4's block gates rely
on the constants. On the `A2B_ONLINE_OPT` path `[c]` is still *consumed* on vacant slices, because
the early boolean addition emits a fixed-stride slice per value and skipping would misalign it.

## Coverage

| circuit | plain (`*_and_ab`) | reshared (`RESHARE_OPT=1`) | a_known (`A_KNOWN_TO_EVALUATORS_OPT=1`) |
|---|---|---|---|
| ripple-carry (RCA) | early stop, skipped triples | early stop, skipped triples | early stop, skipped triples |
| prefix (PPA) | identity substitution, skipped gates | identity substitution, skipped reshares and gates | identity substitution, skipped gates |
| four-way prefix (PPA4) | skipped gates + arity reduction | skipped gates + arity reduction | skipped b-product gates |

All at k = 32. The generic 3PC/4PC adders are covered separately (see below). The `_split` variants
(`ADDITIONAL_PPA_THREADS > 0`) are not covered - see Limitations.

## How each family implements it

### Ripple-carry

Stop at case `31-F` and read `msb = x[F] ^ y[F] ^ carry_(F+1)` - the standard reduced-width-adder
sum bit (verified by 200k-trial simulation over properly bounded values; 0 mismatches). Every case
2..30 carries its own compile-time-selected shortcut, covering any `FRACTIONAL` in `[1, 29]`. The
last executed gate's mask assign is redirected to the adder's spare random so it does not reference
a triple slot beyond the reduced allocation, and beaver retrieval is conditional to match. Saves `F`
AND gates, `F` communication rounds, `F` triples and `F` A2B slice-sends per value group.

### Prefix (Sklansky tree): identity substitution

The prefix combine `(g,p) o (g',p') = (g ^ p&g', p&p')` has identity element `(0, 1)`. Forcing
`(g_i, p_i) := (public 0, public 1)` for slices `1..F` and substituting the output tap
`p_0 := a[F] ^ b[F]` makes the **unchanged** tree compute `p_F ^ G(F+1..31)` - the reduced-width
MSB - because identities drop out of every combine, regardless of tree shape (simulation:
0/200k mismatches). No structural tree changes, no per-F variants.

The subtlety is the `and_ab` **mask algebra**: every AND gate passes a fixed precomputed mask
product keyed to its inputs' *designed* masks, and every wire's designed mask is part of a fixed
XOR chain. The invariant that makes the substitution safe: **every wire must keep its designed
mask** - then all downstream products and PRE recordings stay valid. Only the substituted `g_i`
leaves change mask (to 0); their consumers are patched conditionally:

* odd-slice `g_i` (a direct AND input): the consumer passes mask product `0` instead of
  `triples[X].c` (the actual product with a zero mask IS zero);
* even-slice `g_i` (XOR-folded into `g_L1_i`): the pair-AND's assign drops the `r_i`
  XOR-compensation term, so `g_L1_i` keeps its designed mask;
* `g_1` (spine): `pg_L1_1_2`'s assign folds `r199` in, preserving `g_L2_1`'s designed mask.

p-wires need no fixes: they are `zero_add`-renormalized before every AND.

**Gate skipping.** Every AND whose operands' slice support lies entirely below `F` is rewritten into
a local form: a public constant, or, when one operand is the all-ones constant, the other operand
retargeted to the gate's designed output mask with `zero_add`. `zero_add` is local in the online
phase (one preprocessing delta, not a triple and not a round), which is what lets a gate disappear
without re-deriving the masks of everything downstream. Retargeting is not optional: a bare public
constant carries mask 0 and would break every consumer expecting the designed mask. Three things
move in step with each skipped gate, or the phases desynchronise: `complete_and()` is guarded by the
same predicate, the operand `zero_add`s are dropped (the cut branches retarget the SOURCE wire), and
beaver retrieval becomes conditional - via the per-slot table `cut_skip_threshold[i]`, slot `i`
being skipped exactly when `FRACTIONAL > cut_skip_threshold[i]` - because INIT counts per called
gate and allocates one triple fewer. An assign naming a skipped slot's mask reads the spare random
`r_cut_spare` instead.

In the reshared flavour the identity slices additionally consume no random multiplication (the bake
map `reshare_rt_offset` / `reshares_per_adder` shifts accordingly), and under `RESHARE_OPT_SIM=1`
their input `zero_add`s switch back from `zero_add_local` (which presumes a baked mask; public wires
are unbakeable) to the communicating `zero_add`. The reshared circuit identity-substitutes the
boundary leaf as well, so its thresholds are one lower than the plain circuit's.

### Four-way prefix

PPA4's 3-slice blocks (`B3G = g1 ^ p_i*g_(i+1) ^ p_i*p_(i+1)*g_(i+2)` via dot3/dot4 tuple gates,
then two more W-combine levels) get the value-level treatment: the A2B writes public constants into
the vacant slices and every dot gate computes the reduced function through the completely normal
machinery - a 4-input gate whose low inputs are vacant semantically degenerates to the 2-input
product of its remaining wires. The input `zero_add`s re-mask the constants, so every wire keeps its
designed mask and every tuple product stays valid.

Structural changes only where public wires meet special machinery: reshared `g1` gates at identity
slices are skipped (public-0 output) with the universal mask fold `t1.assign ^= g1_mask`, their
`reshare_a/b` calls and `rt` slots are skipped, and the bake map renumbers kept slices by rank in
the retrieval order `{1,4,7,...,29,22}` (22 last); the boundary slice keeps real wires for the `p0`
output tap, with its tree contributions substituted at the use sites.

**Gate skipping and arity reduction.** Fully-vacant gates are skipped and consume no tuple. The
P-gate beaver-3 slots are dropped in all phases, so allocation and retrieval both shrink and the
consumption *rank* of later slots shifts - `cut_frac_ppa4_b3_skipped_below` is what external offset
arithmetic (the S1 peek, the bake) must use. A gate holding an all-ones operand folds it away and
runs one arity lower on a reserved tuple: `dot4 -> dot3` trades 15 correlated values for 7,
`dot3 -> dot` and `and3 -> and` trade 7 for 3.

A reduction is only real inside a **window** `>= TH_RED && < TH_ZERO`: below the first threshold
nothing is constant, and at the term's own zero threshold the cut removes the gate outright while
INIT still runs the original gate. Reserving across the whole range retrieves tuples nobody consumes
- which shows up as the "only the last test fails" signature. Multi-level: a gate with several
all-ones operands emits one branch per reduction level, the windows tiling the range disjointly, so
at most one level is live for a given `FRACTIONAL` and only its reserved slot is retrieved. Two
limits are deliberate: a dot term never degenerates below a real dot gate (its result must stay
dot-pending for the chain accumulation, and a single surviving wire is standard-form), and an `and`
gate's last level is a pure retarget of the single survivor, so its `complete_and` guard takes that
window too.

### a_known-to-evaluators circuits (the A2B bake path)

In the a_ab decomposition a wire is `a ^ b` with `a` known to the evaluators, so products expand
into a-only terms, mixed `a*b` terms (both local) and b-only products, and only the b-products
consume a beaver tuple. A vacant slice is prepared as `a := all-ones, b := 0`, so every b-product
touching a vacant slice is identically zero and its gate disappears entirely.

Zero-ness is a threshold `z`, the wire being provably zero exactly when `FRACTIONAL >= z`: `b[i]`
gives `i+1`; `a[i]` and a-only products never die; a product takes the `min` of its factors; an XOR
takes the `max`. That is `cut_frac_ppa4_skip(z)`. Gates are cut only when their zero-ness traces
back to the INPUTS without passing through a chain output: a chain output is a finalised wire
carrying an accumulated mask, so cutting its consumer corrupts the heap rather than merely computing
a wrong value. That rule drops 5 of 45 candidates and holds at every `FRACTIONAL`.

The prefix a_ab circuit's tree gates are p-products (all-ones when vacant, not zero), so the
zero-threshold analysis finds nothing there and it uses the identity treatment instead.

Two rules keep this family safe, and both were learned by breaking it:

1. **Leave the local `mult_a_known` gates running.** Handed a share that is literally `(0, 0)` they
   return `(0, 0)` in whichever representation they produce, so nothing mixes standard with
   dot-pending shares. For the same reason `mask_and_send`/`complete` must ALWAYS run, even on a
   chain that has gone entirely zero: those local contributions are still dot-pending and carry
   their own mask assigns, and the finalisation is what converts the accumulator to standard form
   and applies the accumulated mask.
2. **Compensate a cut gate's output mask along XOR-only paths** - not along every chain the gate can
   reach. A dot gate consumed through a `mult_a_known` gate has its mask replaced by that gate's own
   output mask, so it never reaches that chain and compensating there corrupts it. Sharing is heavy
   (36 of 50 dot outputs have more than one consumer), so a gate can need compensating in two chains
   at once; the local `mult_a_known` carriers, which can never be cut, are the safe place to park a
   mask. Every `prepare_remask` also reads its mask from a triple field, so freeing a slot must
   redirect the remasks that named it, not just the gate assigns.

### Generic 3PC/4PC adders

The generic circuits (`programs/functions/adders/rca_msb.hpp`, `ppa_msb_unsafe.hpp`,
`ppa_msb_4_way.hpp`) need nothing protocol-specific:

* **RCA**: the ripple runs from the LSB upwards, so the cut is just "stop at slice `F` and read the
  sum bit there". The last `F` rounds, with their AND gates and communication, never happen.
* **PPA**: identity substitution with EXACT constant tracking. `cst[i]` marks a prefix wire still
  holding the identity `(g,p) = (0,1)`; a wire stays identity only while every slice it aggregates
  does, so gates are skipped at every tree depth, not just at the leaves. A combine whose low wire
  is identity is dropped outright; one whose current wire is identity degenerates to a local copy.
  Completion consults a separate `skipped[]` record, because `cst[]` is mutated during prepare and
  cannot be re-derived afterwards.
* **PPA4**: public constants written into the vacant slices (slice 0 included - it feeds a dot4 -
  exactly as the 2PC A2B does) and the output tap moved to the boundary slice. Its blocks still
  evaluate on the constants, so it is correct but does not save anything yet.

`CUT_FRAC_ELIGIBLE_GENERIC` must name every adder it covers: while it still said `RCA_MSB == 1`, the
PPA cut compiled in and never engaged, showing byte-identical traffic with the flag on and off.

## Stream-count consistency

Every preprocessing stream (random multiplications, boolean triples, beaver tuples,
`preprocessed_outputs`) must have **identical production (PRE), allocation (INIT) and consumption
(LIVE) counts**. INIT counts per *called* gate/reshare stub, so when the circuit skips calls, an
unconditional `retrieve*()` loop in the constructor overruns the generated arrays and corrupts
whatever ran last - the classic symptom being that only the final test or the final adder group
fails, with clean value errors or garbage depending on what the out-of-bounds memory held. Every
skipped gate therefore has its retrieval skipped under the same predicate, and the bake maps
(`reshares_per_adder`, `b3_tuples_per_adder`, `ppa4_zero_add_t3`) use cut-aware counts and ranks.
When a "last test only" failure appears, check the per-stream counters first (`A2B_STREAM_DBG` in
protocol_executer.hpp dumps consumed vs allocated at each phase end; `rmul=651/621` was the first
smoking gun). One more trap from the four-way circuit: `mask_and_send` must be skipped in EVERY
phase including INIT even where the term itself stays counted, since INIT accounts for the send and
keeping it for INIT alone deadlocks the adder.

A welcome side effect for `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1 + TRUNC_DELAYED=0`: the
SecureML trunc image (`l = TRUNC(-r1)` zeroes the top-F mask bits) previously made PPA/PPA4's high
reshared slices unbakeable under `RESHARE_OPT_SIM`, forcing `TRUNC_DELAYED=1`. With the cut those
slices are not reshared at all and every kept slice's bake target lies inside the image, so that
default configuration now works on all three adders.

## Measured savings

func53, `FRACTIONAL=5`, 32-bit, `TRUNC_DELAYED=0`, cut off -> on. Material counts come from the
preprocessing requirement print.

| circuit | preprocessing material | notes |
|---|---|---|
| RCA plain | boolean 8928 -> 7488 (-16.1%) | 27 of 32 rounds; F=8: 24 rounds (-26%); F=12: 20 rounds (-39%) |
| PPA plain | boolean 24768 -> 21312 (-14%) | F=8: -24%; F=12: -38% |
| PPA a_ab | boolean 15840 -> 13536 (-14.5%) | F=8: -25.5% |
| PPA reshared | random mult. 8928 -> 7488; boolean 15840 -> 14688 (-7.3%) | F=12: boolean -20% |
| PPA4 plain | beaver3 6912 -> 6336, beaver4 3744 -> 3456, boolean 4320 -> 4896 | the arity reduction trades one beaver4 (15 values) and one beaver3 (7) for two boolean triples (3 each), net -16 per adder, so the boolean count alone reads as a regression; F=6: beaver3 6336 via multi-level |
| PPA4 a_ab | boolean 9792 -> 8352 (-14.7%), beaver3 4320 -> 3744 (-13.3%) | F=10: -26.5% / -20%; F=12: -35.3% / -26.7% |
| PPA4 reshared | beaver3 6912 -> 6336 (cut + arity), random mult. 3168 -> 2592 | online 0.005516 -> 0.005012 MB |

LeNet (10 images, `FRACTIONAL=5`), party 0, cut off -> on:

| adder | mode | boolean triples | online MB |
|---|---|---|---|
| RCA  | plain    | 2018720 -> 1693120 (-16.1%) | 0.3239 -> 0.2832 (-12.6%) |
| RCA  | a2b      | 1953600 -> 1628000 (-16.7%) | 0.3076 -> 0.2669 (-13.2%) |
| RCA  | reshared | 1953600 -> 1628000 (-16.7%) | 0.3158 -> 0.2751 (-12.9%) |
| PPA  | plain    | 5600320 -> 4818880 (-14.0%) | 1.016 -> 0.9751 (-4.0%) |
| PPA  | a2b      | 3581600 -> 3060640 (-14.5%) | 0.5111 unchanged |
| PPA  | reshared | 3581600 -> 3321120 (-7.3%)  | 0.7635 -> 0.7228 (-5.3%) |
| PPA4 | plain    | 976800 -> 1107040 (arity trade, see above) | 1.146 -> 1.081 (-5.7%) |
| PPA4 | a2b      | 2214080 -> 1888480 (-14.7%) | 0.910 -> 0.8367 (-8.1%) |
| PPA4 | reshared | 260480 unchanged (the saving is in random multiplications) | 1.057 -> 0.9263 (-12.4%) |

`plain` = no A2B/reshare optimizations; `a2b` = `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1
A2B_CONV_BAKE=1` (the first two must be enabled together); `reshared` = `RESHARE_OPT=1`.

ResNet50 (CIFAR-10, 10 images, `FUNCTION_IDENTIFIER=171`, RCA, no other optimizations): cut off
70.00%, 60789760 boolean triples, 8.008 MB online; cut on 80.00%, 50984960 (-16.1%), 6.782 MB
(-15.3%). The 70 -> 80% difference is one image in ten, i.e. noise at this sample size, not an
improvement.

3PC/4PC (func53, generic adders, online MB per party, cut off -> on): trio (5) RCA 0.00582 ->
0.00570 and 0.00494 -> 0.00482; trio PPA 0.00780 -> 0.007512 and 0.00692 -> 0.006632; Quad_OffOn
(12) RCA 0.005948 -> 0.005828, 0.00622 -> 0.00610, 0.00726 -> 0.00714; Quad PPA 0.00924 ->
0.008952, 0.008936 -> 0.008648, 0.00820 -> 0.007912. PPA4 is unchanged on both, as expected.

`MB SENT PRE` is not a useful indicator for this optimization (1.63115 -> 1.63117 for the
ripple-carry cut): under `ROT_PREPROCESSING_OPT` boolean triples cost about 6e-5 MB against about
1.0 MB for the arithmetic ones, so the preprocessing byte total is dominated by other material.
Count triples and rounds instead.

## Validation

* func53: **8/8 in all 18 2PC runs** (three adders x three optimization modes, cut on and off) and
  **8/8 in all 12 3PC/4PC runs** (protocols 5 and 12, three adders, cut on and off), `FRACTIONAL=5`.
* LeNet, 10 images, matched `standard` model and dataset: **100% in all 18 2PC runs**.
* Other `FRACTIONAL`: plain RCA and plain PPA 8/8 at F=8; PPA4 a_ab 8/8 at F=10 and F=12; PPA4 plain
  8/8 at F=6, 8, 9, 10, 11; bake PPA4 8/8 at F=12; reshared PPA4 + MWK 8/8 at F=8 (the boundary
  falling on a `zero_add` slice).
* At F=12 the plain RCA and plain PPA circuits report 7/8 **with the cut on and off alike** - a
  pre-existing fixed-point precision limit (2F = 24 fractional bits leave too little integer
  headroom in 32 bits), not a cut regression. The plain PPA4 fails the same test at F=12; that comes
  from the identity substitution, not from the gate skipping (the unported file fails it too), and
  everything below F=12 is clean.
* Under `RESHARE_OPT_SIM=1` the reshare-check counter confirms the skips actually fire: 52 checks
  per func53 run instead of 62 (PPA), 18 instead of 22 (PPA4), 0 mismatches.

Two pitfalls that once produced whole eras of invalid "validation": stale build objects silently
ignore header changes (`make clean` between configurations), and a flag missing from the Makefile's
`CONFIG_OPTIONS` compiles as OFF, so every "flag-on" run is really a baseline run
(`CUT_FRACTIONAL_BITS_OPT` is listed there now). Model and dataset variants must also match
(`standard` with `standard`): a mismatch costs one image in ten on LeNet and gives chance accuracy
on ResNet50.

## Limitations and open items

* **`BITLENGTH != 32` is blocked below the cut.** Neither 16 nor 64 builds at all: the Cheetah
  preprocessing interface types its buffers by `TRIPLE_BITLEN`, `nn/ConvTriple` is linked as a
  prebuilt 32-bit shared library, and the top-level Makefile never passes that width. The k=8/16
  circuit specializations are therefore untouched - the result could not be validated. The
  prerequisite is rebuilding that library with `-DTRIPLE_BITLEN=<width>` and plumbing the flag
  through.
* **`_split` variants** (`ADDITIONAL_PPA_THREADS > 0`) are not covered, and the blocker is not the
  cut: pristine `ppa_msb_4way_and_a_ab_split.hpp` scores 3/8 at `FRACTIONAL=5` with the cut OFF, and
  the reshared split scores 7/8 - identical with and without the transform. That path fails on its
  own, so extending the cut there cannot be validated.
* **Generic PPA4** (3PC/4PC) is correct under the cut but not yet free: its blocks still evaluate on
  the substituted constants, the stage the 2PC four-way circuit started at before gate skipping and
  arity reduction. Making it free needs the same per-block analysis.
* **The a_ab adder path breaks ResNet50, and it is neither the cut nor the bake.** Bisect at
  `FRACTIONAL=5` with matched weights: RCA/PPA4 without optimizations 70-80%;
  `A_KNOWN_TO_EVALUATORS_OPT=1` alone (online-opt off, bake off, cut off) **10%**; with
  `A2B_ONLINE_OPT=1` and the bake, cut off/on, 10%/10%. func53 passes 8/8 and LeNet holds its
  baseline on the same path, so the failure needs ResNet50's structure - BatchNorm, depth, or the
  residual adds. A related design gap worth noting for whoever picks this up: the bake compensates
  exactly ONE post-GEMM operation (`add_bias`, via `g_bake_bias_l`); nothing invalidates a committed
  mask when BatchNorm or a skip-add changes it between the conv and the ReLU.
* **LeNet cannot verify the 3PC/4PC cut.** The LeNet harness is nondeterministic on both protocols -
  the same binary gives all-parties-0% on some runs and a party-dependent mix of 0% and 100% on
  others - identically with the cut on and off, and reproducibly at this branch's base commit, so it
  predates the branch. func53 is deterministic there and is what the 3PC/4PC table rests on. Also,
  only trio (3PC) and Quad_OffOn (4PC) build this path at all: rep3, sharemind, astra and Tetrad are
  missing `mask_and_send_dot_with_trunc` / `complete_mult_with_trunc`.
