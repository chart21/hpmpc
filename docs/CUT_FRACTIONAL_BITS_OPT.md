# CUT_FRACTIONAL_BITS_OPT: skip MSB-adder work for redundant top slices

Under `TRUNC_DELAYED=0`, the conv/FC layer feeding a ReLU has already truncated its output by
`FRACTIONAL` bits, so the wire's **reconstructed value** is provably bounded within
`BITLENGTH - FRACTIONAL` signed bits - regardless of `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING`
(the bound follows from the truncation, not from any mask construction). The DRELU adder's top
`FRACTIONAL` bit-slices are then redundant, and everything about them can be skipped: the A2B
sharing (P0's per-slice send), the reshares, and the adder gates.

## Slice roles (protocols/beaver_triples.hpp)

With the cut active (`g_cut_frac_active`, set by RELU only - max/min/comparison adders on the
same build run the full circuit):

* slices `[0, F)` - **vacant**: never prepared, shared, reshared, or read.
* slice `F` - **boundary**: its raw wire pair is masked + sent in the A2B prepare (taking over
  slice 0's original role), because the adder's output tap reads it.
* slices `(F, k)` - unchanged.

The A2B prepare/complete loops (online, PRE, and INIT - all three must skip identically) route
slices via `cut_frac_prep_vacant` / `cut_frac_prep_boundary`.

## RCA (sequential ripple)

Stop at case `31-F` and read `msb = x[F] ^ y[F] ^ carry_(F+1)` - the standard reduced-width-adder
sum bit (verified by 200k-trial simulation over properly bounded values; 0 mismatches). Every case
2..30 carries its own compile-time-selected shortcut, covering any `FRACTIONAL` in `[1, 29]`.
Saves `F` AND-gates, `F` communication rounds, and `F` A2B slice-sends per value-group.

## PPA (Sklansky prefix tree): identity substitution

The prefix combine `(g,p) o (g',p') = (g ^ p&g', p&p')` has identity element `(0, 1)`. Forcing
`(g_i, p_i) := (public 0, public 1)` for slices `1..F` and substituting the output tap
`p_0 := a[F] ^ b[F]` makes the **unchanged** tree compute `p_F ^ G(F+1..31)` - the reduced-width
MSB - because identities drop out of every combine, regardless of tree shape (simulation:
0/200k mismatches). No structural tree changes, no per-F variants.

The subtlety is the `and_ab` **mask algebra**: every AND gate passes a fixed precomputed mask
product keyed to its inputs' *designed* masks, and every wire's designed mask is part of a fixed
XOR-chain. The invariant that makes the substitution safe: **every wire must keep its designed
mask** - then all downstream products and PRE recordings stay valid. Only the substituted `g_i`
leaves change mask (to 0); their consumers are patched conditionally:

* odd-slice `g_i` (a direct AND input): the consumer passes mask-product `0` instead of
  `triples[X].c` (the actual product with a zero mask IS zero);
* even-slice `g_i` (XOR-folded into `g_L1_i`): the pair-AND's assign drops the `r_i`
  XOR-compensation term, so `g_L1_i` keeps its designed mask;
* `g_1` (spine): `pg_L1_1_2`'s assign folds `r199` in, preserving `g_L2_1`'s designed mask.

p-wires need no fixes: they are `zero_add`-renormalized before every AND.

Saves `F` reshared AND-gates (and their `rt` consumption + reshare communication) per adder,
plus the A2B slice-sends.

## The invariant that bit twice: stream-count consistency

Every preprocessing stream (random multiplications, boolean triples, `preprocessed_outputs`)
must have **identical production (PRE), allocation (INIT), and consumption (LIVE) counts** -
INIT counts per *called* gate/reshare stub, so when the circuit skips calls, the constructors'
unconditional `retrieve*()` loops overran the generated arrays and corrupted whatever ran last
(the classic symptom: only the final test / final adder group fails, with clean value errors or
garbage depending on what the out-of-bounds memory held). Fixed by making retrieval conditional
to exactly the slots executed gates use:

* PPA: `random_triples[i]` retrieved only for kept slices (bake map `reshare_rt_offset` /
  `reshares_per_adder` shift accordingly under `CUT_FRAC_ELIGIBLE`);
* RCA: `triples[i]` retrieved only for `i < 30-F`, and the last executed gate's mask assign is
  swapped to the adder's spare random `r61` so no executed gate references a slot beyond the
  allocation.

Diagnosing this required a per-stream counter dump (consumed vs produced/allocated at each phase
end) - `rmul=651/621` was the first smoking gun. When a "last test only" failure appears, check
the counters FIRST.

## MWK + TRUNC_DELAYED=0 (the default config) now works for PPA

Previously the SecureML trunc image (`l = TRUNC(-r1)` zeroes the top-F mask bits) made PPA/PPA4's
high reshared slices unbakeable under RESHARE_OPT_SIM, forcing TRUNC_DELAYED=1. With the cut,
those slices are not reshared at all, and every KEPT slice's bake target (numeric bits
`0..K-1-F`) lies inside the trunc image - so `MWK=1 + TRUNC_DELAYED=0 + PPA` passes fully.

## PPA4 (4-way tree): value-level identity substitution

PPA4's 3-slice blocks (`B3G = g1 ^ p_i*g_(i+1) ^ p_i*p_(i+1)*g_(i+2)` via dot3/dot4 tuple gates,
then two more W-combine levels) get the cleanest treatment of all: the A2B writes PUBLIC constants
`a := 1, b := 0` into the vacant slices (so `g = a&b = 0` and `p = a^b = 1` arise automatically),
and every dot gate computes the reduced function through the completely NORMAL machinery - a
4-input gate whose low inputs are vacant semantically degenerates to the 2-input product of its
remaining wires, exactly the arity reduction one would hand-derive, with zero downstream patches.
The input zero_adds re-mask the constants, so every wire keeps its designed mask and every tuple
product stays valid; no gate restructuring, no per-field product conditionals needed.

Structural changes only where public wires meet special machinery:
* reshared `g1` gates at identity slices are skipped (public-0 output) with the universal mask
  fold `t1.assign ^= g1_mask`; their `reshare_a/b` calls and `rt` retrieval slots are skipped, and
  the bake map renumbers kept slices by rank in the retrieval order `{1,4,7,...,29,22}` (22 last);
* under `RESHARE_OPT_SIM=1`, identity slices' input zero_adds switch from `zero_add_local` (which
  presumes a baked mask - public wires are unbakeable) back to the COMMUNICATING `zero_add`, and
  `bake_reshare_mask` skips their beaver3 fields;
* the boundary slice keeps real (masked+sent) wires for the `p0` output tap; its tree
  contributions are substituted at the use sites (`p_F := 1`, zero_add sources swapped to
  constants when `FRACTIONAL == F`).

Because of the constants approach, the A2B vacant handling for ALL adders writes public constants
instead of leaving wires unprepared (benign for RCA/PPA, whose vacant wires are simply unread).

## Validation (all clean builds, FRACTIONAL=5 unless noted)

| Config | func53 | LeNet (10 imgs) |
|---|---|---|
| RCA + cut, MWK=1 TD=0 | 8/8 (checks=2, mm=0) | 100% (checks 2035, mm 2 = padded lanes) |
| PPA + cut, plain SIM=0 / SIM=1 | 8/8 (checks 52 vs 62 uncut, mm=0) | 100% (52910 = 2035x26, mm 52) |
| PPA + cut, MWK=1 TD=0 | 8/8 (mm=0) | 100% |
| PPA4 + cut, plain SIM=0 / SIM=1 | 8/8 (checks 18 = 2x9 kept, mm=0) | 100% (18315 = 2035x9, mm 18) |
| PPA4 + cut, MWK=1 TD=0 | 8/8 (mm=0) | 100% |
| PPA4 + cut, MWK=1 TD=0, F=8 (boundary on a zero_add slice) | 8/8 (checks=16, mm=0) | - |
| flag off (any) | matches all prior baselines | matches |

With this, **MWK=1 + TRUNC_DELAYED=0 (the default config) works on ALL THREE adders** - it
previously required TRUNC_DELAYED=1 for PPA and PPA4.

## Remaining work

* The `RESHARE_OPT=0` twins (`*_and_ab.hpp`) and `A_KNOWN_TO_EVALUATORS_OPT` variants are
  unpatched and guarded off (the cut no-ops there).
* `BITLENGTH != 32` circuits (k=8/16 specializations) untouched (guarded).
* The PPA4 tuple gates still run on the constant inputs (correct but not free); restructuring
  them to actually drop tuple/communication consumption for fully-vacant blocks is a further
  optimization with the same stream-count discipline.

## Historical note: two invalid validation eras

Early versions of this feature were "validated" while (a) stale build objects silently ignored
header changes (`make clean` fixed it), and (b) `CUT_FRACTIONAL_BITS_OPT` was missing from the
Makefile's forwarded-defines list, so `make CUT_FRACTIONAL_BITS_OPT=1` compiled the flag OFF and
every "flag-on" run was actually a baseline run. Both are fixed; every result in the table above
was obtained after both fixes, from clean builds, with the reshare-check counter confirming the
skips actually fire (52 checks per func53 run instead of 62).


## Coverage beyond the reshared family (2026-07)

Originally the cut was gated on `RESHARE_OPT == 1` (later also the A2B bake), so with the plain
circuits — the ones used when neither the reshare simulation nor the bake is on, i.e. the DEFAULT
configuration — the flag was silently inert. It is now eligible there too, with two changes:

1. **Output tap moved** in the four circuits that lacked it (`ppa_msb_unsafe_and_ab`,
   `ppa_msb_unsafe_and_a_ab`, `ppa_msb_4way_and_ab`, `ppa_msb_4way_and_a_ab` + its `_split`):
   `p0 = a[FRACTIONAL] ^ b[FRACTIONAL]` when the cut is active, mirroring the reshared circuits.
   This is the correctness-critical piece: the substituted top slices make the sum bits above the
   boundary meaningless, while the carry INTO the boundary slice is untouched (carries flow from
   less- to more-significant, i.e. from higher slice index to lower), so the boundary sum bit is the
   true sign.

2. **Vacancy on the bake path.** In the plain A2B prepare the vacant slices were already replaced by
   the public constants `a := 1`, `b := 0` (identity for the carry algebra) — which is what makes the
   substitution work and also skips those slices' sharing communication. The `A2B_ONLINE_OPT` prepare
   had no such handling, so under the bake the adder still saw real bits; it now applies the same
   substitution. `[c]` is still *consumed* on vacant slices, because the early boolean addition emits
   a fixed-stride slice per value and skipping would misalign it.

`CUT_FRAC_ADDER_SUPPORTED` names exactly which circuits implement the cut. The plain ripple-carry
circuit (`rca_msb_and_ab.hpp`) does NOT, so it is excluded rather than silently reading the sign at a
substituted slice; the reshared and a_known ripple-carry circuits do implement it and are unaffected.

**What this does and does not save.** Correctness and eligibility are now uniform. The realized
saving today is the A2B input sharing: FRACTIONAL of 32 slices are neither masked nor sent (about 16
percent of that step at FRACTIONAL=5). The adders still *evaluate* the substituted slices — their
gates now compute on constants. Skipping those gates (and their triples) is a further, purely
performance-oriented step: it is not a textual port between circuits, because each circuit has its
own triple layout and each skipped gate's consumer must still receive its designed mask (the reshared
circuits solve this with identity substitution plus per-consumer mask compensation, and their beaver
retrieval is hoisted into the constructor, so the allocation must be adjusted in step). Expected
additional saving, extrapolating from the reshared circuits: roughly 10-20 percent of adder triples
at FRACTIONAL=5, more at larger FRACTIONAL.

Validated at FRACTIONAL=5 (func53 8/8 each, plus LeNet): plain PPA +CUT (LeNet 90%), plain PPA4 +CUT,
bake PPA +CUT, bake PPA4 +CUT, bake RCA +CUT, and the reshare-sim PPA4 +CUT regression.


## All nine circuits (2026-07)

The cut is now implemented in every msb circuit, so `CUT_FRAC_ELIGIBLE` no longer depends on which
optimization family is active - only on the value-level precondition (`TRUNC_DELAYED == 0`, so the
adder input is already truncated and its top FRACTIONAL bits are sign extension) and the width.

| circuit | plain | reshared | a_known (bake) |
|---------|-------|----------|----------------|
| ripple-carry      | early stop + skipped triples | early stop + skipped triples | early stop + skipped triples |
| prefix            | tap moved | identity substitution + skipped reshares | tap moved |
| four-way prefix   | tap moved | identity substitution + skipped reshares | tap moved |

**Ripple-carry (all three flavours): full cut.** The carry chain stops after case
`31 - FRACTIONAL`, the sign is read at the boundary slice (`x[F] ^ y[F] ^ carry_{F+1}`), the last
executed gate's mask assign is redirected to the spare random so it does not reference a triple slot
beyond the reduced allocation, and beaver retrieval is made conditional to match - INIT counts per
CALLED gate, so retrieval that did not shrink in step would overrun the allocation. Measured for
k = 32: FRACTIONAL 5 -> 27 of 32 rounds and 26 of 31 triples (16 percent saved); FRACTIONAL 8 -> 24
rounds, 23 triples (26 percent); FRACTIONAL 12 -> 20 rounds, 19 triples (39 percent).

**Prefix and four-way prefix.** The reshared flavours already carried the full identity substitution
(skipping gates and reshares). The plain and a_known flavours now have the moved output tap, which is
the correctness-critical half, and are correct under the cut - but they still EVALUATE the
substituted slices on constants, so their gate-level saving is still to come. That step is a
per-circuit re-derivation, not a port: substituting `g_i := 0` loses the mask the gate was designed
to output, so every consumer needs compensation (this is why the reshared prefix circuit has ~157 cut
sites and why its boolean-triple retrieval stays unconditional - only reshares are saved there).

Validated at k = 32, func53: all nine circuits 8/8 at FRACTIONAL 5; FRACTIONAL 8 (plain ripple-carry,
plain prefix) 8/8; FRACTIONAL 12 (bake four-way prefix) 8/8; LeNet 90 percent with the plain prefix
circuit and the cut on. FRACTIONAL 12 with the plain ripple-carry circuit reports 7/8 (Conv+ReLU) BOTH
with the cut on and with it off - a pre-existing fixed-point precision limit at that setting (2 x
FRACTIONAL = 24 fractional bits leaves too little integer headroom for that test's accumulation in 32
bits), not a cut regression.


## Measured cost of the substituted slices (does "evaluated on constants" include communication?)

Yes. A beaver AND gate communicates regardless of its inputs: `prepare_and(b, assign, triple_c)`
lowers to `prepare_mult(...)`, which ends in `send_to_live(PNEXT, c.m)`, and `complete_and()` does
the matching receive. The parties cannot drop that exchange unless BOTH know to drop it - which is
exactly what the identity substitution plus consumer compensation encodes. So in the prefix circuits
that only have the moved tap, the substituted slices still consume their beaver triples AND still
perform their online send/receive.

Measured at FRACTIONAL = 5, thirty-two bit, TRUNC_DELAYED = 0, plain (non-reshared, non-a-known)
circuits. Boolean triples from the preprocessing requirement print (func53); online megabytes from
the aggregated per-layer network statistics (LeNet, one image, party 0).

| circuit | boolean triples off -> on | online MB sent off -> on |
|---------|---------------------------|--------------------------|
| ripple-carry    | 8928 -> 7488  (-16.1%) | 0.1099 -> 0.1018  (-7.4%) |
| prefix          | 24768 -> 24768 (unchanged) | 0.1548 -> 0.1507  (-2.6%) |
| four-way prefix | 4320 -> 4320  (unchanged) | - |  <!-- superseded: see the four-way update below (beaver3 6912 -> 6624 at F=5) -->

Reading the table: the ripple-carry cut removes the gates themselves, so both the preprocessing
material and the online traffic fall (and, more importantly on a high-latency link, five of
thirty-two communication ROUNDS disappear at FRACTIONAL = 5). The prefix circuits skip no gates yet -
their triple count is bit-for-bit identical - and the small online drop they do show comes only from
the A2B input sharing, where the vacant slices are never masked or sent. Closing that gap is the
gate-skipping work described above.

### Update: the plain prefix circuit now skips its gates too

The "unchanged" row for the prefix circuit above is out of date. `ppa_msb_unsafe_and_ab.hpp` now
rewrites every AND whose operands are constant into a local form, so those gates consume no beaver
triple and do no online communication:

| FRACTIONAL | boolean triples off -> on | func53 |
|-----------|----------------------------|--------|
| 5  | 24768 -> 21312 (-14%) | 8/8 |
| 8  | 24768 -> 18720 (-24%) | 8/8 |
| 12 | 24768 -> 15264 (-38%) | 7/8, and 7/8 with the cut OFF as well (precision limit, not the cut) |

LeNet over ten images at FRACTIONAL = 5 keeps its 90% baseline with 5600320 -> 4818880 triples and
online 1.016 -> 0.9751 MB.

The rewrite works because `zero_add` is local in the online phase: an AND against the all-ones
constant becomes the other operand retargeted to the gate's designed output mask, costing one
preprocessing delta instead of a triple and a round. Retargeting is not optional - a bare public
constant carries mask 0 and would break every consumer expecting the designed mask. Alongside each
skipped gate, `complete_and()` and the operand `zero_add`s are dropped and beaver retrieval is made
conditional, because INIT counts per called gate and so allocates one triple fewer; an assign naming
a skipped slot's mask reads a spare random, as that mask is no longer generated.

### Update: the plain four-way circuit now skips its gates too

`ppa_msb_4way_and_ab.hpp` carries the same treatment as its reshared sibling, ported gate for gate
(the two are generated from the same circuit, so the guard conditions transplant while the mask
expressions are rebuilt from the plain file's own masks). Beaver-3 tuples, against 6912 with the cut
off:

| FRACTIONAL | beaver3 tuples | func53 |
|-----------|----------------|--------|
| 5  | 6624 (-4.2%)  | 8/8 |
| 8  | 6336 (-8.3%)  | 8/8 |
| 10 | 5760 (-16.7%) | 8/8 |
| 11 | 5760          | 8/8 |
| 12 | 5760          | 7/8 - see below |

LeNet over ten images at FRACTIONAL = 5 holds its 90% baseline with beaver3 1562880 -> 1497760 and
online 1.146 -> 1.105 MB. Beaver-4 tuples are unchanged below FRACTIONAL = 22, where the only
skippable four-way slot starts.

The FRACTIONAL = 12 failure (ReLU on large magnitudes) is NOT the gate skipping: the unported file
fails the same test at FRACTIONAL = 12 with the cut on, so it comes from the pre-existing identity
substitution. Worth chasing separately - everything below 12 is clean.

Two differences from the reshared circuit are worth remembering, because each cost a debugging run.
The reshared circuit identity-substitutes the BOUNDARY leaf as well, so a block ending exactly on
slice FRACTIONAL is still all-identity there; this circuit keeps that leaf raw, since a[F] and b[F]
are the real wires feeding the output tap, so every threshold is one stricter. And `mask_and_send` is
skipped in every phase INCLUDING INIT even where the term itself stays counted - INIT accounts for the
send, so keeping it for INIT alone deadlocks the adder.

What is still on the table here is the arity reduction for MIXED gates: a four-input dot with one
constant operand could fold that operand locally with `mult_public` and run as a three-input gate,
saving 15 correlated values for 7. That is not implemented - the gate's tuple is allocated at its
original arity, so a reduced gate needs a spare lower-arity tuple, which means widening the beaver3
pool and retrieving the extras conditionally. Today a mixed gate simply runs at full arity.

### Update: the four-way a_ab circuit skips its b-product gates

`ppa_msb_4way_and_a_ab.hpp` is covered, and it is the best case of the three. In the a_ab
decomposition a wire is `a ^ b` with `a` known to the evaluators, so products expand into a-only
terms, mixed `a*b` terms - both local - and b-only products, and ONLY the b-products consume a beaver
tuple. The cut prepares a vacant slice as `a := all-ones, b := 0`, so every b-product touching a
vacant slice is identically zero and its gate disappears entirely.

Zero-ness is a threshold `z`, the wire being provably zero exactly when `FRACTIONAL >= z`: `b[i]` gives
`i+1`; `a[i]` and a-only products never die; a product takes the `min` (it dies if either factor dies);
an XOR takes the `max` (it dies only once both terms die). That is precisely `cut_frac_ppa4_skip(z)`.

| | baseline | with cut | |
|---|---|---|---|
| F=5  boolean triples | 9792 | 8352 (-14.7%) | 8/8 |
| F=5  beaver3         | 4320 | 3744 (-13.3%) | |
| F=8  boolean triples | 9792 | 7200 (-26.5%) | 8/8 |
| F=8  beaver3         | 4320 | 3456 (-20.0%) | |
| F=8  online MB       | 0.005296 | 0.004328 (-18.3%) | |

LeNet over ten images at FRACTIONAL = 5 keeps its 90% baseline with boolean triples
2214080 -> 1888480, beaver3 976800 -> 846560, and online **0.910 -> 0.8205 MB (-9.8%)**.

**Generic - no bound.** A wire can be provably zero for two different reasons, and only one is safe to
act on: directly, because a factor is a vacant INPUT slice, or indirectly, because a derived CHAIN
OUTPUT is zero (at FRACTIONAL >= 10 a whole block dies, so B3G_7_9_out really is zero and the level-1
gate consuming it looks cuttable). The second is true but not locally rewritable - a chain output is a
finalised wire carrying an accumulated mask from local carriers, and cutting its consumer corrupted
the heap rather than merely computing a wrong value. Gates are cut only when their zero-ness traces
back to the inputs without passing through a chain output; that drops 5 of 45 candidates and makes the
rule hold everywhere, so savings now GROW with FRACTIONAL:

| FRACTIONAL | boolean | beaver3 |
|---|---|---|
| 5  | 8352 (-14.7%) | 3744 (-13.3%) |
| 10 | 7200 (-26.5%) | 3456 (-20.0%) |
| 12 | 6336 (-35.3%) | 3168 (-26.7%) |

all 8/8, against a cut-off baseline of bool 9792 / beaver3 4320.

Two rules make this family safe, and both were learned by breaking it:

1. **Leave the local `mult_a_known` gates running.** Handed a share that is literally `(0, 0)` they
   return `(0, 0)` in whichever representation they produce, so nothing mixes standard with
   dot-pending shares. For the same reason `mask_and_send`/`complete` must ALWAYS run, even on a chain
   that has gone entirely zero: those local contributions are still dot-pending and still carry their
   own mask assigns, and the finalisation is what converts the accumulator to standard form and
   applies the accumulated mask. Skipping it looks like a free online saving and silently corrupts the
   wire.
2. **Compensate a cut gate's output mask along XOR-only paths.** It is NOT "every chain the gate can
   reach": a dot gate consumed through a `mult_a_known` gate has its mask replaced by that gate's own
   output mask, so it never reaches that chain, and compensating there corrupts it. Sharing is heavy
   (36 of 50 dot outputs have more than one consumer), so a gate genuinely can need compensating in
   two chains at once; the local `mult_a_known` carriers, which can never be cut, are the safe place
   to park the mask.

### Update: the a_ab prefix circuit

`ppa_msb_unsafe_and_a_ab.hpp` is covered too. Its tree gates are p-products, and a vacant p is
all-ones rather than zero, so the zero-threshold analysis finds nothing here - it needs the identity
treatment: an AND against all-ones is the other operand, retargeted to the designed output mask.

Boolean triples, against 15840 with the cut off: **13536 at FRACTIONAL=5 (-14.5%)**, **11808 at
FRACTIONAL=8 (-25.5%)**, both 8/8. LeNet over ten images at FRACTIONAL=5 keeps its 90% baseline with
3581600 -> 3060640 triples and online unchanged at 0.5111 MB. For reference the unmodified file at
CUT=1 was byte-identical to CUT=0 - the cut was previously a complete no-op for this circuit.

Two failures at 3/8 preceded this, both the same shape - a mask read from a slot whose gate no longer
exists. Every `prepare_remask` takes its mask from a triple field and all 15 of its slots are also
AND-gate slots, so freeing an AND's triple left its remask reading an unallocated field: the
freed-slot fallback now sweeps EVERY statement, not just gate assigns and zero_adds. And retargeting
must read the OPERAND, never the wire it was retargeted from, since sources here can still be
dot-pending ahead of their remask.

Keeping every mask carrier alive to make that safe cost more than it saved - each cut gate then traded
one AND for a net extra preprocessing delta and LeNet's online traffic rose to 0.5763 MB. Carriers are
now dropped per OPERAND: with the branch order both-const, A-const, B-const, else, operand A is only
read in the B-const branch, which is taken only when A is not constant, so A's carrier can go exactly
when `FRACTIONAL > (A's highest slice)`.

### Update: arity reduction

Implemented for the four-way circuit. A gate holding an all-ones operand folds it away and runs one
arity lower on a reserved tuple: `dot4 -> dot3` trades 15 correlated values for 7, `dot3 -> dot` and
`and3 -> and` trade 7 for 3.

Reservation is straightforward - `FRACTIONAL` is compile-time and INIT counts per CALLED gate, so a
circuit calling `prepare_dot3` where it called `prepare_dot4` shifts the pool counts by itself; the
arrays get a compile-time overhang and the reserved slots are retrieved under the same predicate that
selects the reduced branch. The reduced gate re-`zero_add`s its operands from their sources onto the
new tuple's fields; the original carriers and slot are skipped on the same predicate.

The catch is that a reduction is only real inside a **window**. Below the first threshold nothing is
constant; at or above the term's own zero threshold the cut removes the gate outright, and because the
dot-term cuts deliberately keep INIT counting, INIT still runs the ORIGINAL gate there. Reserving
across that whole range retrieves tuples nobody consumes - which appeared as the documented "only the
last test fails" signature. Every reduction predicate is therefore `>= TH_RED && < TH_ZERO`, on the
gate, the freed slot, the carriers and the reserved-slot retrieval alike.

Measured at FRACTIONAL=5 against the cut without reduction (bool 4320, beaver3 6624, beaver4 3744):
bool 4896, beaver3 6336, beaver4 3456 - per adder one beaver4 and one beaver3 against two boolean
triples, so `15 + 7 - 2*3 = 16` correlated values saved per adder, 8/8. LeNet at FRACTIONAL=5 holds
90% with beaver3 1497760 -> 1432640, beaver4 846560 -> 781440, boolean 976800 -> 1107040, online
1.105 -> 1.081 MB.

**Multi-level.** A gate can hold several all-ones operands at once, so it emits one branch per
reduction LEVEL: with the thresholds sorted `t1 <= t2 <= ...`, level `j` covers `[t_j, t_j+1)` and
folds away the first `j` operands, the last window closed by the term's own zero threshold. The
windows tile that range and are disjoint, so at most one level is live for a given FRACTIONAL and only
the live one's reserved slot is retrieved.

Two limits are deliberate. A dot term never degenerates below a real dot gate: its result must stay in
dot-pending form for the chain accumulation, and a single surviving wire is standard. An `and` gate's
output IS standard, so its last level is a pure retarget of the single survivor - no tuple, no
communication - and since that level prepares nothing, its `complete_and` guard takes the extra window
too.

Gains appear wherever a 3-slice block has two or more vacant p-operands, i.e. FRACTIONAL one short of
a block boundary, against single-level reduction:

| FRACTIONAL | beaver3 | |
|---|---|---|
| 6 | 6624 -> 6336, online 0.006020 -> 0.005948 MB | 8/8 |
| 9 | 6336 -> 6048 | 8/8 |
| 5 | unchanged - no gate has two vacant p-operands there | 8/8 |

At FRACTIONAL=6 the levels net out exactly as designed: the `and3` degenerates to a bare wire, giving
back a boolean triple, while the `dot4` drops to a two-input dot and spends one - so the boolean count
is flat and a beaver3 disappears. LeNet at FRACTIONAL=5 is identical to single-level (90%, beaver3
1432640, beaver4 781440, online 1.081 MB), as expected at that FRACTIONAL.

### Reshared circuits

`ppa_msb_4way_and_ab_reshared.hpp` now carries the arity reduction too. One convention differs and
matters: it identity-substitutes the BOUNDARY leaf as well, so `p_i` is all-ones once `i <= FRACTIONAL`
where the plain circuit needs `i < FRACTIONAL` - all-ones thresholds are one lower here, the same
one-off the cut thresholds carry. Operands passed raw (no zero_add carrier) are their own source, which
this circuit relies on more than the plain one. At FRACTIONAL=5, RESHARE_OPT=1: beaver3 6912 -> 6624
(cut) -> 6336 (cut + arity), online 0.005516 -> 0.005012 MB, 8/8.

`ppa_msb_unsafe_and_ab_reshared.hpp` (the reshared PPA) now cuts its boolean-triple gates as well:
15840 -> 14688 at FRACTIONAL=5 (-7.3%) and -> 12672 at FRACTIONAL=12 (-20%), 8/8, on top of the random
multiplications it already saved (8928 -> 7488). Two parser gaps had hidden the whole circuit from the
analysis: it writes its identity substitution as a MULTI-LINE ternary (statements are now assembled
logically, accumulating until the parens balance and the text ends in ';'), and the freed-slot
fallback regex matched the TAIL of `random_triples[2].b` and rewrote it to `random_(...)` - it now
requires a name boundary, which also protects `beaver3_tuples[N]` in the other pass. 25 of 39
candidate gates are classified; the rest keep unresolved leaves and are left alone.

The `_split` variants (`ADDITIONAL_PPA_THREADS > 0`) are NOT covered, and the blocker is not the cut:
that path fails on its own. Pristine `ppa_msb_4way_and_a_ab_split.hpp` scores 3/8 at FRACTIONAL=5 with
CUT_FRACTIONAL_BITS_OPT=**0**, and the reshared split scores 7/8 with the cut on - identical to the
scores with the transform applied. Extending the cut there is pointless until the threaded adder path
itself is fixed, and it cannot be validated in the meantime, so the transforms were not committed.

### ResNet50 validation (10 images, CIFAR-10, FUNCTION_IDENTIFIER=171, FRACTIONAL=5)

Use the MATCHED model/dataset pair - `ResNet50_avg_CIFAR-10_standard_best.bin` with
`CIFAR-10_standard_test_images.bin`, i.e. `resnet50_env.sh` unmodified. Pairing the `custom` weights
with the `standard` images (or vice versa) silently yields chance accuracy.

| config | accuracy | boolean triples | online |
|---|---|---|---|
| RCA, no optimizations, cut OFF | 70.00% | 60789760 | 8.008 MB |
| RCA, no optimizations, cut ON  | **80.00%** | 50984960 (-16.1%) | 6.782 MB (-15.3%) |
| PPA4, no optimizations, cut OFF | 80.00% | 29414400 | 32.77 MB |

The cut is correct at ResNet50 scale and delivers its usual savings; the 70 -> 80% difference is one
image out of ten, i.e. noise at this sample size, not an improvement.

**The A_KNOWN_TO_EVALUATORS_OPT (a_ab) adder path breaks ResNet50, and it is not the cut and not the
bake.** Full bisect at FRACTIONAL=5 with matched weights:

| config | accuracy |
|---|---|
| RCA, no optimizations, cut off / on | 70% / 80% |
| PPA4, no optimizations, cut off | 80% |
| a_ab adder ONLY - `A_KNOWN_TO_EVALUATORS_OPT=1`, online-opt OFF, bake OFF, **cut OFF** | **10%** |
| a_ab + `A2B_ONLINE_OPT=1`, bake off, cut on | 0% |
| a_ab + `A2B_ONLINE_OPT=1` + `A2B_CONV_BAKE=1`, cut off / on | 10% / 10% |

The third row is the root cause: with every optimization in this document disabled and the bake
disabled, simply selecting the a_ab adder drops ResNet50 to chance. Disabling the bake alone does NOT
recover it, so the conv-mask bake is a red herring here.

func53 passes 8/8 and LeNet holds 90% on the same a_ab path, so whatever fails needs ResNet50's
structure - BatchNorm, depth, or the residual adds - to appear. Worth noting for whoever picks this
up: the bake compensates for exactly ONE post-GEMM operation, `add_bias` via `g_bake_bias_l`. Nothing
invalidates a committed mask when BatchNorm or a skip-add changes it between the conv and the ReLU,
which is a real gap in that design even though it is not what this bisect is pointing at.

### Full adder x optimization matrix (FRACTIONAL=5)

Every combination of the three adders with the three optimization modes, each run with the cut ON and
OFF. `plain` = no A2B/reshare optimizations; `a2b` = `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1
A2B_CONV_BAKE=1` (the first two must be enabled together); `reshared` = `RESHARE_OPT=1`.

func53 unit tests: **8/8 in all 18 runs**. LeNet, 10 images, matched standard/standard pair:
**100.00% in all 18 runs**. So the cut is correct, and accuracy-neutral, in all nine configurations.

LeNet preprocessing / online, cut off -> on:

| adder | mode | boolean triples | online MB |
|---|---|---|---|
| RCA  | plain    | 2018720 -> 1693120 (-16.1%) | 0.3239 -> 0.2832 (-12.6%) |
| RCA  | a2b      | 1953600 -> 1628000 (-16.7%) | 0.3076 -> 0.2669 (-13.2%) |
| RCA  | reshared | 1953600 -> 1628000 (-16.7%) | 0.3158 -> 0.2751 (-12.9%) |
| PPA  | plain    | 5600320 -> 4818880 (-14.0%) | 1.016 -> 0.9751 (-4.0%) |
| PPA  | a2b      | 3581600 -> 3060640 (-14.5%) | 0.5111 unchanged |
| PPA  | reshared | 3581600 -> 3321120 (-7.3%)  | 0.7635 -> 0.7228 (-5.3%) |
| PPA4 | plain    | 976800 -> 1107040 (see note) | 1.146 -> 1.081 (-5.7%) |
| PPA4 | a2b      | 2214080 -> 1888480 (-14.7%) | 0.910 -> 0.8367 (-8.1%) |
| PPA4 | reshared | 260480 unchanged             | 1.057 -> 0.9263 (-12.4%) |

Two rows need reading carefully. **PPA4 plain is the only cell where boolean triples go UP** - that is
the arity reduction trading one beaver4 (15 correlated values) and one beaver3 (7) for two boolean
triples (3 each) per adder, a net -16 values; judging it on the boolean count alone reads as a
regression. **The reshared circuits spend random multiplications** where the others spend triples, so
their win shows up there instead (func53: PPA 8928 -> 7488, PPA4 3168 -> 2592).

**Model/dataset pairing.** MNIST and CIFAR-10 each ship `standard` and `custom` preprocessed variants
for BOTH the weights and the test set, and they must match. Pairing `LeNet5_MNIST_custom_best.bin`
with `MNIST_standard_test_images.bin` costs one image in ten on some configurations (90% instead of
100%) - verified by running the same build both ways. On ResNet50 the same mismatch is catastrophic
rather than marginal (chance accuracy). Use the env scripts unmodified.

### BITLENGTH != 32 is blocked below the cut

Neither BITLENGTH=16 nor BITLENGTH=64 builds in this repo state, and the cut is not involved. Both
fail identically in `core/generate_beaver_tiples.hpp` with `cannot convert uint16_t* / uint64_t* to
const Iface::uintNN_t*`: the Cheetah preprocessing interface types its buffers as
`std::conditional_t<BIT_LEN == 32, uint32_t, uint64_t>` where `BIT_LEN` comes from `TRIPLE_BITLEN`,
and `nn/ConvTriple` is linked as a PREBUILT shared library (`-L nn/ConvTriple/build/lib`) compiled for
32 bits. The top-level Makefile never passes `TRIPLE_BITLEN`.

So the k=8/16 circuit specializations could be given the same treatment, but no non-32 configuration
can be built or run until ConvTriple is rebuilt for the target width - the result would be
unvalidatable, which is why it has not been written. The prerequisite is rebuilding that library with
`-DTRIPLE_BITLEN=<width>` and plumbing the flag through the top-level Makefile.

### Remaining

The `ppa_msb_unsafe_and_a_ab.hpp` notes below predate the coverage above. It takes its leaves
from `mult_a_known_to_evaluators` followed by `prepare_remask`/`complete_remask`, so those wires are
dot-pending rather than standard until remasked; substituting standard-form constants there mixes the
two share representations and the circuit produces wrong results (3/8, though the triple count does
fall 15840 -> 13536). The four-way circuits use `prepare_dot3`/`prepare_dot4`/`prepare_and3`/
`prepare_and4`, which the rewrite does not handle at all - and a three-input dot with one constant
operand collapses to a two-input AND needing a boolean triple that was never allocated for it.

Note that `MB SENT PRE` barely moves in either case (1.63115 -> 1.63117 for the ripple-carry cut):
under `ROT_PREPROCESSING_OPT` boolean triples are generated very cheaply - the per-type breakdown
shows the boolean material at roughly 6e-5 MB against about 1.0 MB for the arithmetic triples - so
the preprocessing byte total is dominated by other material and is not a useful indicator of this
optimization. Count triples and rounds instead.
