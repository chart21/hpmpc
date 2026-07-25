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

`ppa_msb_unsafe_and_a_ab.hpp` is still NOT covered. It takes its leaves
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
