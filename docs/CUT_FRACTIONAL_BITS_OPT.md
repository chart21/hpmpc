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
