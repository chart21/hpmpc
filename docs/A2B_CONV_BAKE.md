# A2B_CONV_BAKE — baking A2B masks into the conv/FC output

Flag: `A2B_CONV_BAKE` (default 0). Active only when
`A2B_ONLINE_OPT == 1 && A2B_CONV_BAKE == 1 && DATTYPE == BITLENGTH`
(macro `A2B_CONV_BAKE_ACTIVE`, defined in `protocols/beaver_triples.hpp`).

Its purpose is to make the **online-optimized A2B** (`A2B_ONLINE_OPT=1`) correct, in particular
together with the "a known to evaluators" msb adders (`A_KNOWN_TO_EVALUATORS_OPT=1`, which requires
`A2B_ONLINE_OPT=1`). This combination implies `RESHARE_OPT=0` and `RESHARE_OPT_SIM=0`.

See `docs/STATUS.md` for the validation matrix and the TRUNC_DELAYED=1 unit-test artifact.

## Background: how A2B_ONLINE_OPT is supposed to work

For a ReLU we need the sign (msb) of an arithmetic value `v`. ABY2 shares a value as a public
masked value `mv = v + lv` (`lv = lv0 + lv1` the joint mask) plus per-party mask shares. The A2B
runs an msb adder over two boolean inputs:

- `s1 = bool(mv)` — public, both parties hold the same bits, `l = 0`.
- `s2 = bool(-lv)` — secret; the adder computes `bool(mv) ⊞ bool(-lv) = bool(v)` and extracts the msb.

`A2B_ONLINE_OPT` precomputes `s2 = [c] = bool(-lv)` in preprocessing via an **interactive boolean
addition** of each party's `bool(-lv_i)`, so the online phase does no A2B communication.

## The two bugs the bake fixes

1. **Conv-mask desync.** `s1` uses the LIVE mask/send's `mv = v + lv_live`, while `[c]` was built in
   PRE from `lv_pre`. They only cancel if `lv_live == lv_pre`; `A2B_ONLINE_OPT` alone relies on the
   `PSELF` PRNG staying byte-for-byte synced across the two passes, which the conv machinery can break.
2. **Adder beaver-triple mismatch (the real blocker).** The msb adder's beaver triples are generated
   in PRE from the `s2` wire mask (`out.l` of `prepare_A2B_S2`). The unbaked code set PRE
   `out.l = ia` (the boolean-adder *input*) but LIVE `out.l = [c]` (the *output*) — triples generated
   for the wrong mask, garbage msb even with `[c]` and `s1` individually correct.

## The construction

Per party, **before either FUNCTION pass** (same generation stage as the LXLY triples):

1. Draw a random boolean A2B-mask `ia` (`getRandomVal(PSELF)` with the counter saved/restored, so the
   function's own stream stays PRE↔LIVE synced).
2. Derive the conv mask `lz = -untranspose(ia)`; `real_ortho` is self-inverse, so `ortho(-lz) == ia`.
   Under `TRUNC_DELAYED=0` with `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1`, P1's committed mask is
   constrained to the **logical**-truncation image (top FRACTIONAL bits ZERO — `FUNC_TRUNC` is
   `OP_SHIFT_LOG_RIGHT` under `SKIP_PRE=0`) and `ia1` re-derived, since P1's mask is realized as
   `TRUNC(-r1)` there.
3. Run the boolean addition **early**: `[c] = ia0 ⊞ ia1 = bool(-(lz0+lz1)) = bool(-lv)`.

During both FUNCTION passes: every baked conv/FC mask/send emits `lz` (`a2b_bake_conv_mask`,
**index-addressed**: `g_a2b_lz[g_a2b_layer_base + g_bake_batch_offset + e]`, correct for the FC's
linear and the conv's tiled call order), and every A2B-S2 slice emits `[c]` (`a2b_bake_get_c`) — the
same values in both phases, so the mask cancels and the PRE-generated triples match LIVE.
`get_msb_range` snaps `g_a2b_layer_base = g_a2b_c_cursor` at the end (A2B groups are BITLENGTH-value
padded). Out-of-range reads mean "this output never feeds an A2B" (e.g. the final FC before the
reveal) and fall back to a fresh synced-PRNG draw — a committed constant there would make P1's
`r1 = -low` tiny and break the SecureML truncation wrap condition (`B >= |v|`), turning every
negative logit into `+2^(K-F)`.

## How each producer realizes the committed mask

- **Conv/FC with MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1** (a_known_pre paths): P0's mask is a free
  draw and is committed directly; P1's mask is derived from its conv/FC triple share `r1`, so `r1`
  is chosen to match (`r1 = -lz1` under TRUNC_DELAYED=1; `r1 = -((m1<<F)+low)` under TRUNC_DELAYED=0
  so `l1 = TRUNC(-r1) == m1`) and the triple generation forces P1's share to it (`mwk_fix_p1_share`).
- **Conv/FC with MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0 or A_KNOWN=0** (symmetric
  `mask_and_send_dot_with[out]_trunc_with_triple` paths): BOTH parties' masks are free draws
  (truncation applies to the masked share, the mask enters linearly), so both emit the committed
  `lz` at every indexed/`_baked` GEMM call site.
- **Bias**: `add_bias` shifts the output mask by the party's bias-mask share. The conv/FC forwards
  publish the effective bias mask (`g_bake_bias_l`, shared with the RESHARE_OPT_SIM bake) and
  `a2b_bake_conv_mask` subtracts it, so the mask after the bias addition is the committed `lz`.
- **Everything else** - a residual sum, a BatchNorm output, a public-weight layer (which multiplies
  locally and has no fresh mask), a max/min comparison difference - reaches the A2B with some other
  mask. The network marks each ReLU whose input comes straight from a conv/FC (`ReLU::input_baked`,
  set in `SimpleNN::compile` / `ResNet::compile`; max/min clears it for comparisons), and for every
  other input `get_msb_range` first moves the value onto its committed mask with `rebase`: each party
  sends `lz_i - l_i` in preprocessing and adds both deltas to `m` online. It costs one preprocessing
  message per value and party, and nothing online.

### Why the rebase, and what it costs (2026-09-30)

`[c]` must exist before the PRE pass: the msb adder's triples are requested during that single non-interactive sweep
from its input masks (`[c]` among them), and the sweep cannot host the Boolean addition's 31 interactive rounds. Doing
the addition after the sweep is the unbaked configuration, whose triples then belong to the wrong mask (bug 2 above; the
triad all_opt A2bits configs as given classify 0-2 of 10). So every ReLU input's mask has to be committed before the
sweep. A conv/FC with secret weights re-masks its output with a mask each party draws, so it draws the committed `lz`
(free). Every other producer fixes the mask itself (public-weight conv: `W * l_in`; residual sum `l_a + l_b`; pooling;
the BatchNorm layers here), so `rebase` sends `lz_i - l_i` per value and party in preprocessing: a one-time pad of `l_i`
(`lz_i` is fresh and private), 4 B each, nothing online.

CHEETAH ResNet50 (FUNCTION_IDENTIFIER 87/187/287) has 9,006,592 ReLU inputs; 7,300,608 come straight from a conv (BN
fused). UC1 / UC2 rebase the 1,705,984 after the stem's pooling and the four residual additions (13 MiB); UC3 (public
weights) rebases all 9,006,592: 2 x 4 B x 9.0 M = 69 MiB, the measured difference to the (wrong) as-given build.
Avoiding it would need the actual masks before the Boolean addition, i.e. an extra mask-only pass (estimated at the
PRE pass's 0.3-0.5 s single-threaded) to save 22 ms of transfer at 25 Gbit/s.

The one invariant all of this relies on: the counting (INIT) pass must not draw from the PRNG,
because the PRNG is reseeded only after preprocessing and every PRE mask would otherwise be shifted
against LIVE. `a2b_bake_conv_mask` returns 0 there.

## CUT_FRACTIONAL_BITS_OPT under the bake

`CUT_FRAC_ELIGIBLE` does not depend on the optimization family. Under the bake the a_ab circuits cut
like the others: the ripple-carry one stops at the boundary slice, the prefix one skips its identity
gates (15840 -> 13536 boolean triples at FRACTIONAL=5), and the four-way one skips its b-product gates
(see `docs/CUT_FRACTIONAL_BITS_OPT.md`). `[c]` is still consumed on vacant slices, because the early
boolean addition emits a fixed-stride slice per value.

## A_KNOWN=0 baseline fixes (needed before the bake could run there)

1. `GEMM.hpp`: the tiled conv sends retrieve their lxly INDEXED (cursor + index, no advance) for any
   A_KNOWN, but the post-layer cursor bump was gated `A_KNOWN == 1` — everything after the first conv
   read shifted values. Guard corrected.
2. First-layer SecureML wrap: the raw data-owner input (`m = 0` under `SHARE_PREP=1`) makes the
   truncation share pair the bare layer-triple c-shares, whose integer sum systematically wraps on
   negative outputs (`+2^(K-F)` each). `remask_range` (GEMM.hpp) gives the first layer's input split
   masks with a `rebase` onto fresh random masks - preprocessing only.
3. BatchNorm dot dispatch: used the a_known accumulation unconditionally under `BN2D_TRIPLES`; now
   dispatches to `prepare_dot_ex_lxly` for the AB-flavored triples.

## Known limitations

- The plain boolean AND path (`CaseAND`, basic-primitives test) fails at baseline under every
  configuration — pre-existing, independent of the bake, unused by the conv/pool suite and LeNet.

## PPA4 under the bake — what was wrong and how it was fixed

`ppa_msb_4way_and_a_ab.hpp` is only reachable via `A_KNOWN_TO_EVALUATORS_OPT=1`, which requires
`A2B_ONLINE_OPT=1` — a combination that never worked before the bake, so this generated circuit had
never actually been executed. It carried three independent defects, all now repaired (the repairs are
scripted and re-verifiable; see the checkers described below).

**1. Use-before-assignment ordering (42 + 83 sites).** Statements consumed wires before the
statements that computed them, so those gates ran on default-constructed shares. Repaired by a
dependency-preserving topological reorder inside each round (flow/anti/output edges, with the anti
edges of exactly the pairs being repaired suppressed so the graph stays acyclic; stable order
otherwise). Originally applied to k=32 only; now applied to k=8/16 as well — which also corrected
their operand classification for defect 2.

**2. Representation mismatch (212 sites) — the reason the msb was silently wrong.** Two share forms
coexist in these circuits:

  - *standard*: `m` identical on both parties, `v = m (+) l_0 (+) l_1`;
  - *dot-pending*: `m` additive between the parties (`v = m_0 (+) m_1`), `l_i` being the mask the
    finished wire will carry. `prepare_dot*/prepare_and*` produce this form; a chain of them is
    finalized by one `mask_and_send_dot_without_remask` + `complete_and`.

The generator replaced triple-based ANDs (dot-pending) with `mult_a_known_to_evaluators`, which
returns a **standard** share — a drop-in that silently changed representation. XORing a standard
share into a pending chain is wrong twice over: its common `m` cancels between the parties at
completion (`X (+) X == 0`, so the term vanishes entirely), and its value-dependent mask
(`a_pub & l_b`) pollutes the chain's output mask, which PRE cannot predict (PRE's a-known mult
returns an *unset* share). Fixed by emitting those products directly in pending form via new
primitives `mult_a_known_to_evaluators_dot` (standard operand: P0 carries the public part, each party
its own mask part — mirroring how `prepare_dot` carries `mx*my` on P0 only) and
`..._dot_pending` (already-pending operand: scale the pending halves only), both contributing a
**zero** mask. Products that only serve as public multipliers keep the original call.

**3. Chain output masks (29 sites).** Each operand of a `prepare_dotN_and_assign` must carry exactly
the corresponding beaver-tuple field; the generator instead left placeholder randoms (or, for chains
built purely from local products, no mask at all). Since a pending term's mask does not affect the
value it contributes, the correction is free: assign-carrying overloads of the two new primitives let
one designated chain member carry `required (+) current`, so the finished wire has the mask its
consumer requires.

Two independent checkers were used and are worth re-running after any regeneration: a representation
classifier (counts mixed standard/pending XORs — must be 0) and a symbolic mask checker (expands
nested `FUNC_XOR` and compares each dot operand's accumulated mask against the tuple field it is used
with — must be 0). Both were validated against `ppa_msb_4way_and_ab.hpp`, which reports 0 on each.
