# A2B_CONV_BAKE — baking A2B masks into the conv/FC output

Flag: `A2B_CONV_BAKE` (default 0). Active only when
`A2B_ONLINE_OPT == 1 && A2B_CONV_BAKE == 1 && DATTYPE == BITLENGTH`
(macro `A2B_CONV_BAKE_ACTIVE`, defined in `protocols/beaver_triples.hpp`).

Its purpose is to make the **online-optimized A2B** (`A2B_ONLINE_OPT=1`) correct, in particular
together with the "a known to evaluators" msb adders (`A_KNOWN_TO_EVALUATORS_OPT=1`, which requires
`A2B_ONLINE_OPT=1`). This combination implies `RESHARE_OPT=0` and `RESHARE_OPT_SIM=0`.

See `docs/RESHARE_OPT_SIM.md` for the explanation of the TRUNC_DELAYED=1 "6 of 8" unit-test artifact
and `docs/CUT_FRACTIONAL_BITS_OPT.md` for the adder x optimization validation matrix.

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

## How each weight setting realizes the committed mask

- **MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1** (a_known_pre paths): P0's mask is a free draw →
  committed directly; P1's mask is derived from its conv/FC triple share `r1`, so `r1` is
  **prescribed** (`r1 = -lz1` under TRUNC_DELAYED=1; `r1 = -((m1<<F)+low)` under TRUNC_DELAYED=0 so
  `l1 = TRUNC(-r1) == m1`) and the triple generation forces P1's share to it (the delta fix, or the
  AB2P prescription under `MWK_PRESCRIBED_HE`) — reusing the MWK machinery.
- **MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0** and **A_KNOWN=0** (symmetric
  `mask_and_send_dot_with[out]_trunc_with_triple` paths): BOTH parties' masks are free draws
  (truncation applies to the masked share, the mask enters linearly) → both emit the committed `lz`
  at every indexed/`_baked` GEMM call site. No prescription, no image constraint.
- **Bias**: `add_bias` shifts the output mask by the party's bias-mask share (owner `l = -val` under
  `SHARE_PREP=1`, non-owner 0). The conv/FC forwards publish the effective bias mask
  (`g_bake_bias_l`, shared with the RESHARE_OPT_SIM bake) and `a2b_bake_conv_mask` subtracts the
  party's own share, so the total mask after the bias addition equals the committed `lz`.

## MaxPool (and comparisons generally)

Comparison differences carry unbaked masks (linear combinations). Under the bake every A2B consumes
committed `[c]`, so `max_min_msb_range` re-masks the differences through the baked path
(`prepare_dot(1)` + `mask_and_send_dot_baked(e)`) before `get_msb_range` — gated
`A2B_CONV_BAKE_ACTIVE || RESHARE_BAKE_ACTIVE`, because the RESHARE_OPT_SIM bake has the same
requirement (its breakage had been hidden by a vacuous test epsilon; the MaxPool unit test now
compares against the quantized expected with epsilon 0.01). Under RESHARE_OPT_SIM the comparison
adders additionally run with the cut active (`g_cut_frac_active`, like the ReLU), because the SIM
bake's stride and slot mapping are compile-time cut-aware. `COMPUTE_ARGMAX=1` (argmax inside MPC) has the
same structure, but a real run shows it does not need the treatment: 8/8 and LeNet 100%, with the cut
enabled, in both the plain and the bake configurations.

## CUT_FRACTIONAL_BITS_OPT under the bake

`CUT_FRAC_ELIGIBLE` has a second leg: `A2B_ONLINE_OPT == 1 && A_KNOWN_TO_EVALUATORS_OPT == 1`.

- **RCA_MSB_A_AB (k=32): full cut** (ported from the reshared RCA): conditional triple retrieval
  (`i < 30 - FRACTIONAL`), per-case boundary shortcut (`msb = x[F] ^ y[F] ^ carry_{F+1}`),
  last-executed-gate mask-assign swapped to the spare random `r61`. Saves FRACTIONAL rounds and
  triples per adder.
- **PPA a_ab: full cut** — the identity treatment skips the vacant gates and their triples
  (15840 -> 13536 boolean triples at FRACTIONAL=5; see `docs/CUT_FRACTIONAL_BITS_OPT.md`).
- **PPA4 a_ab: FIXED** (was 3/8, now 8/8 + LeNet 90%). See "PPA4 under the bake" below.

## A_KNOWN=0 baseline fixes (needed before the bake could run there)

1. `GEMM.hpp`: the tiled conv sends retrieve their lxly INDEXED (cursor + index, no advance) for any
   A_KNOWN, but the post-layer cursor bump was gated `A_KNOWN == 1` — everything after the first conv
   read shifted values. Guard corrected.
2. First-layer SecureML wrap: the raw data-owner input (`m = 0` under `SHARE_PREP=1`) makes the
   truncation share pair the bare layer-triple c-shares, whose integer sum systematically wraps on
   negative outputs (`+2^(K-F)` each). `remask_range` (GEMM.hpp) re-randomizes the first layer's
   input in place (`is_first`, A_KNOWN=0-gated). The proper protocol fix (pre-truncated dealer
   triples) lives on another branch.
3. BatchNorm dot dispatch: used the a_known accumulation unconditionally under `BN2D_TRIPLES`; now
   dispatches to `prepare_dot_ex_lxly` for the AB-flavored triples.

## Known limitations

- **Residual networks (ResNet)**: every residual block's final ReLU consumes `conv3 + shortcut`,
  whose summed mask matches neither the committed `[c]` (bake) nor the baked reshare shares
  (RESHARE_OPT_SIM — measured on ResNet50: 809,600 of 1,409,440 reshare checks violated). Both
  families are unsupported on residual architectures until the MaxPool-style re-mask treatment is
  applied to non-conv-fed ReLU inputs. LeNet-style chains are fully supported.

- `PUBLIC_WEIGHTS=1` under the bake is unsupported (no mask is committed on the public-weight paths;
  use the RESHARE_OPT_SIM family, which auto-falls back to SIM=0 there).
- `COMPUTE_ARGMAX=1` — see the MaxPool section.
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
