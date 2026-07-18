# A2B_CONV_BAKE — baking A2B masks into the conv/remask output

Flag: `A2B_CONV_BAKE` (default 0). Active only when
`A2B_ONLINE_OPT == 1 && A2B_CONV_BAKE == 1 && DATTYPE == BITLENGTH`
(macro `A2B_CONV_BAKE_ACTIVE`, defined in `protocols/beaver_triples.hpp`).

Its purpose is to make the **online-optimized A2B** (`A2B_ONLINE_OPT=1`) actually correct, in
particular together with the "a known to evaluators" msb adders
(`A_KNOWN_TO_EVALUATORS_OPT=1`, which *requires* `A2B_ONLINE_OPT=1`). This combination implies
`RESHARE_OPT=0` and `RESHARE_OPT_SIM=0`.

## Background: how A2B_ONLINE_OPT is supposed to work

For a ReLU we need the sign (msb) of an arithmetic value `v`. ABY2 shares a value as a public
masked value `mv = v + lv` (`lv = lv0 + lv1` the joint mask) plus per-party mask shares. The A2B
runs an msb adder over two boolean inputs:

- `s1 = bool(mv)` — public, both parties hold the same bits, `l = 0`.
- `s2 = bool(-lv)` — secret; the adder computes `bool(mv) ⊞ bool(-lv) = bool(v)` and extracts the msb.

`A2B_ONLINE_OPT` precomputes `s2 = [c] = bool(-lv)` in preprocessing via an **interactive boolean
addition** of each party's `bool(-lv_i)`, so the online phase does no A2B communication — it just
reads `s1` from the public `mv` and `s2` from `[c]`.

## The two bugs this flag fixes

`A2B_ONLINE_OPT` on its own is broken, for two independent reasons. `A2B_CONV_BAKE` fixes both.

1. **Conv-mask desync.** `s1` uses the LIVE mask/send's `mv = v + lv_live`, while `[c]` was built in
   PRE from `lv_pre`. They only cancel if `lv_live == lv_pre`. `A2B_ONLINE_OPT` alone relies on the
   `PSELF` PRNG staying byte-for-byte synced across the PRE and LIVE passes, which the conv/adder can
   break → garbage msb.

2. **Adder beaver-triple mismatch (the real blocker).** The msb adder's beaver triples are generated
   in PRE from the `s2` wire mask (`out.l` of `prepare_A2B_S2`). The unbaked code set PRE `out.l = ia`
   (the boolean-adder *input*, `bool(-lv_i)`) but LIVE `out.l = [c]` (the *output*, a share of
   `bool(-lv)`). Those are different values, so the triples are generated for the wrong mask and the
   adder produces garbage — **even though `[c]` and `s1` are individually correct** (verified: `[c]`
   un-transposed equals `-lv` for all 32/32 lanes; `s1 = bool(mv)` with `mv == v + lv` exactly).

## The construction

Per party, **before either FUNCTION pass** (same generation stage as the LXLY triples):

1. Draw a random boolean A2B-mask `ia` (`getRandomVal(PSELF)`, with the `PSELF` counter saved and
   restored so the function's own stream stays PRE↔LIVE synced).
2. Derive the conv mask `lz = -untranspose(ia)`. `real_ortho` (`unorthogonalize_boolean`) is
   self-inverse, so `ortho(-lz) == ia`.
3. Run the boolean addition **early**: `[c] = ia0 ⊞ ia1 = bool(-lz0) ⊞ bool(-lz1) = bool(-(lz0+lz1))
   = bool(-lv)`.

Then during both the PRE and LIVE FUNCTION passes:

- every baked conv mask/send emits `lz` (`a2b_bake_conv_mask` → `g_a2b_lz`), so PRE and LIVE agree on
  the mask;
- every A2B-S2 slice emits `[c]` (`a2b_bake_get_c` → `g_a2b_c`), the **same** value in both phases, so
  the adder's PRE triples match what LIVE consumes.

Because `[c] = bool(-lv)` is tied to the committed `lz` by construction (via `ia`), the mask cancels
and the triples line up simultaneously.

### Grouping / cursor realignment

The A2B packs `BITLENGTH` values per sint (`load_shares`) and consumes `BITLENGTH` `[c]` slices even
for a partial (`< BITLENGTH`) group, whereas the conv mask/send advances once per real value. To keep
the next layer's masks and `[c]` drawn from the same `ia` groups, `get_msb_range` snaps
`g_a2b_assign = g_a2b_c_cursor` at the end (identically in PRE and LIVE, so triples still match).

## Files (all gated on `A2B_CONV_BAKE_ACTIVE`)

- `protocols/beaver_triples.hpp`
  - `g_a2b_ia` / `g_a2b_lz` / `g_a2b_c` buffers; `g_a2b_assign` (conv-mask cursor) and
    `g_a2b_c_cursor` (A2B-S2 cursor).
  - `a2b_bake_conv_mask`, `a2b_bake_get_c` (early block, before the buffer decls).
  - `init_a2b_bake` (draw `ia`, derive `lz`, load boolean-addition inputs) and `a2b_bake_store_c`
    (capture `[c]`), placed after the `boolean_addition_triple_*` declarations they reference.
- `protocol_executer.hpp` (pre-PRE generation, ~L386): `init_a2b_bake` →
  `init_booleanAdditionBeaverC` → `generate_beaver_triples(..., "BOOLEANADDITION")` →
  `a2b_bake_store_c`; reset both cursors (also reset before the LIVE pass).
- `protocols/2-PC/aby2/aby2_pre.hpp` `prepare_A2B_S2` (A2B_ONLINE_OPT): under the bake,
  `out.l = a2b_bake_get_c()` (no `ia` recording). The late `complete_preprocessing` BOOLEANADDITION
  generation is guarded off under the bake (it ran early).
- `protocols/2-PC/aby2/aby2_online.hpp` `prepare_A2B_S2` (A2B_ONLINE_OPT): under the bake,
  `out.l = a2b_bake_get_c()`. `mask_and_send_dot_baked` emits `a2b_bake_conv_mask` when its
  `bake_index >= 0`.
- `programs/functions/share_conversion.hpp` `get_msb_range` end: cursor realignment.

## Status

- **Pure ReLU (remask input):** `func53` ReLU(large) passes with `PROTOCOL=4, MWK=0,
  TRUNC_DELAYED=1, RESHARE_OPT=0/SIM=0, A2B_ONLINE_OPT=1, A_KNOWN_TO_EVALUATORS_OPT=1,
  A2B_CONV_BAKE=1, F=5`.
- **Conv/FC output (step 1 done):** with `MWK=1, TRUNC_DELAYED=1` the full `func53` suite passes
  **6/8 — identical to the MWK=1 no-bake baseline** (every ReLU-feeding test passes: ReLU+AvgPool,
  ReLU(large), FC+ReLU, Conv+ReLU; the 2 "failures" are standalone Convolution/BatchNorm, which reveal
  untruncated output under TD=1, exactly as the baseline does). The conv/FC output mask is baked by
  prescribing P1's triple share `r1 = -lz1` (`mwk_choose_r1_no_trunc`) and P0's `l0 = lz0`
  (`a_known_pre_mask_send`), reusing the MWK=1 AB2P prescription. `a2b_bake_conv_mask` is
  **index-addressed** (`g_a2b_lz[g_a2b_layer_base + g_bake_batch_offset + e]`), so it works for the FC
  (linear order) AND the conv (tiled order), and standalone conv/BN layers (no following ReLU) don't
  desync — `g_a2b_layer_base` only advances (to the `[c]` group boundary) when an A2B runs.
- **No regression** to bake-off configs (all changes gated on `A2B_CONV_BAKE_ACTIVE`; RESHARE_OPT_SIM
  and MWK=1 baselines unchanged).

## Remaining: conv/FC output-mask bake (blocks FC+ReLU / Conv+ReLU and TRUNC_DELAYED=0)

`func53` FC+ReLU / Conv+ReLU fail under the bake (both TD=1 and TD=0) because the ReLU's A2B input
is the conv/FC **GEMM output**, whose mask is produced by `mask_and_send_dot_a_known_pre_with_triple`
(and `..._baked`), not the test `remask`. There:

- **P0** picks its mask freely (`l = getRandomVal(PSELF)`) — bakeable to `g_a2b_lz0` directly, like the
  remask.
- **P1** *derives* its mask from the conv/FC triple share `r1`: `l = -r1` (TRUNC_DELAYED=1, via
  `mwk_choose_r1_no_trunc`) or `l = TRUNC(-r1)` (TRUNC_DELAYED=0, via `mwk_choose_r1_trunc`). To bake
  P1's mask to `g_a2b_lz1`, `r1` must be **prescribed** so the derived `l` equals `g_a2b_lz1`. That is
  exactly the RESHARE_OPT_SIM machinery (`construct_mwk_r1_baked`, `mwk_choose_r1_*`, plus the matching
  prescribed conv/FC-triple generation) — just targeting `g_a2b_lz` instead of the reshare mask `rt.a`.
  Note A2B_CONV_BAKE implies RESHARE_OPT=0/SIM=0, so this needs a **parallel A2B version** of that path,
  not reuse of the (no-op) reshare bake.

### TRUNC_DELAYED=0 specifics (the truncation-aware part)

Under TD=0 the conv/FC output is SecureML-truncated by `FRACTIONAL` before the ReLU, so **only P1's
mask is truncated**: `l1 = TRUNC(-r1)`, while `l0` stays full. Consequences, exactly as the existing
`construct_mwk_r1_baked` shows:

- `-r1 := (l1_baked << FRACTIONAL) + low`, `low < 2^FRACTIONAL` free. The trunc image zeroes the top
  `FRACTIONAL` bits, so **only mask bits `0..K-FRACTIONAL-1` are realizable** — i.e. `ortho(Trunc(l1))`
  has its top `FRACTIONAL` slices FIXED (sign-extension). This is the "some bits are fixed" the design
  relies on; it is the same vacancy `CUT_FRACTIONAL_BITS_OPT` already removes from the msb adders.
- The early boolean addition that forms `[c] = ia0 ⊞ ia1` must therefore **cut its top `FRACTIONAL`
  bit-positions** (they are determined sign-extensions, not free) — reduce `num_bits_per_input` in
  `generateBooleanAdditionTriples` by `FRACTIONAL` for the P1/truncated contribution, analogous to the
  adder cut.
- This truncation is inherently **path-local** (conv/FC only). It must NOT be applied globally in
  `init_a2b_bake`, because the pure `remask` mask is full/untruncated — a global truncation-aware `ia`
  would break relu_large. So `init_a2b_bake` stays as-is for the full-mask paths, and the truncated
  P1-mask derivation happens at the conv/FC mask/send (mirroring `mwk_choose_r1_trunc`).

Implementation order: (1) conv/FC output bake TD=1 (prescribe `r1 = -g_a2b_lz1`, testable via Conv/FC+
ReLU) — **DONE**; (2) TD=0 add `TRUNC(-r1)` prescription with the top-`FRACTIONAL` fix + boolean-addition
cut — **partial (see below)**; (3) `MWK=1` — folded into step 1; (4) bias compensation.

### Step 2 (TRUNC_DELAYED=0) — DONE, `func53` 8/8

Implemented and **gated on `A2B_CONV_BAKE_ACTIVE && TRUNC_DELAYED == 0`**:
- `init_a2b_bake` (P1 only): the committed mask `m1` is constrained to the trunc **image**: its top
  `FRACTIONAL` bits are **ZEROED** (`m1 &= 2^(K-F)-1`), and `ia1 = bool(-m1)` is re-derived so
  `[c] = bool(-(lz0+m1))` still holds. P0's mask stays full.
- `mwk_choose_r1_trunc` (A2B branch): prescribes `r1 = -((m1 << F) + low)`, `low < 2^F` fresh, so P1's
  SecureML mask `l1 = TRUNC(-r1) == m1` exactly. P0's `with_trunc` mask baked to `lz0` in the online
  and PRE variants.

**The convention that matters:** `FUNC_TRUNC = OP_TRUNC = OP_SHIFT_LOG_RIGHT<FRACTIONAL>` under
`SKIP_PRE == 0` — a **logical** shift, so the trunc image has its top `FRACTIONAL` bits **zero**, not
sign-extended. The first step-2 attempt sign-extended `m1`; for every mask with bit `K-F-1` set the
committed and actual masks then differed by `2^(K-F)`, so the computed `v = v_true + 2^(K-F)` — small
negatives (top bits all 1) wrapped to reading positive at bit `K-1` while positives stayed positive,
which is exactly the observed "ReLU passes negatives through". Zero-extending fixes it: the K-bit
boolean addition `bool(mv) ⊞ [c]` is then exact, and in the SecureML good case the truncated value is
properly sign-extended, so the msb at bit `K-1` is the true sign. **`func53` `MWK=1 TD=0` passes 8/8**
(bias-carrying FC included). No cut needed for correctness; see below.

**CUT status on this path:** `CUT_FRAC_ELIGIBLE` requires `RESHARE_OPT == 1`, so with the bake
(`RESHARE_OPT = 0`) `CUT_FRACTIONAL_BITS_OPT=1` is inert — the a_ab (`A_KNOWN_TO_EVALUATORS_OPT`)
adders are not covered by the cut yet. Under TD=0 the bake makes P1's `ia1` top-`FRACTIONAL` slices
*determined* (`-m1` is 0 or has its top F bits all 1), so a follow-up optimization can cut those slices
from the early boolean addition and the a_ab adders together (they must agree on the slice count —
`[c]` is currently K slices/sint). Optimization only; correctness holds without it.

### LeNet end-to-end (step 4: bias — DONE; trailing-layer fallback)

`func53` at 8/8 was not yet sufficient for LeNet (initially 20% vs the 90% no-bake baseline on 10
images). Two further fixes, both verified layer-by-layer with `VERIFY_CORRECTNESS=1`:

1. **Bias pre-compensation** (`a2b_bake_conv_mask`). `add_bias` after the GEMM shifts the output mask
   by the party's bias-mask share (under `SHARE_PREP=1` an owner's shared value has `l = −val, m = 0`,
   the non-owner `l = 0`), so the ReLU's A2B saw mask `lz + l_b ≠` committed `lz` — the msb was
   computed on the *pre-bias* value. Fix: the conv/FC layer forwards publish the effective bias mask
   (`g_bake_bias_l`, the same buffer the RESHARE_OPT_SIM bake uses — gates widened to
   `|| A2B_CONV_BAKE_ACTIVE`), and `a2b_bake_conv_mask` subtracts the party's OWN share of it, so the
   total mask after `add_bias` equals the committed `lz`. P1 (non-owner) subtracts 0, preserving its
   TD=0 trunc-image constraint.
2. **Out-of-range fallback = "output never feeds an A2B"**. `g_a2b_lz` covers exactly the INIT-counted
   A2B slices, so a layer with no following ReLU (LeNet's final FC before the reveal) reads past the
   end. Returning a constant there made P1's `r1 = −low` tiny and broke the SecureML trunc wrap
   (`B ≥ |v|` fails) — every *negative* logit came out `+2^(K-F)` (observed: 9 of 10 logits at exactly
   `2^22` with `F=5`). Fix: out-of-range reads fall back to a fresh synced-PRNG draw (baseline
   behavior); `r1 = −((rand<<F)+low)` is uniform and still satisfies `l1 = TRUNC(−r1)` by construction.

**Results (10 MNIST images, `LeNet5_MNIST_custom_best.bin`, `MWK=1`):** bake = **90% = no-bake
baseline**, for BOTH `TRUNC_DELAYED=0` and `TRUNC_DELAYED=1`. `func53`: TD=0 8/8, TD=1 6/8 (= baseline).

### Known limitation: comparisons on unbaked masks (MaxPool / argmax in MPC)

Under the bake, **every** `prepare_A2B_S2` consumes committed `[c]`. For A2B inputs whose masks were
never baked — MaxPool comparison differences, `COMPUTE_ARGMAX=1` — the msb shares are *consistent but
wrong* (valid shares of a wrong bit): selections pick a wrong candidate rather than producing garbage,
which `func53`'s MaxPool test cannot detect (epsilon 0.8 over values 0.05–0.9). LeNet avoids this
(AvgPool only, argmax on revealed logits). Networks with MaxPool or in-MPC argmax need either their
comparison inputs re-masked through a baked path or a hybrid that keeps the original PRE-derived
boolean addition for unbaked A2Bs. Future work, alongside the CUT extension above.

## Support matrix (2026-07, branch dbg; func53 = PPA unit tests, LeNet = func182, 10 MNIST images)

Family flags — A2B bake: `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1 A2B_CONV_BAKE=1 RESHARE_OPT=0
RESHARE_OPT_SIM=0`; RESHARE+SIM: `RESHARE_OPT=1 RESHARE_OPT_SIM=1 CUT_FRACTIONAL_BITS_OPT=1
A2B_ONLINE_OPT=0`. Baseline (both families off, MWK=1 TD=0): func53 8/8, LeNet 90%.

| # | Family      | Setting                  | func53                         | LeNet |
|---|-------------|--------------------------|--------------------------------|-------|
| 1 | A2B bake    | MWK=0 A_KNOWN=1 TD=0     | 6/8 (FC+ReLU, Conv+ReLU)       | 10%   |
| 2 | A2B bake    | MWK=1 A_KNOWN=1 TD=0     | **8/8**                        | **90%** |
| 3 | A2B bake    | PW=1 TD=1                | 3/8 (all ReLU paths)           | 0%    |
| 4 | A2B bake    | A_KNOWN=0 TD=0           | 0/8                            | 20%   |
| 5 | RESHARE+SIM | MWK=0 A_KNOWN=1 TD=0     | **8/8**                        | **90%** |
| 6 | RESHARE+SIM | MWK=1 A_KNOWN=1 TD=0     | **8/8**                        | **90%** |
| 7 | RESHARE+SIM | PW=1 TD=1 (auto SIM=0)   | 6/8 (Conv/BN: TD=1 reveals)    | **90%** |
| 8 | RESHARE+SIM | A_KNOWN=0 TD=0           | 0/8                            | 10%   |

Reading the failures:
- Row 1/3: the A2B bake requires the MWK=1 AB2P prescription for P1's conv/FC triple share; under
  MWK=0 or PW=1 the conv/FC (and under PW even the remask) outputs are unbaked -> the committed `[c]`
  doesn't match -> ReLU msb wrong. Design constraint, not a bug: the bake's supported cell is MWK=1.
- Row 7: PW=1 auto-falls back to SIM=0 (config.h); the 2 func53 fails are the standard TD=1
  untruncated Conv/BatchNorm reveals (same as every TD=1 baseline).
- Rows 4/8: CONTROL with both families OFF also gives 0/8 — A_KNOWN=0 is broken at baseline on this
  branch (the A_KNOWN=0 exact-truncation/dealer-triple work is not on dbg), so these rows say nothing
  about either optimization.

## MWK=0 support + genuine MaxPool (2026-07)

**MWK=0 conv/FC bake.** Under `MWK=0, A_KNOWN=1` the conv/FC outputs use the SYMMETRIC mask/send
(`mask_and_send_dot_with_trunc` via the `with_triple(index)` / `_baked` wrappers): each party's output
mask is a FREE `getRandomVal` draw (SecureML truncation applies to the masked share `m`, the mask `l`
enters linearly after it). So baking needs no share prescription at all — both parties emit their
committed `lz` (`a2b_bake_conv_mask`) at every indexed/`_baked` GEMM call site, in the online and PRE
variants. No trunc-image constraint on either mask (P1's committed zero-extension from the MWK path is
kept — merely a constrained but valid mask). Non-GEMM callers of these variants pass no index → default
`-1` → unaffected. Results: `func53` MWK=0 8/8 (TD=0), 6/8 (TD=1, = baseline); LeNet MWK=0 **90% = the
no-bake baseline for BOTH TD modes**.

**MaxPool under the bakes — test was vacuous, path was broken, both fixed.**
- The MaxPool unit test compared with the global epsilon 0.8 while all candidates lie within 0.8 of
  every window max — a wrong candidate pick could not fail it. It now compares against the fixed-point
  quantized expected with epsilon 0.01.
- The honest test exposed that MaxPool comparisons were broken under BOTH baking schemes (consistent-
  but-wrong msb → wrong candidate picks): the comparison differences carry unbaked masks. Fix in
  `max_min_msb_range` (gated `A2B_CONV_BAKE_ACTIVE || RESHARE_BAKE_ACTIVE`): re-mask the differences
  through the baked path (`prepare_dot(1)` + `mask_and_send_dot_baked(e)`) before `get_msb_range`, so
  the committed `[c]` (A2B bake) or the baked `rt.a` (SIM) lines up with the values.
- Under RESHARE_OPT_SIM one more piece was needed: the SIM bake's stride (`reshares_per_adder`) and
  slice→rt rank mapping are compile-time CUT-aware, but the cut is gated at runtime by
  `g_cut_frac_active`, which only the ReLU set — MaxPool's adders ran UNCUT (31 rts/adder vs the bake's
  26) and misaligned every group. `max_min_msb_range` now sets `g_cut_frac_active` around its
  `get_msb_range` like the ReLU does (the differences of post-trunc bounded values satisfy the same
  top-FRACTIONAL vacancy), which both fixes the alignment and saves the cut gates.
- Verified: func53 8/8 with the tight MaxPool test under A2B bake MWK=0/MWK=1, RESHARE+SIM MWK=0/MWK=1
  (+CUT), and the plain baseline. Remaining RB-MISMATCH reports are the documented benign padding-lane
  case (relu_large's 16 real lanes match, discarded padding lanes differ).
- Still open: `COMPUTE_ARGMAX=1` (argmax_argmin in MPC) has the same unbaked-comparison structure and
  needs the same remask + cut treatment if used under either baking scheme.
