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

CHEETAH ResNet50 (FUNCTION_IDENTIFIER 87/187/287) has 9,006,592 ReLU inputs in 49 ReLUs; 8,003,072 come straight
from a conv (BN fused). UC1 / UC2 rebased (before the residual bake below) the 1,003,520 after the stem's pooling (+ unfused BatchNorm, 200,704) and
after the one residual addition the layout keeps (stage 1, 802,816; the other three are commented out in
`Cheetah_ResNet`): 2 x 4 B x 1.0 M = 7.7 MiB (measured per ReLU, `DBGRELU`, 2026-09-30; an earlier version of this
paragraph said 1,705,984 / four residual additions / 13 MiB, which was wrong). UC3 (public weights) rebases all
9,006,592: 2 x 4 B x 9.0 M = 69 MiB, the measured difference to the (wrong) as-given build. UC3 no longer does:
see the mask-only forward below.

### Privacy fixes (2026-09-30)

Two ways in which committed randomness masked more than one value:

1. **Slot reuse (hpmpc d8a48ae, PIGEON d3e2948).** The committed masks are addressed as `g_a2b_layer_base + e`, and
   the base moves only at an A2B. Every conv/FC took them, so a conv whose output does not go straight into a baked
   ReLU used the slots of other values: the downsample conv and the conv after it (the same base) masked their
   outputs at equal positions with the same mask (the difference of two secret activations became public), and the
   stem conv's 802,816 outputs spilled into the next layers' slots. And the rebase `delta_i = lz_i - l_i` is a
   one-time pad only if `lz_i` masks nothing else: at the residual ReLU, `lz_i` had already masked the residual
   partner (conv #5), so `delta_i = -l_i(downsample)` - both parties learned the downsample conv's mask, and with its
   public masked value its output. Measured in the PRE pass (ImageNet, UC1 and UC2 A2bits): 1,505,280 slots used by
   two convs, all 1,003,520 rebased slots also used by a conv. Fix: `SimpleNN::mark_baked_relu_inputs` marks the
   conv/FC right before a baked ReLU (`bake_output`, not for a ReLU fused into max pooling); only those take committed
   masks (`g_conv_bake`), all others draw fresh ones, and the rebase reads the slot (`a2b_bake_slot_mask`). Afterwards:
   0 reused slots, 8,003,072 conv-used slots (exactly the baked ReLU inputs), rebases on unused slots only.
2. **Replayed generator (this commit).** `init_a2b_bake` drew `ia` with `getRandomVal(PSELF)` and restored the
   generator so that the passes stay in step. The generator runs AES on its own state (output feedback), so the
   passes then drew exactly these values again as their own masks: `lz = -untranspose(ia)` was a fixed function of
   mask shares the same party used elsewhere (conv, gate and bit-injection masks), whose masked values are public
   too. The committed values now come from the same key with the state XORed with a constant, an independent
   sequence.

Neither fix changes traffic or work; the A2bits UC1 / UC2 CIFAR checks classify 4-7 / 10 as before (new output
hashes, since the masks changed).

3. **Narrowed masks in UC1 (hpmpc 841335d, 2026-10-01).** `init_a2b_bake` zeroed the top FRACTIONAL bits of P1's
   committed masks whenever `TRUNC_DELAYED=0`. That is needed where P1's conv/FC mask is the truncation's image of its
   prescribed triple share (`l1 = TRUNC(-r1)`: weights known in preprocessing, `A_KNOWN=1`), and there it reveals
   nothing (P0 forms the masked value from its own share, `v' + r1` with `r1` uniform). With `A_KNOWN=0` (UC1) the
   masks are free draws and P1 sends `TRUNC(m_1) + l_1` for its share `m_1`: with `l_1 < 2^27`, P0 learns an
   interval constraint on `TRUNC(m_1)`, i.e. on the secret value shifted by a known offset (about one bit on average).
   Now only `A2B_P1_IMAGE_MASKS` (P1, `A_KNOWN=1`, `TRUNC_DELAYED=0`) narrows. Traffic and work unchanged; the UC1
   outputs do not depend on the masks (CIFAR hashes bit for bit as before).

### UC3: the mask-only forward (A2B_BAKE_MASK_PASS, default 1)

With public weights no producer can hit a committed mask: a conv's output mask is `W * lambda_in`, a linear function
of earlier bit-injection masks (choosing those so that `W * lambda_in = lz` is a linear system without a solution in
general). But every ReLU input mask is then a function of the input masks and the earlier ReLUs' output masks alone.
So the preprocessing pass first runs the network over the masks (`a2b_mask_forward`, from `SimpleNN::evaluate`):

* the ReLUs record their input masks at their A2B slots and output **committed bit-injection masks** (`g_relu_out`,
  counter-mode values under the party's key, by slot; since round 5 shared with `CHEETAH_CONV_EARLY`); the real passes'
  bit injections take the same ones (`bi_output_mask`, slot per element via `BiSlotScope`);
* all other layers run their normal preprocessing code; their truncations (pooling, delayed conv truncations, the
  data owner's first layer) take their masks from a counter-mode stream under the party's key that restarts with
  every forward (`lin_mask`), since a PSELF draw after a ReLU would differ between the two forwards;
* pre-sends are dropped (`pre_send_to_live`), the triple-type index and the generators are restored, and every
  other preprocessing stream must not have moved (abort otherwise, as for an A2B outside a ReLU);
* then the Boolean addition runs on `bool(-lambda_v)` of the recorded masks (moved from the OT phase into the pass),
  and no ReLU input needs the rebase; in preprocessing every A2B input's mask is checked against the recorded one.

UC3 A2bits, ImageNet: 69 MiB less preprocessing traffic (hpmpc's pass messages 140 -> 75 MB for RCA), no mismatch
in any of the 49 ReLUs.

### UC1 / UC2: residual sums (A2B_BAKE_RESIDUAL, 2026-09-30 / 2026-10-01)

The conv/FC computed last of a residual sum's two addends (the partner) draws `lz - (the other addend's mask)`:
ResNet's forward publishes that mask (`g_bake_res_l`; the identity, or `temp` when a downsample branch finishes at the
sum), and `a2b_bake_conv_mask` subtracts it, so the sum carries `lz`. UC1: the residual sum's 6.1 MiB are gone.

UC2 (weights known in preprocessing, SecureML truncation): P1's conv/FC masks lie in the truncation's image (top
FRACTIONAL bits zero), so `lz_1 - (a free mask)` is not a mask P1 can produce. Until 2026-10-01 P1 drew fresh there
and moved the sum alone (`rebase_p1`, 3.1 MiB). Now the other addend's P1 mask `m_b` is committed too, and P1's
committed mask of the sum is `lz_1 = m_a + m_b` (`m_a` the image value `init_a2b_bake` draws), so the partner draws
`lz_1 - m_b = m_a`, again in the image:

* **conv/FC producer** (the downsample branch's conv or, after it, the block's last conv): draws `m_b` as a PRF value
  (`prf_value(kTweakResidual, k << 40 | j)` under its own key) in the image. Like a fresh `m1`, it is uniform there,
  so its prescribed triple share `r1 = -((m_b << F) + low)` stays uniform;
* **ReLU producer** (the identity of a block without downsample): its committed bit-injection output masks
  (`g_relu_out`), available in builds with a mask-only forward (`CHEETAH_CONV_EARLY`, i.e. every single-batch UC2
  build with packed pipelined convs); without one, that sum keeps `rebase_p1`.

`ResNet::mark_residual_producers` numbers the sums in network order and replays the `Identity_*` events to find each
sum's other addend (the partner is the layer computed last); the producer layer publishes the sum's number while it
runs (`g_res_producer_k`, `g_relu_identity_k`), the sum's ReLU its own (`g_residual_k`), the partner `g_bake_res_k`.
The INIT pass records the sum's A2B slots (`g_residual_sums`: the Boolean addition's slices counted so far) and a ReLU
producer's first slot; `init_a2b_bake` then adds `m_b` on those slots and re-derives `ia = bool(-lz_1)`. Whether a sum
is committed follows from the producer kinds alone (`a2b_residual_committed`), so both parties agree on who sends
what. This reveals nothing new: `lz_1 = m_a + m_b` is exactly the mask the sum's local addition gives, and `m_a`,
`m_b` are independent and uniform in the image, as fresh masks would be.

Measured (flare / polynize, hpmpc 841335d): UC2 A2bits ImageNet, P0's received pass messages 32.97 -> 29.76 MB (RCA),
66.75 -> 63.54 (PPA), 111.8 -> 108.6 (PPA4), exactly the 802,816 x 4 B; UC1 unchanged. CIFAR ResNet50 (16 residual
sums: 4 conv-, 12 ReLU-produced) UC2: 6.13 / 10.63 -> 6.13 / 6.13 MB sent / received, i.e. every residual sum is free
and the pass traffic is symmetric; the P1 invariant check (real weights) passes on every ReLU; 5 / 5 / 6 of 10
(RCA / PPA / PPA4; round 5: 6 / 5 / 5, new hashes since P1's prescribed shares changed), 69 / 100 for RCA on 100
images (UC1: 68 / 100).

The stem stays rebased: with `FUSE_CONV_BN=1` its BatchNorm passes the pooling's output on, whose mask is the
pooling's truncation mask. A BatchNorm with secret parameters could take the committed masks like a conv
(`A2B_BAKE_BN`, BatchNorm through the `_baked` mask call with its beta compensated like a conv bias); in the
multi-batch `FUSE_CONV_BN=0` check that changed the outputs (105 instead of 129 of 192) although every mask matched
its slot - not understood yet, so it is off by default.

In preprocessing, every ReLU input the bake moved by its producer is checked against its committed slot (abort
otherwise). Not for P1 with weights known in preprocessing, `TRUNC_DELAYED=0` and dummy weights: the dummy biases give
P1 a bias mask that its image-constrained mask cannot compensate (so those dummy runs' A2Bs were off at P1 after biased
convs; timing is unaffected, and a real model owner's bias carries no mask at P1).

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
