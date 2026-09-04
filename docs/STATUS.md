# Branch status: 2PC (PROTOCOL=4) optimizations

Scope: two-party ABY2-style sharing (`PROTOCOL=4`), 32-bit, `FRACTIONAL=5` unless stated.
Unit tests are `FUNCTION_IDENTIFIER=53` (convolution / pooling suite, 8 tests); the network test is
LeNet5 on MNIST (`FUNCTION_IDENTIFIER=182`, 10 images).

Per-feature detail lives in `docs/A2B_CONV_BAKE.md` (the A2B mask bake),
`docs/RESHARE_OPT_SIM.md` (the reshare simulation), `docs/CUT_FRACTIONAL_BITS_OPT.md` (the
fractional-bit cut) and `docs/RESNET50_COMMUNICATION.md` (ResNet50 communication and rounds).

## Three optimization families

| family | flags | what it removes |
|---|---|---|
| plain | none of the below | - (baseline) |
| A2B bake | `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1 A2B_CONV_BAKE=1` | the A2B's online communication (implies `RESHARE_OPT=0`) |
| reshare simulation | `RESHARE_OPT=1 RESHARE_OPT_SIM=1` | the reshare's preprocessing send |

`CUT_FRACTIONAL_BITS_OPT=1` composes with all three.

## Support matrix

Unit tests, `TRUNC_DELAYED=0`, all three adders (`RCA_MSB` / `PPA_MSB` / `PPA4_MSB`), with the cut
on and off - **8 of 8 in every cell** (18 configurations):

| weight setting | plain | A2B bake | reshare sim |
|---|---|---|---|
| `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1` | 8/8 | 8/8 | 8/8 |
| `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0` | 8/8 | 8/8 | 8/8 |
| `A_KNOWN=0` | 8/8 | 8/8 | 8/8 |
| `PUBLIC_WEIGHTS=1` (`TRUNC_DELAYED=1`) | 6/8 (artifact, see below) | **unsupported** | 6/8, falls back to `RESHARE_OPT_SIM=0` |

LeNet with the **matched** model/dataset pair (`LeNet5_MNIST_standard_best.bin` with
`MNIST_standard_test_images.bin`): **100% in all 18 configurations**, cut on and off. Pairing the
`custom` weights with the `standard` images costs one image in ten (90%); on ResNet50 the same
mismatch gives chance accuracy. Use the env scripts unmodified.

### LeNet matrix by adder and truncation mode

Re-measured per cell; `b4` = beaver 4-tuples required, nonzero only for PPA4, recorded so a PPA4 row
cannot be a silent PPA fallback. Accuracy is against the `custom`/`standard` pairing, whose
reference for these ten images is 90%.

| # | family | weights / truncation | adder | accuracy | b4 |
|---|---|---|---|---|---|
| 1 | plain | MWK=1 TD=0 | PPA | 90% | 0 |
| 2 | plain | MWK=1 TD=1 | PPA | 90% | 0 |
| 3 | plain | MWK=0 TD=0 | PPA | 90% | 0 |
| 4 | plain | A_KNOWN=0 TD=0 | PPA | 90% | 0 |
| 5 | plain | PW=1 TD=1 | PPA | 90% | 0 |
| 6 | A2B bake | MWK=1 TD=0 | PPA | 90% | 0 |
| 7 | A2B bake | MWK=1 TD=1 | PPA | 90% | 0 |
| 8 | A2B bake | MWK=0 TD=0 | PPA | 90% | 0 |
| 9 | A2B bake | MWK=0 TD=1 | PPA | 90% | 0 |
| 10 | A2B bake | A_KNOWN=0 TD=0 | PPA | 90% | 0 |
| 11 | A2B bake | A_KNOWN=0 TD=1 | PPA | 90% | 0 |
| 12 | A2B bake | PW=1 TD=1 | PPA | **fails** (see below) | 0 |
| 13 | reshare sim | MWK=1 TD=0 | PPA | 90% | 0 |
| 14 | reshare sim | MWK=0 TD=0 | PPA | 90% | 0 |
| 15 | reshare sim | A_KNOWN=0 TD=0 | PPA | 90% | 0 |
| 16 | reshare sim | PW=1 TD=1 | PPA | 90% | 0 |
| 17 | A2B bake | MWK=1 TD=0 | RCA | 90% | 0 |
| 18 | A2B bake | MWK=1 TD=0 | PPA4 | 90% | 195360 |
| 19 | A2B bake + cut | MWK=1 TD=0 | RCA | 90% | 0 |
| 20 | A2B bake + cut | MWK=1 TD=0 | PPA | 90% | 0 |
| 21 | A2B bake + cut | MWK=1 TD=0 | PPA4 | 90% | 195360 |
| 22 | reshare sim | MWK=1 TD=0 | PPA4 | 90% | 846560 |

## Not supported / known broken

* **`PUBLIC_WEIGHTS=1` under the A2B bake.** With public weights the conv/FC layers multiply
  locally, so there is no fresh output mask to commit the bake into and nothing cancels. It builds
  and runs (5/8, LeNet 20%) but is wrong. Public weights are served by the reshare-simulation family,
  which falls back to `RESHARE_OPT_SIM=0` there (documented fallback in config.h) and reaches
  LeNet parity.
* **Residual networks under either baking family.** Every residual block's final ReLU consumes
  `conv3 + shortcut`, and the mask of a sum is the sum of the masks - matching neither the committed
  `[c]` nor the baked reshare shares. Measured on ResNet50: 809,600 of 1,409,440 reshare checks
  violated, and two runs of the same binaries returned 40% and 0%. The MaxPool treatment
  (re-randomize the adder input through the baked path when it is not a direct conv output) is the
  fix; not done. LeNet-style chains are fully supported.
* **The a_ab adder path (`A_KNOWN_TO_EVALUATORS_OPT=1`) breaks ResNet50** independently of the bake
  and the cut - see the bisect in `docs/CUT_FRACTIONAL_BITS_OPT.md`.
* **The plain boolean AND test** (basic-primitives suite, `FUNCTION_IDENTIFIER=54`) fails under
  every configuration including the plain `A_KNOWN=1` baseline. Pre-existing, independent of
  everything here, not exercised by the conv/pool suite or LeNet (whose boolean gates all go through
  the adder and COT paths). Left open.
* **`BITLENGTH != 32`** does not build at all (`nn/ConvTriple` is a prebuilt 32-bit library); the
  `_split` adder variants (`ADDITIONAL_PPA_THREADS > 0`) fail on their own. Both detailed in the cut
  document.
* **Exact truncation for `A_KNOWN=0`** (pre-truncated dealer triples) lives on a different branch.
  Here `A_KNOWN=0` relies on the first-layer re-randomization (`remask_range` in GEMM.hpp), which is
  correct but costs one extra triple and one round for the first layer's inputs.

`COMPUTE_ARGMAX=1` was previously listed as unsupported. It is not: 8/8 and LeNet 100%, verified
with the cut enabled in both the plain and the A2B-bake configurations.

## Why some runs report "Passed 6 out of 8"

Every 6/8 comes from `TRUNC_DELAYED=1`, and the two failing tests are always the standalone
Convolution and BatchNorm tests. Under delayed truncation the layer leaves its output at scale
`2^(2*FRACTIONAL)` and the following ReLU performs the truncation; those two tests reveal the layer
output directly, with no activation after it, so the revealed values are exactly `2^FRACTIONAL`
times the expected ones (expected 0.1625, got 5.5). It is a property of the test, not of the
protocol: the same two tests pass at `TRUNC_DELAYED=0`, every test that contains an activation
passes at `TRUNC_DELAYED=1`, the unmodified baseline shows the same 6/8, and LeNet - whose layers
are all activation-terminated - reaches full accuracy in exactly these configurations.

## Fused BatchNorm needs more than 32 bits

`FUSE_CONV_BN` folds the BatchNorm scale `gamma / sqrt(var + eps)` into the convolution weights.
Measured over all 23.45M conv weights and 26,560 BN channels of the ResNet50 model:

* the scale is below 1 for 99.3% of channels (median 0.033, 90% of channels below 0.25);
* fusion shrinks the median absolute weight from 0.0756 to 0.0016, a factor of about 48;
* weights quantizing to zero, raw -> fused: `FRACTIONAL=5` 30.7% -> 89.3%; `8` 4.0% -> 54.2%;
  `10` 1.0% -> 28.7%; `12` 0.3% -> 11.0%.

So the fused network needs roughly 10-14 fractional bits before its weights survive quantization at
all - while each multiplication carries `2*FRACTIONAL` fractional bits before truncation, so the
32-bit integer headroom is already exhausted at `FRACTIONAL=10` (accuracy collapses there). The two
requirements cannot both be met in a 32-bit word: fused BatchNorm needs `BITLENGTH=64`, for which no
2PC msb-adder specializations exist. The unfused path splits the problem - raw weights survive 5
fractional bits and the tiny scale multiplies large unnormalized inputs - at the price of integer
range (BN running means reach 218), which is why unfused accuracy drops again at `FRACTIONAL=8`.

ResNet50 on CIFAR-10 (10 images, plaintext 74.48%), plain family:

| configuration | accuracy |
|---|---|
| fused BN, `FRACTIONAL=5` / `8` / `10` / `12` | 40% / 40% / 10% / 20% |
| **unfused BN, `FRACTIONAL=5`** | **70%** (best; near the plaintext ceiling) |
| unfused BN, `FRACTIONAL=8` | 20% |

## Fixes this branch made to pre-existing code

* **`A_KNOWN=0` was 0/8 and 10%** because of three independent baseline bugs: the tiled conv's
  indexed triple retrieval needed its cursor bump for any `A_KNOWN` (it was gated to `A_KNOWN=1`);
  the first layer's raw data-owner input (public part 0) made the SecureML truncation wrap on every
  negative output; and the BatchNorm dot used the a-known accumulation even with symmetric triples.
* **The MaxPool unit test was vacuous** (global tolerance 0.8 while every candidate lies within 0.8
  of the window maximum). Tightened to the quantized expected value with tolerance 0.01, which
  exposed that MaxPool comparisons were silently wrong under BOTH baking families - fixed by
  re-masking the comparison differences through the baked path, and by running the comparison adders
  cut under the reshare simulation (whose slot mapping is compile-time cut-aware).
* **`RCA_MSB` was undefined inside the A2B code** (only derived in share_conversion.hpp, included
  after Protocols.h), so the `RESHARE_OPT` reshare-skip always took the PPA branch - wrong for RCA.
  The derivation moved to config.h.
* **The four-way a_ab circuit had never actually run** (it is reachable only through the bake) and
  carried three defects: 125 use-before-assignment orderings, 212 share-representation mismatches
  (local public-times-secret products are standard-form, the chains they were spliced into are
  dot-pending), and 29 missing chain output masks. All repaired; see `docs/A2B_CONV_BAKE.md`.
