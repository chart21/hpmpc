# Branch status: 2PC (PROTOCOL=4) optimizations

Scope: two-party ABY2-style sharing (`PROTOCOL=4`), 32-bit, `FRACTIONAL=5` unless stated. Tests:
the conv/pool unit suite (`FUNCTION_IDENTIFIER=53`, 9 tests), LeNet5 on MNIST (10 images, matched
`standard` model and dataset), and ResNet50 on CIFAR-10 with trained weights (10 images).

Per-feature detail lives in `docs/A2B_CONV_BAKE.md` (the A2B mask bake), `docs/RESHARE_OPT_SIM.md`
(the reshare simulation), `docs/CUT_FRACTIONAL_BITS_OPT.md` (the fractional-bit cut) and
`docs/RESNET50_COMMUNICATION.md` (ResNet50-Cheetah communication and rounds).

## Three optimization families

| family | flags | what it removes |
|---|---|---|
| plain | none of the below | - (baseline) |
| a2b (A2B bake) | `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1 A2B_CONV_BAKE=1` | the A2B's online communication |
| reshare (simulation) | `RESHARE_OPT=1 RESHARE_OPT_SIM=1` | the reshare's preprocessing send |

`CUT_FRACTIONAL_BITS_OPT=1` composes with all three. `BIT_INJECTION_TRUNC_SIM=1` folds the delayed
truncation of `TRUNC_DELAYED=1` into the ReLU's bit injection (one message per value and one round
fewer per ReLU); it is off by default because only ABY2 and trio implement it.

## Correctness

* **Unit tests: 9/9 in every cell** - all three adders x all three families x cut on/off with
  `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1` (18 cells), and trio (3PC) and Quad_OffOn (4PC) x three
  adders x cut on/off (12 cells). Under `TRUNC_DELAYED=1` the result is 7/9: the two tests that
  reveal a layer output before its truncation (see below).
* **LeNet: 100% everywhere tested** - every adder x family, cut on and off (18 cells); A_KNOWN=0,
  MWK=1 and PUBLIC_WEIGHTS=1 under each family; TRUNC_DELAYED=1 with and without
  `BIT_INJECTION_TRUNC_SIM`. At `DATTYPE=256` (8 lanes, 80 images) MWK=1 gives 98.75% against 100%
  for MWK=0.
* **CIFAR-10 ResNet50 (RCA, 10 images):** both families track the plain protocol. Plain adders
  themselves differ by one image in ten (RCA 70%, PPA 70%, PPA4 80% unfused), which is the noise
  band at this sample size:

| BatchNorm | plain | reshare | a2b | reshare, MWK=1 | a2b, MWK=1 | reshare adder, no bake |
|---|---|---|---|---|---|---|
| unfused | 70% | 60% | 60% | | | 70% |
| fused | 30% | 30% | 40% | 40% | 40% | |

Without the unbaked-input handling below both families were garbage on ResNet50 (a2b 10%, reshare
0-40% from run to run); fused BatchNorm is low for every configuration because of the 32-bit
precision limit below.

## How the bake families stay exact

Both bakes only work on a value that comes straight out of a conv/FC mask/send: that is where the
committed A2B mask (a2b) or the rt.a bits in P1's mask (reshare) are written. Every network marks its
ReLUs accordingly (`ReLU::input_baked`); residual sums, BatchNorm outputs, public-weight layers and
max/min comparison differences are not baked, and the protocol falls back to an exact operation for
them:

* a2b: `rebase` moves the value onto its committed mask - one preprocessing message per value and
  party, nothing online;
* reshare: the adder takes the real reshare (and PPA4's real zero_add) - their pre-sends only.

Measured cost: ResNet50-Cheetah (one residual add, one BatchNorm-fed stem ReLU) +4.0 MB
preprocessing per party for a2b and +0.13 MB for reshare; a2b with PUBLIC_WEIGHTS rebases every ReLU
input, +0.26 MB per party on LeNet. Online traffic and rounds are unchanged.

## Not supported / known broken

* **The plain boolean AND test** (basic-primitives suite, `FUNCTION_IDENTIFIER=54`) fails under every
  configuration including the unmodified baseline. Pre-existing, not exercised by the conv/pool suite
  or the networks.
* **`BITLENGTH != 32`** does not build (`nn/ConvTriple` is a prebuilt 32-bit library); the `_split`
  adder variants (`ADDITIONAL_PPA_THREADS > 0`) fail on their own. See the cut document.
* **Exact truncation for `A_KNOWN=0`** (pre-truncated dealer triples) lives on another branch. Here
  the first layer's input gets split masks with a preprocessing-only rebase, which restores the usual
  SecureML truncation analysis.

## Why delayed truncation reports "Passed 7 out of 9"

Under `TRUNC_DELAYED=1` a conv or BatchNorm layer leaves its output at scale `2^(2*FRACTIONAL)` and
the following ReLU (or pool) truncates it. The standalone Convolution and BatchNorm tests reveal the
layer output directly, so they see values exactly `2^FRACTIONAL` times too large. Every test with an
activation or pool after the layer passes, the same two tests pass at `TRUNC_DELAYED=0`, and the
networks - whose layers are all activation-terminated - reach full accuracy.

## Fused BatchNorm needs more than 32 bits

`FUSE_CONV_BN` folds the BatchNorm scale `gamma / sqrt(var + eps)` into the convolution weights.
Over the 23.45M conv weights and 26,560 BN channels of the CIFAR-10 ResNet50 model, the scale is below
1 for 99.3% of channels (median 0.033), and fusion shrinks the median absolute weight from 0.0756 to
0.0016. Weights quantizing to zero, raw -> fused: `FRACTIONAL=5` 30.7% -> 89.3%; `8` 4.0% -> 54.2%;
`10` 1.0% -> 28.7%; `12` 0.3% -> 11.0%.

The fused network therefore needs 10-14 fractional bits, while each product carries `2*FRACTIONAL`
fractional bits in a 32-bit word and overflows from `FRACTIONAL=10` on. Fused BatchNorm needs
`BITLENGTH=64` (2PC: docs/BITLENGTH64.md). Unfused BatchNorm at
`FRACTIONAL=5` is the 32-bit sweet spot (70% on CIFAR-10, plaintext 74.48%).

## Fixes this branch made to pre-existing code

* **`A_KNOWN=0` was 0/8 and 10%**: the tiled conv's indexed triple retrieval needed its cursor bump
  for any `A_KNOWN`; the first layer's raw data-owner input made the SecureML truncation wrap on
  every negative output; and the BatchNorm dot used the a-known accumulation with symmetric triples.
* **AvgPool under `FUSE_RELU_AVG=1`** skipped its division even when no ReLU had absorbed it -
  ResNet50-Cheetah's stem (Conv -> AvgPool -> BN) was 9x too large.
* **`RCA_MSB` was undefined inside the A2B code**, so the RESHARE_OPT reshare-skip always took the
  PPA branch - wrong for RCA.
* **The MaxPool unit test was vacuous** (tolerance 0.8 while every candidate lies within 0.8 of the
  window maximum). With a real tolerance it exposed that comparisons were wrong under both bakes.
* **The four-way a_ab circuit had never run** (it is reachable only through the bake) and carried
  125 use-before-assignment orderings, 212 share-representation mismatches and 29 missing chain
  output masks; see `docs/A2B_CONV_BAKE.md`.
