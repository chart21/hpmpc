# Runtime of the 2PC variants: single batch and multi-batch, two machine pairs

Scope: ABY2 (`PROTOCOL=4`) with CHEETAH preprocessing (`PRE=1`, `A_KNOWN=1`), 32-bit, `FRACTIONAL=5`,
ResNet50 on CIFAR-10 with the trained model (`Cifar_adam_001`) and the first images of the test set.
Every variant is measured for preprocessing and online time, with the output hash checked for
reproducibility and against a serial build.

## Machines

| pair | CPU | SIMD | cores / threads | RAM | link |
|---|---|---|---|---|---|
| flare / polynize | AMD EPYC 9354 (Zen 4) | AVX-512, VAES, IFMA | 32 / 64 | 752 GB | 25 Gbit/s direct, ice 2.6.7, MTU 9700 (10.0.4.1 / .2) |
| algofi / goracle | AMD EPYC 7543 (Zen 3) | AVX2, AES-NI | 32 / 64 | 512 GB | 25 Gbit/s direct, ice 2.6.7, MTU 9700 (10.0.1.1 / .2) |

P0 (model owner) runs on flare / algofi, P1 (data owner) on polynize / goracle. Each machine builds its
own binaries (`-march=native`); ConvTriple is rebuilt per machine as well.

## Variants

| axis | values | flags |
|---|---|---|
| MSB adder | RCA, PPA, PPA4 | `RCA_MSB=1` (`FUNCTION_IDENTIFIER=71`), `PPA_MSB=1` (171), `PPA4_MSB=1` (271) |
| family | plain, reshare, reshare + sim, a2b, a2b + a known to evaluators | -, `RESHARE_OPT=1`, `RESHARE_OPT=1 RESHARE_OPT_SIM=1`, `A2B_ONLINE_OPT=1 A2B_CONV_BAKE=1`, `A2B_ONLINE_OPT=1 A_KNOWN_TO_EVALUATORS_OPT=1 A2B_CONV_BAKE=1` |
| batch norm | unfused, fused | `FUSE_CONV_BN=0`, `1` |

`A_KNOWN_TO_EVALUATORS_OPT` switches the reshare off (`config.h`). `A2B_CONV_BAKE` is only active for
`DATTYPE == BITLENGTH`, i.e. single batch; in multi-batch the a2b families run unbaked.

Common flags: `PROTOCOL=4 BITLENGTH=32 FRACTIONAL=5 PRE=1 MODELOWNER=P_0 DATAOWNER=P_1 FUSE_RELU_AVG=1
TRUNC_DELAYED=0 PUBLIC_WEIGHTS=0 CHEETAH_CONV_PACKED=1 CHEETAH_CONV_PIPELINE=1 A_KNOWN=1
MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0 CUT_FRACTIONAL_BITS_OPT=1 SEND_BUFFER=100000 RECV_BUFFER=100000`.

| mode | batch | flags |
|---|---|---|
| single batch | 10 images, one process | `DATTYPE=32 NUM_INPUTS=10 CHEETAH_THREADS=32 ADDITIONAL_GEMM_THREADS=31 ADDITIONAL_RELU_THREADS=31 RNG_AHEAD=1` |
| multi-batch | 192 images, 24 processes x 8 lanes | `DATTYPE=256 PROCESS_NUM=24 NUM_INPUTS=1 CHEETAH_THREADS=3 ADDITIONAL_GEMM_THREADS=1 ADDITIONAL_RELU_THREADS=1` |

Every variant also has a serial reference build (no `ADDITIONAL_*_THREADS`, no `RNG_AHEAD`): its
output hash must equal the threaded build's.

## Method

* Build: `vb.sh` compiles one variant with the Makefile's compiler line into `/root/vx/<name>.p<party>`
  (each machine only its own party, 20 in parallel, about 20 s each).
* Run: `mxall.sh` runs every variant on a pair and appends one CSV row: preprocessing and online time
  (maximum over processes), accuracy, a digest of the output hashes (`PRINT_OUTPUT_HASH=1`), the
  preprocessing phases (conv, BN, bool, COT, multiplexer, FC; average per process), the online time of
  ReLU, conv and BN layers, traffic, the time P0 waited for data and the number of rounds.
* Seeds are fixed (ConvTriple `PRG_SEED` 42 / 43, OT `emp::DetSeedScope`, hpmpc `SRNG_SEED`): repeated
  runs of a variant give identical output hashes, and the threaded build gives the same hash as the serial
  one (see Correctness). The hashes differ between the two machine pairs: the SRNG draws 512-bit AES
  blocks on Zen 4 (VAES-512) and 256-bit blocks on Zen 3, i.e. a different stream from the same seed.
* Two rounds: round 1 at hpmpc `7539e53` / ConvTriple `3d134b0`, round 2 at `945a693` (flare / polynize) and
  `26c5a9b` (algofi / goracle) / ConvTriple `e3fc972` with the fixes listed below. Single batch runs twice per
  variant (median), multi-batch once. PPA4 reshare + sim fused (single batch) was re-measured at `992e66b`;
  its racy round-2 rows are in `res_*_superseded.csv`.
* Round 3: every PPA4 build at ConvTriple `ed4f83f` (shared-OT Beaver tuples), threaded and serial, both pairs
  (`res_*_r3.csv`). Round 4: the whole matrix at hpmpc `f75b82e` / ConvTriple `3e80748` (batched BN), threaded
  builds on both pairs and serial references on flare / polynize (`res_*_r4.csv`). The multi-batch a2b
  families were rerun with the multi-batch A2B bake (`64404c3`, `res_*_ab.csv`): flare / polynize with
  `Cifar_adam_001`, algofi / goracle with the AdamW model. `res_*_r4a.csv` is round 4 with those rows.
* Round 5 (final): single batch at hpmpc `a404661` / ConvTriple `fa7b952` / flexNN `b1ce619` (concurrent conv-triple
  evaluation) on both pairs, `res_*_r5.csv` (the `*_k0` rows are the A_KNOWN=0 builds on flare / polynize); multi-batch at
  `97772ef` (lane-batched conv triples) on both pairs, `res_*_r5m.csv`; AdamW accuracy of all 60 builds at `97772ef` on
  flare / polynize, `res_fp_final_wd.csv`. The 30 single-batch hashes of round 5 equal round 4's, which equal the serial builds'.
* Accuracy: the same threaded builds with the AdamW model (`MX_MODEL=wd`), single batch as 100-image builds.
* Multi-batch traffic: the 24 processes' log lines interleave, so `vsum.sh` takes the median of the intact
  per-process lines times the process count (every process sends the same). Round 1's multi-batch traffic
  column summed fragments and is wrong; round 2's CSVs were corrected from the logs.

The harness scripts are in `scripts/variants/` (`vb.sh`, `vball.sh`, `vrun.sh`, `vsum.sh`, `mxall.sh`,
`analyze.py`, `doc_tables.py`, `acc_table.py`); raw CSVs and the generated tables are in `docs/variant_data/`.

## Summary

* **Single batch** (10 images, final round 5): preprocessing on flare / polynize 1.3-1.8 s (RCA), 1.9-2.2 s (PPA) and
  2.1-2.5 s (PPA4) with fused BN, 1.6-2.0 / 2.2-2.4 / 2.3-2.7 s unfused (the a2b families are the upper end);
  algofi / goracle needs 1.13-1.25x. Online 0.35-0.50 s (flare / polynize) and 0.31-0.59 s (algofi / goracle).
  Online is bound by round latency (RCA 1,641-1,801 rounds, PPA 665-829, PPA4 514-674): with the threaded ReLU
  levels, PPA and PPA4 are slightly faster online than RCA. The reshare families add 49-53 rounds and change little.
  Fastest in total: RCA plain fused, 1.32 s + 0.39 s.
* **Multi-batch** (192 images, 24 processes, final round 5 with lane-batched conv triples): preprocessing 7.8-11.7 s (RCA),
  11.0-15.0 s (PPA) and 13.9-19.5 s (PPA4) on flare / polynize, online 0.54-0.81 s; algofi / goracle needs 13.6-28.6 s
  (1.46-1.75x) and 0.81-1.07 s (compute-bound: AVX2 instead of AVX-512 in the HE and GEMM kernels).
* **Speedups during the study**: CHEETAH channels without Nagle (a2b with PPA / PPA4 and PPA4 reshare + sim
  8.4-10.8 s -> 3.5-5.0 s single batch), OT packs sized for the whole preprocessing, bit-packed COT multiply,
  CHEETAH_THREADS > 1 in multi-batch (port stride bug; 25 -> 14 s), Beaver 3/4-tuples with one OT per shared
  factor (PPA4 4.35 -> 2.68 s single batch unfused, 31.5 -> 22.6 s multi-batch), BN triples of all layers in
  one product (unfused RCA 2.51 -> 1.94 s single batch), several conv-triple chunks evaluated at once (single batch
  -0.26..0.55 s, 10-26%), and in multi-batch one convolution for all 8 lanes (-4.5..6.3 s, 19-38%; P0 traffic -50..62%).
* **Correctness**: every variant's threaded build gives the serial build's output, and repeated runs are
  bit-identical (60 of 60 on flare / polynize in round 4; the final single-batch hashes are unchanged, the lane batching
  changes the multi-batch triples' random shares and thus those hashes). Fixed during the study: a race in P0's A2B on the
  worker pool (PPA4 reshare + sim fused, `992e66b`) and the a2b families in multi-batch, whose A2B mask bake
  only existed for DATTYPE == BITLENGTH (13-29 of 192 correct before `64404c3`, now as accurate as plain).
* **Accuracy** (AdamW ResNet50, standard truncation, final builds): 62-72 of 100 images for all 30 variants in single batch
  (plaintext 76, Trio 71), 118-131 of 192 (61-68%) for all 30 variants in multi-batch (plaintext 138, 71.9%), fused BN
  included; A_KNOWN=0 64-72 / 117-133, public weights 63-73 / 115-131. With the older `Cifar_adam_001` model the probabilistic truncation in the 32-bit ring cost 10-17
  points (Trio 56% vs 73%).
* **CUT_FRACTIONAL_BITS_OPT**: every run above uses it (`=1`). It removes 12% of RCA's online rounds and 8-11%
  of the ReLU traffic; runtime and accuracy change within noise (see "CUT_FRACTIONAL_BITS_OPT, A_KNOWN=0 and public
  weights" below).
* **Public weights** (`TRUNC_DELAYED=1 BIT_INJECTION_TRUNC_SIM=1`): 0.92-2.03 s preprocessing per 10 images, 3.5-14.3 s
  per 192, online 0.27-0.42 s; AdamW accuracy 63-73 of 100 and 115-131 of 192 after the unfused-BN fix (`4dd999b`).
  **A_KNOWN=0** works after two fixes (threaded GEMM, BN triples over x - mu; see below).
* **Concurrent conv-triple evaluation** (ConvTriple `3286b6b` / `fa7b952`): AB2 was slower than AB although it does half the
  work, because its weight holder evaluated one chunk at a time. Now three chunks are evaluated at once: the single-batch
  conv phase halves, and AB2 is faster than AB (isolated CIFAR suite 0.18-0.19 s vs 0.21 s).

## Results (final, round 5)

Median of two runs (single batch) or one run (multi-batch), threaded builds, `Cifar_adam_001` (the runtime does not
depend on the weights). "MB pre" is P0's preprocessing traffic summed over processes; accuracy on 10 / 192 images of
this model (see Accuracy for the AdamW model, which is the reference). This model is sensitive to the truncation noise:
flare / polynize PPA plain and reshare unfused classify 82 and 98 of 192 here, against 114-115 on algofi / goracle and
121-129 with the AdamW model on the same builds. Round 4's tables (before the concurrent evaluation and the lane batching)
are in `docs/variant_data/matrix_round4.md` (round 2: `matrix_round2.md`).
### Single batch (10 images, one process)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 1.55 | 1.92 | 0.452 | 0.590 | 1752 | 229 | 6/10 | 7/10 |
| rca | reshare | unfused | 1.61 | 1.93 | 0.492 | 0.554 | 1801 | 229 | 6/10 | 7/10 |
| rca | reshare + sim | unfused | 1.59 | 1.96 | 0.462 | 0.558 | 1801 | 229 | 7/10 | 7/10 |
| rca | a2b | unfused | 2.02 | 2.39 | 0.464 | 0.444 | 1752 | 256 | 8/10 | 7/10 |
| rca | a2b + AKTE | unfused | 2.02 | 2.35 | 0.498 | 0.444 | 1752 | 256 | 7/10 | 7/10 |
| ppa | plain | unfused | 2.15 | 2.63 | 0.428 | 0.516 | 776 | 268 | 8/10 | 8/10 |
| ppa | reshare | unfused | 2.16 | 2.65 | 0.464 | 0.538 | 829 | 268 | 8/10 | 8/10 |
| ppa | reshare + sim | unfused | 2.25 | 2.71 | 0.487 | 0.528 | 829 | 268 | 7/10 | 8/10 |
| ppa | a2b | unfused | 2.42 | 2.86 | 0.434 | 0.409 | 776 | 283 | 7/10 | 8/10 |
| ppa | a2b + AKTE | unfused | 2.36 | 2.74 | 0.455 | 0.412 | 780 | 283 | 7/10 | 8/10 |
| ppa4 | plain | unfused | 2.42 | 2.89 | 0.424 | 0.483 | 625 | 321 | 5/10 | 6/10 |
| ppa4 | reshare | unfused | 2.29 | 2.87 | 0.397 | 0.502 | 674 | 321 | 5/10 | 6/10 |
| ppa4 | reshare + sim | unfused | 2.59 | 3.12 | 0.458 | 0.506 | 674 | 337 | 6/10 | 6/10 |
| ppa4 | a2b | unfused | 2.67 | 3.10 | 0.421 | 0.405 | 625 | 335 | 5/10 | 8/10 |
| ppa4 | a2b + AKTE | unfused | 2.54 | 2.94 | 0.467 | 0.434 | 625 | 300 | 7/10 | 6/10 |
| rca | plain | fused | 1.32 | 1.65 | 0.393 | 0.374 | 1641 | 140 | 4/10 | 4/10 |
| rca | reshare | fused | 1.33 | 1.66 | 0.404 | 0.407 | 1690 | 140 | 4/10 | 4/10 |
| rca | reshare + sim | fused | 1.37 | 1.65 | 0.397 | 0.382 | 1690 | 140 | 4/10 | 4/10 |
| rca | a2b | fused | 1.82 | 2.11 | 0.394 | 0.384 | 1641 | 168 | 4/10 | 4/10 |
| rca | a2b + AKTE | fused | 1.76 | 2.07 | 0.404 | 0.378 | 1641 | 168 | 4/10 | 5/10 |
| ppa | plain | fused | 1.94 | 2.37 | 0.369 | 0.363 | 665 | 180 | 4/10 | 4/10 |
| ppa | reshare | fused | 1.93 | 2.41 | 0.409 | 0.372 | 718 | 180 | 4/10 | 4/10 |
| ppa | reshare + sim | fused | 2.01 | 2.45 | 0.381 | 0.377 | 718 | 180 | 4/10 | 4/10 |
| ppa | a2b | fused | 2.21 | 2.55 | 0.375 | 0.332 | 665 | 194 | 5/10 | 4/10 |
| ppa | a2b + AKTE | fused | 2.11 | 2.52 | 0.360 | 0.335 | 669 | 194 | 4/10 | 4/10 |
| ppa4 | plain | fused | 2.13 | 2.61 | 0.345 | 0.344 | 514 | 232 | 4/10 | 4/10 |
| ppa4 | reshare | fused | 2.18 | 2.58 | 0.375 | 0.355 | 563 | 232 | 4/10 | 4/10 |
| ppa4 | reshare + sim | fused | 2.26 | 2.83 | 0.376 | 0.362 | 563 | 248 | 4/10 | 4/10 |
| ppa4 | a2b | fused | 2.49 | 2.80 | 0.358 | 0.310 | 514 | 247 | 4/10 | 4/10 |
| ppa4 | a2b + AKTE | fused | 2.32 | 2.69 | 0.408 | 0.342 | 514 | 212 | 4/10 | 4/10 |

### Multi-batch (192 images, 24 processes x 8 lanes)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 9.08 | 15.37 | 0.805 | 0.838 | 1738 | 4459 | 131/192 | 114/192 |
| rca | reshare | unfused | 8.95 | 15.41 | 0.699 | 0.863 | 1787 | 4459 | 126/192 | 111/192 |
| rca | reshare + sim | unfused | 8.96 | 15.42 | 0.654 | 0.871 | 1787 | 4459 | 126/192 | 111/192 |
| rca | a2b | unfused | 11.75 | 18.84 | 0.709 | 0.913 | 1738 | 4875 | 115/192 | 108/192 |
| rca | a2b + AKTE | unfused | 11.20 | 18.15 | 0.736 | 0.828 | 1738 | 4875 | 116/192 | 110/192 |
| ppa | plain | unfused | 12.10 | 19.29 | 0.619 | 0.831 | 758 | 4597 | 82/192 | 114/192 |
| ppa | reshare | unfused | 12.08 | 19.42 | 0.638 | 0.858 | 807 | 4574 | 98/192 | 115/192 |
| ppa | reshare + sim | unfused | 12.05 | 19.43 | 0.667 | 0.858 | 807 | 4574 | 98/192 | 115/192 |
| ppa | a2b | unfused | 14.98 | 22.83 | 0.629 | 0.902 | 758 | 5013 | 110/192 | 126/192 |
| ppa | a2b + AKTE | unfused | 13.43 | 20.83 | 0.625 | 0.868 | 758 | 4944 | 109/192 | 125/192 |
| ppa4 | plain | unfused | 16.90 | 25.01 | 0.591 | 0.806 | 611 | 5813 | 102/192 | 124/192 |
| ppa4 | reshare | unfused | 16.52 | 25.05 | 0.597 | 0.842 | 660 | 5813 | 103/192 | 122/192 |
| ppa4 | reshare + sim | unfused | 16.47 | 24.95 | 0.659 | 0.869 | 660 | 6123 | 103/192 | 122/192 |
| ppa4 | a2b | unfused | 19.48 | 28.56 | 0.560 | 0.882 | 611 | 6229 | 124/192 | 126/192 |
| ppa4 | a2b + AKTE | unfused | 14.81 | 22.82 | 0.585 | 0.921 | 611 | 5350 | 128/192 | 115/192 |
| rca | plain | fused | 7.85 | 13.64 | 0.646 | 0.976 | 1632 | 2763 | 69/192 | 70/192 |
| rca | reshare | fused | 7.83 | 13.67 | 0.778 | 1.014 | 1681 | 2763 | 71/192 | 71/192 |
| rca | reshare + sim | fused | 7.83 | 13.73 | 0.641 | 1.065 | 1681 | 2763 | 71/192 | 71/192 |
| rca | a2b | fused | 10.58 | 17.13 | 0.634 | 1.020 | 1632 | 3179 | 66/192 | 72/192 |
| rca | a2b + AKTE | fused | 10.15 | 16.49 | 0.649 | 1.038 | 1632 | 3179 | 69/192 | 73/192 |
| ppa | plain | fused | 11.03 | 17.58 | 0.561 | 0.903 | 652 | 2901 | 72/192 | 73/192 |
| ppa | reshare | fused | 11.01 | 17.94 | 0.576 | 0.903 | 701 | 2878 | 73/192 | 70/192 |
| ppa | reshare + sim | fused | 11.03 | 17.77 | 0.593 | 1.005 | 701 | 2878 | 73/192 | 70/192 |
| ppa | a2b | fused | 13.93 | 21.26 | 0.589 | 0.977 | 652 | 3317 | 72/192 | 71/192 |
| ppa | a2b + AKTE | fused | 12.27 | 19.15 | 0.556 | 1.019 | 652 | 3248 | 70/192 | 70/192 |
| ppa4 | plain | fused | 15.64 | 23.35 | 0.580 | 0.947 | 505 | 4117 | 72/192 | 71/192 |
| ppa4 | reshare | fused | 15.57 | 23.33 | 0.536 | 0.968 | 554 | 4117 | 73/192 | 71/192 |
| ppa4 | reshare + sim | fused | 15.56 | 23.32 | 0.548 | 0.926 | 554 | 4427 | 73/192 | 71/192 |
| ppa4 | a2b | fused | 18.47 | 27.03 | 0.551 | 0.906 | 505 | 4533 | 74/192 | 71/192 |
| ppa4 | a2b + AKTE | fused | 13.91 | 21.39 | 0.574 | 0.965 | 505 | 3653 | 71/192 | 71/192 |

## ImageNet (ResNet50, one image)

The HE preprocessing steps and the online steps of the paper (`docs/paper`) use `FUNCTION_IDENTIFIER=87`: CHEETAH's
ResNet50 layout for 1000 classes, input 224x224 padded to 230x230, dummy model and image (value-independent runtime).
Flags beyond the `config.h` defaults: `NUM_INPUTS=1 DATTYPE=32 PRE=1 CHEETAH_CONV_PACKED=1 CHEETAH_CONV_PIPELINE=1
CHEETAH_THREADS=32 FUSE_CONV_BN=1 FUSE_RELU_AVG=1 RCA_MSB=1 MODELOWNER=P_0 DATAOWNER=P_1` (plus the defaults `A_KNOWN=1
PUBLIC_WEIGHTS=0 TRUNC_DELAYED=0 MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0` set explicitly); ConvTriple `TRIPLE_ZERO=ON`,
`TRIPLE_FERRET=b12`. The online steps add `CUT_FRACTIONAL_BITS_OPT=1 SEND_BUFFER=100000 RECV_BUFFER=100000
ADDITIONAL_GEMM_THREADS=31 ADDITIONAL_RELU_THREADS=31 RNG_AHEAD=1`.

| same build (ConvTriple `d6c65f9`), median of 3 | algofi / goracle | flare / polynize | ratio |
|---|---|---|---|
| preprocessing (s) | 4.26 | 3.31 | 1.29 |
| key setup / conv / AND triples (s) | 1.15 / 1.74 / 0.34 | 1.01 / 1.13 / 0.29 | 1.13 / 1.55 / 1.17 |
| triple traffic P0 sent / received (MB) | 305.5 / 281.9 | 305.5 / 281.9 | 1 |
| online (s), before the online changes (delayed-ACK stalls) | 4.77 | 3.08 (2.43-3.21) | |
| online (s), final online build | - | 0.69-0.73 (1,820 rounds) | |

PPA4 on ImageNet (flare / polynize, before the shared-OT tuples): preprocessing 10.8-11.4 s, online 1.05-1.06 s.

## Changes made during the study

| commit | repository | what | effect |
|---|---|---|---|
| `22a2c58` | hpmpc | `stream_parallel` cursors for every index-addressed preprocessing stream (Boolean / arithmetic / AB2 triples, Beaver 3/4-tuples, random multiplications); adders of every kind constructed in parallel | PPA / PPA4 / reshare adders on threads without races (they consume Beaver tuples and random multiplications) |
| `7539e53`, ConvTriple `3d134b0` | both | `Iface::ot_demand_hint`: the OT packs are sized for the whole preprocessing, not the first request | PPA4, reshare and a2b get enough packs for their tuples |
| ConvTriple `1e460c2` | ConvTriple | `TCP_NODELAY` on every CHEETAH channel (was only with `CHEETAH_WAN_OPT=1`) | the small flushes of the random OTs and the COT multiply no longer wait for delayed ACKs: each of the 31 COT-multiply calls of the A2B bake 170 ms -> 11 ms with one ferret thread per pack; PPA a2b 8.4 -> 3.5 s, PPA4 reshare + sim 8.9 -> 5.0 s (single batch) |
| ConvTriple `ec6e49c` | ConvTriple | `cot_multiply_shares` on bit-packed random OTs (`send/recv_rot_bits`) | same bits (hashes unchanged), less memory traffic; ferret extension counters, `CHEETAH_OT_GROUP` |
| `ee5028a` | hpmpc | CHEETAH channel port stride `(n(n-1)+1) * PROCESS_NUM` | `CHEETAH_THREADS > 1` works with 24 processes (before: shared connections, hangs); multi-batch preprocessing 25 -> 14 s with 3 threads |
| `b16c6e9` | hpmpc | connection start: flag and broadcast under the waiters' mutex | lost-wakeup hang of single processes |
| `96c3a85` | hpmpc | close the listening socket right after `accept` | the next phase's peer could connect into the old listener's backlog: hang at the start of the live phase |
| ConvTriple `d350ae8` | ConvTriple | `emp::DetSeedScope`: OT PRGs seeded per pack and worker | bit-reproducible runs (the COT / multiplexer shares are OT outputs and are truncated share-wise) |
| `8d5c543` | hpmpc | `RELU_RANDOM` test (`FUNCTION_IDENTIFIER=59`) | 32,768 random values over all magnitudes: RCA, PPA, PPA4 exact in 2PC and Trio |
| `945a693` | hpmpc | `AVG_RECIP_EXTRA_BITS` (default 0): more fractional bits for the 1/denom that `FUSE_RELU_AVG` folds into the ReLU | 1/9 is folded as 0.125 with 5 bits |
| `992e66b` | hpmpc | the A2B group count of P0's PPA4 reshare + sim bake (`g_a2b_s1_pending`) is a per-worker index stream; `STREAM_PARALLEL_GEMM` / `_CTOR` / `_RELU` switches | threaded PPA4 reshare + sim fused runs are bit-identical to serial (before: a different output every run) |
| ConvTriple `ed4f83f` (hpmpc `5ce0021`) | ConvTriple | Beaver 3/4-tuples: `cot_outer_multiply` - the products with a common factor (3-tuple: ac, bc, abc; 4-tuple: {a, b, ab} x {c, d, cd}) from one random OT per direction whose k message bits (`send/recv_rot_bitplanes`) carry all k products, the fresh random c taken from the OT choices | per 3-tuple 6 -> 2 OTs and 6 -> 3 sent bits, per 4-tuple 18 -> 6 OTs and 18 -> 12 bits; tuple phases per process (multi-batch) 19.0 -> 9.2 s; PPA4 preprocessing single batch 4.35 -> 3.19 s (unfused) / 3.46 -> 2.62 s (fused), multi-batch 31.5 -> 21.7 s |
| ConvTriple `3e80748`, hpmpc `f75b82e` | both | `CHEETAH_BN_BATCHED`: the BN triples of all layers in one elementwise product (`generateBNTriplesBatched`) instead of 53 latency-bound calls | unfused BN phase 0.75 -> 0.19 s single batch, 1.16 -> 0.85 s per process multi-batch; RCA unfused single batch 2.51 -> 1.99 s |
| `64404c3` | hpmpc | A2B conv-mask bake for DATTYPE > BITLENGTH (value-wise negation, `orthogonalize_arithmetic` packing) | the a2b families work in multi-batch: 13-29 -> 99-132 of 192 |
| `4dd999b`, flexNN `ce50579` | both | public weights, unfused BN: the BN truncates the first conv's delayed output (data-owner sharing) exactly on the owner side (`trunc_a_known_in_place`) | ResNet50 `PUBLIC_WEIGHTS=1` unfused BN: 0/10 -> 6/10 (plaintext 6/10); 63-73 of 100, 118-131 of 192 over the variants |
| `f2f55b0`, flexNN `b1ce619` | both | threaded GEMM as the serial one (A_KNOWN=0 conv triple share and cursor; public weights' first-layer owner-side truncation); A_KNOWN=0 BN triples over the masks of x - mu | A_KNOWN=0 correct in threaded builds and with unfused BN (1 / 0 -> 5 / 7 of 10); public weights with `TRUNC_DELAYED=0` no longer deadlock |
| ConvTriple `3286b6b`, `fa7b952` (hpmpc `a404661`) | ConvTriple | conv triples: several chunks evaluated at once (3 evaluator threads, ordered sender, same PRNG calls) for the AB2 weight holder and for AB | isolated CIFAR suite AB2 0.34-0.39 -> 0.18-0.19 s (AB 0.26-0.30 -> 0.21 s), ImageNet AB2 0.73 s (AB 1.03 -> 0.82 s); single-batch conv phase 0.54-0.65 -> 0.26-0.31 s, same output hashes |
| `97772ef` (ConvTriple `fa7b952`) | hpmpc | `CHEETAH_CONV_LANES`: in multi-batch the conv triples of all 8 lanes and all layers as one packed convolution with lane 0's weight masks (all lanes share the masks -w by construction; `CHEETAH_CONV_LANES_CHECK=1` verifies it), pipelined | conv phase per process 8.4-8.7 -> 3.8-3.9 s, P0 preprocessing traffic -50..62%, RCA plain unfused multi-batch 13.7 -> 8.9 s; AdamW accuracy unchanged in kind (see Accuracy) |
| `499076f` | hpmpc | comparison tests declare their inputs unbaked (`g_msb_input_baked = false`) | `RELU_RANDOM` with PPA4 reshare + sim at DATTYPE 32: ~35% wrong (the SIM skip assumed a conv bake) -> 0 of 4,096 |

## Correctness

* Threaded vs serial: the output hash of the threaded build equals the serial build's for all 60 variants
  (30 single batch, 30 multi-batch) on both pairs in round 2 and 3 (PPA4), and on flare / polynize in round 4.
  Until `992e66b` PPA4 reshare + sim with fused BN was the exception: its runs
  differed from each other and from serial on both pairs. Bisected with the `STREAM_PARALLEL_*` switches,
  the cause was P0's `prepare_A2B_S1`. For the SIM zero_add skip, it takes each group's slice masks from the
  Beaver 3-tuples the group's adder will consume, at position `g_a2b_s1_pending` (groups prepared so far). On
  the worker pool that count was one shared global: workers used execution-order counts and raced on the
  increment, so masks and tuples disagreed. It is now an index stream with a cursor per worker, reset serially
  after the prepare level. Only fused BN was affected, because only a baked conv output (fused BN) switches
  the SIM path on; with the GEMM levels serial the pool threads woke one after another and hid the race.
  After the fix: 5 of 5 threaded runs give the serial hash (3 on algofi / goracle, 2 on flare / polynize).
* The multi-batch "mismatches" of round 1 were serial runs that lost one process (184 of 192 images; its hash is
  that of an empty output) to the connection hangs fixed in `b16c6e9` / `96c3a85`. The other 23 processes matched,
  and reruns of all three serial references match the threaded builds exactly. `analyze.py` now counts such runs
  as failed.
* The hashes differ between the two pairs (see Method), so they are compared within a pair only.
* Round 2 reproduces round 1's hashes in 44 of 45 single-batch and 25 of 26 multi-batch variants compared
  (the exceptions are PPA4 reshare + sim fused, nondeterministic before `992e66b`, and a broken a2b multi-batch variant):
  the TCP and COT-multiply changes do not change values.
* a2b in multi-batch was wrong until `64404c3` (13-29 of 192 correct): `A2B_CONV_BAKE_ACTIVE` required
  `DATTYPE == BITLENGTH`, and `A2B_ONLINE_OPT` without the bake is broken (`docs/A2B_CONV_BAKE.md`). The bake's
  mask commit cast each transposed word to a `Datatype`; it now negates value by value and packs with
  `orthogonalize_arithmetic`, the inverse of the A2B's own slicing, which is the identity for DATTYPE == BITLENGTH:
  single-batch hashes are unchanged (checked for RCA a2b and PPA4 a2b + AKTE). Multi-batch a2b / a2b + AKTE now
  reach 99-123 of 192 unfused and 67-73 fused on flare / polynize (`Cifar_adam_001`; plain 110-121 / 70-72), and
  117-132 on algofi / goracle (AdamW; plain 116-129); threaded equals serial (two variants checked). Still
  excluded: `MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1` (its prescribed shares treat a `Datatype` as one word).

## CUT_FRACTIONAL_BITS_OPT, A_KNOWN=0 and public weights

**CUT_FRACTIONAL_BITS_OPT** (all runs above use it). Quick check on the plain family, same code as round 4
(`res_fp_c0.csv`, `res_ag_c0wd.csv`, `res_ag_c1wd.csv`):

| single batch, unfused BN, flare / polynize | rounds CUT / no CUT | ReLU MB (P0) CUT / no CUT | pre s CUT / no CUT | online s CUT / no CUT |
|---|---|---|---|---|
| RCA | 1,752 / 1,997 | 20.8 / 23.3 | 1.94 / 1.98 | 0.52 / 0.48 |
| PPA | 776 / 789 | 32.6 / 36.8 | 2.56 / 2.55 | 0.44 / 0.49 |
| PPA4 | 625 / 625 | 20.1 / 21.8 | 2.68 / 2.71 | 0.42 / 0.42 |

CUT removes 12% of RCA's rounds and 8-11% of the ReLU traffic for every adder; preprocessing and online time
differ within run-to-run noise. Accuracy is unchanged: with the AdamW model on algofi / goracle, the six plain
variants in multi-batch (192 images) and single batch (100 images) classify 1,134 of 1,752 with CUT and 1,138
without (64.7% / 65.0%).

**Public weights** (`PUBLIC_WEIGHTS=1 TRUNC_DELAYED=1 BIT_INJECTION_TRUNC_SIM=1`, `res_fp_kpw.csv`, accuracy
`res_ag_pwwd.csv`). No conv / FC / BN triples: preprocessing 0.92-2.03 s per 10 images and 3.5-14.3 s per 192
images (16-41% and 22-61% less than the same variant with secret weights in the final builds), online 0.27-0.41 s and
0.33-0.42 s (9-33% and 30-50% less). AdamW accuracy: 63-73 of 100 (single batch) and 115-131 of 192 (multi-batch). Unfused
BN needed `4dd999b`: with public weights and delayed truncation the first conv's output is still the data owner's
sharing (the other party's mask is 0), and the BN's `trunc_pr_in_place` wrapped on it (0 of 10 correct).

**A_KNOWN=0** (`res_fp_kpw.csv`, `res_fp_r5.csv`: `*_k0`; accuracy `res_ag_k0wd.csv`): AB triples for the linear layers,
1.4-1.9x the preprocessing traffic (RCA unfused, 10 images: 229 -> 428 MB). It first classified at chance (0-1 of 10). That
was not the truncation: a bisection over the added optimizations (`res_fp_kb.csv`) found two bugs.
- The threaded GEMM (`GEMM_threaded.hpp`) sent A_KNOWN=0 outputs without the conv triple's share and left the triple cursor
  behind (the serial GEMM was right): without the extra threads 5 of 10 were correct. Fixed in `f2f55b0`.
- Unfused BN: the AB BN triple was set up over the masks of x, but the product's operand is x - mu, whose mask contains the
  model owner's mask of mu; the missing lambda_s * lambda_mu = s * mu made every output s * x + beta (func53 BN test failed).
  Fixed in flexNN `b1ce619` (A_KNOWN=1 output hashes unchanged).
After both, 10 images: fused 5/10, unfused 7/10 (plaintext 6/10); all 30 variants below.

**Public weights, exact failure conditions** (`res_ag_pq.csv`, 10 images, AdamW; plaintext 6/10):
- Unfused BN with `TRUNC_DELAYED=1` was wrong with and without `BIT_INJECTION_TRUNC_SIM` (1/10 and 0/10): the BN truncated the
  first conv's delayed output, still the data owner's sharing, with the SecureML local truncation. Fixed in `4dd999b` / flexNN
  `ce50579`: now 5/10 and 6/10.
- With `TRUNC_DELAYED=0` every threaded build (fused and unfused) hung in preprocessing: the threaded GEMM truncated the first
  layer with the regular truncation while the completion expected the data owner's exact one. Fixed in `f2f55b0`: unfused
  7/10, fused 5/10.

## Accuracy

Accuracy is measured with the AdamW-trained ResNet50 (`adam_001_wd/ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin`,
74.35% on the full test set, `python nn/Pygeon/download_pretrained.py adam_001_wd`), with the standard truncation
(`TRUNC_APPROACH=0`, `TRUNC_DELAYED=0`). Every variant is above 60%: 62-72 of the first 100 images in single
batch (plaintext 76), 118-131 of the first 192 in multi-batch (plaintext 138). Runs: `MX_MODEL=wd mxall.sh` with the final
builds (`97772ef`, lane-batched multi-batch conv triples) on flare / polynize, single batch with 100-image builds (`a_*`,
`NUM_INPUTS=100`), `res_fp_final_wd.csv`. Before the lane batching (round 4 builds, multi-batch on algofi / goracle) the
ranges were 62-73 and 116-132.

| setting | images | correct |
|---|---|---|
| plaintext, float (PyTorch, `/root/plain_eval_wd.py`) | 100 / 192 | 76 / 138 (71.9%) |
| Trio, standard config (32 bit, FRACTIONAL 5, probabilistic truncation) | 100 | 71 |
| Trio, `TRUNC_APPROACH=1` | 100 | 70 |
| 2PC, single batch, all 30 variants | 100 | 62-72 |
| 2PC, multi-batch, all 30 variants | 192 | 118-131 (a2b families: 14-25 before `64404c3`) |
| 2PC, A_KNOWN=0, single / multi-batch, all 30 variants | 100 / 192 | 64-72 / 117-133 |
| 2PC, public weights, single / multi-batch, all 30 variants | 100 / 192 | 63-73 / 115-131 |

| adder | family | BN | single batch, 100 images | multi-batch, 192 images | pre s (100 images) | online s (100 images) |
|---|---|---|---|---|---|---|
| rca | plain | unfused | 67/100 (67%) | 131/192 (68%) | 8.6 | 1.89 |
| rca | reshare | unfused | 69/100 (69%) | 128/192 (67%) | 8.4 | 2.12 |
| rca | reshare + sim | unfused | 67/100 (67%) | 128/192 (67%) | 8.3 | 1.89 |
| rca | a2b | unfused | 67/100 (67%) | 126/192 (66%) | 11.8 | 1.97 |
| rca | a2b + AKTE | unfused | 70/100 (70%) | 122/192 (64%) | 11.7 | 1.97 |
| ppa | plain | unfused | 66/100 (66%) | 121/192 (63%) | 11.4 | 1.86 |
| ppa | reshare | unfused | 65/100 (65%) | 129/192 (67%) | 11.0 | 1.96 |
| ppa | reshare + sim | unfused | 66/100 (66%) | 129/192 (67%) | 11.7 | 1.99 |
| ppa | a2b | unfused | 69/100 (69%) | 129/192 (67%) | 15.2 | 2.05 |
| ppa | a2b + AKTE | unfused | 65/100 (65%) | 123/192 (64%) | 13.0 | 2.04 |
| ppa4 | plain | unfused | 68/100 (68%) | 127/192 (66%) | 15.1 | 1.87 |
| ppa4 | reshare | unfused | 65/100 (65%) | 128/192 (67%) | 15.2 | 1.86 |
| ppa4 | reshare + sim | unfused | 71/100 (71%) | 128/192 (67%) | 16.7 | 1.98 |
| ppa4 | a2b | unfused | 65/100 (65%) | 125/192 (65%) | 18.9 | 2.02 |
| ppa4 | a2b + AKTE | unfused | 67/100 (67%) | 124/192 (65%) | 15.8 | 2.15 |
| rca | plain | fused | 64/100 (64%) | 120/192 (62%) | 6.5 | 1.39 |
| rca | reshare | fused | 65/100 (65%) | 125/192 (65%) | 6.5 | 1.42 |
| rca | reshare + sim | fused | 68/100 (68%) | 125/192 (65%) | 6.7 | 1.41 |
| rca | a2b | fused | 71/100 (71%) | 130/192 (68%) | 10.2 | 1.50 |
| rca | a2b + AKTE | fused | 66/100 (66%) | 122/192 (64%) | 10.1 | 1.40 |
| ppa | plain | fused | 66/100 (66%) | 121/192 (63%) | 9.9 | 1.42 |
| ppa | reshare | fused | 66/100 (66%) | 118/192 (61%) | 9.6 | 1.43 |
| ppa | reshare + sim | fused | 67/100 (67%) | 118/192 (61%) | 9.9 | 1.48 |
| ppa | a2b | fused | 72/100 (72%) | 118/192 (61%) | 13.4 | 1.48 |
| ppa | a2b + AKTE | fused | 67/100 (67%) | 126/192 (66%) | 11.3 | 1.45 |
| ppa4 | plain | fused | 68/100 (68%) | 119/192 (62%) | 13.2 | 1.30 |
| ppa4 | reshare | fused | 64/100 (64%) | 123/192 (64%) | 13.4 | 1.37 |
| ppa4 | reshare + sim | fused | 62/100 (62%) | 123/192 (64%) | 14.9 | 1.47 |
| ppa4 | a2b | fused | 67/100 (67%) | 123/192 (64%) | 16.9 | 1.35 |
| ppa4 | a2b + AKTE | fused | 65/100 (65%) | 121/192 (63%) | 13.9 | 1.60 |

* Fused BN is as accurate as unfused with this model (single batch 62-72 vs 65-71, multi-batch 118-130 vs
  121-131). With the older model it was not (30-40% single batch, 69-73 of 192), so fusion's 32-bit range is a
  property of the weights, not of the protocol.
* The spread between variants (62-72) is truncation noise: the variants draw different amounts of randomness,
  so different values wrap. The adders are exact (`RELU_RANDOM`: 0 of 32,768 wrong). With the older model, one
  variant moved between 112 and 126 of 192 over three SRNG seeds.
* The 100-image builds also give single-batch throughput on flare / polynize: preprocessing 6.5-18.9 s and online
  1.3-2.2 s per 100 images (fused RCA 6.5 s + 1.4 s), against 1.3-2.7 s and 0.35-0.50 s per 10 images.

### Older model (`Cifar_adam_001`)

The runtime matrix used `Cifar_adam_001/ResNet50_avg_CIFAR-10_standard_best.bin` (74.48% in PyTorch, 73% on the first
100 and 69.8% on the first 192 images). Its 2PC accuracy is lower: 55-68% of 192 unfused, 30-40% single batch and
69-73 of 192 with fused BN.

| setting | protocol | images | accuracy |
|---|---|---|---|
| plaintext, float | PyTorch | 100 | 73% |
| plaintext, weights, BN scales and activations rounded / truncated to 5 fractional bits | PyTorch | 100 | 75% |
| standard config (32 bit, FRACTIONAL 5, probabilistic truncation), dbg branch | Trio | 100 | 56% |
| same, master branch (`155c935`) | Trio | 100 | 56% |
| 64 bit, FRACTIONAL 12 | Trio | 100 | 73% |
| 32 bit, `TRUNC_APPROACH=1` (reduced-slack truncation) | Trio | 100 | 78% |
| 32 bit, `TRUNC_DELAYED=1` | Trio | 100 | 58% |
| 32 bit, FRACTIONAL 8 | Trio | 100 | 13% (overflow) |

* The fixed-point quantization costs nothing (75% in plaintext). The loss comes from the probabilistic
  (SecureML-style) truncation in a 32-bit ring: a local truncation of a value with 10 fractional bits wraps with
  probability about |x| / 2^32, and a wrapped activation is off by 2^27. With 64 bits or reduced-slack truncation
  the accuracy is at the plaintext level; master gives the same 56%. The AdamW model is far less sensitive to it
  (Trio 71% vs 56%).
* ABY2 has no reduced-slack or exact truncation (`prepare_trunc_2k_inputs` / `prepare_B2A` are missing), so the
  2PC variants cannot use `TRUNC_APPROACH=1`.
