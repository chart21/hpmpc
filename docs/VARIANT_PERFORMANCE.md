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
* Accuracy: the same threaded builds with the AdamW model (`MX_MODEL=wd`), single batch as 100-image builds.
* Multi-batch traffic: the 24 processes' log lines interleave, so `vsum.sh` takes the median of the intact
  per-process lines times the process count (every process sends the same). Round 1's multi-batch traffic
  column summed fragments and is wrong; round 2's CSVs were corrected from the logs.

The harness scripts are in `scripts/variants/` (`vb.sh`, `vball.sh`, `vrun.sh`, `vsum.sh`, `mxall.sh`,
`analyze.py`, `doc_tables.py`, `acc_table.py`); raw CSVs and the generated tables are in `docs/variant_data/`.

## Summary

* **Single batch** (10 images, round 4): preprocessing on flare / polynize 1.6-2.2 s (RCA), 2.3-2.5 s (PPA) and
  2.5-2.8 s (PPA4) with fused BN, 1.9-2.4 / 2.6-2.9 / 2.7-3.1 s unfused (the a2b families are the upper end);
  algofi / goracle is 1-16% slower. Online 0.35-0.52 s (flare / polynize) and 0.32-0.59 s (algofi / goracle).
  Online is bound by round latency (RCA 1,641-1,801 rounds, PPA 665-829, PPA4 514-674): with the threaded ReLU
  levels, PPA and PPA4 are slightly faster online than RCA. The reshare families add 49-53 rounds and change little.
* **Multi-batch** (192 images, 24 processes, round 4): preprocessing 12.5-16.6 s (RCA), 15.7-19.8 s (PPA) and
  18.7-24.4 s (PPA4) on flare / polynize, online 0.54-0.79 s; algofi / goracle needs 21-36 s (1.4-1.7x) and
  0.8-1.1 s (compute-bound: AVX2 instead of AVX-512 in the HE and GEMM kernels).
* **Speedups during the study**: CHEETAH channels without Nagle (a2b with PPA / PPA4 and PPA4 reshare + sim
  8.4-10.8 s -> 3.5-5.0 s single batch), OT packs sized for the whole preprocessing, bit-packed COT multiply,
  CHEETAH_THREADS > 1 in multi-batch (port stride bug; 25 -> 14 s), Beaver 3/4-tuples with one OT per shared
  factor (PPA4 4.35 -> 2.68 s single batch unfused, 31.5 -> 22.6 s multi-batch), BN triples of all layers in
  one product (unfused RCA 2.51 -> 1.94 s single batch).
* **Correctness**: every variant's threaded build gives the serial build's output, and repeated runs are
  bit-identical (60 of 60 on flare / polynize in round 4). Fixed during the study: a race in P0's A2B on the
  worker pool (PPA4 reshare + sim fused, `992e66b`) and the a2b families in multi-batch, whose A2B mask bake
  only existed for DATTYPE == BITLENGTH (13-29 of 192 correct before `64404c3`, now as accurate as plain).
* **Accuracy** (AdamW ResNet50, standard truncation): 62-73% of 100 images for all 30 variants in single batch
  (plaintext 76%, Trio 71%), 60-69% of 192 for all 30 variants in multi-batch (plaintext 71.9%), fused BN
  included. With the older `Cifar_adam_001` model the probabilistic truncation in the 32-bit ring cost 10-17
  points (Trio 56% vs 73%).
* **CUT_FRACTIONAL_BITS_OPT**: every run above uses it (`=1`). It removes 12% of RCA's online rounds and 8-11%
  of the ReLU traffic; runtime and accuracy change within noise (see "CUT_FRACTIONAL_BITS_OPT, A_KNOWN=0 and public
  weights" below).
* **Public weights** (`TRUNC_DELAYED=1 BIT_INJECTION_TRUNC_SIM=1`): 0.92-2.03 s preprocessing per 10 images, 3.5-14.3 s
  per 192, online 0.27-0.42 s; AdamW accuracy 63-73 of 100 and 115-131 of 192 after the unfused-BN fix (`4dd999b`).
  **A_KNOWN=0** runs but classifies at chance: ABY2 lacks exact / reduced-slack truncation.

## Results (round 4)

Median of two runs (single batch) or one run (multi-batch), threaded builds, `Cifar_adam_001` (the runtime does not
depend on the weights). "MB pre" is P0's preprocessing traffic summed over processes; accuracy on 10 / 192 images of
this model (see Accuracy for the AdamW model). The multi-batch a2b rows are the `64404c3` builds; their algofi /
goracle runs used the AdamW model, so their accuracy is in the Accuracy section. Per-phase times, the serial
comparison and per-pair tables: `docs/variant_data/matrix_round4.md` (round 2: `matrix_round2.md`).
### Single batch (10 images, one process)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 1.94 | 2.15 | 0.515 | 0.568 | 1752 | 229 | 6/10 | 7/10 |
| rca | reshare | unfused | 2.07 | 2.12 | 0.465 | 0.590 | 1801 | 229 | 6/10 | 7/10 |
| rca | reshare + sim | unfused | 1.99 | 2.18 | 0.483 | 0.575 | 1801 | 229 | 7/10 | 7/10 |
| rca | a2b | unfused | 2.41 | 2.56 | 0.485 | 0.453 | 1752 | 256 | 8/10 | 7/10 |
| rca | a2b + AKTE | unfused | 2.40 | 2.59 | 0.478 | 0.449 | 1752 | 256 | 7/10 | 7/10 |
| ppa | plain | unfused | 2.56 | 2.89 | 0.439 | 0.520 | 776 | 268 | 8/10 | 8/10 |
| ppa | reshare | unfused | 2.65 | 2.87 | 0.443 | 0.516 | 829 | 268 | 8/10 | 8/10 |
| ppa | reshare + sim | unfused | 2.56 | 2.96 | 0.461 | 0.536 | 829 | 268 | 7/10 | 8/10 |
| ppa | a2b | unfused | 2.88 | 3.10 | 0.454 | 0.413 | 776 | 283 | 7/10 | 8/10 |
| ppa | a2b + AKTE | unfused | 2.72 | 2.96 | 0.437 | 0.407 | 780 | 283 | 7/10 | 8/10 |
| ppa4 | plain | unfused | 2.68 | 3.11 | 0.417 | 0.477 | 625 | 321 | 5/10 | 6/10 |
| ppa4 | reshare | unfused | 2.84 | 3.04 | 0.409 | 0.512 | 674 | 321 | 5/10 | 6/10 |
| ppa4 | reshare + sim | unfused | 2.92 | 3.35 | 0.435 | 0.523 | 674 | 337 | 6/10 | 6/10 |
| ppa4 | a2b | unfused | 3.07 | 3.31 | 0.430 | 0.406 | 625 | 335 | 5/10 | 8/10 |
| ppa4 | a2b + AKTE | unfused | 2.86 | 3.18 | 0.458 | 0.412 | 625 | 300 | 7/10 | 6/10 |
| rca | plain | fused | 1.66 | 1.82 | 0.403 | 0.389 | 1641 | 140 | 4/10 | 4/10 |
| rca | reshare | fused | 1.64 | 1.85 | 0.407 | 0.374 | 1690 | 140 | 4/10 | 4/10 |
| rca | reshare + sim | fused | 1.84 | 1.86 | 0.433 | 0.371 | 1690 | 140 | 4/10 | 4/10 |
| rca | a2b | fused | 2.22 | 2.30 | 0.441 | 0.394 | 1641 | 168 | 4/10 | 4/10 |
| rca | a2b + AKTE | fused | 2.14 | 2.33 | 0.388 | 0.365 | 1641 | 168 | 4/10 | 5/10 |
| ppa | plain | fused | 2.29 | 2.59 | 0.385 | 0.364 | 665 | 180 | 4/10 | 4/10 |
| ppa | reshare | fused | 2.40 | 2.58 | 0.369 | 0.367 | 718 | 180 | 4/10 | 4/10 |
| ppa | reshare + sim | fused | 2.31 | 2.65 | 0.391 | 0.364 | 718 | 180 | 4/10 | 4/10 |
| ppa | a2b | fused | 2.50 | 2.82 | 0.377 | 0.343 | 665 | 194 | 5/10 | 4/10 |
| ppa | a2b + AKTE | fused | 2.44 | 2.77 | 0.390 | 0.339 | 669 | 194 | 4/10 | 4/10 |
| ppa4 | plain | fused | 2.61 | 2.76 | 0.349 | 0.342 | 514 | 232 | 4/10 | 4/10 |
| ppa4 | reshare | fused | 2.53 | 2.77 | 0.359 | 0.362 | 563 | 232 | 4/10 | 4/10 |
| ppa4 | reshare + sim | fused | 2.80 | 3.06 | 0.388 | 0.370 | 563 | 248 | 4/10 | 4/10 |
| ppa4 | a2b | fused | 2.78 | 3.07 | 0.355 | 0.317 | 514 | 247 | 4/10 | 4/10 |
| ppa4 | a2b + AKTE | fused | 2.74 | 2.89 | 0.402 | 0.347 | 514 | 212 | 4/10 | 4/10 |

### Multi-batch (192 images, 24 processes x 8 lanes)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 13.71 | 22.85 | 0.787 | 0.884 | 1738 | 8884 | 110/192 | 116/192 |
| rca | reshare | unfused | 13.74 | 22.88 | 0.748 | 0.858 | 1787 | 8884 | 111/192 | 116/192 |
| rca | reshare + sim | unfused | 13.94 | 22.87 | 0.760 | 0.868 | 1787 | 8884 | 111/192 | 116/192 |
| rca | a2b | unfused | 16.58 | 26.41 | 0.770 | 0.940 | 1738 | 9300 | 100/192 | - |
| rca | a2b + AKTE | unfused | 16.02 | 25.65 | 0.681 | 0.912 | 1738 | 9300 | 99/192 | - |
| ppa | plain | unfused | 16.85 | 26.87 | 0.743 | 0.853 | 758 | 9022 | 121/192 | 117/192 |
| ppa | reshare | unfused | 17.03 | 26.90 | 0.728 | 0.824 | 807 | 8999 | 118/192 | 114/192 |
| ppa | reshare + sim | unfused | 16.84 | 26.84 | 0.678 | 0.911 | 807 | 8999 | 118/192 | 114/192 |
| ppa | a2b | unfused | 19.77 | 30.07 | 0.623 | 0.956 | 758 | 9438 | 123/192 | - |
| ppa | a2b + AKTE | unfused | 18.03 | 28.07 | 0.599 | 0.809 | 758 | 9369 | 122/192 | - |
| ppa4 | plain | unfused | 22.63 | 32.55 | 0.652 | 0.799 | 611 | 10238 | 117/192 | 129/192 |
| ppa4 | reshare | unfused | 21.41 | 32.33 | 0.569 | 0.899 | 660 | 10238 | 112/192 | 132/192 |
| ppa4 | reshare + sim | unfused | 21.49 | 32.50 | 0.616 | 0.845 | 660 | 10548 | 112/192 | 132/192 |
| ppa4 | a2b | unfused | 24.41 | 35.67 | 0.565 | 0.797 | 611 | 10654 | 117/192 | - |
| ppa4 | a2b + AKTE | unfused | 19.94 | 30.10 | 0.619 | 0.905 | 611 | 9774 | 118/192 | - |
| rca | plain | fused | 12.52 | 20.91 | 0.599 | 0.903 | 1632 | 7188 | 71/192 | 73/192 |
| rca | reshare | fused | 12.59 | 20.90 | 0.606 | 1.051 | 1681 | 7188 | 72/192 | 72/192 |
| rca | reshare + sim | fused | 12.62 | 20.99 | 0.617 | 1.042 | 1681 | 7188 | 72/192 | 72/192 |
| rca | a2b | fused | 15.76 | 24.62 | 0.601 | 1.067 | 1632 | 7604 | 68/192 | - |
| rca | a2b + AKTE | fused | 14.81 | 23.94 | 0.622 | 0.972 | 1632 | 7604 | 70/192 | - |
| ppa | plain | fused | 15.75 | 24.84 | 0.659 | 1.002 | 652 | 7326 | 72/192 | 72/192 |
| ppa | reshare | fused | 15.67 | 25.08 | 0.585 | 0.970 | 701 | 7303 | 72/192 | 72/192 |
| ppa | reshare + sim | fused | 15.74 | 25.03 | 0.664 | 0.971 | 701 | 7303 | 72/192 | 72/192 |
| ppa | a2b | fused | 18.55 | 28.67 | 0.598 | 1.085 | 652 | 7742 | 68/192 | - |
| ppa | a2b + AKTE | fused | 16.98 | 26.47 | 0.550 | 0.995 | 652 | 7673 | 67/192 | - |
| ppa4 | plain | fused | 20.23 | 30.64 | 0.583 | 0.912 | 505 | 8542 | 70/192 | 72/192 |
| ppa4 | reshare | fused | 20.16 | 30.66 | 0.582 | 1.038 | 554 | 8542 | 71/192 | 71/192 |
| ppa4 | reshare + sim | fused | 20.41 | 30.52 | 0.543 | 0.913 | 554 | 8852 | 71/192 | 71/192 |
| ppa4 | a2b | fused | 23.15 | 34.14 | 0.561 | 0.992 | 505 | 8958 | 73/192 | - |
| ppa4 | a2b + AKTE | fused | 18.65 | 28.54 | 0.594 | 0.970 | 505 | 8078 | 70/192 | - |

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
images (27-53% and 38-75% less than the same variant with secret weights), online 0.27-0.41 s and 0.33-0.42 s
(5-29% and 29-50% less). AdamW accuracy: 63-73 of 100 (single batch) and 115-131 of 192 (multi-batch). Unfused
BN needed `4dd999b`: with public weights and delayed truncation the first conv's output is still the data owner's
sharing (the other party's mask is 0), and the BN's `trunc_pr_in_place` wrapped on it (0 of 10 correct).

**A_KNOWN=0** (`res_fp_kpw.csv`, `*_k0`): AB triples for the linear layers, 1.4-1.9x the preprocessing traffic (RCA
unfused, 10 images: 229 -> 428 MB). Single batch 1.50-2.83 s preprocessing, 3-18% *faster* than A_KNOWN=1 (both parties
encrypt, so the pipelined convolutions keep both busy: conv 0.65 -> 0.48 s); multi-batch 13.8-26.4 s, 3-16% slower
(three threads per process). Online 0.34-0.52 s and 0.56-0.73 s.
Accuracy is at chance level (0-1 of 10 in every configuration tried, also with `TRUNC_DELAYED=1` and the
truncation in the bit injection): with both masks shared, the probabilistic local truncation of the linear
layers' outputs wraps. It needs exact or reduced-slack truncation for ABY2. The runtimes are valid because the
runtime does not depend on the values.

## Accuracy

Accuracy is measured with the AdamW-trained ResNet50 (`adam_001_wd/ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin`,
74.35% on the full test set, `python nn/Pygeon/download_pretrained.py adam_001_wd`), with the standard truncation
(`TRUNC_APPROACH=0`, `TRUNC_DELAYED=0`). Every working variant is above 60%: 62-73 of the first 100 images in single
batch (plaintext 76), 116-132 of the first 192 in multi-batch (plaintext 138). Runs: `MX_MODEL=wd mxall.sh`, single
batch with 100-image builds (`a_*`, `NUM_INPUTS=100`) on flare / polynize, multi-batch on algofi / goracle.

| setting | images | correct |
|---|---|---|
| plaintext, float (PyTorch, `/root/plain_eval_wd.py`) | 100 / 192 | 76 / 138 (71.9%) |
| Trio, standard config (32 bit, FRACTIONAL 5, probabilistic truncation) | 100 | 71 |
| Trio, `TRUNC_APPROACH=1` | 100 | 70 |
| 2PC, single batch, all 30 variants | 100 | 62-73 |
| 2PC, multi-batch, the 18 variants without a2b | 192 | 116-129 |
| 2PC, multi-batch, a2b families (`64404c3`) | 192 | 117-132 (14-25 before the fix) |

| adder | family | BN | single batch, 100 images | multi-batch, 192 images | pre s (100 images) | online s (100 images) |
|---|---|---|---|---|---|---|
| rca | plain | unfused | 68/100 (68%) | 124/192 (65%) | 11.7 | 1.98 |
| rca | reshare | unfused | 70/100 (70%) | 121/192 (63%) | 11.5 | 1.96 |
| rca | reshare + sim | unfused | 67/100 (67%) | 121/192 (63%) | 11.5 | 1.98 |
| rca | a2b | unfused | 68/100 (68%) | 119/192 (62%) | 15.3 | 2.04 |
| rca | a2b + AKTE | unfused | 72/100 (72%) | 130/192 (68%) | 14.9 | 1.94 |
| ppa | plain | unfused | 66/100 (66%) | 117/192 (61%) | 15.7 | 1.88 |
| ppa | reshare | unfused | 71/100 (71%) | 121/192 (63%) | 14.4 | 1.91 |
| ppa | reshare + sim | unfused | 68/100 (68%) | 121/192 (63%) | 14.8 | 1.92 |
| ppa | a2b | unfused | 65/100 (65%) | 127/192 (66%) | 18.0 | 2.13 |
| ppa | a2b + AKTE | unfused | 62/100 (62%) | 132/192 (69%) | 16.1 | 2.17 |
| ppa4 | plain | unfused | 68/100 (68%) | 129/192 (67%) | 25.2 | 1.79 |
| ppa4 | reshare | unfused | 67/100 (67%) | 128/192 (67%) | 25.5 | 1.94 |
| ppa4 | reshare + sim | unfused | 71/100 (71%) | 128/192 (67%) | 25.6 | 2.00 |
| ppa4 | a2b | unfused | 71/100 (71%) | 129/192 (67%) | 29.9 | 1.95 |
| ppa4 | a2b + AKTE | unfused | 73/100 (73%) | 132/192 (69%) | 21.0 | 2.12 |
| rca | plain | fused | 64/100 (64%) | 120/192 (62%) | 8.7 | 1.40 |
| rca | reshare | fused | 65/100 (65%) | 119/192 (62%) | 8.3 | 1.41 |
| rca | reshare + sim | fused | 68/100 (68%) | 119/192 (62%) | 8.2 | 1.43 |
| rca | a2b | fused | 71/100 (71%) | 122/192 (64%) | 12.5 | 1.45 |
| rca | a2b + AKTE | fused | 66/100 (66%) | 127/192 (66%) | 11.9 | 1.42 |
| ppa | plain | fused | 66/100 (66%) | 117/192 (61%) | 11.8 | 1.46 |
| ppa | reshare | fused | 66/100 (66%) | 120/192 (62%) | 11.6 | 1.44 |
| ppa | reshare + sim | fused | 67/100 (67%) | 120/192 (62%) | 11.6 | 1.57 |
| ppa | a2b | fused | 72/100 (72%) | 125/192 (65%) | 15.2 | 1.46 |
| ppa | a2b + AKTE | fused | 67/100 (67%) | 123/192 (64%) | 13.0 | 1.45 |
| ppa4 | plain | fused | 67/100 (67%) | 116/192 (60%) | 22.7 | 1.34 |
| ppa4 | reshare | fused | 66/100 (66%) | 122/192 (64%) | 22.5 | 1.36 |
| ppa4 | reshare + sim | fused | 63/100 (63%) | 122/192 (64%) | 24.5 | 1.49 |
| ppa4 | a2b | fused | 69/100 (69%) | 117/192 (61%) | 25.5 | 1.39 |
| ppa4 | a2b + AKTE | fused | 71/100 (71%) | 126/192 (66%) | 18.3 | 1.71 |

* Fused BN is as accurate as unfused with this model (single batch 63-72 vs 62-73, multi-batch 116-122 vs
  117-129). With the older model it was not (30-40% single batch, 69-73 of 192), so fusion's 32-bit range is a
  property of the weights, not of the protocol.
* The spread between variants (62-73) is truncation noise: the variants draw different amounts of randomness,
  so different values wrap. The adders are exact (`RELU_RANDOM`: 0 of 32,768 wrong). With the older model, one
  variant moved between 112 and 126 of 192 over three SRNG seeds.
* The 100-image builds also give single-batch throughput on flare / polynize: preprocessing 8.2-30 s and online
  1.3-2.2 s per 100 images (fused RCA 8.2 s + 1.4 s), against 1.7-4.4 s and 0.36-0.51 s per 10 images.

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
