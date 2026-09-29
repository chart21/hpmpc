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
* Accuracy: the same threaded builds with the AdamW model (`MX_MODEL=wd`), single batch as 100-image builds.
* Multi-batch traffic: the 24 processes' log lines interleave, so `vsum.sh` takes the median of the intact
  per-process lines times the process count (every process sends the same). Round 1's multi-batch traffic
  column summed fragments and is wrong; round 2's CSVs were corrected from the logs.

The harness scripts are in `scripts/variants/` (`vb.sh`, `vball.sh`, `vrun.sh`, `vsum.sh`, `mxall.sh`,
`analyze.py`, `doc_tables.py`, `acc_table.py`); raw CSVs and the generated tables are in `docs/variant_data/`.

## Summary

* **Single batch** (10 images, round 2): preprocessing on flare / polynize 1.7-2.2 s (RCA), 2.3-2.6 s (PPA) and
  2.8-3.8 s (PPA4) with fused BN, 2.5-3.0 / 3.0-3.3 / 3.5-4.4 s unfused; algofi / goracle is 4-18% slower.
  Online 0.36-0.51 s (flare / polynize) and 0.33-0.58 s (algofi / goracle). Online is bound by round latency
  (RCA 1,641-1,801 rounds, PPA 665-829, PPA4 514-674): with the threaded ReLU levels, PPA and PPA4 are slightly
  faster online than RCA. The reshare families add 49-53 rounds and change little.
* **Multi-batch** (192 images, 24 processes): preprocessing 12.5-17 s (RCA), 15.6-20 s (PPA) and 21-34 s (PPA4)
  on flare / polynize, online 0.49-0.71 s; algofi / goracle needs 21-47 s and 0.76-1.07 s, 1.4-1.8x more
  (compute-bound: AVX2 instead of AVX-512 in the HE and GEMM kernels).
* **Fixes that changed runtimes** (round 1 -> round 2): CHEETAH channels without Nagle (a2b with PPA / PPA4
  and PPA4 reshare + sim preprocessing 8.4-10.8 s -> 3.5-5.0 s single batch), OT packs sized for the whole
  preprocessing, bit-packed COT multiply, CHEETAH_THREADS > 1 in multi-batch (port stride bug; 25 -> 14 s).
* **Correctness**: every variant's threaded build gives the serial build's output, and repeated runs are
  bit-identical. PPA4 reshare + sim with fused BN differed between runs until `992e66b` (a shared counter
  in P0's A2B on the worker pool). The a2b families are wrong in multi-batch (13-27 of 192): the A2B mask bake
  only exists for DATTYPE == BITLENGTH.
* **Accuracy** (AdamW ResNet50, standard truncation): 62-73% of 100 images for all 30 variants in single batch
  (plaintext 76%, Trio 71%), 60-67% of 192 in multi-batch (plaintext 71.9%), fused BN included. With the older
  `Cifar_adam_001` model the probabilistic truncation in the 32-bit ring cost 10-17 points (Trio 56% vs 73%).

## Results (round 2)

Median of two runs (single batch) or one run (multi-batch), threaded builds, `Cifar_adam_001` (the runtime does not
depend on the weights). "MB pre" is P0's preprocessing traffic summed over processes; accuracy on 10 / 192 images of
this model (see Accuracy for the AdamW model). Per-phase times, the serial comparison and per-pair tables are in
`docs/variant_data/matrix_round2.md`.
### Single batch (10 images, one process)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 2.51 | 2.72 | 0.474 | 0.557 | 1752 | 230 | 6/10 | 6/10 |
| rca | reshare | unfused | 2.48 | 2.68 | 0.497 | 0.584 | 1801 | 230 | 6/10 | 7/10 |
| rca | reshare + sim | unfused | 2.54 | 2.65 | 0.464 | 0.571 | 1801 | 230 | 6/10 | 7/10 |
| rca | a2b | unfused | 2.99 | 3.11 | 0.514 | 0.438 | 1752 | 258 | 6/10 | 7/10 |
| rca | a2b + AKTE | unfused | 2.96 | 3.12 | 0.514 | 0.456 | 1752 | 258 | 7/10 | 7/10 |
| ppa | plain | unfused | 3.00 | 3.43 | 0.465 | 0.504 | 776 | 269 | 8/10 | 8/10 |
| ppa | reshare | unfused | 3.07 | 3.43 | 0.439 | 0.531 | 829 | 269 | 8/10 | 9/10 |
| ppa | reshare + sim | unfused | 3.24 | 3.45 | 0.446 | 0.544 | 829 | 269 | 8/10 | 9/10 |
| ppa | a2b | unfused | 3.14 | 3.61 | 0.449 | 0.423 | 776 | 284 | 8/10 | 8/10 |
| ppa | a2b + AKTE | unfused | 3.25 | 3.53 | 0.433 | 0.417 | 780 | 284 | 7/10 | 8/10 |
| ppa4 | plain | unfused | 4.35 | 4.72 | 0.425 | 0.500 | 625 | 387 | 6/10 | 6/10 |
| ppa4 | reshare | unfused | 4.23 | 4.76 | 0.480 | 0.557 | 674 | 387 | 5/10 | 6/10 |
| ppa4 | reshare + sim | unfused | 4.35 | 4.99 | 0.497 | 0.532 | 674 | 398 | 6/10 | 6/10 |
| ppa4 | a2b | unfused | 4.37 | 4.94 | 0.435 | 0.412 | 625 | 402 | 6/10 | 7/10 |
| ppa4 | a2b + AKTE | unfused | 3.54 | 3.84 | 0.475 | 0.433 | 625 | 315 | 8/10 | 6/10 |
| rca | plain | fused | 1.73 | 1.85 | 0.426 | 0.402 | 1641 | 140 | 4/10 | 4/10 |
| rca | reshare | fused | 1.72 | 1.84 | 0.416 | 0.378 | 1690 | 140 | 4/10 | 4/10 |
| rca | reshare + sim | fused | 1.76 | 1.90 | 0.408 | 0.380 | 1690 | 140 | 4/10 | 4/10 |
| rca | a2b | fused | 2.18 | 2.29 | 0.423 | 0.375 | 1641 | 168 | 4/10 | 4/10 |
| rca | a2b + AKTE | fused | 2.23 | 2.29 | 0.427 | 0.360 | 1641 | 168 | 4/10 | 5/10 |
| ppa | plain | fused | 2.35 | 2.58 | 0.373 | 0.376 | 665 | 180 | 4/10 | 4/10 |
| ppa | reshare | fused | 2.25 | 2.65 | 0.372 | 0.378 | 718 | 180 | 4/10 | 4/10 |
| ppa | reshare + sim | fused | 2.35 | 2.64 | 0.376 | 0.361 | 718 | 180 | 4/10 | 4/10 |
| ppa | a2b | fused | 2.62 | 2.84 | 0.384 | 0.333 | 665 | 194 | 5/10 | 4/10 |
| ppa | a2b + AKTE | fused | 2.44 | 2.68 | 0.373 | 0.327 | 669 | 194 | 4/10 | 4/10 |
| ppa4 | plain | fused | 3.46 | 3.91 | 0.360 | 0.363 | 514 | 297 | 4/10 | 4/10 |
| ppa4 | reshare | fused | 3.36 | 3.90 | 0.422 | 0.389 | 563 | 297 | 4/10 | 4/10 |
| ppa4 | reshare + sim | fused | 3.60 | 4.09 | 0.382 | 0.366 | 563 | 308 | 4/10 | 4/10 |
| ppa4 | a2b | fused | 3.81 | 4.18 | 0.375 | 0.326 | 514 | 312 | 4/10 | 4/10 |
| ppa4 | a2b + AKTE | fused | 2.80 | 3.06 | 0.410 | 0.361 | 514 | 225 | 4/10 | 4/10 |

### Multi-batch (192 images, 24 processes x 8 lanes)

| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |
|---|---|---|---|---|---|---|---|---|---|---|
| rca | plain | unfused | 13.92 | 22.98 | 0.615 | 0.974 | 1738 | 8884.0 | 112/192 | 114/192 |
| rca | reshare | unfused | 13.88 | 23.08 | 0.659 | 0.974 | 1787 | 8884.0 | 109/192 | 112/192 |
| rca | reshare + sim | unfused | 14.04 | 23.02 | 0.635 | 1.037 | 1787 | 8884.0 | 109/192 | 112/192 |
| rca | a2b | unfused | 17.04 | 26.48 | 0.614 | 0.867 | 1738 | 9300.0 | 25/192 | 26/192 |
| rca | a2b + AKTE | unfused | 16.19 | 26.01 | 0.607 | 1.038 | 1738 | 9300.0 | 18/192 | 24/192 |
| ppa | plain | unfused | 17.22 | 26.95 | 0.668 | 1.055 | 758 | 9021.8 | 120/192 | 110/192 |
| ppa | reshare | unfused | 17.27 | 27.28 | 0.617 | 1.028 | 807 | 8998.8 | 121/192 | 113/192 |
| ppa | reshare + sim | unfused | 17.06 | 27.10 | 0.616 | 0.895 | 807 | 8998.8 | 121/192 | 113/192 |
| ppa | a2b | unfused | 20.11 | 30.42 | 0.590 | 1.017 | 758 | 9437.8 | 25/192 | 16/192 |
| ppa | a2b + AKTE | unfused | 18.33 | 28.63 | 0.519 | 1.008 | 758 | 9369.0 | 19/192 | 16/192 |
| ppa4 | plain | unfused | 31.54 | 44.89 | 0.490 | 0.757 | 611 | 11380.0 | 122/192 | 131/192 |
| ppa4 | reshare | unfused | 31.15 | 44.31 | 0.530 | 0.758 | 660 | 11334.1 | 116/192 | 134/192 |
| ppa4 | reshare + sim | unfused | 30.98 | 44.43 | 0.527 | 0.899 | 660 | 11540.5 | 116/192 | 134/192 |
| ppa4 | a2b | unfused | 33.63 | 46.94 | 0.507 | 0.802 | 611 | 11727.2 | 27/192 | 16/192 |
| ppa4 | a2b + AKTE | unfused | 22.97 | 34.26 | 0.571 | 0.995 | 611 | 10168.0 | 15/192 | 27/192 |
| rca | plain | fused | 12.47 | 20.92 | 0.604 | 0.974 | 1632 | 7187.9 | 71/192 | 73/192 |
| rca | reshare | fused | 12.48 | 20.86 | 0.707 | 0.994 | 1681 | 7187.9 | 72/192 | 72/192 |
| rca | reshare + sim | fused | 12.46 | 20.94 | 0.613 | 1.035 | 1681 | 7187.9 | 72/192 | 72/192 |
| rca | a2b | fused | 15.46 | 24.56 | 0.641 | 0.989 | 1632 | 7603.9 | 16/192 | 13/192 |
| rca | a2b + AKTE | fused | 14.76 | 23.86 | 0.584 | 1.067 | 1632 | 7603.9 | 16/192 | 18/192 |
| ppa | plain | fused | 15.68 | 24.87 | 0.535 | 0.952 | 652 | 7325.6 | 72/192 | 72/192 |
| ppa | reshare | fused | 15.63 | 25.05 | 0.628 | 0.900 | 701 | 7302.6 | 72/192 | 72/192 |
| ppa | reshare + sim | fused | 16.07 | 24.98 | 0.574 | 0.963 | 701 | 7302.6 | 72/192 | 72/192 |
| ppa | a2b | fused | 18.59 | 28.38 | 0.569 | 1.034 | 652 | 7741.7 | 26/192 | 21/192 |
| ppa | a2b + AKTE | fused | 16.79 | 26.34 | 0.561 | 0.981 | 652 | 7672.8 | 25/192 | 21/192 |
| ppa4 | plain | fused | 30.21 | 43.00 | 0.534 | 0.903 | 505 | 9684.0 | 69/192 | 69/192 |
| ppa4 | reshare | fused | 29.73 | 42.13 | 0.575 | 0.791 | 554 | 9638.0 | 73/192 | 73/192 |
| ppa4 | reshare + sim | fused | 29.55 | 42.27 | 0.540 | 0.852 | 554 | 9844.5 | 73/192 | 73/192 |
| ppa4 | a2b | fused | 31.58 | 44.80 | 0.565 | 0.795 | 505 | 10031.1 | 14/192 | 12/192 |
| ppa4 | a2b + AKTE | fused | 21.39 | 32.35 | 0.561 | 1.042 | 505 | 8471.9 | 15/192 | 15/192 |

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

## Correctness

* Threaded vs serial: the output hash of the threaded build equals the serial build's for all 60 variants
  (30 single batch, 30 multi-batch) on both pairs. Until `992e66b` PPA4 reshare + sim with fused BN was the exception: its runs
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
* a2b in multi-batch is wrong (13-27 of 192 correct): `A2B_CONV_BAKE_ACTIVE` requires `DATTYPE == BITLENGTH`,
  and `A2B_ONLINE_OPT` without the bake is known to be broken (`docs/A2B_CONV_BAKE.md`).

## Accuracy

Accuracy is measured with the AdamW-trained ResNet50 (`adam_001_wd/ResNet50_avg_AdamW_d05_wd003_lr0001_ep100_acc74_35.bin`,
74.35% on the full test set, `python nn/Pygeon/download_pretrained.py adam_001_wd`), with the standard truncation
(`TRUNC_APPROACH=0`, `TRUNC_DELAYED=0`). Every working variant is above 60%: 62-73 of the first 100 images in single
batch (plaintext 76), 116-129 of the first 192 in multi-batch (plaintext 138). Runs: `MX_MODEL=wd mxall.sh`, single
batch with 100-image builds (`a_*`, `NUM_INPUTS=100`) on flare / polynize, multi-batch on algofi / goracle.

| setting | images | correct |
|---|---|---|
| plaintext, float (PyTorch, `/root/plain_eval_wd.py`) | 100 / 192 | 76 / 138 (71.9%) |
| Trio, standard config (32 bit, FRACTIONAL 5, probabilistic truncation) | 100 | 71 |
| Trio, `TRUNC_APPROACH=1` | 100 | 70 |
| 2PC, single batch, all 30 variants | 100 | 62-73 |
| 2PC, multi-batch, the 18 variants without a2b | 192 | 116-129 |
| 2PC, multi-batch, a2b families | 192 | 14-25 (wrong, see Correctness) |

| adder | family | BN | single batch, 100 images | multi-batch, 192 images | pre s (100 images) | online s (100 images) |
|---|---|---|---|---|---|---|
| rca | plain | unfused | 68/100 (68%) | 124/192 (65%) | 11.7 | 1.98 |
| rca | reshare | unfused | 70/100 (70%) | 121/192 (63%) | 11.5 | 1.96 |
| rca | reshare + sim | unfused | 67/100 (67%) | 121/192 (63%) | 11.5 | 1.98 |
| rca | a2b | unfused | 68/100 (68%) | 24/192 (12%) | 15.3 | 2.04 |
| rca | a2b + AKTE | unfused | 72/100 (72%) | 23/192 (12%) | 14.9 | 1.94 |
| ppa | plain | unfused | 66/100 (66%) | 117/192 (61%) | 15.7 | 1.88 |
| ppa | reshare | unfused | 71/100 (71%) | 121/192 (63%) | 14.4 | 1.91 |
| ppa | reshare + sim | unfused | 68/100 (68%) | 121/192 (63%) | 14.8 | 1.92 |
| ppa | a2b | unfused | 65/100 (65%) | 22/192 (11%) | 18.0 | 2.13 |
| ppa | a2b + AKTE | unfused | 62/100 (62%) | 16/192 (8%) | 16.1 | 2.17 |
| ppa4 | plain | unfused | 68/100 (68%) | 129/192 (67%) | 25.2 | 1.79 |
| ppa4 | reshare | unfused | 67/100 (67%) | 128/192 (67%) | 25.5 | 1.94 |
| ppa4 | reshare + sim | unfused | 71/100 (71%) | 128/192 (67%) | 25.6 | 2.00 |
| ppa4 | a2b | unfused | 71/100 (71%) | 22/192 (11%) | 29.9 | 1.95 |
| ppa4 | a2b + AKTE | unfused | 73/100 (73%) | 24/192 (12%) | 21.0 | 2.12 |
| rca | plain | fused | 64/100 (64%) | 120/192 (62%) | 8.7 | 1.40 |
| rca | reshare | fused | 65/100 (65%) | 119/192 (62%) | 8.3 | 1.41 |
| rca | reshare + sim | fused | 68/100 (68%) | 119/192 (62%) | 8.2 | 1.43 |
| rca | a2b | fused | 71/100 (71%) | 18/192 (9%) | 12.5 | 1.45 |
| rca | a2b + AKTE | fused | 66/100 (66%) | 14/192 (7%) | 11.9 | 1.42 |
| ppa | plain | fused | 66/100 (66%) | 117/192 (61%) | 11.8 | 1.46 |
| ppa | reshare | fused | 66/100 (66%) | 120/192 (62%) | 11.6 | 1.44 |
| ppa | reshare + sim | fused | 67/100 (67%) | 120/192 (62%) | 11.6 | 1.57 |
| ppa | a2b | fused | 72/100 (72%) | 25/192 (13%) | 15.2 | 1.46 |
| ppa | a2b + AKTE | fused | 67/100 (67%) | 16/192 (8%) | 13.0 | 1.45 |
| ppa4 | plain | fused | 67/100 (67%) | 116/192 (60%) | 22.7 | 1.34 |
| ppa4 | reshare | fused | 66/100 (66%) | 122/192 (64%) | 22.5 | 1.36 |
| ppa4 | reshare + sim | fused | 63/100 (63%) | 122/192 (64%) | 24.5 | 1.49 |
| ppa4 | a2b | fused | 69/100 (69%) | 14/192 (7%) | 25.5 | 1.39 |
| ppa4 | a2b + AKTE | fused | 71/100 (71%) | 17/192 (9%) | 18.3 | 1.71 |

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
