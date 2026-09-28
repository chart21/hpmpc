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
  variant (median), multi-batch once.

The harness scripts are in `scripts/variants/` (`vb.sh`, `vball.sh`, `vrun.sh`, `vsum.sh`, `mxall.sh`,
`analyze.py`); raw CSVs and the generated tables are in `docs/variant_data/`.

## Summary

* **Single batch** (10 images): preprocessing 1.7-2.7 s for RCA and PPA, 3.1-5 s for the a2b families and
  PPA4; online 0.35-0.58 s. Online is bound by round latency (RCA 1,752 rounds, PPA 776, PPA4 625): with the
  threaded ReLU levels, PPA and PPA4 are now faster online than RCA, and the a2b families save the A2B round
  per ReLU. Both machine pairs are within about 10% of each other.
* **Multi-batch** (192 images, 24 processes): preprocessing 12-25 s (RCA / PPA) and 31-47 s (PPA4) on
  flare / polynize, online 0.5-0.7 s; algofi / goracle is 1.5-1.9x slower (compute-bound, no AVX-512).
* **Fixes that changed runtimes** (round 1 -> round 2): CHEETAH channels without Nagle (a2b with PPA / PPA4
  and PPA4 reshare + sim preprocessing 8.4-10.8 s -> 3.5-5.0 s single batch), OT packs sized for the whole
  preprocessing, bit-packed COT multiply, CHEETAH_THREADS > 1 in multi-batch (port stride bug; 25 -> 14 s).
* **Correctness**: every variant's threaded build gives the serial build's output, and repeated runs are
  bit-identical, except PPA4 reshare + sim with fused BN (runs differ, open). The a2b families are wrong in
  multi-batch (13-27 of 192): the A2B mask bake only exists for DATTYPE == BITLENGTH.
* **Accuracy**: all 2PC variants lose about 10-17 points to plaintext (plaintext 73% on the first 100 images,
  69.8% on the first 192). The cause is the probabilistic truncation in the 32-bit ring, not the fixed-point
  quantization and not the adders; Trio with reduced-slack truncation reaches 78%. See Accuracy below.

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

## Correctness

* Threaded vs serial: the output hash of the threaded build equals the serial build's for all 30 single-batch
  variants on flare / polynize except PPA4 reshare + sim with fused BN, whose two runs differ from each other
  (also in round 2 and on algofi / goracle). In multi-batch, two serial runs showed one process with a
  different hash (rare, not reproduced in three reruns) and two runs lost a process to the connection hangs
  fixed in `b16c6e9` / `96c3a85`.
* Round 2 reproduces round 1's hashes in 44 of 45 single-batch and 25 of 26 multi-batch variants compared
  (the exceptions are the nondeterministic PPA4 reshare + sim fused and a broken a2b multi-batch variant):
  the TCP and COT-multiply changes do not change values.
* a2b in multi-batch is wrong (13-27 of 192 correct): `A2B_CONV_BAKE_ACTIVE` requires `DATTYPE == BITLENGTH`,
  and `A2B_ONLINE_OPT` without the bake is known to be broken (`docs/A2B_CONV_BAKE.md`).

## Accuracy

The trained model (`ResNet50_avg_CIFAR-10_standard_best`, 74.48% on the full test set in PyTorch) reaches
73% on the first 100 and 69.8% on the first 192 test images in plaintext (`/root/plain_eval.py`).

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
| 2PC variants, 32 bit, probabilistic truncation | ABY2 | 192 | 55-68% (RCA 112-126 of 192 over SRNG seeds 0-2) |

* The fixed-point quantization itself costs nothing (75% in plaintext). What costs 17 points is the
  probabilistic (SecureML-style) truncation in a 32-bit ring: a local truncation of a value with 10
  fractional bits wraps with probability about |x| / 2^32, and a wrapped activation is off by 2^27. With 64
  bits or with reduced-slack truncation the accuracy is at the plaintext level. The branch is not the cause:
  master gives the same 56%.
* The adders are exact (`RELU_RANDOM`: 0 of 32,768 wrong for RCA, PPA, PPA4, 2PC and Trio). The differences
  between adders in the matrix (e.g. 112 vs 131 of 192) are the truncation noise: they consume different
  amounts of randomness, and SRNG seeds 0-2 alone move RCA between 112 and 126.
* ABY2 has no reduced-slack or exact truncation (`prepare_trunc_2k_inputs` / `prepare_B2A` are missing), so
  the 2PC variants cannot use `TRUNC_APPROACH=1` yet; implementing it is the way to plaintext accuracy at 32 bits.
* Fused BN is low in every variant (30-40% single batch, 69-73 of 192 multi-batch): fusion needs more than
  32 bits (`docs/STATUS.md`).
