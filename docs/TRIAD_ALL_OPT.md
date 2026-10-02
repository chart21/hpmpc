# Triad 2PC single-batch all_opt configs: optimized runtime

Scope: the six single-batch configs of `measurements/configs/artifacts/triad/2pc/single_batch/2PC_single_batch_all_opt_*.conf`
(A2bits, A2bits A_Not_Known, A2bits Public, reshared, reshared A_Not_Known, Reshared Public), each with
`FUNCTION_IDENTIFIER=87,187,287` (ResNet50 on one ImageNet image in CHEETAH's layout, 230x230 input, RCA / PPA / PPA4
adder) and `COMPRESS=0,1`: 36 builds. Runtime is always measured with dummy model and data (`MODELOWNER=-1
DATAOWNER=-1`, as in the configs); the runtime does not depend on the values (`TRIPLE_ZERO=ON`).

Use cases (the labels of the tables and the paper): **UC1** weights known to none (`A_KNOWN=0`, the `A_Not_Known`
configs), **UC2** weights known to one party, the model owner (`A_KNOWN=1`, here also
`MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1`: the plain A2bits / reshared configs), **UC3** weights known to all
(public weights). Tables are sorted UC1 -> UC3.

## Builds

* `conf`: the config as given (plus `PRINT_OUTPUT_HASH=1 NET_WAIT_STATS=1 USE_CUDA_GEMM=0` for the statistics), at hpmpc
  `161cf1d`. Conv triples use CHEETAH's `HomConv2DSS` (the config does not set `CHEETAH_CONV_PACKED`).
* `fin` (CSV `fin2`): the config plus every optimization of this branch, at hpmpc `aeab3ae` / ConvTriple `fa7b952` / flexNN `b1ce619`:
  `CHEETAH_CONV_PACKED=1 CHEETAH_CONV_PIPELINE=1 CUT_FRACTIONAL_BITS_OPT=1 ADDITIONAL_RELU_THREADS=24 RNG_AHEAD=1
  SEND_BUFFER=100000 RECV_BUFFER=100000` and, for the A2bits configs (`A2B_ONLINE_OPT=1`), `A2B_CONV_BAKE=1`. The
  code-level optimizations are on by default (packed/pipelined/concurrent conv triples, exact wire format, shared-OT
  Beaver tuples, batched BN, `TCP_NODELAY`, element-parallel ReLU levels, the pool spin and the vectorized GEMM below).
  `ADDITIONAL_RELU_THREADS` must equal the config's `ADDITIONAL_GEMM_THREADS=24` (one worker pool); 31/31 was not faster.
* `fin64`: `fin` with `CHEETAH_THREADS=64` instead of the config's 32 (a tuning option, see below; measured at `ce0608a`,
  whose worker pool differs only in the online wait). The CSV `fin` is `ce0608a` (pause spin, shared cache line).
* `opt`: `fin`'s flags at `161cf1d`, i.e. before this round's changes (profiling baseline).

Scripts: build lists from `gen.py`, harness `scripts/variants/` (`vb.sh`, `vball.sh`, `mxall.sh`); CSVs in
`docs/variant_data/triad/`.

## Correctness

Checked on CIFAR-10 (ResNet50 71/171/271 with the same flags, `MODELOWNER=P_0 DATAOWNER=P_1`, AdamW model, 10 images):

* COMPRESS=0: all 18 configs correct, 4-7 of 10 (plaintext 6 of 10); the output hashes are identical before and after
  this round's changes (the GEMM change is bit-exact).
* **The A2bits configs as given are not correct** (2026-09-30): 0-2 of 10 in UC1, UC2 and UC3 and every adder
  (`cconf_*` in `res_fp_cchk.csv`). They set `A2B_ONLINE_OPT=1` without `A2B_CONV_BAKE`; the unbaked path generates
  the MSB adder's Beaver triples in the preprocessing pass for the Boolean addition's input `ia` while the online phase
  uses its output `[c]` (bug 2 of `docs/A2B_CONV_BAKE.md`). The optimized flags without the bake (`cnob_*`) are equally
  wrong (UC3: the same output hashes as as-given); with the bake, 4-7 of 10. The reshared configs are correct as given
  (4-7 of 10). This is why as-given A2bits UC3 sends 69 MiB less in preprocessing (see Communication).
* COMPRESS=1 is not accurate by construction and is only kept running (user decision): the ReLU sees bits 12..19
  (`REDUCED_BITLENGTH_m=12, k=20`), and with `FRACTIONAL=5` the dropped carry of the low 12 bits flips the sign of
  any activation below 128.0. Plain RCA without reshare, MWK or any optimization of this branch classifies 0 of 10
  (COMPRESS=0: 5 of 10).
* UC2 (`MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1`, A2bits and reshared): correct in all adders, 4-7 of 10.

## Profile (flare / polynize, `opt` builds)

Preprocessing is bound by OT generation. Per inference (A2bits RCA, 4.46 s): OT pack setup 0.9 s (32 ferret packs),
AND triples 0.24 s, the A2B bake's boolean addition 1.46 s (31 carry levels over 9.0M values; one level pays a ferret
extension of all packs, 0.74 s), conv triples 0.71 s, the preprocessing pass of the network 0.46 s (its ReLU levels
run on one thread), COT/multiplexer/FC 0.16 s. PPA4 adds the Beaver 3/4-tuples (1.2 + 1.9-2.0 s); reshared PPA4 pays
six ferret extension rounds after setup (1.7 s of OT generation). In a CPU profile over all threads, ferret's LPN step
takes 31-44%, the MITCCRH key schedules 8-11% and MPCOT 10-13%.

The online phase is not bandwidth-bound and, after this round, not compute-bound either: 84% of the online CPU
samples are idle pool workers; RCA's 1,714 rounds at ~0.3 ms are close to its floor. Turning off the NIC's interrupt
coalescing cut the ping from 0.25 to 0.08 ms but the RCA online phase only by 7% (not used for the numbers below);
busy-polling the receive path (`NET_SPIN_US`) changed nothing.

## Changes in this round (hpmpc `ce0608a`)

| change | effect |
|---|---|
| `GEMM_POOL_SPIN`: pool workers wait ~20 ms (was ~100 us) before sleeping on the futex, single process only (`ce0608a`) | the workers are still waiting when the next circuit level arrives: Zen 4 ImageNet online -17..20% (A2bits RCA 0.67 -> 0.55 s, PPA4 0.77 -> 0.62 s) |
| counters on their own cache lines, idle workers wait with MONITORX/MWAITX (`aeab3ae`, `GEMM_POOL_MWAITX`) | with the long spin, completions on the shared cache line slowed Zen 3's public-weight builds by up to 35% (reshared public RCA 0.51 -> 0.69 s); aligned: 0.55 s. MWAITX vs pause: equal within noise on both CPUs, no SMT contention |
| `GEMM_fast.hpp`: the GEMM accumulation for A_KNOWN=1 (MWK and AB2) as uint32 products with a 4x32 register-blocked kernel, weight operand cached across a layer's calls, fused packing and accumulation | bit-identical; Zen 3 ImageNet conv online -24..28% (online -5..12%); CIFAR MWK online 0.41-0.44 -> 0.34-0.38 s; ~15x less online CPU on Zen 4 (no wall-time change there: not on the critical path) |
| `A2B_CONV_BAKE_ACTIVE` needs a full-width ReLU | COMPRESS=1 A2bits builds aborted (the bake committed masks for 8 of 32 slices); they now run the A2B unbaked |
| `stream_parallel_for` reports call site and stream on a misuse | found the COMPRESS abort |

Tried and not kept: `CHEETAH_OT_GROUP=2` (32 packs with 2 ferret threads: worse than 64 packs), `NET_SPIN_US`, 31/31
worker threads.

## Results

ImageNet ResNet50, one image, dummy weights; fp = flare / polynize (Zen 4), ag = algofi / goracle (Zen 3). Final data
(2026-10-01): `conf` = the config as given (res_*_final4.csv, 3 runs alternating with the round-3 code), `fin` = the
configs plus every optimization at the round-7 code, hpmpc `92a2c0e` / ConvTriple `8d8f244` / flexNN `417f12b`, without
`RNG_AHEAD` (res_*_final9.csv, `fin9_`, 3 runs); `fin64`: `CHEETAH_THREADS=64` at `ce0608a` (earlier round).

* Every build is faster than as given online on both pairs, and in preprocessing all but two UC3 COMPRESS=1 builds on
  ag, 1-2% slower (a2bpw_ppa_c1 1.88 vs 1.86 s, rspw_ppa_c1 2.01 vs 1.97 s; code untouched by rounds 4-7: spread).
* UC1 and UC2: preprocessing 11.7-17.3 -> 1.7-5.5 s on fp (2.9-7.1x), 15.1-21.9 -> 2.7-6.9 s on ag (3.0-6.0x); online
  0.77-1.55 -> 0.34-0.63 s on fp (2.0-3.0x), 0.72-1.73 -> 0.24-0.63 s on ag (2.2-4.2x). UC2 online 0.34-0.56 s (fp),
  0.24-0.48 s (ag).
* UC3: preprocessing 1.05-1.21x faster (ag 0.98-1.13x), online 1.5-2.2x (ag 1.4-2.2x).
* `RNG_AHEAD` is no longer in the optimized set: at the round-3 code it saved ~0.04 s online on Zen 4 (median) but cost
  up to 0.17 s on Zen 3 (UC3 COMPRESS=1: 0.51 vs 0.36 s; the online draws wait, most likely for the ring's producer).
* `CHEETAH_THREADS=64` (earlier round): preprocessing -8% (median, fp) / -4% (ag), up to -20% for PPA4, +19..30% for UC3
  RCA; a per-config choice, not a default.
* Correctness: all 18 COMPRESS=0 CIFAR builds classify 3-7 of 10 at the final code (`res_fp_pc10.csv`).

### COMPRESS=0

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 14.84 | 3.72 | 3.92 | 18.88 | 4.44 | 4.90 | 1.179 | 0.492 | 1.256 | 0.510 |
| UC1, A2bits | PPA | 16.11 | 4.72 | 5.11 | 19.71 | 5.58 | 6.50 | 1.323 | 0.501 | 1.343 | 0.504 |
| UC1, A2bits | PPA4 | 17.28 | 5.24 | 5.77 | 21.85 | 6.15 | 7.11 | 1.371 | 0.597 | 1.440 | 0.634 |
| UC1, reshared | RCA | 13.37 | 2.08 | 2.63 | 17.05 | 3.30 | 3.62 | 1.392 | 0.605 | 1.363 | 0.565 |
| UC1, reshared | PPA | 15.20 | 3.90 | 4.38 | 19.40 | 5.09 | 5.82 | 1.550 | 0.632 | 1.588 | 0.590 |
| UC1, reshared | PPA4 | 17.06 | 5.46 | 5.96 | 21.69 | 6.86 | 8.06 | 1.550 | 0.597 | 1.726 | 0.553 |
| UC2, A2bits | RCA | 13.82 | 3.68 | 3.77 | 17.53 | 4.48 | 4.83 | 1.155 | 0.430 | 1.007 | 0.381 |
| UC2, A2bits | PPA | 15.00 | 4.67 | 5.22 | 18.61 | 5.69 | 6.41 | 1.200 | 0.447 | 1.114 | 0.352 |
| UC2, A2bits | PPA4 | 16.54 | 5.23 | 5.67 | 20.05 | 6.19 | 7.06 | 1.242 | 0.531 | 1.204 | 0.481 |
| UC2, reshared | RCA | 11.90 | 2.02 | 2.55 | 16.12 | 3.17 | 3.53 | 1.191 | 0.515 | 1.148 | 0.386 |
| UC2, reshared | PPA | 14.04 | 3.85 | 4.44 | 18.51 | 5.14 | 5.69 | 1.362 | 0.557 | 1.285 | 0.442 |
| UC2, reshared | PPA4 | 16.17 | 5.51 | 6.03 | 20.77 | 6.83 | 7.89 | 1.506 | 0.536 | 1.485 | 0.420 |
| UC3, A2bits | RCA | 3.75 | 3.38 | 3.05 | 3.75 | 3.67 | 3.40 | 0.917 | 0.507 | 0.878 | 0.480 |
| UC3, A2bits | PPA | 4.70 | 4.39 | 4.34 | 4.92 | 4.85 | 5.12 | 1.028 | 0.461 | 0.968 | 0.452 |
| UC3, A2bits | PPA4 | 6.15 | 5.78 | 4.96 | 6.62 | 6.14 | 5.69 | 1.064 | 0.565 | 1.024 | 0.578 |
| UC3, reshared | RCA | 2.05 | 1.76 | 2.32 | 2.31 | 2.04 | 2.76 | 1.071 | 0.610 | 0.918 | 0.520 |
| UC3, reshared | PPA | 3.69 | 3.25 | 3.32 | 4.34 | 3.96 | 4.07 | 1.128 | 0.586 | 1.093 | 0.498 |
| UC3, reshared | PPA4 | 5.45 | 4.95 | 4.42 | 6.55 | 6.07 | 5.28 | 0.971 | 0.514 | 0.887 | 0.428 |

### COMPRESS=1

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 13.20 | 2.13 | 2.58 | 16.99 | 3.25 | 3.38 | 0.845 | 0.410 | 0.924 | 0.403 |
| UC1, A2bits | PPA | 13.13 | 2.11 | 2.59 | 16.92 | 3.26 | 3.45 | 0.954 | 0.430 | 0.941 | 0.401 |
| UC1, A2bits | PPA4 | 13.31 | 2.19 | 2.75 | 16.94 | 3.36 | 3.63 | 0.933 | 0.429 | 0.989 | 0.443 |
| UC1, reshared | RCA | 12.52 | 1.78 | 2.05 | 16.77 | 2.78 | 2.97 | 0.876 | 0.442 | 0.988 | 0.432 |
| UC1, reshared | PPA | 13.12 | 2.24 | 2.67 | 17.15 | 3.29 | 3.59 | 1.164 | 0.470 | 1.228 | 0.467 |
| UC1, reshared | PPA4 | 13.81 | 2.22 | 3.44 | 17.77 | 3.36 | 4.66 | 1.295 | 0.441 | 1.497 | 0.441 |
| UC2, A2bits | RCA | 11.71 | 2.06 | 2.41 | 15.49 | 3.16 | 3.33 | 0.765 | 0.345 | 0.719 | 0.255 |
| UC2, A2bits | PPA | 11.95 | 1.98 | 2.51 | 15.69 | 3.14 | 3.39 | 0.824 | 0.364 | 0.730 | 0.244 |
| UC2, A2bits | PPA4 | 12.18 | 2.13 | 2.64 | 15.73 | 3.18 | 3.59 | 0.810 | 0.338 | 0.764 | 0.269 |
| UC2, reshared | RCA | 11.74 | 1.65 | 2.02 | 15.08 | 2.68 | 2.87 | 0.775 | 0.395 | 0.795 | 0.312 |
| UC2, reshared | PPA | 11.87 | 2.19 | 2.51 | 15.93 | 3.03 | 3.55 | 0.984 | 0.401 | 0.919 | 0.311 |
| UC2, reshared | PPA4 | 12.39 | 2.20 | 3.40 | 16.41 | 3.11 | 4.51 | 1.181 | 0.392 | 1.274 | 0.307 |
| UC3, A2bits | RCA | 1.77 | 1.62 | 1.71 | 1.86 | 1.82 | 1.91 | 0.576 | 0.383 | 0.522 | 0.370 |
| UC3, A2bits | PPA | 1.82 | 1.73 | 1.74 | 1.86 | 1.88 | 1.97 | 0.608 | 0.378 | 0.549 | 0.360 |
| UC3, A2bits | PPA4 | 2.04 | 1.82 | 1.87 | 2.05 | 2.01 | 2.11 | 0.638 | 0.389 | 0.565 | 0.383 |
| UC3, reshared | RCA | 1.36 | 1.13 | 1.25 | 1.40 | 1.29 | 1.31 | 0.616 | 0.397 | 0.524 | 0.357 |
| UC3, reshared | PPA | 1.68 | 1.55 | 1.61 | 1.97 | 2.01 | 1.74 | 0.658 | 0.412 | 0.577 | 0.380 |
| UC3, reshared | PPA4 | 1.94 | 1.75 | 1.81 | 2.12 | 1.92 | 1.88 | 0.711 | 0.383 | 0.585 | 0.377 |

## UC3: as given vs optimized (2026-09-30)

Question: why does the as-given A2bits UC3 build look slightly better in preprocessing (Fig. 16 on Zen 3, Fig. 17)?

* **Traffic**: optimized A2bits UC3 COMPRESS=0 sends 69 MiB more in preprocessing: the A2B bake rebases all 9.0 M ReLU
  inputs (none comes straight from a masked conv with public weights), one word per value and party. As given skips the
  rebase and is **wrong** (0-2 of 10 on CIFAR, see Correctness); nothing correct can skip it with this design.
* **Time, 5 interleaved runs per build** (`res_fp_ab.csv`, `res_ag_ab.csv`): Zen 4: optimized faster in all 12 UC3
  builds (paired median -0.02..-0.20 s). Zen 3: reshared -0.08..-0.31 s; A2bits COMPRESS=0 +0.01..+0.08 s (within the
  spread); A2bits COMPRESS=1 +0.11..+0.14 s (5 of 5 runs slower).
* **Ablation on Zen 3** (`res_ag_abl.csv`, builds `list_ag_abl.txt`: fin minus one flag each, as-given flags at the
  current code `c2_`): only `ADDITIONAL_RELU_THREADS` matters: without it A2bits UC3 RCA COMPRESS=1 preprocesses in 1.84
  s (fin 1.97, conf 1.85) but runs online in 0.54 s (fin 0.39). The code changes since 161cf1d (`c2_`) are neutral.
* **Where**: phase timestamps (temporary instrumentation) put the extra ~0.12 s in P0's `complete_preprocessing`
  (0.44-0.49 vs 0.32-0.34 s); triple generation and the preprocessing pass are unchanged; both main threads use the same
  CPU time. The flag adds a stream-cursor check to every `getRandomVal` and `retrieve_output_share_{bool,arithmetic}`
  call, also in the preprocessing share, which never runs in a parallel level.
* **Tried, not kept**: hook-free `_serial` versions of these functions for `aby2_pre.hpp` (bit-identical outputs on six
  seeded CIFAR builds and two multi-batch builds). Zen 3: COMPRESS=1 -0.13..-0.17 s, but COMPRESS=0 +0.09..+0.13 s
  (incl. reshared UC3, no bake); retrievals only (`pr`): the same; `getRandomVal` only (`ps`): neutral; Zen 4: neutral
  (`res_ag_pg.csv`, `res_ag_pq.csv`, `res_ag_prs.csv`, `res_fp_pq.csv`). A code-layout effect of +-0.1 s on Zen 3, not
  a net gain, so the code stays at `aeab3ae`.

## Communication

MiB sent plus received by P0 (both directions; P1's counters mirror P0's), from the fp logs; every value is identical
in both reruns and on ag. Source `docs/variant_data/triad/comm_fp.csv` (as given), `comm_fp6.csv` (final builds at
the round-5 code), `comm_fp7.csv` (UC1 / UC2 A2bits at the round-6 code) `comm_fp8.csv` (all final builds at
the round-6 code) and `comm_fp9.csv` (round 7, the same bytes); triple counters in MiB, network counters
in 10^6 bytes, converted; plot data `docs/paper/data/triad_comm_c{0,1}.dat` (`make_triad.py`), paper Fig. 17.
`triples`: HE + OT generation incl. key exchange; `pre`: triples + the network's preprocessing pass.

* UC1 and UC2: preprocessing 1.35-1.86x less with COMPRESS=0, 1.61-1.92x with COMPRESS=1. Conv triples ~1,000 ->
  479 MiB (UC2), 2,003 -> 958 MiB (UC1). MWK builds also send P1's share corrections (4 B per conv output,
  42 MiB); the pipelined build sends them after the conv batch, so the log counts them under `FC`.
* CUT: PPA4 needs 12 MiB (reshared) / 74 MiB (A2bits, with the bake) fewer multi-input AND tuples; online -4..8% for the
  UC1 / UC2 COMPRESS=0 builds except A2bits PPA4. Not eligible in UC3 (TRUNC_DELAYED=1).
* A2B bake (A2bits, COMPRESS=0): UC3 sends exactly what as given sends (the mask-only forward, round 4; round 3 rebased
  all 9.0 M ReLU inputs, +69 MiB), and computes correctly. UC1 / UC2 rebase only the stem's 200,704 ReLU inputs after
  its pooling (1.5 MiB); the residual sum is baked since rounds 5 (UC1) / 6 (UC2). With COMPRESS=1 the bake is not
  active (reduced bit lengths).

### Communication, COMPRESS=0

| use case, config | adder | triples conf | triples fin | pre conf | pre fin | pre factor | online conf | online fin | online factor |
|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 2414 | 1368 | 2480 | 1426 | 1.74 | 220.9 | 210.2 | 1.05 |
| UC1, A2bits | PPA | 2475 | 1430 | 2595 | 1551 | 1.67 | 338.9 | 321.8 | 1.05 |
| UC1, A2bits | PPA4 | 2710 | 1591 | 2935 | 1798 | 1.63 | 207.9 | 207.9 | 1.00 |
| UC1, reshared | RCA | 2219 | 1174 | 2285 | 1230 | 1.86 | 255.2 | 239.2 | 1.07 |
| UC1, reshared | PPA | 2311 | 1266 | 2435 | 1389 | 1.75 | 373.3 | 348.7 | 1.07 |
| UC1, reshared | PPA4 | 3022 | 1965 | 3225 | 2150 | 1.50 | 242.3 | 232.7 | 1.04 |
| UC2, A2bits | RCA | 1453 | 930 | 1518 | 986 | 1.54 | 178.5 | 167.8 | 1.06 |
| UC2, A2bits | PPA | 1514 | 991 | 1633 | 1111 | 1.47 | 296.5 | 279.3 | 1.06 |
| UC2, A2bits | PPA4 | 1749 | 1152 | 1973 | 1359 | 1.45 | 165.5 | 165.5 | 1.00 |
| UC2, reshared | RCA | 1258 | 735 | 1323 | 790 | 1.68 | 212.8 | 196.7 | 1.08 |
| UC2, reshared | PPA | 1350 | 827 | 1502 | 949 | 1.58 | 330.8 | 306.2 | 1.08 |
| UC2, reshared | PPA4 | 2061 | 1526 | 2312 | 1710 | 1.35 | 200.0 | 190.3 | 1.05 |
| UC3, A2bits | RCA | 406 | 406 | 478 | 478 | 1.00 | 142.2 | 142.2 | 1.00 |
| UC3, A2bits | PPA | 468 | 468 | 593 | 593 | 1.00 | 260.3 | 260.3 | 1.00 |
| UC3, A2bits | PPA4 | 703 | 703 | 933 | 933 | 1.00 | 129.3 | 129.3 | 1.00 |
| UC3, reshared | RCA | 212 | 212 | 284 | 284 | 1.00 | 176.6 | 176.6 | 1.00 |
| UC3, reshared | PPA | 304 | 304 | 462 | 462 | 1.00 | 294.6 | 294.6 | 1.00 |
| UC3, reshared | PPA4 | 886 | 886 | 1143 | 1143 | 1.00 | 163.6 | 163.6 | 1.00 |

### Communication, COMPRESS=1

| use case, config | adder | triples conf | triples fin | pre conf | pre fin | pre factor | online conf | online fin | online factor |
|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 2249 | 1204 | 2264 | 1219 | 1.86 | 169.3 | 169.3 | 1.00 |
| UC1, A2bits | PPA | 2249 | 1204 | 2270 | 1225 | 1.85 | 188.6 | 188.6 | 1.00 |
| UC1, A2bits | PPA4 | 2268 | 1223 | 2313 | 1268 | 1.82 | 162.9 | 162.9 | 1.00 |
| UC1, reshared | RCA | 2166 | 1121 | 2181 | 1136 | 1.92 | 177.9 | 177.9 | 1.00 |
| UC1, reshared | PPA | 2219 | 1174 | 2241 | 1196 | 1.87 | 197.2 | 197.2 | 1.00 |
| UC1, reshared | PPA4 | 2318 | 1273 | 2353 | 1308 | 1.80 | 171.5 | 171.5 | 1.00 |
| UC2, A2bits | RCA | 1288 | 765 | 1302 | 779 | 1.67 | 126.9 | 126.9 | 1.00 |
| UC2, A2bits | PPA | 1288 | 765 | 1308 | 785 | 1.67 | 146.2 | 146.2 | 1.00 |
| UC2, A2bits | PPA4 | 1307 | 785 | 1351 | 828 | 1.63 | 120.5 | 120.5 | 1.00 |
| UC2, reshared | RCA | 1206 | 683 | 1219 | 696 | 1.75 | 135.5 | 135.5 | 1.00 |
| UC2, reshared | PPA | 1258 | 735 | 1286 | 756 | 1.70 | 154.8 | 154.8 | 1.00 |
| UC2, reshared | PPA4 | 1357 | 834 | 1402 | 868 | 1.61 | 129.1 | 129.1 | 1.00 |
| UC3, A2bits | RCA | 242 | 242 | 262 | 262 | 1.00 | 90.6 | 90.6 | 1.00 |
| UC3, A2bits | PPA | 242 | 242 | 268 | 268 | 1.00 | 110.0 | 110.0 | 1.00 |
| UC3, A2bits | PPA4 | 261 | 261 | 311 | 311 | 1.00 | 84.2 | 84.2 | 1.00 |
| UC3, reshared | RCA | 159 | 159 | 180 | 180 | 1.00 | 99.2 | 99.2 | 1.00 |
| UC3, reshared | PPA | 212 | 212 | 245 | 245 | 1.00 | 118.6 | 118.6 | 1.00 |
| UC3, reshared | PPA4 | 289 | 289 | 340 | 340 | 1.00 | 92.8 | 92.8 | 1.00 |

## What is left (profile 2026-09-30, flare / polynize, hpmpc `aeab3ae`)

Profile before round 3 (below). At the final code (`phases2_fp.log`, paper Table 15) the pass takes 0.25-0.64 s over the
18 builds (was 0.43-0.86 s) and the conv triples run alongside it, leaving a tail of 0.28-0.60 s (1.0-1.1 s for reshared
PPA4, where the HE runs at about half speed next to the pass); items 3 and 4 below are therefore done in part.

Phase timestamps of the 18 COMPRESS=0 builds (temporary instrumentation, `phases_fp.log`, table: `docs/paper/make_phases.py`)
and CPU profiles of UC2 A2bits RCA, UC2 reshared PPA4, UC3 reshared RCA:

* OT: setup of the 32 ferret packs 0.84-0.90 s in every build; tuples after it 0.25-0.33 s (RCA) but 3.5 s (PPA4: six
  more extension rounds of all 32 packs; PPA: four; A2bits RCA: two, from the bake). LPN 33-41% of all CPU samples, AES
  key schedules 8-11%. The partly used last round wastes at most one round (~0.3 s).
* Bake Boolean addition (A2bits): 1.5 s (31 COT-multiplication rounds of 15-22 ms + an extension round) and 0.25 s of
  serial input preparation.
* Serial main-thread work in preprocessing: ~1.4 s CPU (per-sample profile): PRE pass 0.43-0.86 s (ReLU levels on one
  thread), triple bookkeeping 0.2-0.3 s, conv tiling search + polyphase weights 0.09 s.
* UC1 / UC2 conv triples (0.71-0.84 s) start only after the PRE pass (it fixes their input masks); no overlap with OT.
* Online: RCA waits 0.15-0.18 s in 1,714-2,012 rounds; main thread: pool hand-offs 0.11 s, serial bit transposition
  (`real_ortho`) 0.05 s, layer glue 0.06 s.
* HE packing (`docs/paper/he_model.py`, reproduces 244.1 + 234.9 MiB): dense packing in the same format would be
  234.7 MiB; N = 8192 never helps; rounding input ciphertexts is limited by the 64-bit flooding (12 bits headroom).
  Output repacking (dense co = 1 inputs + automorphism-based packing of outputs) would reach ~240 MiB with ~2.3e5 key
  switches (merge tree; stages 3-4 alone: 1.2e5 key switches, -169 MiB). Not implemented.

## Round 3 (2026-09-30): conv triples alongside the pass, parallel preprocessing levels

hpmpc `1421df0`, ConvTriple `5b568e9` (both default on):

* `CHEETAH_CONV_ASYNC` (single batch, packed + pipelined convs): the batched conv triples run on their own thread from
  the start of the ABY2 preprocessing pass; the HE pipeline takes layer i once `SetupConv2dTriples` has recorded its
  masks (ConvTriple: `ready(i)` callback of `generateConvTriplesPackedBatch`); `complete_preprocessing` joins before the
  next generator uses the CHEETAH channels.
* `STREAM_PARALLEL_PRE` (with `ADDITIONAL_RELU_THREADS`): the A2B, adder and bit-injection levels of the preprocessing
  pass run on the worker pool (`stream_parallel_pre_for`). The pass's append-only streams (triple types, AB/AB2 triple
  inputs, Boolean-addition / COT / multiplexer inputs, pre-send buffer, stored outputs) get per-worker cursors (`tl_pre`),
  reads use the online `tl_stream` cursors; each worker's use of every stream is checked against the first element's.
  GEMM levels stay serial in preprocessing (MWK pushes into vectors there).
* Correctness: all 18 COMPRESS=0 CIFAR checks (`res_fp_pcif.csv`) and two multi-batch builds (`res_fp_pm2.csv`) give the
  earlier output hashes bit for bit.
* flare / polynize, ImageNet, 3 interleaved runs (`res_fp_as.csv`, `res_fp_ap.csv`; fin2 = aeab3ae, as = async only,
  ap = both): UC1 A2bits RCA 4.54 -> 4.17 -> 4.04 s, UC1 reshared PPA4 6.87 -> 6.37 -> 6.16 s, UC2 A2bits RCA 4.51 ->
  4.12 -> 4.09 s, UC2 reshared PPA4 7.02 -> 6.41 -> 6.16 s, UC3 A2bits RCA 3.72 -> 3.49 s, UC3 reshared RCA 1.92 ->
  1.68 s; online unchanged within noise.

## Round 4 (2026-09-30): P1's bake in the pass, output repacking, privacy fixes, UC3 bake

hpmpc `a57871b`..`c26d99f`, ConvTriple `7778034`, flexNN (PIGEON) `f41b3ba`; flare / polynize only (algofi / goracle
were re-imaged by another user).

* **Reshared PPA4 tail** (`a57871b`). The ~1 s of conv triples after the pass was not the HE competing with the pass:
  P1's pass took 1.28-1.43 s against P0's 0.38-0.50 s, and the HE waits for P1's input masks. 77% of P1's pass was its
  serial GEMM level, where P1 bakes the reshare bits into every conv output mask (RESHARE_OPT_SIM, P1 only); under
  CUT_FRACTIONAL_BITS_OPT the PPA4 rank helper scanned the retrieval order for each of the 32 slices of each of the
  7.3 M outputs. Compile-time (numeric bit, tuple offset) tables: P1's pass 0.55-0.66 s, reshared PPA4 preprocessing
  UC2 6.0 -> 5.5-5.7 s, UC1 6.1-6.4 -> 5.4-5.7 s; CIFAR hashes unchanged.
* **Output repacking** (`CHEETAH_CONV_REPACK=1`, ConvTriple `conv_repack`, default off): the packed convs run in a ring
  of N = 8192 with a special prime; inputs hold channels interleaved (C slots per position, a power of two), one
  filter per product, and the evaluator merges the products of C filters into one dense ciphertext with PackLWEs'
  tree (C - 1 automorphisms, products scaled by C^-1 mod q first), then floods / masks / truncates as before. Galois
  keys once (P1's, both parties' for AB; 1.8 MB each, the public keys of the new ring 0.36 MB). The tiling minimizes
  bytes with each automorphism priced at `CONV_REPACK_KS_BYTES` (default 2048): at 0 it takes 148,622 automorphisms
  for 264 MiB (6.3 s of HE), at 256 53,000 for 271 MiB (3.0 s), at 2048 26,000 for 285 MiB (2.1 s), against 479 MiB
  in 0.71 s without repacking (ConvTriple test, 53 ImageNet convs, 32 threads, all exact for AB2 and AB). Powers of
  two leave 23% of the coefficients unused at 56x56 ... 7x7 (49 * 2^k positions), which the model (`he_model.py`,
  dense outputs) did not count. End to end (`res_fp_rp.csv`, 2 interleaved runs): preprocessing traffic UC1 -18..-31%
  (1,229-2,150 -> 845-1,766 MiB), UC2 -11..-24% (790-1,710 -> 598-1,518 MiB), preprocessing +1.3..+1.7 s, online
  unchanged. Break-even about 1.2 Gbit/s (UC2) and 2.1 Gbit/s (UC1). UC2 output hashes unchanged; UC1 CIFAR 100
  images 72 / 64 (71 / 67 without). The price per automorphism end to end (`res_fp_ks.csv`, `comm_ks.csv`, 2 runs,
  `CONV_REPACK_KS_BYTES` at run time): UC2 A2bits RCA 780 / 786 / 800 MiB in 9.8 / 6.3 / 5.5 s for 0 / 256 / 2048,
  UC1 reshared RCA 804 / 817 / 845 MiB in 8.0 / 4.7 / 3.9 s, UC2 reshared PPA4 1,498 / 1,504 / 1,518 MiB in 11.2 /
  7.9 / 7.0 s: the last 14-41 MiB cost 4 s.
* **Privacy fixes in the A2B bake** (`d8a48ae`, `3c70834`, details in `docs/A2B_CONV_BAKE.md`): committed slots were
  reused by convs not feeding a baked ReLU (1,505,280 slots masked two values, and the rebase revealed the
  downsample conv's mask), and the committed values replayed the passes' own generator stream. Fixed without any
  change in traffic or time; the performance numbers of rounds 1-3 stand. Multi-batch (`res_fp_pm3.csv`): A2bits RCA 129 / 192
  (115 before, new hash), reshared PPA4 the same hash as before.
* **UC3 bake for free** (`A2B_BAKE_MASK_PASS`, default on): a mask-only forward in the pass records every ReLU's input
  mask before the Boolean addition, so no ReLU input is rebased. UC3 A2bits: preprocessing traffic -69 MiB (RCA 546 ->
  478, PPA 661 -> 593, PPA4 1,002 -> 933 MiB), preprocessing 3.54 -> 3.52, 4.63 -> 4.54, 5.92 -> 5.82 s, online
  0.63 -> 0.58, 0.63 -> 0.56, 0.68 -> 0.64 s (`res_fp_mp.csv`, 3 runs). CIFAR 100 images: 63 / 66 (71 / 63 without).
* ReLU-input counts corrected: UC1 / UC2 rebase 1,003,520 inputs (stem + one residual sum), 7.7 MiB, not 1.7 M /
  13 MiB (the CHEETAH layout keeps one of its four residual sums). These could be baked too (stem BatchNorm like a
  conv, the residual partner drawing `lz - l_identity`), see `docs/A2B_CONV_BAKE.md`.

## Round 5 (2026-09-30): conv triples alongside the OT phase, residual sums baked

hpmpc `26119e6`, `3673aad`; ConvTriple `ccb84d8`; flexNN (PIGEON) `74f85d8`, `49b7658`; flare / polynize.

* **Why:** during the OT phase only about 20 of the 64 hardware threads are busy (`/proc/stat` every 50 ms, phase
  stamps; `cpu_phases_fp.txt`): UC2 A2bits RCA 19.6 over its 3.0 s OT phase, reshared RCA 21.9 over 1.3 s; the pass
  keeps 46-50 busy, the conv tail after it 34-35.
* **`CHEETAH_CONV_EARLY`** (default 1; secret weights, packed + pipelined convs, single batch): every ReLU's outputs
  get committed masks (counter-mode values under the party's key, by slot, `g_relu_out`; both phases' bit injections
  take them), so every conv input (a ReLU output, or the network input) is known before the pass. The pass first runs
  the network over the masks (`mask_forward`, generalized from the UC3 mask pass, on a copy of the input): the convs
  record their triple inputs (`RecordConv2dInputs`) and stop, as do FC and BatchNorm. Then the conv triples start on
  four channels of their own (`Keys::get_side_ios`, ports after the regular ones) and the OT phase runs, moved from
  `preprocess_circuit` into the pass (`run_ot_phase`); then the real pass, which checks each conv's inputs against the
  recorded ones. At 32 HE threads the OT phase slowed down by about what the conv triples gained (OT setup stage +0.3 s);
  a sweep of threads and nice values (`res_fp_en.csv`) gave half the threads (`CHEETAH_CONV_EARLY_THREADS`), normal
  priority (nice 10-19 made the conv triples the tail of the PPA4 builds).
* **Residual sums (A2B bake, `A2B_BAKE_RESIDUAL`, default 1):** the conv/FC computed last draws `lz - (the other
  addend's mask)` (ResNet's forward publishes it: the identity, or `temp` when a downsample branch finishes there), so
  the sum carries `lz`. UC1 needs no rebase there; in UC2 (weights known in preprocessing, SecureML truncation) P1's
  masks lie in the truncation's image, so P1 draws fresh and moves the sum alone (`rebase_p1`, P0's delta is zero).
  The stem stays rebased: with `FUSE_CONV_BN=1` its BatchNorm passes the pooling's output on. A BatchNorm bake
  (`A2B_BAKE_BN`) works mechanically but changed the outputs of the multi-batch `FUSE_CONV_BN=0` check (105 instead of
  129 of 192) for reasons not understood yet: off by default (no triad build uses `FUSE_CONV_BN=0`).
* **Check:** in preprocessing every baked ReLU input's mask share must equal its committed slot. It found that with
  dummy weights (`MODELOWNER=-1`) P1 holds a bias mask, which the UC2 bake (P1's masks in the truncation's image) cannot
  compensate: UC2 A2bits ImageNet dummy runs had P1 off after biased convs all along (their outputs never mattered;
  timing unaffected; a real model owner's bias has no mask at P1). The check skips that case.
* **Results** (`res_fp_b.csv`, `comm_b.csv`, 3 interleaved runs against round 4): preprocessing -0.03..-0.34 s over the
  12 UC1 / UC2 builds (RCA -0.18..-0.34), online -0.02..-0.07 s (the bit injections draw no masks online); preprocessing
  traffic -6.1 MiB (UC1 A2bits) / -3.1 MiB (UC2 A2bits), reshared unchanged. CIFAR: all 12 builds and UC3 classify as
  before (`res_fp_bc2.csv`), A2bits hashes unchanged by the residual bake; on the CIFAR ResNet50 (a residual sum per
  block) hpmpc's preprocessing messages 21.5 -> 12.5 MB (UC1) / 21.3 -> 16.8 MB (UC2). Multi-batch: bit for bit as round
  4 (`res_fp_pm6.csv`).
* **Final rerun, both pairs** (algofi / goracle re-imaged and set up again): at hpmpc `f353bb2` (`res_*_final6.csv`)
  Zen 3's builds with short OT phases (COMPRESS=1 UC1 / UC2, reshared RCA) were 0.36-0.70 s slower than round 3 - the
  conv triples at half the threads became their tail (`res_ag_th.csv`: with 32 threads 0.4-0.7 s faster, the long-OT
  PPA4 build prefers 16). hpmpc `7913adb` / ConvTriple `8d8f244`: the limit (`conv_threads_now`) is lifted when the OT
  phase returns. At that code (`res_*_final7.csv`, 3 runs of all 36 builds per pair, the paper's Table 14 / Fig. 16):
  against round 3 preprocessing median -0.15 s on Zen 4 (-0.80..+0.09) and -0.03 s on Zen 3 (-0.81..+0.20; a few
  short-OT UC1 / UC2 builds up to +0.11 s, UC3 reshared - untouched by rounds 4-5 - up to +0.20 s: noise / layout);
  online median -0.03 / -0.02 s. As given -> final: UC1 / UC2 preprocessing 11.7-17.3 -> 1.7-5.5 s (3.0-7.2x) on Zen 4,
  15.1-21.9 -> 2.7-7.0 s (3.0-6.0x) on Zen 3; online 2.0-3.0x / 2.1-4.1x; UC3 preprocessing 1.1-1.2x / 1.0-1.1x, online
  1.5-2.0x / 1.5-2.2x (`docs/paper/triad_ranges.py`). Traffic (`comm_fp6.csv`, Fig. 17): UC3 A2bits now sends exactly
  what as given sends (the mask-only forward), and computes correctly.

## Round 6 (2026-10-01): UC2 residual sums without rebase, UC1 masks

hpmpc `841335d`, flexNN (PIGEON) `417f12b`; flare / polynize.

* **UC2 residual sums:** P1's conv/FC masks lie in the SecureML truncation's image, so its residual partner could not
  draw `lz_1 - (a free mask)` and P1 moved the sum alone (`rebase_p1`, 3.1 MiB). Now the other addend's P1 mask `m_b`
  is committed as well (a conv/FC producer: a PRF value in the image; a ReLU producer: its committed bit-injection
  outputs from the mask-only forward), P1's committed mask of the sum is `lz_1 = m_a + m_b`, and the partner draws
  `m_a`, in the image. Which sums are committed follows from the network (both parties agree); details in
  `docs/A2B_CONV_BAKE.md`. ImageNet UC2 A2bits: P0's received pass messages -3.21 MB (802,816 x 4 B; RCA 32.97 ->
  29.76 MB), UC1 unchanged, COMPRESS=1 unchanged (no bake there); UC1 and UC2 now rebase only the stem's 200,704 ReLU
  inputs after its pooling. CIFAR ResNet50 (16 residual sums, 12 of them ReLU-produced) UC2: pass messages 6.13 /
  10.63 -> 6.13 / 6.13 MB sent / received; the P1 invariant check (real weights) passes everywhere; 5 / 5 / 6 of 10
  (RCA / PPA / PPA4, round 5 6 / 5 / 5, new hashes: P1's prescribed shares changed), 69 / 100 on 100 images
  (`res_fp_pc6.csv`, `comm_fp7.csv`).
* **UC1 privacy fix:** P1's committed masks were narrowed to the truncation's image with every `TRUNC_DELAYED=0` build.
  In UC1 (`A_KNOWN=0`) the masks are free draws and P1 sends `TRUNC(m_1) + l_1`, which a narrowed `l_1` does not
  fully hide from P0. Now only `A_KNOWN=1` narrows. UC1 outputs bit for bit as before (they do not depend on the masks).
* **Timing:** an A/B series right after the change (`res_fp_r6ab.csv`, `res_fp_r6n.csv`) had UC2 unchanged but UC1 A2bits
  RCA 0.1-0.2 s slower than round 5, all in P1's OT pack setup, and a control build with the old narrowing as fast as
  round 5; since the OT packs are set up before `init_a2b_bake` runs, this looked like a code-layout effect. It was
  not: in the final rerun below the identical binary (same md5) preprocesses in 3.70-3.73 s, faster than round 5
  (3.77-3.80 s). Series-to-series variation of P1's OT setup.
* **Final rerun, both pairs** (hpmpc `e1066ce` = code `841335d`, ConvTriple `8d8f244`, flexNN `417f12b`;
  `res_*_final8.csv`, 3 runs of all 36 builds per pair; the paper's Table 14 / Fig. 16): against round 5 (final7)
  preprocessing median +0.01 s on Zen 4 (-0.21..+0.10) and -0.00 s on Zen 3 (-0.12..+0.16), online within 0.04 s.
  As given -> final: UC1 / UC2 preprocessing 11.7-17.3 -> 1.6-5.5 s (3.0-7.3x) on Zen 4, 15.1-21.9 -> 2.7-7.0 s
  (3.0-6.0x) on Zen 3; online 2.0-3.2x / 2.1-4.3x; UC3 preprocessing 1.05-1.21x / 0.98-1.14x, online 1.5-1.9x /
  1.4-2.2x (`docs/paper/triad_ranges.py`). One build is not faster than as given: UC3 A2bits RCA COMPRESS=0 on Zen 3,
  3.82 vs 3.75 s (round 5: 3.77) - the correct bake's mask-only forward, which the as-given (wrong) build skips.
  Traffic (`comm_fp8.csv`): every build sends what round 5 sent, except UC2 A2bits COMPRESS=0 (-3.21 MB).
* **Phases at the final code** (`phases8_fp.log`, 18 instrumented COMPRESS=0 builds x 2, `docs/paper/make_phases.py`,
  the paper's Table 15; patch `docs/variant_data/triad/phpatch3.py` adds `ot_start` / `ot_end` around the OT phase, which runs inside the pass
  with secret weights): no conv tail is left in any build (round 3: 0.28-1.1 s); OT setup 0.85-1.04 s (about 0.1 s more
  with the conv triples alongside), tuples 0.26-0.35 s (RCA) / 1.6-3.5 s (PPA4), the bake's Boolean addition
  1.42-1.52 s, the pass's own work 0.25-0.75 s (UC3 A2bits 0.55-0.75 with its mask-only forward), online waiting
  0.06-0.20 s.
* **CIFAR at the final code** (`res_fp_pc8.csv`, all 18 COMPRESS=0 builds, 10 images): 3-7 of 10 (plaintext 6); every
  hash as in round 5 except UC2 A2bits (round 6 changed P1's prescribed shares).

## Round 7 (2026-10-01): what the bake costs, its runtime, repeated committed masks

hpmpc `b4577f5`, `92a2c0e`; flare / polynize, algofi / goracle.

* **What the bake costs** (`res_fp_bake8.csv`, round 6; `res_fp_bake10.csv` + `comm_fp9.csv` / `comm_nb10.csv`, round 7):
  the 9 A2bits COMPRESS=0 builds against the same flags with `A2B_CONV_BAKE=0` (the as-given A2B: same Boolean addition,
  after the pass, wrong signs), 3 interleaved runs. Triple and online traffic identical byte for byte; the pass sends
  +0.80 MB per direction in UC1 / UC2 (the stem's 200,704 remasked ReLU inputs, 1.53 MiB = 0.1%), nothing in UC3.
  Round 6: preprocessing +0.03..+0.17 s, online +0.01..+0.10 s. The online cost was the baked A2B's serial prepare /
  complete (one global [c] cursor), the preprocessing cost partly its serial commitment.
* **Baked A2B on the pool** (`b4577f5`): [c] addressed by value (`tl_a2b_c`), prepare / complete on the worker pool.
  **Parallel commitment**: ia from `prf_value(kTweakA2bIa, slot)` on `A2B_BAKE_INIT_THREADS` (16) threads.
  **Residual bake off with public weights** (`msb_input_residual()`): UC3 with `A2B_BAKE_MASK_PASS=0` aborted in the bake
  check (a public-weight conv draws no mask).
* **Repeated committed masks** (`92a2c0e`, privacy): `BUFFER_SIZE` was `AES_DATTYPE / DATTYPE` without parentheses, so
  `prf_value` took `k / 512 / 32` and `(k % 512) / 32`: each value came back for 32 consecutive k, 16 distinct per 16,384.
  Affected: the mask-only forward's truncation masks (round 4), every committed ReLU output mask (round 5: UC3 and the
  CHEETAH_CONV_EARLY builds of UC1 / UC2), UC2's committed residual masks (round 6). No timing, traffic or accuracy number
  changes (masks never change the work); it surfaced when b4577f5 drew the commitment from `prf_value` too and P1's
  narrowed UC2 masks came out tiny (CIFAR 0-2/10; bisected with `A2B_BAKE_PAR_PREP` / `A2B_BAKE_INIT_PRF` knobs, since
  removed). Parenthesized; `prf_value_check()` aborts if neighbouring values repeat.
* **Results**: CIFAR, all 18 builds 3-7 of 10 (`res_fp_pc10.csv`), multi-batch A2bits 130/192 (`res_fp_pm10.csv`). The bake
  at round 7 (`res_fp_bake10.csv`): preprocessing -0.10..+0.09 s with secret weights (UC1 -0.05..-0.10, UC2
  +0.03..+0.09), +0.06..+0.16 s with public weights (mask-only forward); online 0.00..-0.07 s (faster with the bake).
  Final rerun of all 36 builds, both pairs (`res_*_final9.csv`, paper Table 14 / Fig. 16): against round 6 preprocessing
  median +-0.00 s (fp) / -0.01 s (ag), online median -0.01 s, up to -0.10 s. Phase profile at round 7 (`phases9_fp.log`,
  Table 15). ConvTriple `6980819` (GPU evaluator, see below) leaves the CPU path bit for bit (all 18 CIFAR hashes as
  `pc10`, `res_fp_pc11.csv`; timing `res_fp_ct11.csv`).

## GPU (2026-10-01, workstation cmucl771615)

RTX 4000 Ada (20 GB, sm_89), 2x Xeon Silver 4410Y (24 cores, 48 threads), CUDA 12.6; both parties on the machine
(loopback). Data: `docs/variant_data/gpu/` (`ws_convtriples.log`, `ws_hpmpc.csv`).

* **GPU evaluator in the packed pipeline** (ConvTriple `6980819`, `TRIPLE_GPU` builds, `CONV_GPU=0` off):
  `packed_gpu::Engine` (src/core/conv_packed_gpu.cu) builds each weight polynomial in the shared memory of its NTT kernel,
  transforms the inputs' c0, sums products (128-bit, both ciphertext halves per thread), inverse-transforms; SEAL's
  NTT replicated with SEAL's tables (c1 arrives in SEAL's NTT form; the triples are the CPU's values). Pinned staging.
  The host keeps parsing, flooding, masking, truncation, wire format; repacking stays on the CPU.
* **Conv triples in isolation** (`cheetah_conv_triple_test imagenet --pipelined`, 16 threads per party, median of 3, all
  exact): old per-layer troy path 5.26 s AB2 / 10.22 s AB (655 / 1,310 MiB); CPU packed 2.18 / 3.02 s (479 / 958 MiB); GPU
  evaluator 0.94 / 1.45 s. Profile (nsys, AB2): weight NTTs 0.30 s, products 0.17 s, copies 0.09 s of GPU time.
* **Whole inference** (final flags but 16 HE/OT threads and 11 workers per party; the configurations' 32 / 24 oversubscribe
  the shared machine: online 7.0-7.3 s): CPU vs GPU conv triples, median of 3: UC1 A2bits RCA pre 12.38 / 11.53 s,
  UC2 A2bits RCA 11.61 / 11.48, UC1 reshared RCA 7.43 / 6.82, UC2 reshared RCA 6.46 / 6.56; online unchanged (0.47-0.68 s).
  Preprocessing is the OT phase (ferret pack setup 2.4 s; A2bits: the bake's Boolean addition 4.8-5.1 s); the conv
  triples (CPU 4.6-6.4 s, GPU 1.4-2.3 s) run alongside it; 24 OT threads per party are slower (setup 3.7 s).
* **Online on the GPU**: `GEMM_FAST_GPU=1` (core/cuda/gemm_fast_gpu.cu: GEMM_FAST's uint32 product via CUTLASS, the weight
  operand kept on the device, exact; `GEMM_FAST_GPU_CHECK=1` recomputes 64 sampled entries per product): UC2 online 0.46 s
  either way; the conv layers' 0.2 s are per-output mask work and the exchange. `USE_CUDA_GEMM=2` (CUTLASS conv) bypasses
  the per-output mask step and breaks the A2B bake (check aborts). core/cuda: the uint16_t CUTLASS instantiation does not
  build with current CUTLASS.
* **Ferret on the GPU** (ConvTriple `0adcaec`, `55869c1`, `fb812f3`, `111e7be`; `TRIPLE_GPU` builds; data
  `docs/variant_data/gpu/ws_ot_tuples.txt`, `ws_hpmpc_ot.log`). All bit for bit emp's (`FERRET_GPU_CHECK=1` compares every
  tree and output with emp's CPU code, `ROT_GPU_CHECK=1` every hashed bit with emp's MITCCRH); the bytes on the wire are
  unchanged, so either party may run either path. Seeded CIFAR-10 ResNet50 (AdamW model, 10 images) gives the CPU build's
  output hashes (UC2 A2bits RCA `c50b7335ee17cd7b`, UC1 reshared RCA `58568ef2bef45216`, CPU with `CHEETAH_OT_GROUP=4`).
  1. LPN step (src/ot/lpn_gpu.cu): one thread per group of 4 outputs, 10 AES blocks (T-table replicated per lane in shared
     memory, 32 KiB), 40 gathers from the k = 238,000-block table (L2-resident); trailing outputs (`__compute1`) too.
  2. MPCOT (`FerretCOT<IO::NetIO>::extend` specialization, hpmpc_interface.cpp): emp's objects keep the seeds and the
     pre-OTs; the GPU builds all 2,507 trees level by level (sender from the seeds, receiver from its messages; per-level
     even/odd sums by warp shuffles + atomics; punctured pair fixed per level), the LPN step runs on the leaves in place.
     The tree messages of a channel go out as one message (emp flushes after every tree: 2,507 sends per extension).
  3. Outputs stay on the device (`FerretCOT<IO::NetIO>::rcot` specialization): a per-instance device buffer (164 MiB)
     holds each extension into `ot_data`; the host gets the last M outputs (next pre-OTs) at once, other ranges when a host
     consumer asks (pieces >= 1M COTs). `send/recv_rot_bits` and `_bitplanes` (97% of all COTs: 1.0 G of 1.04 G in UC2
     A2bits) register a consumer: the GPU computes MITCCRH<8> (key s ^ makeBlock(gid, 0) per OT, schedule on the fly) and
     returns only the bit planes. `ROT_GPU=0`: outputs to the host (then `ot_data` is pinned; `FERRET_PIN=0/1`).
  4. OT packs: with cheap extensions the 16 one-channel packs' setup dominated (2.0-2.5 s); GPU builds default to 4 packs
     of 4 channels (`CHEETAH_OT_GROUP` overrides; 8 / 16 channels: setup 0.6 / 0.5 s but MUX and pre slower).
  * Tuple test (one pack, 2e8 Boolean triples): CPU 38.6 s; GPU LPN 22.3-24.2; + MPCOT 17.6-20.0; + direct copies, one
    message per channel 14.4-15.1; + ROT bits on the GPU 6.2-6.6 s.
  * Whole inference (final session, median of 3; CPU / GPU conv only (`LPN_GPU=0`) / GPU all): UC1 A2bits RCA pre 13.66 /
    12.26 / 6.21 s, UC1 reshared 7.62 / 6.97 / 4.32, UC2 A2bits 12.56 / 11.86 / 5.73, UC2 reshared 6.73 / 6.58 / 3.89.
    Online 0.65 / 0.66 / 0.67, 0.69 / 0.68 / 0.76, 0.46 / 0.48 / 0.53, 0.51 / 0.51 / 0.56: 0.02-0.07 s higher with the GPU
    OT path; not pinning (FERRET_PIN=0 same), not warm-up (5 s pause before online same), no OT work online, no memory
    pressure; unexplained on the shared machine. Peak device memory 15.0 GB (both parties, 16 packs).
  * Steps end to end (UC2 A2bits RCA, separate sessions): CPU 11.5-12.6; GPU conv 11.5-11.9; + LPN 9.8; + MPCOT 8.0-8.2;
    + direct copies / one message 7.5; + ROT bits 7.0; + 4-channel packs 5.7 s.
  * Instrumentation: OT pack line has ferret setup / first-extension / MPCOT / LPN sums; `OT consumer` lines at
    disconnect give COTs and time per SilentOT consumer.
* Next: the packs' setup (base OTs, IKNP, first extension: 0.8-1.2 s), the A2bits Boolean addition rounds, the block
  consumers (cam_cc, rm_rc: 3% of COTs) on the GPU.

## Multi-batch and the online conv on the GPU (2026-10-02, workstation)

hpmpc `88c800b` (ConvTriple `1f5b71e`, flexNN `0404226`); data `docs/variant_data/gpu/ws_multibatch.log`.

* **Multi-batch with MODELWEIGHTS_KNOWN** (ImageNet DATTYPE=256, 8 images per process; the final builds set MWK):
  three single-batch optimizations were off. (1) CHEETAH_CONV_LANES excluded MWK: now P1's prescribed shares are put in
  per layer after the batched product (bit-exact with the per-layer path: the prescribed shares fix the triples; lane
  premise check silent; only lane 0's weights are extracted). (2) The A2B bake excluded MWK in multi-batch
  (mwk_choose_r1_trunc treated a register as one word) -> the non-bake A2B_ONLINE_OPT path ran, which is broken (known:
  its [c] does not match the online mask; single batch with A2B_CONV_BAKE=0 is broken too, 0-2 / 10): multi-batch UC2
  A2bits CIFAR-10 (AdamW) 2-4 / 32. Now -r1 = (m1 << F) + low lane by lane (OP_SHIFT_LEFT / OP_AND / PROMOTE; same
  values for one word: single-batch hash 687e1473bd unchanged): 20 / 32 (UC1 reshared 21 / 32). (3) CHEETAH_CONV_SIDE: the
  conv triples run after the pass on 4 side channels (Keys::get_side_ios) alongside BOOLEANADDITION / COT / MUX (the side
  thread must not disconnect the regular channels at the end: CHEETAH_DISCONNECT race, fixed).
* **GEMM_FAST for lanes**: one GEMM with L = DATTYPE / 32 columns per output (weights the same in every lane, checked per
  layer, else the share-level loop); GEMM_FAST_GPU likewise. Bit-exact (multi-batch CIFAR hash ef20263b20 = GEMM_FAST=0).
* **Online conv**: step timers (P0, one ImageNet image) products 0.08 of 0.21 s, mask/send 0.05, output zeroing 0.03,
  bias add 0.03, im2col 0.02, bake bias-mask expansion 0.01. Zeroing and bias add now on the GEMM threads, one bias mask
  per channel (g_bake_bias_rep), and GEMM_FAST_GPU builds the column matrix on the GPU (accumulate_conv + conv_fast_gpu:
  only the input operand travels, the host im2col is skipped; CUTLASS uint32 GEMM). Hashes unchanged. Conv layers online
  ~0.21 -> 0.15-0.18 s (CPU) / 0.13-0.14 s (GPU product); UC2 online GPU 0.425 vs CPU 0.426 s. Multi-batch UC2 conv layers
  0.77-1.19 s (GPU product) vs 1.10-1.36 (CPU), before 1.5-1.7 s.
* **CHEETAH_RELEASE_OT**: the OT packs (ferret's host / device buffers) are released after the preprocessing; no zero
  fill of ot_data when the outputs stay on the device.
* **Results** (median; single batch 3 runs, multi-batch P=1 2 runs; CPU / GPU with GEMM_FAST_GPU=1): single batch pre UC1
  A2bits 13.29 / 6.04, UC1 rs 7.73 / 4.33, UC2 A2bits 12.33 / 5.74, UC2 rs 7.14 / 3.92 s; online 0.59 / 0.64, 0.64 / 0.68,
  0.43 / 0.43, 0.45 / 0.44. Multi-batch (8 images) pre 89.59 / 36.80, 50.92 / 24.69, 79.31 / 33.04, 43.90 / 20.31 s
  (2.1-2.4x); online 4.14 / 3.97, 4.32 / 4.40, 3.19 / 2.97, 3.29 / 3.07. Before the multi-batch work: UC2 A2bits 86.8 / 49.1 s.
* **Scaling on the workstation** (62 GB, both parties, other users): one process per party peaks at 20-26 GB (GPU) /
  26-30 GB (CPU, both parties); two GPU processes per party fit (42 GB) and take 61.5 s for 16 images (33.0 for 8); two CPU
  processes per party do not fit. mbw.sh aborts a run when available memory drops below 3 GB (other users' jobs took up to
  54 GB). Multi-batch preprocessing is GPU-bound: GPU 90-100 % busy in the OT phase (Boolean triples 9.8 s + bake Boolean
  addition 12.3 s of 33 s), CPU ~27 %.
* Fixed during this work: download_range used a pooled context without bounce buffers (rare "copy out: invalid
  argument" abort); Keys triple stats under a mutex (two generator threads).
* Next: faster GPU AES (OT kernels bound the multi-batch preprocessing), the online activations (linear in the lanes).

## Output repacking: the estimate before round 4

Needs key switching, hence a special prime; at N = 4096 the 109-bit data modulus (2^32 plaintexts, 64-bit flooding)
already uses the whole 128-bit budget, so N = 8192 (60 + 49 data + 60 special bits). `docs/paper/he_model.py`: with
co = 1 layouts (dense inputs) and outputs packed after the evaluation, 134 + 108 = 243 MiB per HE product instead of
479 MiB, plus 4.7 MiB of Galois keys once (13 keys). `docs/paper/repack_estimate.py` (measured traffic of everything
else): preprocessing traffic -22..-38% in UC1 (1,230-2,150 -> 766-1,686 MiB), -14..-29% in UC2 (790-1,710 -> 558-1,478
MiB), UC3 unchanged; with the online phase -19..-32% and -12..-23%.
Runtime (`docs/paper/sealbench.cpp`, one thread: apply_galois at N = 8192 583 us on Zen 4, 1,308 us on Zen 3; calibrated
against the measured conv phase): evaluation about unchanged (0.64 vs 0.71 s on Zen 4), repacking +1.1 s with a merge
tree (7.2e4 key switches) or +8.2 s with traces (5.4e5) on 32 threads (Zen 3: +1.7 / +12.4 s). Net about +1.0 s (Zen 4)
/ +1.8 s (Zen 3) per inference for 236 MiB less per HE product: break-even ~1.9 Gbit/s (Zen 3: ~1.1 Gbit/s).
