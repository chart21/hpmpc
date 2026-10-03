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

## Round 8 (2026-10-02): the bake's Boolean addition (flare / polynize)

hpmpc `ddc082f` (ConvTriple `70f3eab`); data `docs/variant_data/triad/round8/` (README there).

* **Where the time went** (per-round log, UC2 A2bits RCA): 31 dependent rounds, each generating its own random OTs
  (cot_multiply_shares: 9M ANDs per round, per OT worker in chunks of 1M / 32 workers / 8 bytes, each chunk with
  its own ROT calls and round trip), and every few rounds a ferret extension round of all instances inside the chain
  (workstation: 7 rounds of ~175 ms (GPU) / 2 rounds of 2 s (CPU); flare: 15-22 ms rounds + an extension round).
* **A2B_ADDER_BATCH** (default on): Iface::boolCOTMultRoundsBegin generates the random OTs of all rounds first, one
  rot_bits call per pack and direction (cot_multiply_shares' order: reversed instance, then straight), on a thread of its
  own started next to init_a2b_bake (a2b_adder_prestart, not with the UC3 mask pass); boolCOTMultRound computes the
  corrections of all ranges on the pool, exchanges them over A2B_ROUND_CHANNELS = 4 channels (1 / 32 measured the
  same), computes the shares. hpmpc's adder loops run on bit-position-contiguous arrays (transposed once each way): the
  value-major [i][r] loops with a 32-word stride took 0.14 s, more than the exchanges (0.08 s). Bit-exact with
  respect to the round function (same CIFAR hashes across these refactorings; the random OTs differ from the old path).
* **A2B_ADDER_CUT** (default on): under CUT_FRACTIONAL_BITS_OPT every A2B replaces [c]'s top FRACTIONAL slices by 0, so
  the Boolean addition stops at bit 26: 26 instead of 31 rounds, -16 % OTs. Only if every conversion takes the cut
  (g_a2b_full_width, set by the INIT pass for any conversion without it). CUT_FRAC_ELIGIBLE and g_a2b_full_width now
  live in core/generate_beaver_tiples.hpp, which beaver_triples.hpp includes first (the first version saw the macro
  undefined and never cut).
* **Results** (same code, flags off / on, median of 2): COMPRESS=0 A2bits pre UC2 RCA 3.70 / 3.19, PPA 4.69 / 4.04, PPA4
  5.26 / 4.60; UC1 RCA 3.73 / 3.17, PPA 4.68 / 4.18, PPA4 5.33 / 4.58; UC3 RCA 3.46 / 3.01, PPA 4.42 / 3.83, PPA4 5.73 / 5.14
  s (-0.45..-0.75); COMPRESS=1 -0.05..-0.12 (PPA4 UC2 +0.04, noise); online unchanged. Boolean addition 1.44-1.52 ->
  1.04-1.21 s: ~1.0 s random OTs (one extension round of all 64 instances: bool triples 450M + adder 468M COTs > the
  640M left after setup; + hashing), 0.1 s rounds. CIFAR (AdamW, 10 images): all A2bits builds 4-7 / 10;
  multi-batch 24 x 8: 134 / 192 (MWK=0), 124 / 192 (MWK=1). Zen 4 only (algofi / goracle booked).
* Not done: B2A of random bits (daBits) would halve the adder's OTs and need one round, but P1's committed UC2 residual
  masks are prescribed (lz_1 = m_a + m_b), which daBits cannot give; a log-depth adder needs more ANDs (more OTs);
  routing the adder to fewer packs would save CPU (fewer extensions) but not wall time (each pack's two instances
  extend one after the other on its single channel).

## TS1 / TS_Mix in 2PC (2026-10-02): reduced-slack truncation without online communication

`TRUNC_APPROACH=4` (TS_Mix: TS1 fused into the ReLUs, TS{L} elsewhere) and `=1` (TS1: also the poolings) work with
PROTOCOL 4 (`TS1_FUSED_ACTIVE`, needs `TRUNC_DELAYED=1` and the A2B bake, i.e. A2bits with COMPRESS=0; `#error`
otherwise). Data `docs/variant_data/triad/ts1/`.

* **Protocol.** A delayed ReLU input z = m - lambda (scale 2^2F) is truncated by Truncation Untangled's Fig. 9 on
  u = -z: c = -m + offset is public (both parties hold m, so c', MSB(c) and the XOR with r_msb are local), and r =
  nu = -lambda is what the bake's Boolean addition already adds. Its sum-bit shares give the carries (carry share =
  [c] ^ own input): r' = (a_0 >> F) + (a_1 >> F) + w_t - K w and r_msb = MSB(nu), a_i = nu_i mod 2^(l-1),
  K = 2^(l-1-F). Preprocessing per value: w and r_msb to arithmetic shares by COTs of F bits (they are only used
  times K), and the bit injection's extra product [lambda_b r_msb] by an (F+1)-bit multiplexer (ConvTriple
  `generateCOT` / `do_multiplex` take a bit width). The truncated value is M - (la + sK r_msb) with M, s public: the
  bit injection takes it with the products [lambda_b la] (its usual one) + sK [lambda_b r_msb]. No message of its own.
* **Shifted design: the cut in the Boolean addition too** (default with CUT_FRACTIONAL_BITS_OPT; `beaver_triples.hpp`
  `g_ts1_la`). The full design above needs the carry into bit l-1 of nu_0 + nu_1, so the bake's Boolean addition ran
  full width (31 AND rounds, all slices) although the ReLU's A2B only reads the low l - F. Instead the Boolean
  addition adds the locally shifted shares rho_i = (f nu_i mod 2^l) >> F (f a public factor, 1 without a pooling)
  over l' = l - F bits, i.e. with the cut (26 rounds): (f z) >> F = (f m >> F) + rho + E mod 2^l', E in {0, 1, 2} the
  dropped low carries. The A2B converts y1 = M0 + rho with the cut, M0 = ((f m + 2^(F-1)) >> F) + 1 (rounded and
  centred for two random mask shares, so the dropped carries cost no bias), and TS1 runs in the small ring 2^l' on
  -y1 (offset 2^(l'-2), 1-bit slack) and lifts y1 >> t' to l bits: with t' = 0 the A2B and the bit injection see the
  same value (DReLU exact for it, the ReLU never outputs -1), off by -1 / 0 / +1 from the floor (27 / 1001 / 991 of
  the unit test's 2019 positive inputs), mean +0.03 LSB from the real value.
* **TS1_LOW_CARRY=1** (w_t, default 0) keeps the full design (full-width Boolean addition, [c] shifted by F slices for
  the cut A2B) and also converts w_t (a COT of l - 1 bits per value): errors 0 / +1 as in 3PC, mean -0.01 LSB.
* **Pooling (TRUNC_APPROACH=1, `TS1_FOLD_POOL`).** An average pooling fused into a ReLU (FUSE_RELU_AVG), one that
  runs between a conv and the next ReLU (PIGEON: it only sums and leaves 1/denom to the ReLU, `g_pending_denom`) and a
  uniform AdaptiveAvgPool after the last ReLU (`fuse_relu_pools`, also in `Cheetah_ResNet::compile` now) are divided
  by the ReLU's TS1: f = 1/denom (F bits), t' = F (delayed input) and the lift truncates by t' with the exact w_t of
  those values only (31-bit COT; without it the small averages lose half an LSB on average: CIFAR 69 of 256 at
  F = 5). The COTs and the multiplexer take F + t' bits per ReLU, not the largest t' for all. TS_Mix keeps TS{L} for
  the poolings: the pool divides its own bits only, a pending truncation stays pending for the next ReLU's TS1.
  TRUNC_APPROACH=1 aborts at any other truncation outside a ReLU (a stand-alone TS1 needs a message of its own).
* **UC3's first ReLU.** Its inputs carry the data owner's input sharing (one mask share 0, the other the value),
  which the shifted design's centring does not suit (+1 on every zero input). The mask-only forward leaves their
  committed masks and the ReLU moves the inputs onto them (the rebase message, as without the mask pass).
* **TS{L} gets the cut for UC3 too: `A2B_DELAYED_CUT=1`** (default; 2PC, TRUNC_APPROACH 0, TRUNC_DELAYED=1, A2B
  bake). UC3 (public weights) runs TS{L} with TD=1, so its ReLUs converted the untruncated z: no cut. Now a delayed
  ReLU's A2B converts the locally truncated value (m >> F, and the bake adds rho_i = nu_i >> F, the shifted design
  with f = 1, t' = 0) with the cut, and the bit injection takes z as before (BIT_INJECTION_TRUNC_SIM folds the
  truncation there). DReLU of values in [0, 2^(F+1)) (real value below 2^(1-F)) may come out 0. And in TD=1 builds
  (TS1, TS_Mix, A2B_DELAYED_CUT) a ReLU whose input a pooling has truncated already takes the cut as with TD=0: the
  stem ReLU of UC3 TS{L} ran full width and forced the whole Boolean addition to full width.
* **Bug fixes on the way (all TD=1 builds, Cheetah_ResNet = functions 87/187/287):** the downsample branch runs at the
  block's start; the downsample conv took the identity scaled by 2^F and left `delayed` set, so the main branch's
  first conv truncated the block input. Now the downsample conv takes the block input as it is and OP_Finish restores
  the main branch's state (UC3 TS{L} online -6.1 MiB, -4 rounds, and the values are right). `Cheetah_ResNet` had no
  ReLU/pool fusion of its own compile (no effect so far: its ReLUs are not followed by poolings, but TS1 needs the
  AdaptiveAvgPool fusion).
* **Tests** (func 59, flare / polynize): `RELU_TS1` (random |v| < 2^30, 2^29 with the cut, both signs; error per value
  against the real value), `RELU_TS1_AVG9` (pooling factor 1/9 with t' = F), `RELU_TS1_AVG9_TRUNCATED` (input
  truncated already), `RELU_DCUT`: 0 wrong of 4096 (DATTYPE 32) and of 32768 (DATTYPE 256); mean errors +0.028 (shifted),
  -0.007 (w_t), +0.001 / +0.10 (AVG9 / AVG9 truncated), -0.04 LSB (DCUT). `RELU_RANDOM` now draws |v| < 2^(l-1-F)
  where the cut applies. Single-batch unit tests need `CHEETAH_CONV_EARLY=0 A2B_BAKE_MASK_PASS=0`.
* **ImageNet, final code** (these builds lack the configs' `RNG_AHEAD=1`: for the configs as such see the next
  section; flare / polynize, dummy weights, `res_im_final.csv` / `comm_im_final.csv`, median of 3; P0's
  traffic sent + received, MiB; online seconds: `res_online.csv`, median of 4 after the bit-injection change below).
  TS{L} is the build as given (TD=0 in UC1 / UC2, TD=1 with BIT_INJECTION_TRUNC_SIM in UC3).

| UC, adder | pre MiB TS{L} / TS_Mix / +w_t / TS1 | pre s | online MiB (TS1) | rounds (TS1) | online s TS{L} / TS_Mix / TS1 |
|---|---|---|---|---|---|
| UC1 RCA | 1341/1378/1433/1379 | 3.17/3.38/3.53/3.49 | 210.2 (209.4) | 1818 (1817) | 0.477/0.467/0.457 |
| UC1 PPA | 1464/1501/1555/1502 | 4.21/4.30/4.43/4.40 | 321.8 (321.0) | 982 (981) | 0.459/0.466/0.455 |
| UC1 PPA4 | 1704/1741/1824/1742 | 4.63/4.87/5.40/4.84 | 207.9 (207.1) | 730 (729) | 0.573/0.566/0.550 |
| UC2 RCA | 922/959/1013/960 | 3.10/3.31/3.44/3.49 | 167.8 (167.0) | 1714 (1713) | 0.395/0.395/0.399 |
| UC2 PPA | 1045/1082/1136/1083 | 4.21/4.34/4.45/4.29 | 279.3 (278.6) | 878 (877) | 0.409/0.396/0.393 |
| UC2 PPA4 | 1285/1321/1405/1323 | 4.57/4.77/5.42/4.91 | 165.6 (164.8) | 626 (625) | 0.498/0.485/0.476 |
| UC3 RCA | 421/460/514/461 | 2.96/3.17/3.22/3.23 | 125.3 (124.5) | 1714 (1713) | 0.447/0.437/0.428 |
| UC3 PPA | 544/582/637/584 | 3.84/4.04/3.96/4.14 | 237.0 (236.1) | 878 (877) | 0.465/0.450/0.436 |
| UC3 PPA4 | 784/822/906/823 | 4.53/4.64/5.16/4.71 | 123.2 (122.4) | 626 (625) | 0.558/0.560/0.543 |

  TS_Mix costs +36-39 MiB of preprocessing over TS{L} (RCA: +2.8% UC1, +4.0% UC2, +9.3% UC3; the first version with
  the full-width Boolean addition +57-58), the same online traffic and rounds, and +0.09-0.24 s preprocessing (the four
  narrow COTs per ReLU value: two B2A, the multiplexer's two). TS1 adds 1-2 MiB (the poolings' w_t; before the
  per-width COTs +25) and removes the last pooling's message (-0.8 MiB, -1 round). w_t costs another 54-84 MiB and up
  to 0.65 s. Online the three take the same time (TS{L} 0.395-0.573 s, TS_Mix 0.395-0.566, TS1 0.393-0.550).
* **UC3 TS{L} with A2B_DELAYED_CUT** (`_xl0` without):

| adder | pre MiB | pre s | online MiB | rounds |
|---|---|---|---|---|
| RCA | 452 -> 421 | 3.11 -> 2.96 | 136.1 -> 125.3 | 1959 -> 1714 |
| PPA | 565 -> 544 | 3.87 -> 3.84 | 254.2 -> 237.0 | 887 -> 878 |
| PPA4 | 894 -> 784 | 5.29 -> 4.53 | 123.2 = | 626 = |

* **Larger F** (RCA, `res_f.csv` / `comm_f.csv`, median of 3). TS1 has no wrap failures, so F can grow (accuracy
  below), and the cut adder has l - F bits: per ReLU 31 - F RCA rounds online and in the Boolean addition.

| UC | TS{L} F=5 | TS{L} F=8 | TS_Mix F=5 | TS_Mix F=8 | TS_Mix F=10 |
|---|---|---|---|---|---|
| UC1 rounds / online MiB / pre MiB | 1818 / 210.2 / 1341 | 1671 / 203.7 / 1323 | 1818 / 210.2 / 1378 | 1671 / 203.7 / 1376 | 1573 / 199.3 / 1372 |
| UC2 | 1714 / 167.8 / 922 | 1567 / 161.3 / 903 | 1714 / 167.8 / 959 | 1567 / 161.3 / 956 | 1469 / 157.0 / 952 |
| UC3 | 1714 / 125.3 / 421 | 1567 / 118.9 / 403 | 1714 / 125.3 / 460 | 1567 / 118.9 / 457 | 1469 / 114.6 / 453 |

  (TS_Mix's preprocessing hardly shrinks: its COTs widen with F.)
* **Accuracy** (CIFAR-10, AdamW ResNet50, the first 256 test images, plaintext 189; `res_acc256.csv`, seeded runs):

| UC | variant | F = 5 | F = 8 | F = 10 |
|---|---|---|---|---|
| UC1 | TS{L} | 164 | 167 |  |
| UC1 | TS_Mix | 164 | 190 |  |
| UC1 | TS_Mix + w_t | 158 |  |  |
| UC1 | TS1 | 161 | 187 |  |
| UC1 | TS1 + w_t | 165 |  |  |
| UC2 | TS{L} | 159 | 169 | 44 |
| UC2 | TS_Mix | 164 | 186 | 188 |
| UC2 | TS_Mix + w_t | 157 | 186 | 183 |
| UC2 | TS1 | 161 | 187 | 189 |
| UC2 | TS1 + w_t | 165 | 186 | 188 |
| UC3 | TS{L}, no A2B_DELAYED_CUT | 166 |  |  |
| UC3 | TS{L} | 162 | 139 |  |
| UC3 | TS_Mix | 157 | 188 |  |
| UC3 | TS_Mix + w_t | 159 |  |  |
| UC3 | TS1 | 156 | 186 |  |
| UC3 | TS1 + w_t | 165 |  |  |

  At F = 5 every variant classifies 156-166 (fixed-point precision, not truncation errors, limits it; the spread is
  noise: each variant draws different randomness). w_t makes no consistent difference (UC1 / UC2 / UC3 at F = 5: TS1
  +4 / +4 / +9, TS_Mix -6 / -7 / +2; UC2 at F = 8 / 10: TS1 -1 / -1, TS_Mix 0 / -5). At F = 8 TS_Mix and TS1 reach
  the plaintext model (186-190) and TS{L} does not (167 / 169 / 139: its wrap probability grows with 2^(2F)); at
  F = 10 TS{L} collapses (44) and TS1 stays at 188-189.
* **Online compute.** TS_Mix's online phase took 0.02-0.04 s longer than TS{L}'s in UC1 / UC2 (same waiting): each
  TS1 ReLU zero-filled two per-value arrays (M and sK) that the A2B transform wrote and the bit injection read. The
  shifted design's lift needs only M0, which is the A2B's public input and stays in the inputs' m: the bit injection
  now computes M and sK from it (`ts1_lift_shift`), with outputs identical bit for bit (6 CIFAR builds). Their online
  compute (online minus waiting) fell by 0.02-0.04 s and is now at or below TS{L}'s (median of 4).
* Not supported: COMPRESS=1 and reshared builds (no bake: TS1 would need a Boolean addition of its own over the
  ReLU inputs' masks, ~+157 MiB; A2B_DELAYED_CUT works there since the next section), TS{L} with TD=1 for UC1 / UC2 still aborts in
  the bake check (pre-existing).

## All 36 all_opt configs with TS1 / TS_Mix (2026-10-03, flare / polynize)

Every config of `list_fin2.txt` at the final code, with each truncation variant that builds (ImageNet, dummy
weights, 3 interleaved runs; data `docs/variant_data/triad/ts1/allopt/`).

* **Which variants build.** TS_Mix and TS1 take their carries from the A2B bake, so they exist for the nine A2bits
  COMPRESS=0 configs only. The reshared configs have no bake, and COMPRESS=1 converts bits 12..19 only (no bake, no
  cut); there TS1 would need a Boolean addition of the mask shares of its own (~1 s and ~150 MiB of preprocessing,
  the A2bits bake's cost) or an online message, i.e. it would cost more than TS{L} at the same F. They keep TS{L}.
* **Correction.** The TS1 ImageNet matrix of the previous section came from round 8's build lists, which lack the
  configs' `RNG_AHEAD=1` (the online phase's RNG thread); its online seconds are not the configs'. The table below is
  rebuilt from `list_fin2.txt`.
* **New: the delayed cut for the reshared UC3 configs** (`A2B_DCUT_SHARE_ACTIVE`, part of `A2B_DELAYED_CUT=1`). UC3
  runs TS{L} with TD=1, so its ReLUs converted the untruncated value without the cut; A2B_DELAYED_CUT needed the bake.
  Without the bake the A2B converts the parties' additive shares, P0's m - l_0 and P1's -l_1: each party now shifts
  its own share right by F (`g_a2b_share_shift`, prepare_A2B_S1 / S2 in the preprocessing and online pass), their sum
  is trunc(z) minus 0 or 1, and the cut applies; the bit injection takes z as before. DReLU of values in [0, 2^F)
  may be 0. Not where RESHARE_OPT_SIM baked P1's reshare material into the unshifted share (public weights never do).
  Unit test `RELU_DCUT` (func 59, RCA): 0 wrong of 4096 / 32768 (DATTYPE 256), mean error -0.001 LSB; CIFAR (AdamW,
  256 images) with / without: RCA 170 / 162, PPA 161 / 155, PPA4 169 / 165. ImageNet: RCA 2008 -> 1763 rounds,
  online 170.4 -> 154.3 MiB, 0.550 -> 0.501 s, pre 268 -> 257 MiB; PPA online 288.5 -> 263.8 MiB, pre 442 -> 436;
  PPA4 online 157.5 -> 147.9 MiB, pre 1096 -> 1062 MiB, 5.08 -> 4.93 s, but online 0.423 -> 0.491 s (below).
* **PPA4's cut costs online compute.** With the cut, PPA4's ReLUs take 30-46 ms more local compute online in every
  family, although they send less (5 runs, `res_pc.csv`): UC2 reshared (TD=0) 0.454 s with the cut vs 0.407 s without
  (`CUT_FRACTIONAL_BITS_OPT=0`; pre 5.42 vs 5.63 s, 1639 vs 1718 MiB), UC3 reshared 0.480 vs 0.432 s, UC3 A2bits 0.510
  vs 0.481 s (pre 4.48 vs 5.23 s, 784 vs 894 MiB). Preprocessing plus online is lower with the cut in all three, so it
  stays on; where online latency alone counts, PPA4 without the cut is 0.03-0.05 s faster. A whole-run perf profile
  does not resolve it (the online phase is under 1% of the samples). RCA and PPA do not show it.
* **Results** (`ao_table.md`; `*`: least traffic of the config):

| config | truncation | pre s | online s | rounds | pre MiB | online MiB |
|---|---|---|---|---|---|---|
| UC1 A2bits RCA | TS{L} | 3.18 | 0.498 | 1818 | 1341* | 210.2 |
|  | TS_Mix | 3.45 | 0.453 | 1818 | 1378 | 210.2 |
|  | TS1 | 3.52 | 0.465 | 1817 | 1379 | 209.4 |
|  | TS1, F=8 | 3.44 | 0.452 | 1670 | 1377 | 202.9* |
| UC1 A2bits RCA COMPRESS=1 | TS{L} | 2.16 | 0.376 | 887 | 1163 | 169.4 |
| UC1 A2bits PPA | TS{L} | 4.17 | 0.463 | 982 | 1464 | 321.8 |
|  | TS_Mix | 4.31 | 0.466 | 982 | 1501 | 321.8 |
|  | TS1 | 4.46 | 0.457 | 981 | 1502 | 321.0 |
|  | TS1, F=8 | 3.63 | 0.430 | 978 | 1448* | 308.0* |
| UC1 A2bits PPA COMPRESS=1 | TS{L} | 2.16 | 0.370 | 746 | 1169 | 188.6 |
| UC1 A2bits PPA4 | TS{L} | 4.70 | 0.520 | 730 | 1704* | 207.9 |
|  | TS_Mix | 4.88 | 0.511 | 730 | 1741 | 207.9 |
|  | TS1 | 5.04 | 0.506 | 729 | 1742 | 207.1* |
|  | TS1, F=8 | 4.86 | 0.494 | 729 | 1730 | 207.1* |
| UC1 A2bits PPA4 COMPRESS=1 | TS{L} | 2.23 | 0.373 | 642 | 1211 | 162.9 |
| UC2 A2bits RCA | TS{L} | 3.16 | 0.421 | 1714 | 922* | 167.8 |
|  | TS_Mix | 3.36 | 0.405 | 1714 | 959 | 167.8 |
|  | TS1 | 3.45 | 0.397 | 1713 | 960 | 167.0 |
|  | TS1, F=8 | 3.37 | 0.387 | 1566 | 958 | 160.5* |
| UC2 A2bits RCA COMPRESS=1 | TS{L} | 2.04 | 0.298 | 783 | 743 | 126.9 |
| UC2 A2bits PPA | TS{L} | 4.14 | 0.405 | 878 | 1045 | 279.3 |
|  | TS_Mix | 4.22 | 0.390 | 878 | 1082 | 279.3 |
|  | TS1 | 4.36 | 0.384 | 877 | 1083 | 278.6 |
|  | TS1, F=8 | 3.54 | 0.380 | 874 | 1029* | 265.7* |
| UC2 A2bits PPA COMPRESS=1 | TS{L} | 2.10 | 0.319 | 642 | 750 | 146.2 |
| UC2 A2bits PPA4 | TS{L} | 4.55 | 0.475 | 626 | 1285* | 165.6 |
|  | TS_Mix | 4.87 | 0.467 | 626 | 1321 | 165.6 |
|  | TS1 | 4.91 | 0.459 | 625 | 1323 | 164.8* |
|  | TS1, F=8 | 4.79 | 0.443 | 625 | 1310 | 164.8* |
| UC2 A2bits PPA4 COMPRESS=1 | TS{L} | 2.19 | 0.295 | 538 | 792 | 120.4 |
| UC3 A2bits RCA | TS{L} | 2.94 | 0.447 | 1714 | 421* | 125.3 |
|  | TS_Mix | 3.10 | 0.427 | 1714 | 460 | 125.3 |
|  | TS1 | 3.17 | 0.452 | 1713 | 461 | 124.5 |
|  | TS1, F=8 | 3.15 | 0.413 | 1566 | 459 | 118.1* |
| UC3 A2bits RCA COMPRESS=1 | TS{L} | 1.59 | 0.336 | 783 | 244 | 84.5 |
| UC3 A2bits PPA | TS{L} | 3.93 | 0.451 | 878 | 544 | 237.0 |
|  | TS_Mix | 4.14 | 0.445 | 878 | 582 | 237.0 |
|  | TS1 | 4.17 | 0.431 | 877 | 584 | 236.1 |
|  | TS1, F=8 | 3.39 | 0.389 | 874 | 530* | 223.4* |
| UC3 A2bits PPA COMPRESS=1 | TS{L} | 1.57 | 0.362 | 642 | 251 | 103.9 |
| UC3 A2bits PPA4 | TS{L} | 4.52 | 0.514 | 626 | 784* | 123.2 |
|  | TS_Mix | 4.59 | 0.521 | 626 | 822 | 123.2 |
|  | TS1 | 4.75 | 0.499 | 625 | 823 | 122.4* |
|  | TS1, F=8 | 4.56 | 0.521 | 625 | 811 | 122.4* |
| UC3 A2bits PPA4 COMPRESS=1 | TS{L} | 1.75 | 0.344 | 538 | 293 | 78.1 |
| UC1 reshared RCA | TS{L} | 2.15 | 0.533 | 1867 | 1175 | 239.2 |
| UC1 reshared RCA COMPRESS=1 | TS{L} | 1.82 | 0.390 | 936 | 1084 | 178.0 |
| UC1 reshared PPA | TS{L} | 3.91 | 0.585 | 1004 | 1330 | 348.7 |
| UC1 reshared PPA COMPRESS=1 | TS{L} | 2.37 | 0.435 | 795 | 1141 | 197.2 |
| UC1 reshared PPA4 | TS{L} | 5.51 | 0.538 | 779 | 2059 | 232.7 |
| UC1 reshared PPA4 COMPRESS=1 | TS{L} | 2.25 | 0.422 | 691 | 1249 | 171.5 |
| UC2 reshared RCA | TS{L} | 2.21 | 0.444 | 1763 | 756 | 196.7 |
| UC2 reshared RCA COMPRESS=1 | TS{L} | 1.63 | 0.326 | 832 | 665 | 135.5 |
| UC2 reshared PPA | TS{L} | 3.89 | 0.480 | 900 | 911 | 306.2 |
| UC2 reshared PPA COMPRESS=1 | TS{L} | 2.34 | 0.363 | 691 | 722 | 154.8 |
| UC2 reshared PPA4 | TS{L} | 5.51 | 0.444 | 675 | 1639 | 190.3 |
| UC2 reshared PPA4 COMPRESS=1 | TS{L} | 2.27 | 0.323 | 587 | 830 | 129.0 |
| UC3 reshared RCA | TS{L} as before (no delayed cut) | 1.76 | 0.550 | 2008 | 268 | 170.4 |
|  | TS{L} + delayed cut | 1.76 | 0.501 | 1763 | 257* | 154.3* |
| UC3 reshared RCA COMPRESS=1 | TS{L} | 1.21 | 0.341 | 832 | 167 | 93.1 |
| UC3 reshared PPA | TS{L} as before (no delayed cut) | 3.30 | 0.498 | 903 | 442 | 288.5 |
|  | TS{L} + delayed cut | 3.16 | 0.512 | 900 | 436* | 263.8* |
| UC3 reshared PPA COMPRESS=1 | TS{L} | 1.57 | 0.360 | 691 | 230 | 112.4 |
| UC3 reshared PPA4 | TS{L} as before (no delayed cut) | 5.08 | 0.423 | 675 | 1096 | 157.5 |
|  | TS{L} + delayed cut | 4.93 | 0.491 | 675 | 1062* | 147.9* |
| UC3 reshared PPA4 COMPRESS=1 | TS{L} | 1.73 | 0.352 | 587 | 321 | 86.7 |

* **Minimal per config.**
  * A2bits COMPRESS=0, F = 5: TS{L} has the least preprocessing (TS_Mix / TS1 +36-39 MiB, +0.07-0.36 s); online
    traffic and rounds are equal (TS1 -0.8 MiB, -1 round), online times within noise (TS_Mix / TS1 at or below TS{L}).
  * A2bits COMPRESS=0 with TS1 at F = 8 (plaintext accuracy on CIFAR, see above; the ImageNet ResNet's stem pooling
    before its ReLU needs TS1, not TS_Mix, at F >= 8): the least online traffic in all nine (RCA -148 rounds and
    -7.2 to -7.3 MiB, PPA -13.6 to -13.8 MiB, PPA4 -0.8 MiB) and the shortest online phase in 8 of 9; PPA also
    preprocesses 0.54-0.60 s faster with 14-16 MiB less (its 24-bit adder), RCA and PPA4 send 25-38 MiB more.
  * Reshared and COMPRESS=1: TS{L} (the only variant); reshared UC3 now with the delayed cut.

## TE0 / TE1 in 2PC (2026-10-03): exact truncation fused into the ReLU

`TRUNC_APPROACH=2` (TE0) and `=3` (TE1) did not compile for PROTOCOL 4 (the generic forms need `prepare_B2A`,
`prepare_trunc_exact_xmod2t`, which ABY2 does not implement). Now both run in TS1's framework (`TE_FUSED_ACTIVE`;
`TRUNC_DELAYED=1`, the A2B bake, i.e. A2bits with COMPRESS=0, `A_KNOWN_TO_EVALUATORS_OPT=1`), exact, with no online
message of their own (hpmpc 31c5270, PIGEON 77b0644; `Ts1Range::te` in `protocols/beaver_triples.hpp`).

* **Value.** z = m + nu (scale 2^2F), a = m mod 2^F. TS1's full design (offset 2^(l-1), with w_t) gives
  y = floor(z / 2^F) + [a != 0] - c_t for z >= 0, c_t = [a + (nu mod 2^F) >= 2^F] the carry into bit F of m + nu.
  Only z >= 0 matters (DReLU zeroes the rest), so the wrap of m + nu is linear (MSB(m) or MSB(nu)) and no slack is
  needed. The bit injection takes y - [a != 0] + c_t; [a != 0] = (a + 2^F - 1) >> F is public.
* **c_t online, no extra rounds.** The MSB of an (F + 1)-bit a-known adder of (0, a) and (0, nu mod 2^F): Bool(m) and
  the bake's [c], low slices, prepared from the untransformed m and the unshifted [c] (not counted again in INIT,
  `g_a2b_no_count`), stepped in the same rounds as the ReLU's adder (`run_msb_adders`, `g_te_low_out`; `TE_LOW_ADDER`:
  RCA builds take the folded a-known RCA (F - 1 messages), PPA / PPA4 builds since hpmpc 8c2adac the a-known four-way
  PPA (4 / 6 messages at F = 5 / 8 in 2 rounds, against the Sklansky PPA's 10 / 19 in 5), `zero_add_adders/low/`).
  c_t = m_c ^ lambda_c enters the bit injection linearly: public part + m_c, mask part - (1 - 2 m_c) [lambda_c], product
  - (1 - 2 m_c) [lambda_b lambda_c]; preprocessing per value (`te_generate_products`): [lambda_c] and [lambda_b
  lambda_c] as B2As (COTs of l - 1 bits) of lambda_c and of d = lambda_b lambda_c, whose cross terms take two 1-bit
  COTs: 2 l bits instead of 3 l - 1 (a full-width multiplexer, 2 l, for the product).
* **DReLU.** TE1 of A = (m >> F) + (nu >> F) = trunc(z) - c_t with the cut (A2B xform `TeCut`: m >> F, [c] shifted by F
  slices): A and trunc(z) differ only where the output is 0 either way. TE0 of z at full width (no cut), exact for
  every z. Poolings fold into TS1's shifted design (as TRUNC_APPROACH 1; PIGEON's pooling folds extended).
* **Online cost over TS1:** TE1 the low adder's F ANDs per value; TE0 additionally the rounds of the full-width adder.
* **Tests** (func 59 `RELU_TE0` / `RELU_TE1`, 4096 random values, TE0 |v| < 2^(l-1), TE1 < 2^(l-2)): 0 wrong and every
  positive input exact, RCA / PPA / PPA4 at 32 bits (PPA / PPA4 with the narrow cut adders, below) and at 64 bits.
* **Accuracy** (CIFAR-10, AdamW ResNet50, the first 256 test images, plaintext 189; `res_acc.csv`): TE0 and TE1 give
  bit-identical outputs (both exact) and the same in UC1-3: 158 at F = 5, 185 at F = 8. Same run: TS{L} UC2 159 / 169,
  TS1 UC2 161 / 187 (as in the TS1 table). Exact truncation floors (mean error -0.23 LSB from the real value against
  TS1's rounding): no loss at F = 8.
* **ImageNet** (A2bits all_opt configurations, F = 5, flare / polynize, dummy weights, median of 3 interleaved runs,
  hpmpc 4b43984, `docs/variant_data/triad/campaign10/`; P0's traffic sent + received, MiB: ConvTriple's counters are
  MiB, hpmpc's 10^6 bytes; the tables before this one divided both by 1.048576, which put the HE / OT part 4.6% low).
  The table has all five schemes (TS{L} = the configuration as given). TE1's online phase takes about as long as TS1's (0.37-0.46 vs 0.35-0.44 s) for +8.4 MiB
  online traffic with every adder and the same RCA rounds (PPA4 +3, PPA +35); preprocessing +0.2-1.2 s and +146 MiB
  (RCA), +165 (PPA), +226 (PPA4, whose low adder takes a 3-tuple): the full-width Boolean addition, the (l - 1)-bit COTs
  of w_t, lambda_c and lambda_b lambda_c, the low adder. TE0: +288 rounds with RCA, up to +0.09 s online (PPA4).
  Against the previous version (hpmpc 8c0ff0f, `res5_imte.csv`): TE1 -4.3 MiB (RCA) / -12.6 MiB (PPA, PPA4) online,
  PPA4 -56 rounds; preprocessing -20 MiB (RCA, the B2A products) / -389 MiB (PPA4, the a-known narrow adder). The
  earlier table (8c0ff0f) is in git history.
* **Formal protocols:** `docs/paper/trunc_formal.tex` (TS1, TE1, TE0 as protocols, Lemma 1-3 with proofs, Theorem 2;
  Lemma 2 and TE1's output checked exhaustively at l = 8, F = 2, 3).

| build | scheme | pre MiB | pre s | online MiB | online s | rounds |
|---|---|---|---|---|---|---|
| UC1 RCA | TS{L} | 1,406 | 3.26 | 207.9 | 0.477 | 1769 |
|  | TS_Mix | 1,445 | 3.43 | 207.9 | 0.460 | 1769 |
|  | TS1 | 1,446 | 3.50 | 207.1 | 0.442 | 1768 |
|  | TE1 | 1,592 | 3.85 | 215.5 | 0.462 | 1768 |
|  | TE0 | 1,631 | 4.33 | 228.1 | 0.501 | 2056 |
| UC1 PPA | TS{L} | 1,508 | 4.25 | 306.7 | 0.451 | 949 |
|  | TS_Mix | 1,547 | 4.43 | 306.7 | 0.437 | 949 |
|  | TS1 | 1,548 | 4.60 | 305.9 | 0.427 | 948 |
|  | TE1 | 1,713 | 4.84 | 314.3 | 0.436 | 983 |
|  | TE0 | 1,734 | 4.90 | 345.8 | 0.480 | 1021 |
| UC1 PPA4 | TS{L} | 1,772 | 4.51 | 201.6 | 0.426 | 730 |
|  | TS_Mix | 1,811 | 4.69 | 201.6 | 0.440 | 730 |
|  | TS1 | 1,812 | 4.87 | 200.8 | 0.428 | 729 |
|  | TE1 | 2,038 | 6.01 | 209.2 | 0.439 | 732 |
|  | TE0 | 2,074 | 6.11 | 215.5 | 0.517 | 732 |
| UC2 RCA | TS{L} | 966 | 3.26 | 165.6 | 0.397 | 1665 |
|  | TS_Mix | 1,005 | 3.35 | 165.6 | 0.410 | 1665 |
|  | TS1 | 1,006 | 3.45 | 164.8 | 0.401 | 1664 |
|  | TE1 | 1,152 | 3.74 | 173.2 | 0.411 | 1664 |
|  | TE0 | 1,191 | 4.21 | 185.8 | 0.420 | 1952 |
| UC2 PPA | TS{L} | 1,068 | 4.24 | 264.4 | 0.384 | 845 |
|  | TS_Mix | 1,107 | 4.43 | 264.4 | 0.383 | 845 |
|  | TS1 | 1,108 | 4.42 | 263.6 | 0.385 | 844 |
|  | TE1 | 1,273 | 4.86 | 272.0 | 0.391 | 879 |
|  | TE0 | 1,294 | 4.89 | 303.5 | 0.409 | 917 |
| UC2 PPA4 | TS{L} | 1,333 | 4.42 | 159.2 | 0.361 | 626 |
|  | TS_Mix | 1,371 | 4.60 | 159.2 | 0.351 | 626 |
|  | TS1 | 1,372 | 4.75 | 158.4 | 0.347 | 625 |
|  | TE1 | 1,598 | 5.97 | 166.8 | 0.374 | 628 |
|  | TE0 | 1,634 | 6.26 | 173.1 | 0.455 | 628 |
| UC3 RCA | TS{L} | 441 | 2.94 | 123.2 | 0.455 | 1665 |
|  | TS_Mix | 481 | 3.06 | 123.2 | 0.460 | 1665 |
|  | TS1 | 483 | 3.28 | 122.4 | 0.414 | 1664 |
|  | TE1 | 628 | 3.46 | 130.7 | 0.436 | 1664 |
|  | TE0 | 667 | 3.95 | 143.4 | 0.462 | 1952 |
| UC3 PPA | TS{L} | 543 | 3.94 | 222.0 | 0.407 | 845 |
|  | TS_Mix | 583 | 4.13 | 222.0 | 0.426 | 845 |
|  | TS1 | 585 | 4.24 | 221.3 | 0.413 | 844 |
|  | TE1 | 749 | 4.50 | 229.6 | 0.423 | 879 |
|  | TE0 | 770 | 4.54 | 261.1 | 0.461 | 917 |
| UC3 PPA4 | TS{L} | 808 | 4.37 | 116.7 | 0.434 | 626 |
|  | TS_Mix | 848 | 4.56 | 116.7 | 0.386 | 626 |
|  | TS1 | 849 | 4.62 | 116.0 | 0.387 | 625 |
|  | TE1 | 1,075 | 5.67 | 124.4 | 0.413 | 628 |
|  | TE0 | 1,111 | 5.80 | 130.7 | 0.507 | 628 |

* **The 32-bit PPA / PPA4 cut under the bake was wrong** for inputs near the cut's limit (114 of 4096; TE1 and TS1
  with these adders inherited it): fixed by narrow adders (`CUT_NARROW_32`, `docs/BITLENGTH64.md`).

## Round 9 (2026-10-03): the generated a-known circuits

The circuit generator (llm_test 54d7446, a3c8176) now emits the a-known four-way PPA correctly at any width, so A2bits
PPA4 takes it again at the cut (it had taken the AB circuit since `CUT_NARROW_32`; `A2BITS_PPA4_AB=1` keeps that):

* **One mask share per dot group** (llm_test a3c8176): a dot group's pending members take mask 0 except one leaf of the
  root's XOR chain, which carries the root's mask; the 27-bit a-known four-way adder draws 56 instead of 197 random values
  (the parallel ReLU constructors take them from a serially filled stream). Online gap to the AB circuit 0.1 -> 0.02-0.03 s.
* **Folded a-known RCA:** carry[k-2] = x1 y1 ^ x1 x2 y2 ^ x2 (y1 y2) in one dot group instead of remasking the a-known LSB
  carry (one round and message per adder, the same triples); 32-bit A2bits RCA under the bake takes the narrow cut adder
  (`CUT_FRAC_NARROW32`): 49 fewer rounds (UC2 1,665 instead of 1,714; 64 bits 2,890 instead of 2,939), -2.2 MiB online;
  online the same within the spread in the LAN (campaign10: 0.477 / 0.397 / 0.455 against 0.479 / 0.403 / 0.452 s with
  `CUT_NARROW_32=0`, UC1-3; an earlier single comparison, res6.csv, had shown 0.01-0.07 s): 49 rounds are about 15 ms here.
* **ImageNet, A2bits PPA4, a-known vs AB** (same run, `res7.csv`): preprocessing 1,772 / 1,333 / 808 vs 2,156 / 1,716 /
  1,191 MiB and 4.50 / 4.49 / 4.31 vs 5.69 / 5.53 / 5.34 s; online 0.438 / 0.361 / 0.417 vs 0.412 / 0.339 / 0.389 s, same
  traffic and rounds. 64 bits: 3,674 / 2,740 / 1,635 vs 4,502 / 3,568 / 2,464 MiB, 8.24 / 8.31 / 7.68 vs 11.26 / 11.08 /
  10.80 s, online 0.695 / 0.610 / 0.695 vs 0.648 / 0.585 / 0.695 s.

## Campaign at the round-9 code (2026-10-03): the artifact configurations, truncation schemes, cut variants

hpmpc 4b43984 (ConvTriple 82b1400, PIGEON 77b0644), flare / polynize, 100 builds from
`measurements/configs/artifacts/2pc_optimizations/gen_configs.py` (`docs/variant_data/triad/campaign10/`: `list_c8.txt`,
`run8.sh`, `res8.csv`, `comm8.csv`, `ana10.py`), three interleaved runs each. The 36 all_opt builds give the Zen 4
columns of the paper's Table / Figure (`res_fp_final10.csv`, `comm_fp10.csv`, `make_triad.py`): UC1 / UC2 preprocessing
1.7-5.5 s (3.0-7.1x faster than as given), online 0.30-0.57 s (2.2-3.6x); UC3 1.05-1.4x / 1.7-2.5x. A reference build
from the old lists (`ref_it_a2b_rca_xl`) sends exactly what the config build sends.

**The cut** (MiB; none: `CUT_FRACTIONAL_BITS_OPT=0`; identity: `CUT_NARROW_32=0`; narrow: the default; reshared: no bake,
identity cut):

| build | cut | pre MiB | OT MiB | pre s | online MiB | online s | rounds |
|---|---|---|---|---|---|---|---|
| UC1 A2bits RCA | none | 1,436 | 406 | 3.28 | 220.9 | 0.490 | 2063 |
|  | identity | 1,404 | 384 | 3.22 | 210.2 | 0.479 | 1818 |
|  | narrow | 1,406 | 384 | 3.26 | 207.9 | 0.477 | 1769 |
| UC1 A2bits PPA | none | 1,551 | 467 | 4.31 | 338.9 | 0.450 | 991 |
|  | identity | 1,529 | 445 | 4.29 | 321.8 | 0.471 | 982 |
|  | narrow | 1,508 | 445 | 4.25 | 306.7 | 0.451 | 949 |
| UC1 A2bits PPA4 | none | 1,892 | 702 | 5.56 | 207.9 | 0.520 | 730 |
|  | identity | 1,777 | 607 | 4.72 | 207.9 | 0.525 | 730 |
|  | narrow | 1,772 | 619 | 4.51 | 201.6 | 0.426 | 730 |
| UC2 A2bits RCA | none | 996 | 406 | 3.25 | 178.4 | 0.446 | 1959 |
|  | identity | 964 | 384 | 3.15 | 167.8 | 0.403 | 1714 |
|  | narrow | 966 | 384 | 3.26 | 165.6 | 0.397 | 1665 |
| UC2 A2bits PPA | none | 1,111 | 467 | 4.19 | 296.5 | 0.402 | 887 |
|  | identity | 1,090 | 445 | 4.12 | 279.3 | 0.394 | 878 |
|  | narrow | 1,068 | 445 | 4.24 | 264.4 | 0.384 | 845 |
| UC2 A2bits PPA4 | none | 1,452 | 702 | 5.48 | 165.6 | 0.462 | 626 |
|  | identity | 1,337 | 607 | 4.51 | 165.6 | 0.454 | 626 |
|  | narrow | 1,333 | 619 | 4.42 | 159.2 | 0.361 | 626 |
| UC3 A2bits RCA | none | 471 | 406 | 2.97 | 136.1 | 0.503 | 1959 |
|  | identity | 439 | 384 | 2.98 | 125.3 | 0.452 | 1714 |
|  | narrow | 441 | 384 | 2.94 | 123.2 | 0.455 | 1665 |
| UC3 A2bits PPA | none | 586 | 467 | 3.86 | 254.2 | 0.427 | 887 |
|  | identity | 565 | 445 | 3.81 | 237.0 | 0.449 | 878 |
|  | narrow | 543 | 445 | 3.94 | 222.0 | 0.407 | 845 |
| UC3 A2bits PPA4 | none | 927 | 702 | 5.15 | 123.2 | 0.486 | 626 |
|  | identity | 812 | 607 | 4.45 | 123.2 | 0.483 | 626 |
|  | narrow | 808 | 619 | 4.37 | 116.7 | 0.434 | 626 |
| UC1 reshared RCA | none | 1,240 | 211 | 2.23 | 255.2 | 0.577 | 2112 |
|  | identity | 1,229 | 211 | 2.14 | 239.2 | 0.541 | 1867 |
| UC1 reshared PPA | none | 1,389 | 303 | 4.11 | 373.3 | 0.593 | 1007 |
|  | identity | 1,389 | 303 | 3.97 | 348.7 | 0.570 | 1004 |
| UC1 reshared PPA4 | none | 2,180 | 1,014 | 5.74 | 242.3 | 0.536 | 779 |
|  | identity | 2,150 | 1,002 | 5.54 | 232.7 | 0.525 | 779 |
| UC2 reshared RCA | none | 800 | 211 | 2.17 | 212.8 | 0.511 | 2008 |
|  | identity | 790 | 211 | 2.10 | 196.7 | 0.466 | 1763 |
| UC2 reshared PPA | none | 979 | 303 | 4.08 | 330.8 | 0.479 | 903 |
|  | identity | 949 | 303 | 3.97 | 306.2 | 0.488 | 900 |
| UC2 reshared PPA4 | none | 1,789 | 1,014 | 5.57 | 200.0 | 0.419 | 675 |
|  | identity | 1,710 | 1,002 | 5.45 | 190.3 | 0.463 | 675 |
| UC3 reshared RCA | none | 278 | 211 | 1.78 | 170.4 | 0.529 | 2008 |
|  | identity | 267 | 211 | 1.68 | 154.3 | 0.514 | 1763 |
| UC3 reshared PPA | none | 456 | 303 | 3.34 | 288.5 | 0.506 | 903 |
|  | identity | 450 | 303 | 3.21 | 263.8 | 0.499 | 900 |
| UC3 reshared PPA4 | none | 1,137 | 885 | 4.92 | 157.5 | 0.428 | 675 |
|  | identity | 1,102 | 879 | 5.00 | 147.9 | 0.488 | 675 |

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
