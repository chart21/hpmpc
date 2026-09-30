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
(2026-09-30): `conf` = the config as given (res_*_final4.csv, 3 runs alternating with the new code), `fin` = the configs
plus every optimization at hpmpc `1421df0` / ConvTriple `5b568e9`, without `RNG_AHEAD` (res_*_final5.csv, `fin4_`, 3 runs);
`fin64`: `CHEETAH_THREADS=64` at `ce0608a` (earlier round).

* Every build is faster than as given in both phases on both pairs.
* UC1 and UC2: preprocessing 11.7-17.3 -> 1.9-6.2 s on fp (2.6-6.7x), 15.1-21.9 -> 2.7-7.8 s on ag (2.8-6.0x); online
  0.77-1.55 -> 0.39-0.70 s on fp (1.8-2.8x), 0.72-1.73 -> 0.27-0.70 s on ag (2.0-3.8x). UC2 online 0.39-0.62 s (fp),
  0.27-0.54 s (ag).
* UC3: preprocessing 1.0-1.2x faster, online 1.4-1.9x (ag 1.4-2.1x).
* `RNG_AHEAD` is no longer in the optimized set: at the final code it saves ~0.04 s online on Zen 4 (median) but costs
  up to 0.17 s on Zen 3 (UC3 COMPRESS=1: 0.51 vs 0.36 s; the online draws wait, most likely for the ring's producer).
* `CHEETAH_THREADS=64` (earlier round): preprocessing -8% (median, fp) / -4% (ag), up to -20% for PPA4, +19..30% for UC3
  RCA; a per-config choice, not a default.
* Correctness: all 18 COMPRESS=0 CIFAR builds give the reference hashes at the final code (4-7 of 10 correct).

### COMPRESS=0

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 14.84 | 4.16 | 3.92 | 18.88 | 5.03 | 4.90 | 1.179 | 0.648 | 1.256 | 0.601 |
| UC1, A2bits | PPA | 16.11 | 4.91 | 5.11 | 19.71 | 6.06 | 6.50 | 1.323 | 0.654 | 1.343 | 0.563 |
| UC1, A2bits | PPA4 | 17.28 | 5.47 | 5.77 | 21.85 | 6.51 | 7.11 | 1.371 | 0.700 | 1.440 | 0.703 |
| UC1, reshared | RCA | 13.37 | 2.33 | 2.63 | 17.05 | 3.35 | 3.62 | 1.392 | 0.680 | 1.363 | 0.587 |
| UC1, reshared | PPA | 15.20 | 4.02 | 4.38 | 19.40 | 5.27 | 5.82 | 1.550 | 0.680 | 1.588 | 0.616 |
| UC1, reshared | PPA4 | 17.06 | 6.19 | 5.96 | 21.69 | 7.82 | 8.06 | 1.550 | 0.680 | 1.726 | 0.615 |
| UC2, A2bits | RCA | 13.82 | 4.03 | 3.77 | 17.53 | 4.88 | 4.83 | 1.155 | 0.538 | 1.007 | 0.416 |
| UC2, A2bits | PPA | 15.00 | 4.87 | 5.22 | 18.61 | 5.94 | 6.41 | 1.200 | 0.543 | 1.114 | 0.416 |
| UC2, A2bits | PPA4 | 16.54 | 5.38 | 5.67 | 20.05 | 6.35 | 7.06 | 1.242 | 0.622 | 1.204 | 0.535 |
| UC2, reshared | RCA | 11.90 | 2.34 | 2.55 | 16.12 | 3.22 | 3.53 | 1.191 | 0.580 | 1.148 | 0.430 |
| UC2, reshared | PPA | 14.04 | 4.01 | 4.44 | 18.51 | 5.11 | 5.69 | 1.362 | 0.606 | 1.285 | 0.445 |
| UC2, reshared | PPA4 | 16.17 | 6.16 | 6.03 | 20.77 | 7.54 | 7.89 | 1.506 | 0.584 | 1.485 | 0.440 |
| UC3, A2bits | RCA | 3.75 | 3.51 | 3.05 | 3.75 | 3.73 | 3.40 | 0.917 | 0.638 | 0.878 | 0.596 |
| UC3, A2bits | PPA | 4.70 | 4.41 | 4.34 | 4.92 | 4.80 | 5.12 | 1.028 | 0.630 | 0.968 | 0.552 |
| UC3, A2bits | PPA4 | 6.15 | 5.99 | 4.96 | 6.62 | 6.35 | 5.69 | 1.064 | 0.693 | 1.024 | 0.668 |
| UC3, reshared | RCA | 2.05 | 1.69 | 2.32 | 2.31 | 2.05 | 2.76 | 1.071 | 0.604 | 0.918 | 0.532 |
| UC3, reshared | PPA | 3.69 | 3.20 | 3.32 | 4.34 | 3.80 | 4.07 | 1.128 | 0.585 | 1.093 | 0.527 |
| UC3, reshared | PPA4 | 5.45 | 4.90 | 4.42 | 6.55 | 5.66 | 5.28 | 0.971 | 0.509 | 0.887 | 0.472 |

### COMPRESS=1

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 13.20 | 2.30 | 2.58 | 16.99 | 3.17 | 3.38 | 0.845 | 0.447 | 0.924 | 0.422 |
| UC1, A2bits | PPA | 13.13 | 2.33 | 2.59 | 16.92 | 3.23 | 3.45 | 0.954 | 0.450 | 0.941 | 0.424 |
| UC1, A2bits | PPA4 | 13.31 | 2.53 | 2.75 | 16.94 | 3.36 | 3.63 | 0.933 | 0.470 | 0.989 | 0.444 |
| UC1, reshared | RCA | 12.52 | 1.86 | 2.05 | 16.77 | 2.81 | 2.97 | 0.876 | 0.481 | 0.988 | 0.447 |
| UC1, reshared | PPA | 13.12 | 2.37 | 2.67 | 17.15 | 3.30 | 3.59 | 1.164 | 0.498 | 1.228 | 0.497 |
| UC1, reshared | PPA4 | 13.81 | 3.05 | 3.44 | 17.77 | 4.10 | 4.66 | 1.295 | 0.506 | 1.497 | 0.534 |
| UC2, A2bits | RCA | 11.71 | 2.31 | 2.41 | 15.49 | 3.14 | 3.33 | 0.765 | 0.395 | 0.719 | 0.269 |
| UC2, A2bits | PPA | 11.95 | 2.46 | 2.51 | 15.69 | 3.18 | 3.39 | 0.824 | 0.394 | 0.730 | 0.289 |
| UC2, A2bits | PPA4 | 12.18 | 2.45 | 2.64 | 15.73 | 3.24 | 3.59 | 0.810 | 0.390 | 0.764 | 0.291 |
| UC2, reshared | RCA | 11.74 | 1.94 | 2.02 | 15.08 | 2.69 | 2.87 | 0.775 | 0.418 | 0.795 | 0.312 |
| UC2, reshared | PPA | 11.87 | 2.30 | 2.51 | 15.93 | 3.13 | 3.55 | 0.984 | 0.441 | 0.919 | 0.323 |
| UC2, reshared | PPA4 | 12.39 | 3.01 | 3.40 | 16.41 | 3.86 | 4.51 | 1.181 | 0.419 | 1.274 | 0.331 |
| UC3, A2bits | RCA | 1.77 | 1.61 | 1.71 | 1.86 | 1.75 | 1.91 | 0.576 | 0.384 | 0.522 | 0.379 |
| UC3, A2bits | PPA | 1.82 | 1.67 | 1.74 | 1.86 | 1.75 | 1.97 | 0.608 | 0.380 | 0.549 | 0.372 |
| UC3, A2bits | PPA4 | 2.04 | 1.77 | 1.87 | 2.05 | 1.86 | 2.11 | 0.638 | 0.405 | 0.565 | 0.399 |
| UC3, reshared | RCA | 1.36 | 1.16 | 1.25 | 1.40 | 1.20 | 1.31 | 0.616 | 0.390 | 0.524 | 0.358 |
| UC3, reshared | PPA | 1.68 | 1.56 | 1.61 | 1.97 | 1.80 | 1.74 | 0.658 | 0.408 | 0.577 | 0.392 |
| UC3, reshared | PPA4 | 1.94 | 1.80 | 1.81 | 2.12 | 1.86 | 1.88 | 0.711 | 0.382 | 0.585 | 0.390 |

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
in both reruns and on ag. Source `docs/variant_data/triad/comm_fp.csv` (triple counters in MiB, network counters in
10^6 bytes, converted), plot data `docs/paper/data/triad_comm_c{0,1}.dat` (`make_triad.py`), paper Fig. 17.
`triples`: HE + OT generation incl. key exchange; `pre`: triples + the network's preprocessing pass.

* UC1 and UC2: preprocessing 1.35-1.86x less with COMPRESS=0, 1.61-1.92x with COMPRESS=1. Conv triples ~1,000 ->
  479 MiB (UC2), 2,003 -> 958 MiB (UC1). MWK builds also send P1's share corrections (4 B per conv output,
  42 MiB); the pipelined build sends them after the conv batch, so the log counts them under `FC`.
* CUT: PPA4 needs 12 MiB (reshared) / 74 MiB (A2bits, with the bake) fewer multi-input AND tuples; online -4..8% for the
  UC1 / UC2 COMPRESS=0 builds except A2bits PPA4. Not eligible in UC3 (TRUNC_DELAYED=1).
* A2B bake in UC3 (A2bits, COMPRESS=0): +69 MiB in the preprocessing pass, the rebase of all 9.0 M ReLU inputs (one word
  per value and party); online unchanged. The as-given build skips it and is not correct.

### Communication, COMPRESS=0

| use case, config | adder | triples conf | triples fin | pre conf | pre fin | pre factor | online conf | online fin | online factor |
|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 2414 | 1368 | 2480 | 1432 | 1.73 | 220.9 | 210.2 | 1.05 |
| UC1, A2bits | PPA | 2475 | 1430 | 2595 | 1557 | 1.67 | 338.9 | 321.8 | 1.05 |
| UC1, A2bits | PPA4 | 2710 | 1591 | 2935 | 1805 | 1.63 | 207.9 | 207.9 | 1.00 |
| UC1, reshared | RCA | 2219 | 1174 | 2285 | 1230 | 1.86 | 255.2 | 239.2 | 1.07 |
| UC1, reshared | PPA | 2311 | 1266 | 2435 | 1389 | 1.75 | 373.3 | 348.7 | 1.07 |
| UC1, reshared | PPA4 | 3022 | 1965 | 3225 | 2150 | 1.50 | 242.3 | 232.7 | 1.04 |
| UC2, A2bits | RCA | 1453 | 930 | 1518 | 992 | 1.53 | 178.5 | 167.8 | 1.06 |
| UC2, A2bits | PPA | 1514 | 991 | 1633 | 1118 | 1.46 | 296.5 | 279.3 | 1.06 |
| UC2, A2bits | PPA4 | 1749 | 1152 | 1973 | 1365 | 1.45 | 165.5 | 165.5 | 1.00 |
| UC2, reshared | RCA | 1258 | 735 | 1323 | 790 | 1.68 | 212.8 | 196.7 | 1.08 |
| UC2, reshared | PPA | 1350 | 827 | 1502 | 949 | 1.58 | 330.8 | 306.2 | 1.08 |
| UC2, reshared | PPA4 | 2061 | 1526 | 2312 | 1710 | 1.35 | 200.0 | 190.3 | 1.05 |
| UC3, A2bits | RCA | 406 | 406 | 478 | 546 | 0.87 | 142.2 | 142.2 | 1.00 |
| UC3, A2bits | PPA | 468 | 468 | 593 | 661 | 0.90 | 260.3 | 260.3 | 1.00 |
| UC3, A2bits | PPA4 | 703 | 703 | 933 | 1002 | 0.93 | 129.3 | 129.3 | 1.00 |
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
