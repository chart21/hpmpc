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

ImageNet ResNet50, one image, dummy weights; median of two runs; fp = flare / polynize (Zen 4), ag = algofi / goracle
(Zen 3). `conf`: the config as given; `fin`: optimized (hpmpc `aeab3ae`); `fin64`: optimized with `CHEETAH_THREADS=64`
(at `ce0608a`). UC3 rows (conf, fin): median of five interleaved runs (`res_{fp,ag}_ab.csv`, 2026-09-30).

* UC1 and UC2: preprocessing 11.3-17.4 -> 2.2-7.1 s on fp (2.3-5.5x), 15.5-21.7 -> 3.0-8.8 s on ag; online 0.77-1.57 ->
  0.34-0.67 s on fp (2.0-2.7x), 0.73-1.75 -> 0.27-0.68 s on ag (up to 3.3x).
* UC2 (weights known in preprocessing): online 0.34-0.59 s (fp), 0.27-0.48 s (ag).
* UC3: preprocessing is OT-bound and changes little (see the next section); online 1.3-2.4x faster.
* Against the same flags before this round (`opt`, 161cf1d), online is 1.0-1.5x faster on both pairs.
* `CHEETAH_THREADS=64`: preprocessing -8% (median, fp) / -4% (ag), up to -20% for PPA4 builds, but +19..30% for the
  UC3 RCA builds (64 OT packs to set up for little OT demand). A per-config choice, not a default.
* Correctness at the final commit: all 18 COMPRESS=0 CIFAR builds give the reference hashes (4-7 of 10 correct).

### COMPRESS=0

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 15.13 | 4.51 | 3.92 | 18.69 | 5.41 | 4.90 | 1.265 | 0.619 | 1.295 | 0.571 |
| UC1, A2bits | PPA | 16.00 | 5.44 | 5.11 | 20.12 | 6.55 | 6.50 | 1.340 | 0.607 | 1.361 | 0.596 |
| UC1, A2bits | PPA4 | 17.38 | 6.23 | 5.77 | 21.28 | 7.28 | 7.11 | 1.398 | 0.646 | 1.505 | 0.585 |
| UC1, reshared | RCA | 13.19 | 2.91 | 2.63 | 17.51 | 3.85 | 3.62 | 1.347 | 0.578 | 1.450 | 0.566 |
| UC1, reshared | PPA | 14.98 | 4.63 | 4.38 | 19.59 | 5.86 | 5.82 | 1.572 | 0.649 | 1.596 | 0.603 |
| UC1, reshared | PPA4 | 17.00 | 7.05 | 5.96 | 21.66 | 8.77 | 8.06 | 1.553 | 0.673 | 1.751 | 0.683 |
| UC2, A2bits | RCA | 13.69 | 4.60 | 3.77 | 17.49 | 5.31 | 4.83 | 1.107 | 0.546 | 1.063 | 0.446 |
| UC2, A2bits | PPA | 14.80 | 5.45 | 5.22 | 18.65 | 6.43 | 6.41 | 1.229 | 0.527 | 1.086 | 0.432 |
| UC2, A2bits | PPA4 | 16.18 | 6.16 | 5.67 | 20.63 | 7.13 | 7.06 | 1.260 | 0.589 | 1.242 | 0.481 |
| UC2, reshared | RCA | 11.85 | 2.82 | 2.55 | 16.27 | 3.73 | 3.53 | 1.233 | 0.537 | 1.173 | 0.422 |
| UC2, reshared | PPA | 14.02 | 4.59 | 4.44 | 18.39 | 5.97 | 5.69 | 1.411 | 0.562 | 1.411 | 0.428 |
| UC2, reshared | PPA4 | 16.11 | 7.10 | 6.03 | 20.74 | 8.67 | 7.89 | 1.488 | 0.560 | 1.549 | 0.464 |
| UC3, A2bits | RCA | 3.83 | 3.66 | 3.05 | 3.74 | 3.81 | 3.40 | 0.899 | 0.664 | 0.861 | 0.580 |
| UC3, A2bits | PPA | 4.61 | 4.62 | 4.34 | 4.94 | 4.96 | 5.12 | 1.037 | 0.610 | 0.946 | 0.567 |
| UC3, A2bits | PPA4 | 6.17 | 6.08 | 4.96 | 6.63 | 6.63 | 5.69 | 1.058 | 0.668 | 1.036 | 0.657 |
| UC3, reshared | RCA | 1.95 | 1.92 | 2.32 | 2.31 | 2.12 | 2.76 | 1.064 | 0.577 | 0.923 | 0.552 |
| UC3, reshared | PPA | 3.67 | 3.51 | 3.32 | 4.36 | 4.12 | 4.07 | 1.107 | 0.537 | 1.150 | 0.485 |
| UC3, reshared | PPA4 | 5.40 | 5.27 | 4.42 | 6.37 | 6.05 | 5.28 | 0.965 | 0.447 | 0.915 | 0.461 |

### COMPRESS=1

| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1, A2bits | RCA | 12.92 | 2.74 | 2.58 | 17.03 | 3.56 | 3.38 | 0.835 | 0.395 | 0.982 | 0.450 |
| UC1, A2bits | PPA | 12.87 | 2.90 | 2.59 | 16.76 | 3.67 | 3.45 | 0.899 | 0.403 | 0.977 | 0.404 |
| UC1, A2bits | PPA4 | 13.18 | 2.95 | 2.75 | 16.89 | 3.79 | 3.63 | 0.893 | 0.401 | 0.997 | 0.431 |
| UC1, reshared | RCA | 12.61 | 2.28 | 2.05 | 16.52 | 3.03 | 2.97 | 0.911 | 0.459 | 0.965 | 0.464 |
| UC1, reshared | PPA | 13.24 | 3.02 | 2.67 | 17.36 | 3.87 | 3.59 | 1.115 | 0.511 | 1.258 | 0.473 |
| UC1, reshared | PPA4 | 13.58 | 3.66 | 3.44 | 17.68 | 4.94 | 4.66 | 1.282 | 0.542 | 1.538 | 0.569 |
| UC2, A2bits | RCA | 11.61 | 2.65 | 2.41 | 15.95 | 3.54 | 3.33 | 0.792 | 0.347 | 0.726 | 0.288 |
| UC2, A2bits | PPA | 11.62 | 2.69 | 2.51 | 15.47 | 3.60 | 3.39 | 0.794 | 0.345 | 0.730 | 0.268 |
| UC2, A2bits | PPA4 | 12.07 | 2.92 | 2.64 | 15.72 | 3.69 | 3.59 | 0.930 | 0.351 | 0.761 | 0.296 |
| UC2, reshared | RCA | 11.31 | 2.21 | 2.02 | 15.67 | 2.99 | 2.87 | 0.773 | 0.393 | 0.787 | 0.311 |
| UC2, reshared | PPA | 12.17 | 2.88 | 2.51 | 16.59 | 3.75 | 3.55 | 0.971 | 0.415 | 0.916 | 0.345 |
| UC2, reshared | PPA4 | 12.33 | 3.73 | 3.40 | 16.53 | 4.80 | 4.51 | 1.210 | 0.459 | 1.266 | 0.414 |
| UC3, A2bits | RCA | 1.79 | 1.76 | 1.71 | 1.87 | 2.01 | 1.91 | 0.571 | 0.364 | 0.522 | 0.390 |
| UC3, A2bits | PPA | 1.86 | 1.78 | 1.74 | 1.89 | 2.01 | 1.97 | 0.631 | 0.376 | 0.539 | 0.395 |
| UC3, A2bits | PPA4 | 2.03 | 1.96 | 1.87 | 2.04 | 2.15 | 2.11 | 0.631 | 0.389 | 0.565 | 0.404 |
| UC3, reshared | RCA | 1.33 | 1.29 | 1.25 | 1.39 | 1.31 | 1.31 | 0.623 | 0.391 | 0.506 | 0.382 |
| UC3, reshared | PPA | 1.71 | 1.64 | 1.61 | 1.94 | 1.83 | 1.74 | 0.637 | 0.391 | 0.579 | 0.412 |
| UC3, reshared | PPA4 | 1.90 | 1.88 | 1.81 | 2.13 | 1.95 | 1.88 | 0.703 | 0.388 | 0.591 | 0.423 |

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
