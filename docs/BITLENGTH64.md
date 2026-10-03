# 2PC (ABY2) with BITLENGTH = 64

`BITLENGTH=64 DATTYPE=64` now runs the 2PC configurations end to end (PROTOCOL 4, ROT_PREPROCESSING_OPT, the
packed HE convolutions). Two things were missing, both in the preprocessing.

## 1. HE triples modulo 2^64 (ConvTriple)

The convolution and FC triples come from BFV with the plaintext modulus t = 2^l (Cheetah's coefficient encoding),
and SEAL's plaintext moduli have at most 60 bits: there were no 64-bit conv triples at all (and ConvTriple's
`constants.hpp` did not even compile with `TRIPLE_BITLEN=64`, `1 << 64`).

`PackedConv2D::setUpWide` (`nn/ConvTriple/src/core/conv_packed.cpp`, called by `Keys` when `BIT_LEN == 64`) gives the
packed convolutions their own context and does t = 2^64 outside SEAL:

* N = 8192, q = three 55-bit primes (165 bits; 128-bit security allows 218), own keys, public keys exchanged.
  SEAL's context gets a placeholder plaintext modulus.
* Encoding round(q m / 2^64) exactly (`add_scaled`: q = D 2^64 + r, so round(q m / 2^64) = D m + round(r m / 2^64)),
  for the inputs, the evaluator's own share and the masks. Exact scaling removes BFV's (q mod t) K error term, so the
  product's noise is sum (e + rho) w over up to 2^19.4 weight terms per output coefficient (ResNet50's widest layers),
  with centered 64-bit weight shares (`Ntt::lift`): below 2^78 except with probability 2^-40 (13 sigma), 2^86 worst case.
* The evaluator floods at the input level with up to Delta / 16 = 2^96 (18-19 bits above the products' noise, as
  Cheetah's 32-bit parameters: 2^64 against 2^45.4), switches to two primes (q' < 2^110), adds a public-key encryption
  of zero there, subtracts its mask, composes the coefficients (CRT) and keeps their high bits (c0 truncated below
  Delta' / 8; c1 so that c1's truncation times the ternary secret stays below Delta' / 8 at 14 sigma). Total noise under
  5/16 Delta'.
* The decryptor reassembles the residues, computes c0 + c1 s per prime, composes, and m = round(x 2^64 / q')
  (a long double estimate corrected with the exact remainder).
* FC layers go through the same evaluator as 1x1 convolutions of 1x1 images. gemini's FC, conv and BN stay in SEAL's
  plaintext space: with 64 bits they throw (BN: use `FUSE_CONV_BN=1`; convs: `CHEETAH_CONV_PACKED=1`). No GPU
  evaluator (N = 4096 only).
* Repacking (`CHEETAH_CONV_REPACK=1`): the same data modulus plus a 53-bit special prime (218 bits, the limit), the
  outputs packed densely with Galois automorphisms.

### Traffic

A product's query has 1.5x the bits per coefficient of the 32-bit one (165 against 109 bits), its response ~1.9x
(two primes kept, 64-bit plaintext), and N = 8192 doubles the slots per ciphertext; with the sparse output layout the
response cost per output element grows with sqrt(N) (32-bit triples at N = 8192: 1.31x the traffic of N = 4096, 629
against 479 MiB on ImageNet, `CHEETAH_CONV_POLY_N=8192`). ImageNet conv triples (AB2, dummy weights, laptop, 8 threads):

| | traffic | HE time |
|---|---|---|
| 32 bits, N = 4096 (default) | 479 MiB | 5.4 s |
| 32 bits, N = 8192 | 629 MiB | 6.2 s |
| 32 bits, repacking | 285 MiB | 18.0 s |
| 64 bits, 180-bit q (first version) | 1073 MiB | |
| 64 bits, 165-bit q | 1013 MiB | 10.2 s |
| 64 bits, repacking | 464 MiB | 28.9 s |


Build: `TRIPLE_BITLEN=64 ./build_cpu.sh` in nn/ConvTriple (build dir `build64`); hpmpc's Makefile and
`scripts/variants/vb.sh` link `build64` for `BITLENGTH=64`, and config.h sets `TRIPLE_BITLEN` from `BITLENGTH` (the
interface's `UINT_TYPE` makes a mismatch a link error).

Check (`cheetah_conv_triple_test`, two parties on one host, c1 + c2 == conv(x1 + x2, w1 + w2) exactly, random
64-bit operands): `small` and `cifar` suites, AB2 and AB, layer by layer and pipelined: 0 failed. Cost on the CIFAR-10
ResNet50 convs (23 shapes, batch 10, 8 threads, pipelined):

| | AB2 time | AB2 traffic | AB time | AB traffic |
|---|---|---|---|---|
| 32 bits | 0.65 s | 98.7 MiB | 0.98 s | 197.5 MiB |
| 64 bits | 1.12 s | 235.3 MiB | 1.78 s | 470.6 MiB |

## 2. The ROT-preprocessing adders at k = 64

The MSB and full adders of `programs/functions/adders/zero_add_adders` were generated for k = 8 / 16 / 32 only. The
circuit generator (generate_hpp.py, a separate checkout; it reproduces the imported 8/16/32-bit files bit for bit)
now writes their 64-bit specializations through `scripts/circuits/gen_64bit_adders.py` (`--check` compares):

* RCA and PPA (Sklansky), plain / reshared / a-known: the generator's generic builders.
* The 4-way PPA (PPA4), which the generator builds by hand per width: 64-bit trees in the script, 4 AND levels
  (21 groups of 3 slices, blocks of 4 groups, a superblock of 4 blocks plus the tail, one dot gate; 3 levels of these
  gates cover at most 48 slices). `is_ppa4_reshared(64, i)`: the first slice of each group.
* The a-known PPA4 (`PPA_MSB_4Way_A_AB`, A2bits) is left out: the generator's form needs the hand fixes of hpmpc
  4ec9292 (operand order, dot-pending products, chain masks). With 64 bits the A2bits builds take the AB circuit, with
  the public m as a share of mask 0 (`share_conversion.hpp`).
* The reshare bake (`RESHARE_OPT_SIM`) at 64 bits: RESHARE_BAKE_ACTIVE and the party-local 3-tuples for BITLENGTH 64
  too; the bake's maps for the 64-bit circuits and, with the cut, the narrow ones (`wide_ppa4_width`,
  `ppa4_reshared_b3_count_wide`; RCA and PPA take the 32-bit formulas). The generator's SIM branches of the reshared
  PPA4 re-mask with `zero_add_local` (as the 8/16/32-bit files: the ordinary zero_add for inputs whose mask is not
  baked); the script checks them against the bake's maps (reshared slices j % 3 == 1 with rt[(j - 1) / 3], skipped
  zero_adds at j % 3 == 2 with 3-tuple 2 (j - 2) / 3). At 54 and 48 bits (F = 10, 16) the generator skips other
  zero_adds (4-tuple .d fields, which the bake cannot serve): there they zero_add normally
  (`ppa4_reshared_sim_za_wide`), the reshares stay baked. func 53 (P1's reshare check): 51 / 51 (RCA, PPA), 17 / 17
  (PPA4) matched; CIFAR 25 of 32; P1's ReLU preprocessing traffic -15% for PPA (CIFAR, 32 images).
* `single_row_ortho` (the bit injection's one-word transpose) for DATTYPE = BITLENGTH = 64 too.
* Arithmetic triples (secret-by-secret products: max pooling, ...): gemini's elementwise product is limited to SEAL's
  plaintext space, so with 64 bits every such product came out random (func 53 MaxPool and func 54 Multiplication
  failed; the CIFAR ResNet50 has no max pooling, ImageNet's has one). ConvTriple now multiplies by Gilboa over the
  silent COTs (one COT of width 64 - j per bit j).

## 3. CUT_FRACTIONAL_BITS_OPT at 64 bits: narrow adders

The 32-bit cut identity-substitutes the full-width circuits by hand (public constants on the top FRACTIONAL slices,
gate skipping with mask retargeting, docs/CUT_FRACTIONAL_BITS_OPT.md). At 64 bits the MSB adder is simply a narrower
one: the A2B prepares the full width as before (vacant slices public constants, the boundary slice FRACTIONAL masked
and sent like slice 0), and the generated adder of width 64 - F runs on slices F..63 (`get_msb_range`,
`cut_frac_narrow_on`): the MSB of the low 64 - F bits of the sum, which is the sign of a value with F bits of sign
extension. Correct by construction, no per-gate edits.

* `gen_64bit_adders.py` writes the narrow adders of all eight MSB families (RCA, PPA, PPA4; plain / reshared / a-known,
  the a-known PPA4 again taking the AB circuit) for F in {8, 10, 12, 14, 16, 18, 20, 24} to
  `zero_add_adders/narrow64/<family>.hpp` (one `#if FRACTIONAL == F` block per width) and `narrow64/widths.h`
  (`CUT_FRAC_NARROW64_HAVE`, which `CUT_FRAC_ELIGIBLE` checks: other F run the full width). The 4-way tree is now
  width-generic (groups of 3, blocks of up to 4 per level; reproduces the committed 64-bit tree exactly), and the script
  checks that each reshared circuit reshares the slices the A2B prepare skips (`is_ppa4_reshared` for k > 32: the first
  slice of each group; `ppa4_reshared_at` maps the full-width prepare to the narrow adder's positions).
* Everything downstream of the cut works unchanged at 64 bits: A2B_ADDER_CUT (the bake's Boolean addition stops at
  64 - F sum bits: 51 instead of 63 rounds at F = 12), A2B_DELAYED_CUT (UC3's TS{L}), TS1's shifted design.
* Not with the split four-way circuits (`ADDITIONAL_PPA_THREADS > 0`).

Savings at F = 12 (func 59, 4096 values): RCA 63 -> 51 ANDs per value (Boolean triples -19%), PPA -20%, PPA4 -19%
(Boolean triples) / -19% (3- and 4-tuples); RCA online rounds 64 -> 52 per ReLU.

`scripts/circuits/check_wire_order.py` verifies wire order statically (every wire complete before it is read), which found the
reshare bug fixed by hand in ac3440c and the a-known PPA4 order defects of 4ec9292 in the raw generator output.

## Tests

* func 59 (`RELU`, `RELU_RANDOM` with |v| = 2^e (1 + mantissa), e up to 62, with the cut up to 62 - F), BITLENGTH 64,
  FRACTIONAL 12, the list_fin2 flags of A2bits and reshared UC1 / UC2 / UC3 with RCA, PPA and PPA4: 18 builds, 0 of 4096
  wrong each, without and with the cut (UC3 also `RELU_DCUT`, A2B_DELAYED_CUT).
* CIFAR-10, AdamW ResNet50, the first 32 test images, FRACTIONAL 12 (laptop, both parties, 8 threads): A2bits UC2 RCA,
  reshared UC1 RCA, A2bits UC1 PPA, reshared UC3 RCA, A2bits UC2 PPA4, reshared UC2 PPA4: 25 of 32 each; 32 bits with
  FRACTIONAL 5 (A2bits UC2 RCA): 21 of 32. Preprocessing 43-88 s and online 9-22 s at 64 bits against 23 s and 14 s.
  With the cut: A2bits UC2 RCA, reshared UC1 RCA, A2bits UC2 PPA4, reshared UC3 RCA, reshared UC2 PPA: 25 of 32 each.

## ImageNet: 64 against 32 bits

The 18 all_opt builds with COMPRESS=0 (list_fin2) at 64 bits (F = 12) and 32 bits (F = 5), interleaved on flare / polynize
(hpmpc 4c2e446, dummy weights, median of 3; P0's traffic sent + received, MiB; `docs/variant_data/triad/te_bit64/res3_x64.csv`,
`comm3_x64.csv`, `tables.py x64`). Preprocessing traffic 1.7-2.1x (HE 2.1x, OT 1.7-2.1x), online traffic 2.0x,
preprocessing time 1.5-2.0x, online time 1.6-1.9x; RCA 51 instead of 26 AND levels per ReLU. With repacking
(`CHEETAH_CONV_REPACK=1`) UC2 A2bits RCA sends 1265 instead of 1789 MiB (HE 524 instead of 1050), 1.37x the 32-bit
traffic, for 6.60 instead of 4.82 s of preprocessing (UC1: 1631 MiB, 6.76 s).

| build | pre MiB 32 / 64 | pre s | online MiB | online s | rounds |
|---|---|---|---|---|---|
| UC1 A2bits RCA | 1,341 / 2,679 | 3.29 / 4.93 | 210 / 418 | 0.50 / 0.87 | 1818 / 3043 |
| UC1 A2bits PPA | 1,443 / 2,891 | 4.39 / 6.69 | 307 / 620 | 0.47 / 0.78 | 949 / 998 |
| UC1 A2bits PPA4 | 2,068 / 4,317 | 5.75 / 11.50 | 202 / 403 | 0.39 / 0.66 | 730 / 779 |
| UC2 A2bits RCA | 922 / 1,789 | 3.29 / 4.82 | 168 / 333 | 0.43 / 0.79 | 1714 / 2939 |
| UC2 A2bits PPA | 1,023 / 2,000 | 4.57 / 6.62 | 264 / 535 | 0.38 / 0.67 | 845 / 894 |
| UC2 A2bits PPA4 | 1,648 / 3,426 | 5.64 / 11.11 | 159 / 318 | 0.34 / 0.58 | 626 / 675 |
| UC3 A2bits RCA | 421 / 735 | 2.88 / 4.35 | 125 / 249 | 0.49 / 0.81 | 1714 / 2939 |
| UC3 A2bits PPA | 523 / 946 | 4.06 / 6.04 | 222 / 450 | 0.41 / 0.78 | 845 / 894 |
| UC3 A2bits PPA4 | 1,147 / 2,372 | 5.47 / 10.77 | 117 / 233 | 0.41 / 0.69 | 626 / 675 |
| UC1 reshared RCA | 1,175 / 2,409 | 2.13 / 3.65 | 239 / 474 | 0.52 / 1.00 | 1867 / 3092 |
| UC1 reshared PPA | 1,330 / 2,627 | 3.97 / 6.04 | 349 / 676 | 0.59 / 0.95 | 1004 / 1047 |
| UC1 reshared PPA4 | 2,059 / 4,047 | 5.57 / 9.16 | 233 / 459 | 0.54 / 0.88 | 779 / 828 |
| UC2 reshared RCA | 756 / 1,518 | 2.13 / 3.30 | 197 / 389 | 0.46 / 0.88 | 1763 / 2988 |
| UC2 reshared PPA | 911 / 1,735 | 3.99 / 5.94 | 306 / 591 | 0.49 / 0.78 | 900 / 943 |
| UC2 reshared PPA4 | 1,639 / 3,156 | 5.50 / 9.14 | 190 / 374 | 0.45 / 0.74 | 675 / 724 |
| UC3 reshared RCA | 257 / 469 | 1.71 / 2.81 | 154 / 304 | 0.51 / 0.92 | 1763 / 2988 |
| UC3 reshared PPA | 436 / 734 | 3.24 / 4.87 | 264 / 506 | 0.49 / 0.86 | 900 / 943 |
| UC3 reshared PPA4 | 1,062 / 1,992 | 4.91 / 8.22 | 148 / 289 | 0.48 / 0.83 | 675 / 724 |

## Truncation at 64 bits

TS1 / TS_Mix and the new TE0 / TE1 (`docs/TRIAD_ALL_OPT.md`) run at 64 bits: func 59 `RELU_TE0` / `RELU_TE1` exact
for RCA, PPA and PPA4 (TE1 with the narrow cut adders), TS1's pooling tests pass. CIFAR-10 (AdamW ResNet50, the first
256 test images, UC2 A2bits RCA, flare / polynize): TS{L} as given 188 at F = 12 and 189 at F = 16, TE1 and TE0 188 / 189:
the plaintext model's 189 (32 bits: TS{L} 159 / 169 at F = 5 / 8, TE 158 / 185).

## Found on the way (32 bits): the identity-substituted cut under the A2B bake

`RELU_RANDOM` with the 32-bit PPA and PPA4 adders under the A2B bake with the cut (A2bits, `CUT_FRACTIONAL_BITS_OPT=1`,
`TRUNC_DELAYED=0`) got 114 of 4096 wrong, DReLU flipped (positive inputs gave 0, negative ones passed), identically at
hpmpc 810ebc8. The failures sit below the cut's limit 2^(31-F) and their rate grows with |v| (F = 2: 2^22..2^28, F = 5:
2^16..2^25, F = 8: 2^14..2^22; 40-45% in the top octave). Narrowed down (func 59, UC2 A2bits PPA): the same 114 with the
AB circuit (`A_KNOWN_TO_EVALUATORS_OPT=0`), with `A2B_ADDER_CUT=0` and with `A2B_ADDER_BATCH=0`; 0 wrong with
`A2B_CONV_BAKE=0` or `CUT_FRACTIONAL_BITS_OPT=0`; RCA (which never reads the vacant slices) and the 64-bit narrow
adders get 0 under the bake with the cut. So the identity-substituted 32-bit prefix circuits (PPA and PPA4 alike) go
wrong with the bake's [c]; TE1 and TS1 with these adders inherited it (TE1 PPA: 49 of 4096).

**Fix (`CUT_NARROW_32=1`, default):** under the bake, the 32-bit PPA / PPA4 cut runs narrow adders of 32 - F bits as at
64 bits (`narrow32/`, F = 2..12: the a-known and AB PPA, and for PPA4 the AB four-way circuit, since the a-known one
has no generated form; the public m as a share of mask 0). func 59: 0 of 4096 wrong for A2bits PPA / PPA4 in UC1-3,
TE1 and TS1. Cost (4096 values, UC2 A2bits): PPA 4% fewer AND triples, 6% less online traffic; PPA4 5% less online
traffic but more tuples (Beaver triples 120k -> 50k, 3-tuples 54k -> 83k, 4-tuples 12k -> 50k: about 15 bytes of OT
material more per value). The reshared and plain builds (no bake) keep the identity circuits, which pass.
