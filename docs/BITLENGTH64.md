# 2PC (ABY2) with BITLENGTH = 64

`BITLENGTH=64 DATTYPE=64` now runs the 2PC configurations end to end (PROTOCOL 4, ROT_PREPROCESSING_OPT, the
packed HE convolutions). Two things were missing, both in the preprocessing.

## 1. HE triples modulo 2^64 (ConvTriple)

The convolution and FC triples come from BFV with the plaintext modulus t = 2^l (Cheetah's coefficient encoding),
and SEAL's plaintext moduli have at most 60 bits: there were no 64-bit conv triples at all (and ConvTriple's
`constants.hpp` did not even compile with `TRIPLE_BITLEN=64`, `1 << 64`).

`PackedConv2D::setUpWide` (`nn/ConvTriple/src/core/conv_packed.cpp`, called by `Keys` when `BIT_LEN == 64`) gives the
packed convolutions their own context and does t = 2^64 outside SEAL:

* N = 8192, q = three 60-bit primes (180 bits; 128-bit security allows 218), own keys, public keys exchanged.
  SEAL's context gets a placeholder plaintext modulus.
* Encoding round(q m / 2^64) exactly (`add_scaled`: q = D 2^64 + r, so round(q m / 2^64) = D m + round(r m / 2^64)),
  for the inputs, the evaluator's own share and the masks. Exact scaling removes BFV's (q mod t) K error term, so the
  product's noise is (e + 1/2) |w|_1 <= 2^81 for 64-bit weight shares (centered lift, `Ntt::lift`).
* The evaluator floods at the input level with up to 2^112 (31 bits above the products' noise; Cheetah's 32-bit
  flooding has 16), switches to two primes (q' < 2^120), adds a public-key encryption of zero there, subtracts its
  mask, composes the coefficients (CRT) and keeps their high bits: c0's used coefficients 68 bits, c1 83 bits. Noise
  budget at q': flooding below 2^53 after the switch, the truncations of c0 and of c1 s below 2^52 each, Delta'/2 = 2^55.
* The decryptor reassembles the residues, computes c0 + c1 s per prime, composes, and m = round(x 2^64 / q')
  (a long double estimate corrected with the exact remainder).
* FC layers go through the same evaluator as 1x1 convolutions of 1x1 images. gemini's FC, conv and BN stay in SEAL's
  plaintext space: with 64 bits they throw (BN: use `FUSE_CONV_BN=1`; convs: `CHEETAH_CONV_PACKED=1`). No repacking
  (`CHEETAH_CONV_REPACK`), no GPU evaluator (N = 4096 only).

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
* 32-bit-only, so off at 64 bits: `CUT_FRACTIONAL_BITS_OPT` (CUT_FRAC_ELIGIBLE), the reshare bake
  (`RESHARE_OPT_SIM`: RESHARE_BAKE_ACTIVE, the party-local 3-tuples, the generator's SIM branches of the reshared PPA4,
  keyed on RESHARE_BAKE_ACTIVE), and with them TS1's cut design.

`scripts/circuits/check_wire_order.py` verifies wire order statically (every wire complete before it is read), which found the
reshare bug fixed by hand in ac3440c and the a-known PPA4 order defects of 4ec9292 in the raw generator output.

## Tests

* func 59 (`RELU`, `RELU_RANDOM` with |v| = 2^e (1 + mantissa), e up to 62), BITLENGTH 64, FRACTIONAL 12, the
  list_fin2 flags of A2bits and reshared UC1 / UC2 / UC3 with RCA, PPA and PPA4: 18 builds, 0 of 4096 wrong each.
* CIFAR-10, AdamW ResNet50, the first 32 test images, FRACTIONAL 12 (laptop, both parties, 8 threads): A2bits UC2 RCA,
  reshared UC1 RCA, A2bits UC1 PPA, reshared UC3 RCA, A2bits UC2 PPA4, reshared UC2 PPA4: 25 of 32 each; 32 bits with
  FRACTIONAL 5 (A2bits UC2 RCA): 21 of 32. Preprocessing 43-88 s and online 9-22 s at 64 bits against 23 s and 14 s.

## Found on the way (32 bits, not changed)

`RELU_RANDOM` with the 32-bit A2bits PPA and PPA4 adders (UC2, CUT_FRACTIONAL_BITS_OPT, TRUNC_DELAYED=0) gets 114 of
4096 wrong, all with |v| between 2^16 and 2^25 (more towards 2^25), identically at hpmpc 810ebc8; RCA gets 0. The
a-known PPA / PPA4 cut paths mis-handle large values below the cut's limit 2^26 (activations of 2^11 and more at
FRACTIONAL 5, rare in the networks). Open.
