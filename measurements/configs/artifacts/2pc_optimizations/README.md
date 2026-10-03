# 2PC (ABY2) all_opt configurations

Configs for `measurements/run_config.py` (one `KEY=V1,V2` per line; every combination is built with `make` and run):

```
python3 measurements/run_config.py measurements/configs/artifacts/2pc_optimizations/single_batch -p <party> -a <ip P0> -b <ip P1>
```

The files are written by `gen_configs.py` (edit that, not the `.conf` files). Every optimization flag is set
explicitly, also those whose `config.h` default is on, so a changed default cannot switch one off unnoticed; the
Makefile forwards all of them (`CONFIG_OPTIONS`, extended for `A2B_ADDER_BATCH`, `A2B_ADDER_CUT`, `A2B_BAKE_RESIDUAL`,
`A2B_BAKE_MASK_PASS`, `CHEETAH_CONV_ASYNC`, `CHEETAH_CONV_EARLY`, `CHEETAH_CONV_SIDE`, `CHEETAH_CONV_REPACK`,
`CHEETAH_CONV_POLY_N`, `CHEETAH_RELEASE_OT`). The workload is ImageNet ResNet50 with dummy weights
(`FUNCTION_IDENTIFIER` 87 / 187 / 287: RCA / PPA / PPA4 MSB adders, `MODELOWNER=DATAOWNER=-1`), i.e. runtime and
communication; accuracy needs real weights and data (`MODEL_DIR`, ... and owners `P_0` / `P_1`, see
`docs/variant_data/triad/ts1/list_acc256.txt`).

## Use cases and ReLU families

| | weights | flags |
|---|---|---|
| UC1 | secret, unknown in preprocessing | `A_KNOWN=0 MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=0` (AB conv triples) |
| UC2 | secret, known to the model owner in preprocessing | `A_KNOWN=1 MODELWEIGHTS_KNOWN_DURING_PREPROCESSING=1` |
| UC3 | public | `PUBLIC_WEIGHTS=1 TRUNC_DELAYED=1 BIT_INJECTION_TRUNC_SIM=1` |

* **A2bits**: the A2B with a public operand (`A_KNOWN_TO_EVALUATORS_OPT=1 A2B_ONLINE_OPT=1`) and the A2B bake
  (`A2B_CONV_BAKE=1`, with `A2B_BAKE_RESIDUAL`, `A2B_BAKE_MASK_PASS`): no online A2B communication.
* **reshared**: the reshared MSB adders with the reshare bake (`RESHARE_OPT=1 RESHARE_OPT_SIM=1`).

Common: fused layers (`FUSE_RELU_AVG`, `FUSE_CONV_BN`, `FUSE_DOT`), the optimized bit injection, ROT preprocessing,
the cut of the redundant top slices (`CUT_FRACTIONAL_BITS_OPT`, `A2B_ADDER_CUT`, `A2B_DELAYED_CUT`), the batched
Boolean addition (`A2B_ADDER_BATCH`), the packed / pipelined / asynchronous conv triples started early
(`CHEETAH_CONV_PACKED`, `CHEETAH_CONV_PIPELINE`, `CHEETAH_CONV_ASYNC`, `CHEETAH_CONV_EARLY`, lanes and side channels for
multi-batch), `RNG_AHEAD`, 24 GEMM / ReLU threads, 32 Cheetah threads, 100 kB send / receive buffers.

## Folders

* `single_batch/`: the six all_opt configurations (A2bits / reshared x UC1-3), 32 bits, `FRACTIONAL=5`, `COMPRESS=0,1`.
* `single_batch_WAN/`: the same builds for WAN runs. Shape both hosts with
  `LATENCY_MS=20 BANDWIDTH_MBIT=200 bash measurements/network_shaping/shape_network_alt.sh` (all non-loopback
  interfaces) and reset afterwards with `LATENCY_MS=-1 BANDWIDTH_MBIT=-1`. `CHEETAH_CONV_POLY_N=4096` is the faster
  ring in the WAN as well (flare / polynize, 20 ms / 200 Mbit/s: preprocessing 30.4 s against 33.5 s with N = 8192 in
  UC2 A2bits RCA, 37.0 against 43.3 s in UC1; `docs/variant_data/triad/conv_poly_n/`). Output repacking
  (`CHEETAH_CONV_REPACK=1`) halves the conv-triple traffic for more HE time and may pay off on slower links.
* `bitlength64/`: `BITLENGTH=DATTYPE=64`, `FRACTIONAL=12,16` (the cut's narrow adders exist for F in
  {8, 10, 12, 14, 16, 18, 20, 24}), `COMPRESS=0`. ConvTriple must be built with `TRIPLE_BITLEN=64`
  (`./build_cpu.sh` in `build64`); see `docs/BITLENGTH64.md`.
* `multi_batch/`: 24 processes x `DATTYPE=256` (8 images per process, ImageNet RCA), one Cheetah thread per process.
* `truncation/`: the truncation approaches fused into the ReLUs (A2bits only: they need the A2B bake;
  `TRUNC_DELAYED=1`, `COMPRESS=0`): `TRUNC_APPROACH` 4 (TS_Mix), 1 (TS1), 3 (TE1), 2 (TE0), at 32 bits with
  `FRACTIONAL=5,8` and at 64 bits with `FRACTIONAL=12,16`. TE0 / TE1 are exact (`docs/TRIAD_ALL_OPT.md`). The 32-bit
  PPA / PPA4 cut under the A2B bake runs narrow adders (`CUT_NARROW_32=1`): the identity-substituted circuits it replaced
  DReLU wrongly for some inputs near the cut's limit (`docs/BITLENGTH64.md`, "Found on the way").
