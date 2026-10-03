#!/usr/bin/env python3
"""Writes the 2PC all_opt configs of this folder (measurements/run_config.py format: KEY=V1,V2 per line, no comments).

Every optimization flag is set explicitly, also those whose config.h default is on, so that a default change cannot
switch one off silently (the Makefile forwards all of them, CONFIG_OPTIONS). See README.md.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

COMMON = [
    ("FUNCTION_IDENTIFIER", "87,187,287"),  # ImageNet ResNet50 with RCA / PPA / PPA4 (dummy weights: runtime)
    ("PROTOCOL", "4"), ("PRE", "1"), ("SKIP_PRE", "0"),
    ("BITLENGTH", "32"), ("DATTYPE", "32"), ("FRACTIONAL", "5"), ("NUM_INPUTS", "1"),
    ("MODELOWNER", "-1"), ("DATAOWNER", "-1"),
    # layer fusion and the bit injection
    ("OPTIMIZED_BIT_INJECTION_RELU", "1"), ("BIT_INJECTION_PREPROCESSING_OPT", "1"), ("ROT_PREPROCESSING_OPT", "1"),
    ("FUSE_RELU_AVG", "1"), ("FUSE_CONV_BN", "1"), ("FUSE_DOT", "1"), ("INTERLEAVE_COMM", "1"), ("MSB0_OPT", "1"),
    # HE / OT preprocessing (ConvTriple)
    ("CHEETAH_BOOL_OT_TYPE", "0"), ("CHEETAH_THREADS", "32"), ("CHEETAH_CONV_TYPE", "0"), ("CHEETAH_CONV_PACKED", "1"),
    ("CHEETAH_CONV_PIPELINE", "1"), ("CHEETAH_CONV_ASYNC", "1"), ("CHEETAH_CONV_EARLY", "1"), ("CHEETAH_CONV_LANES", "1"),
    ("CHEETAH_CONV_SIDE", "1"), ("CHEETAH_CONV_POLY_N", "4096"), ("CHEETAH_CONV_REPACK", "0"), ("CHEETAH_BN_BATCHED", "1"),
    ("CHEETAH_DISCONNECT", "1"), ("CHEETAH_RELEASE_OT", "1"), ("CHEETAH_WAN_OPT", "0"),
    # the MSB adders
    ("CUT_FRACTIONAL_BITS_OPT", "1"), ("CUT_NARROW_32", "1"), ("A2B_ADDER_BATCH", "1"), ("A2B_ADDER_CUT", "1"),
    ("A2B_DELAYED_CUT", "1"), ("A2BITS_PPA4_AB", "0"), ("TE_LOW_ADDER", "-1"),
    ("COMPRESS", "0,1"),
    # threads, buffers
    ("ADDITIONAL_GEMM_THREADS", "24"), ("ADDITIONAL_RELU_THREADS", "24"), ("RNG_AHEAD", "1"),
    ("SEND_BUFFER", "100000"), ("RECV_BUFFER", "100000"), ("USE_CUDA_GEMM", "0"),
    ("TRUNC_APPROACH", "0"),
]
A2BITS = [("A_KNOWN_TO_EVALUATORS_OPT", "1"), ("A2B_ONLINE_OPT", "1"), ("A2B_CONV_BAKE", "1"), ("A2B_BAKE_RESIDUAL", "1"),
          ("A2B_BAKE_MASK_PASS", "1"), ("RESHARE_OPT", "0"), ("RESHARE_OPT_SIM", "0")]
RESHARED = [("A_KNOWN_TO_EVALUATORS_OPT", "0"), ("A2B_ONLINE_OPT", "0"), ("A2B_CONV_BAKE", "0"), ("RESHARE_OPT", "1"),
            ("RESHARE_OPT_SIM", "1")]
UC = {  # UC1: no weights known, UC2: weights known in preprocessing, UC3: public weights
    "UC1": [("A_KNOWN", "0"), ("MODELWEIGHTS_KNOWN_DURING_PREPROCESSING", "0"), ("PUBLIC_WEIGHTS", "0"),
            ("TRUNC_DELAYED", "0")],
    "UC2": [("A_KNOWN", "1"), ("MODELWEIGHTS_KNOWN_DURING_PREPROCESSING", "1"), ("PUBLIC_WEIGHTS", "0"),
            ("TRUNC_DELAYED", "0")],
    "UC3": [("A_KNOWN", "1"), ("MODELWEIGHTS_KNOWN_DURING_PREPROCESSING", "1"), ("PUBLIC_WEIGHTS", "1"),
            ("TRUNC_DELAYED", "1"), ("BIT_INJECTION_TRUNC_SIM", "1")],
}


def write(sub, name, *parts, **over):
    d = {}
    for part in parts:
        for k, v in part:
            d[k] = v
    d.update(over)
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    with open(os.path.join(HERE, sub, name + ".conf"), "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in d.items()))


def main():
    for uc, flags in UC.items():
        for mode, mflags in (("A2bits", A2BITS), ("reshared", RESHARED)):
            base = (COMMON, mflags, flags)
            write("single_batch", f"2PC_all_opt_{mode}_{uc}", *base)
            # WAN: the same builds with output repacking (N = 8192 with a special prime: 40% less conv-triple traffic,
            # 14-20% faster preprocessing at 20 ms / 200 Mbit/s, 1.0-1.5 s slower in the LAN; N = 4096 cannot repack at
            # 128-bit security); shape the links with measurements/network_shaping/shape_network_alt.sh
            write("single_batch_WAN", f"2PC_WAN_all_opt_{mode}_{uc}", *base, CHEETAH_CONV_REPACK="1")
            # 64 bits: the narrow cut adders exist for FRACTIONAL 8, 10, 12, 14, 16, 18, 20, 24; no COMPRESS
            write("bitlength64", f"2PC_64bit_all_opt_{mode}_{uc}", *base, BITLENGTH="64", DATTYPE="64",
                  FRACTIONAL="12,16", COMPRESS="0")
            # multi-batch: 24 processes x DATTYPE 256 (8 images per process)
            write("multi_batch", f"2PC_multi_batch_all_opt_{mode}_{uc}", *base, FUNCTION_IDENTIFIER="87",
                  DATTYPE="256", PROCESS_NUM="24", CHEETAH_THREADS="1")
        # truncation approaches fused into the ReLUs (2PC: A2bits with the A2B bake, full-width ReLUs, TRUNC_DELAYED=1):
        # 4 TS_Mix, 1 TS1, 3 TE1, 2 TE0
        write("truncation", f"2PC_trunc_all_opt_A2bits_{uc}", COMMON, A2BITS, flags, TRUNC_APPROACH="4,1,3,2",
              TRUNC_DELAYED="1", COMPRESS="0", FRACTIONAL="5,8")
        write("truncation", f"2PC_64bit_trunc_all_opt_A2bits_{uc}", COMMON, A2BITS, flags, TRUNC_APPROACH="4,1,3,2",
              TRUNC_DELAYED="1", COMPRESS="0", BITLENGTH="64", DATTYPE="64", FRACTIONAL="12,16")


if __name__ == "__main__":
    main()
