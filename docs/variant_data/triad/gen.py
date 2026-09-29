#!/usr/bin/env python3
"""gen.py: build lists for the triad 2PC single-batch all_opt configs.
conf_*: the config as given (ImageNet, dummy model/data); opt_*: + all additional optimizations;
cif_*: the opt flags on CIFAR-10 ResNet50 (71/171/271) with real model/data (P_0 / P_1), 10 images, for correctness."""
import os, sys
D = "/home/q662180/workspace/hpmpc_merge/measurements/configs/artifacts/triad/2pc/single_batch"
SHORT = {"A2bits": "a2b", "A2bits_A_Not_Known": "a2bk0", "A2bits_Public": "a2bpw",
         "reshared": "rs", "reshared_A_Not_Known": "rsk0", "Reshared_Public": "rspw"}
ADD = {87: "rca", 187: "ppa", 287: "ppa4"}
OPT = ["CHEETAH_CONV_PACKED=1", "CHEETAH_CONV_PIPELINE=1", "CUT_FRACTIONAL_BITS_OPT=1", "ADDITIONAL_RELU_THREADS=24",
       "RNG_AHEAD=1", "SEND_BUFFER=100000", "RECV_BUFFER=100000"]
MEAS = ["PRINT_OUTPUT_HASH=1", "NET_WAIT_STATS=1", "USE_CUDA_GEMM=0"]
out = {"conf": [], "opt": [], "cif": []}
for key, short in SHORT.items():
    lines = [l.strip() for l in open(f"{D}/2PC_single_batch_all_opt_{key}.conf") if l.strip()]
    kv = dict(l.split("=", 1) for l in lines)
    for f in (87, 187, 287):
        for c in ("0", "1"):
            base = [f"{k}={v}" for k, v in kv.items() if k not in ("FUNCTION_IDENTIFIER", "COMPRESS")]
            base += [f"FUNCTION_IDENTIFIER={f}", f"COMPRESS={c}"]
            name = f"{short}_{ADD[f]}_c{c}"
            out["conf"].append(f"conf_{name} " + " ".join(base + MEAS))
            opt = OPT + (["A2B_CONV_BAKE=1"] if kv.get("A2B_ONLINE_OPT") == "1" else [])
            out["opt"].append(f"opt_{name} " + " ".join(base + opt + MEAS))
            cb = [x for x in base if not x.split("=")[0] in ("FUNCTION_IDENTIFIER", "MODELOWNER", "DATAOWNER", "NUM_INPUTS")]
            cb += [f"FUNCTION_IDENTIFIER={f - 16}", "MODELOWNER=P_0", "DATAOWNER=P_1", "NUM_INPUTS=10"]
            out["cif"].append(f"cif_{name} " + " ".join(cb + opt + MEAS))
for k, v in out.items():
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"list_{k}.txt"), "w").write("\n".join(v) + "\n")
    print(k, len(v))
