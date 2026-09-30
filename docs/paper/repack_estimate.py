#!/usr/bin/env python3
"""repack_estimate.py: total ImageNet traffic of the COMPRESS=0 all_opt builds (P0 sent + received, MiB, comm_fp.csv)
before and after output repacking (variant B of he_model.py at N = 8192: 242.6 MiB of conv triples per HE product,
4.72 MiB of Galois keys once per evaluating party; UC1's AB triples need both products)."""
import csv
MIB = 1e6 / 2**20
rows = {}
for r in csv.DictReader(open("../variant_data/triad/comm_fp.csv")):
    if r["tag"].startswith("fin2_") and r["tag"].endswith("_c0_r1"):
        v = {k: float(x) for k, x in r.items() if k != "tag"}
        rows[r["tag"][5:-6]] = (v["trip_sent"] + v["trip_recv"] + v["keyex_sent"] + v["keyex_recv"]
                                + (v["pre_sent"] + v["pre_recv"]) * MIB, (v["on_sent"] + v["on_recv"]) * MIB)
CONV = {"UC1": 958.2, "UC2": 479.1, "UC3": 0.0}       # measured conv-triple traffic
NEW = {"UC1": 2 * 242.6, "UC2": 242.6, "UC3": 0.0}
KEYS = {"UC1": 2 * 4.72, "UC2": 4.72, "UC3": 0.0}
lab = [("a2bk0", "UC1", "A2bits"), ("rsk0", "UC1", "reshared"), ("a2b", "UC2", "A2bits"), ("rs", "UC2", "reshared"),
       ("a2bpw", "UC3", "A2bits"), ("rspw", "UC3", "reshared")]
out = []
for c, u, name in lab:
    for a in ("rca", "ppa", "ppa4"):
        pre, on = rows[f"{c}_{a}"]
        pre2 = pre - CONV[u] + NEW[u] + KEYS[u]
        out.append((u, f"{u}, {name}, {a.upper()}", pre, pre2, on))
for u, n, pre, pre2, on in out:
    print(f"{n:22} pre {pre:6.0f} -> {pre2:6.0f} MiB ({(pre2 - pre) / pre:+.0%}) | with online {pre + on:6.0f} -> {pre2 + on:6.0f} ({(pre2 - pre) / (pre + on):+.0%})")
for u in ("UC1", "UC2", "UC3"):
    s = [x for x in out if x[0] == u]
    print(f"{u}: preprocessing {min(x[2] for x in s):.0f}-{max(x[2] for x in s):.0f} -> {min(x[3] for x in s):.0f}-{max(x[3] for x in s):.0f} MiB, "
          f"{min((x[3] - x[2]) / x[2] for x in s):+.0%} to {max((x[3] - x[2]) / x[2] for x in s):+.0%}; "
          f"with online {min((x[3] - x[2]) / (x[2] + x[4]) for x in s):+.0%} to {max((x[3] - x[2]) / (x[2] + x[4]) for x in s):+.0%}")
