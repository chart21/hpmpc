#!/usr/bin/env python3
"""acc_table.py SINGLE.csv MULTI.csv: accuracy of every variant with the AdamW model (MX_MODEL=wd runs).

SINGLE: the 100-image single-batch builds (a_*_t2, one process, NUM_INPUTS=100); MULTI: the multi-batch
builds (m_*_t2, 24 processes x 8 lanes = 192 images). Also lists the 100-image preprocessing / online time.
"""
import csv
import sys

ADDERS = ["rca", "ppa", "ppa4"]
FAMILIES = [("plain", "plain"), ("r", "reshare"), ("rs", "reshare + sim"), ("a", "a2b"), ("ak", "a2b + AKTE")]


def load(path):
    rows = {}
    for row in csv.DictReader(open(path)):
        if row.get("tag") in (None, "missing") or not row.get("pre"):
            continue
        rows.setdefault(row["name"], []).append(row)
    return rows


def acc(rows):
    if not rows:
        return "-"
    r = rows[0]
    ok, n = int(r["acc_ok"]), int(r["acc_n"])
    return f"{ok}/{n} ({100 * ok / n:.0f}%)"


def secs(rows, key, d):
    return f"{float(rows[0][key]):.{d}f}" if rows else "-"


if __name__ == "__main__":
    single, multi = load(sys.argv[1]), load(sys.argv[2])
    print("| adder | family | BN | single batch, 100 images | multi-batch, 192 images | "
          "pre s (100 images) | online s (100 images) |")
    print("|---|---|---|---|---|---|---|")
    for fuse in ("0", "1"):
        for a in ADDERS:
            for f, fname in FAMILIES:
                s = single.get(f"a_{a}_{f}_f{fuse}_t2", [])
                m = multi.get(f"m_{a}_{f}_f{fuse}_t2", [])
                print(f"| {a} | {fname} | {'fused' if fuse == '1' else 'unfused'} | {acc(s)} | {acc(m)} | "
                      f"{secs(s, 'pre', 1)} | {secs(s, 'online', 2)} |")
