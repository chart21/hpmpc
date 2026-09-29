#!/usr/bin/env python3
"""doc_tables.py res_fp.csv res_ag.csv [SUFFIX]: compact per-variant tables of both host pairs (median over runs).

Columns: preprocessing and online seconds on flare/polynize (fp) and algofi/goracle (ag), communication rounds,
preprocessing traffic sent by P0 (MB, sum over processes), accuracy on each pair. SUFFIX (default t2) selects
the threaded builds of that round.
"""
import csv
import statistics
import sys

ADDERS = ["rca", "ppa", "ppa4"]
FAMILIES = [("plain", "plain"), ("r", "reshare"), ("rs", "reshare + sim"), ("a", "a2b"), ("ak", "a2b + AKTE")]


def load(paths):
    rows = {}
    for p in paths:
        for row in csv.reader(open(p)):
            if len(row) < 12 or row[0] not in ("fp", "ag") or row[1] == "name":
                continue
            rows.setdefault((row[0], row[1]), []).append(row)
    return rows


def med(rs, i):
    v = [float(r[i]) for r in rs if r[i]]
    return statistics.median(v) if v else None


def table(rows, mode, suffix):
    out = ["| adder | family | BN | pre fp | pre ag | online fp | online ag | rounds | MB pre | acc fp | acc ag |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fuse in ("0", "1"):
        for a in ADDERS:
            for f, fname in FAMILIES:
                n = f"{mode}_{a}_{f}_f{fuse}_{suffix}"
                fp, ag = rows.get(("fp", n), []), rows.get(("ag", n), [])
                ref = fp or ag
                if not ref:
                    continue
                cell = lambda rs, i, d: "-" if med(rs, i) is None else f"{med(rs, i):.{d}f}"
                acc = lambda rs: f"{rs[0][6]}/{rs[0][7]}" if rs and rs[0][6] else "-"
                rounds = ref[0][-2].split(";")[-1] if ";" in ref[0][-2] else ""
                mb = ref[0][-4] if len(ref[0]) >= 16 else ref[0][-3]
                out.append(f"| {a} | {fname} | {'fused' if fuse == '1' else 'unfused'} | {cell(fp, 4, 2)} | "
                           f"{cell(ag, 4, 2)} | {cell(fp, 5, 3)} | {cell(ag, 5, 3)} | {rounds} | {mb} | {acc(fp)} | {acc(ag)} |")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    suffix = "t2"
    if args and not args[-1].endswith(".csv"):
        suffix = args.pop()
    rows = load(args)
    for mode, title in (("s", "Single batch (10 images, one process)"), ("m", "Multi-batch (192 images, 24 processes x 8 lanes)")):
        print(f"\n### {title}\n")
        print(table(rows, mode, suffix))
