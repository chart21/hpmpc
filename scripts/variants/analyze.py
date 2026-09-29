#!/usr/bin/env python3
"""analyze.py res_fp.csv res_ag.csv ...: tables of the variant matrix (mxall.sh output).

Per mode (s = single batch, m = multi-batch) and host pair: preprocessing / online time (median over the
repetitions), accuracy, rounds, traffic, and whether the threaded build's output hash equals the serial one.
"""
import csv
import statistics
import sys
from collections import defaultdict

ADDERS = ["rca", "ppa", "ppa4"]
FAMILIES = ["plain", "r", "rs", "a", "ak"]
FAMILY_NAME = {"plain": "plain", "r": "reshare", "rs": "reshare+sim", "a": "a2b", "ak": "a2b+AKTE"}


def load(paths):
    runs = defaultdict(list)  # (pair, name) -> rows
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("errors") is None and row.get("wait_s;rounds") is not None:
                    # rows of the first harness version lack mb_live: shift the last columns back
                    row["errors"], row["wait_s;rounds"], row["mb_live"] = row["wait_s;rounds"], row["mb_live"], ""
                if row.get("tag") in (None, "missing") or not row.get("pre"):
                    runs[(row["pair"], row["name"])].append(None)
                    continue
                runs[(row["pair"], row["name"])].append(row)
    return runs


def med(rows, key):
    vals = [float(r[key]) for r in rows if r and r.get(key) not in (None, "")]
    return statistics.median(vals) if vals else None


def cell(rows, full=None):
    ok = [r for r in rows if r]
    if full:  # a run that lost a process (fewer images classified) is a failed run, not a hash mismatch
        ok = [r for r in ok if int(r["acc_n"] or 0) == full]
    if not ok:
        return None
    d = {
        "pre": med(ok, "pre"),
        "online": med(ok, "online"),
        "acc": f"{ok[0]['acc_ok']}/{ok[0]['acc_n']}",
        "hash": {r["hash"] for r in ok},
        "errors": sum(int(r["errors"] or 0) for r in ok),
        "mb_pre": med(ok, "mb_pre"),
        "mb_live": med(ok, "mb_live"),
    }
    w = ok[0]["wait_s;rounds"].split(";")
    d["rounds"] = w[1] if len(w) > 1 else ""
    ph = ok[0]["pre_phases_CONV;BN;BOOL;COT;MUX;FC"].split(";")
    d["phases"] = ph
    lv = ok[0]["live_ms_ACT;CONV;BN"].split(";")
    d["live"] = lv
    return d


def fmt(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def table(runs, pair, mode):
    out = []
    out.append(f"\n### {pair}, {'single batch' if mode == 's' else 'multi-batch'}\n")
    out.append("| adder | family | BN | pre s | online s | acc | rounds | MB pre | MB live | "
               "pre: conv / BN / bool / COT / MUX s | live ms: ReLU / conv / BN | hash = serial |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fuse in ("0", "1"):
        for a in ADDERS:
            for fam in FAMILIES:
                tr = runs.get((pair, f"{mode}_{a}_{fam}_f{fuse}_t"), [])
                rr = runs.get((pair, f"{mode}_{a}_{fam}_f{fuse}_r"), [])
                full = max([int(x["acc_n"] or 0) for x in tr + rr if x] or [0])
                t, r = cell(tr, full), cell(rr, full)
                bn = "fused" if fuse == "1" else "unfused"
                if t is None:
                    out.append(f"| {a} | {FAMILY_NAME[fam]} | {bn} | missing / failed | | | | | | | | |")
                    continue
                same = "- (no complete serial run)" if r is None else (
                    "yes" if t["hash"] == r["hash"] and len(t["hash"]) == 1 else "NO")
                if len(t["hash"]) > 1:
                    same += " (runs differ)"
                if t["errors"]:
                    same += f" ({t['errors']} errors)"
                out.append(
                    f"| {a} | {FAMILY_NAME[fam]} | {bn} | {fmt(t['pre'])} | {fmt(t['online'], 3)} | {t['acc']} | "
                    f"{t['rounds']} | {fmt(t['mb_pre'], 0)} | {fmt(t['mb_live'], 0)} | "
                    f"{' / '.join(t['phases'][:5])} | {' / '.join(t['live'])} | {same} |")
    return "\n".join(out)


def serial_speedup(runs, pair, mode):
    rows = []
    for fuse in ("0", "1"):
        for a in ADDERS:
            for fam in FAMILIES:
                t = cell(runs.get((pair, f"{mode}_{a}_{fam}_f{fuse}_t"), []))
                r = cell(runs.get((pair, f"{mode}_{a}_{fam}_f{fuse}_r"), []))
                if t and r:
                    rows.append((f"{a}/{FAMILY_NAME[fam]}/{'fused' if fuse == '1' else 'unfused'}",
                                 r["pre"], t["pre"], r["online"], t["online"]))
    out = [f"\n### threads vs serial, {pair}, {'single batch' if mode == 's' else 'multi-batch'}\n",
           "| variant | pre serial | pre threads | online serial | online threads |", "|---|---|---|---|---|"]
    for v, a, b, c, d in rows:
        out.append(f"| {v} | {fmt(a)} | {fmt(b)} | {fmt(c, 3)} | {fmt(d, 3)} |")
    return "\n".join(out)


def machines(runs, mode):
    out = [f"\n### flare/polynize vs algofi/goracle, {'single batch' if mode == 's' else 'multi-batch'} (threaded builds)\n",
           "| variant | pre fp | pre ag | ag / fp | online fp | online ag | ag / fp |", "|---|---|---|---|---|---|---|"]
    for fuse in ("0", "1"):
        for a in ADDERS:
            for fam in FAMILIES:
                n = f"{mode}_{a}_{fam}_f{fuse}_t"
                f = cell(runs.get(("fp", n), []))
                g = cell(runs.get(("ag", n), []))
                if not f or not g:
                    continue
                out.append(f"| {a}/{FAMILY_NAME[fam]}/{'fused' if fuse == '1' else 'unfused'} | {fmt(f['pre'])} | "
                           f"{fmt(g['pre'])} | {fmt(g['pre'] / f['pre'])} | {fmt(f['online'], 3)} | "
                           f"{fmt(g['online'], 3)} | {fmt(g['online'] / f['online'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    # --suffix t2: the threaded builds are named *_t2 (round 2); the serial references stay *_r
    args = sys.argv[1:]
    SUFFIX = "t"
    if args[:1] == ["--suffix"]:
        SUFFIX, args = args[1], args[2:]
    runs = load(args)
    if SUFFIX != "t":
        runs = {(p, n[:-len(SUFFIX)] + "t" if n.endswith("_" + SUFFIX) else (n if not n.endswith("_t") else n + "_round1")): v
                for (p, n), v in runs.items()}
    pairs = sorted({p for p, _ in runs})
    for mode in ("s", "m"):
        for pair in pairs:
            print(table(runs, pair, mode))
        for pair in pairs:
            print(serial_speedup(runs, pair, mode))
        if len(pairs) > 1:
            print(machines(runs, mode))
