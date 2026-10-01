#!/usr/bin/env python3
"""make_triad.py: data/triad_c{0,1}.dat from docs/variant_data/triad/res_{fp,ag}_{conf,fin2,ab}.csv (medians)."""
import csv, statistics as st
from pathlib import Path
VD, OUT = Path("../variant_data/triad"), Path("data")
# use cases: UC1 weights known to none (A_KNOWN=0), UC2 known to one (the model owner: A_KNOWN=1, weights known in
# preprocessing), UC3 known to all (public weights); rows sorted by use case, then protocol, then adder
CONF = [("a2bk0", "UC1, A2bits"), ("rsk0", "UC1, reshared"), ("a2b", "UC2, A2bits"), ("rs", "UC2, reshared"),
        ("a2bpw", "UC3, A2bits"), ("rspw", "UC3, reshared")]
ADD = [("rca", "RCA"), ("ppa", "PPA"), ("ppa4", "PPA4")]
def load(f):
    d = {}
    p = VD / f
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("pre") and r.get("online") and float(r["online"]) > 0:
                d.setdefault(r["name"].split("_", 1)[1], []).append(r)
    return d
def load_prefix(f, prefix):
    return {n: [r for r in rows if r["name"].startswith(prefix + "_")] for n, rows in load(f).items()
            if any(r["name"].startswith(prefix + "_") for r in rows)}
res = {(p, k): load(f"res_{p}_{'fin2' if k == 'fin' else k}.csv") for p in ("fp", "ag") for k in ("conf", "fin")}
# the final rerun (res_*_final4.csv: as given and the final code, three interleaved runs of every build) where present;
# else the earlier runs, UC3 from five interleaved runs (res_*_ab.csv)
for p in ("fp", "ag"):
    if (VD / f"res_{p}_final4.csv").exists():
        res[(p, "conf")], res[(p, "fin")] = load_prefix(f"res_{p}_final4.csv", "conf"), load_prefix(f"res_{p}_final4.csv", "fin3")
        if (VD / f"res_{p}_final5.csv").exists():  # the optimized builds without RNG_AHEAD (final flag set)
            res[(p, "fin")] = load_prefix(f"res_{p}_final5.csv", "fin4")
        if (VD / f"res_{p}_final6.csv").exists():  # the same flags at the round-5 code (hpmpc f353bb2)
            res[(p, "fin")] = load_prefix(f"res_{p}_final6.csv", "fin6")
        if (VD / f"res_{p}_final7.csv").exists():  # round-5 code, conv threads following the OT phase (hpmpc 7913adb)
            res[(p, "fin")] = load_prefix(f"res_{p}_final7.csv", "fin7")
        if (VD / f"res_{p}_final8.csv").exists():  # round-6 code (hpmpc 841335d)
            res[(p, "fin")] = load_prefix(f"res_{p}_final8.csv", "fin8")
        if (VD / f"res_{p}_final9.csv").exists():  # round-7 code (hpmpc 92a2c0e)
            res[(p, "fin")] = load_prefix(f"res_{p}_final9.csv", "fin9")
    elif (VD / f"res_{p}_ab.csv").exists():
        res[(p, "conf")].update(load_prefix(f"res_{p}_ab.csv", "conf"))
        res[(p, "fin")].update(load_prefix(f"res_{p}_ab.csv", "fin2"))
def med(p, k, n, c):
    rows = res[(p, k)].get(n)
    return f"{st.median(float(r[c]) for r in rows):.3f}" if rows else "nan"
for c in ("0", "1"):
    with open(OUT / f"triad_c{c}.dat", "w") as f:
        f.write("y label conffp finfp confag finag onconffp onfinfp onconfag onfinag\n")
        y = 0
        for key, name in CONF:
            for a, an in ADD:
                n = f"{key}_{a}_c{c}"
                f.write(f"{y} {{{name}, {an}}} " + " ".join(med(p, k, n, col) for col in ("pre", "online")
                        for p, k in (("fp", "conf"), ("fp", "fin"), ("ag", "conf"), ("ag", "fin"))) + "\n")
                y += 1
print("wrote", [f"triad_c{c}.dat" for c in "01"])

# Communication: data/triad_comm_c{0,1}.dat from comm_fp.csv (P0's counters on flare; every value is identical in
# both reruns, on both pairs, and P1's sent/received mirror P0's). MB sent plus received by P0, i.e. both
# directions: preprocessing = triple generation (HE + OT, incl. key exchange) + the network's preprocessing pass.
# Units: the triple counters (trip_*, keyex_*) are MiB (ConvTriple's Utils::to_MB), the network's (pre_*, on_*) are
# 10^6 bytes (core/utils/print.hpp); everything is written in MiB, as in the rest of the paper.
MIB = 1e6 / 2**20
comm = {}
comm_rows = list(csv.DictReader(open(VD / "comm_fp.csv")))
if (VD / "comm_fp6.csv").exists():  # the final builds at the round-5 code
    comm_rows += list(csv.DictReader(open(VD / "comm_fp6.csv")))
if (VD / "comm_fp7.csv").exists():  # UC1 / UC2 A2bits at the round-6 code (UC2: residual sums without rebase_p1)
    comm_rows += list(csv.DictReader(open(VD / "comm_fp7.csv")))
if (VD / "comm_fp8.csv").exists():  # all final builds at the round-6 code
    comm_rows += list(csv.DictReader(open(VD / "comm_fp8.csv")))
if (VD / "comm_fp9.csv").exists():  # all final builds at the round-7 code
    comm_rows += list(csv.DictReader(open(VD / "comm_fp9.csv")))
for r in comm_rows:
    v = {k: float(x) for k, x in r.items() if k != "tag"}
    comm[r["tag"].rsplit("_r", 1)[0]] = {
        "trip": v["trip_sent"] + v["trip_recv"] + v["keyex_sent"] + v["keyex_recv"],
        "pass": (v["pre_sent"] + v["pre_recv"]) * MIB, "online": (v["on_sent"] + v["on_recv"]) * MIB}
for c in ("0", "1"):
    with open(OUT / f"triad_comm_c{c}.dat", "w") as f:
        f.write("y label preconf prefin tripconf tripfin onconf onfin\n")
        y = 0
        for key, name in CONF:
            for a, an in ADD:
                n = f"{key}_{a}_c{c}"
                g = comm[f"conf_{n}"]
                o = comm.get(f"fin9_{n}", comm.get(f"fin8_{n}", comm.get(f"r6_{n}", comm.get(f"fin6_{n}", comm[f"fin2_{n}"]))))
                f.write(f"{y} {{{name}, {an}}} {g['trip'] + g['pass']:.1f} {o['trip'] + o['pass']:.1f} "
                        f"{g['trip']:.1f} {o['trip']:.1f} {g['online']:.1f} {o['online']:.1f}\n")
                y += 1
print("wrote", [f"triad_comm_c{c}.dat" for c in "01"])

# rows of tab:triad (triad.tex): ranges per use case, COMPRESS and pair, from the figure data
names = {"UC1": "UC1, known to none", "UC2": "UC2, known to one", "UC3": "UC3, known to all"}
tab = []
for uc in ("UC1", "UC2", "UC3"):
    for c in ("0", "1"):
        rows = [l.split("}")[1].split() for l in list(open(OUT / f"triad_c{c}.dat"))[1:] if l.split("{")[1].startswith(uc)]
        cols = "conffp finfp confag finag onconffp onfinfp onconfag onfinag".split()
        v = {k: [float(r[i]) for r in rows] for i, k in enumerate(cols)}
        rg = lambda k, d: f"{min(v[k]):.{d}f}--{max(v[k]):.{d}f}"
        for i, (p, nm) in enumerate((("fp", "Zen 4"), ("ag", "Zen 3"))):
            lead = f"{names[uc]}, \\flag{{COMPRESS={c}}} ({len(rows)})" if i == 0 else ""
            tab.append(f"{lead} & {nm} & {rg('conf' + p, 1)} & {rg('fin' + p, 1)} & {rg('onconf' + p, 2)} & {rg('onfin' + p, 2)}\\\\")
# \bottomrule is part of the file: after \input, the table cannot take a \noalign
open(OUT / "triad_tab.tex", "w").write("\n".join(tab) + "\n\\bottomrule\n")
print("wrote triad_tab.tex")
