#!/usr/bin/env python3
"""make_triad.py: data/triad_c{0,1}.dat from docs/variant_data/triad/res_{fp,ag}_{conf,fin}.csv (medians)."""
import csv, statistics as st
from pathlib import Path
VD, OUT = Path("../variant_data/triad"), Path("data")
CONF = [("a2b", "A2bits"), ("a2bk0", "A2bits, $A$ shared"), ("a2bpw", "A2bits, public"),
        ("rs", "reshared"), ("rsk0", "reshared, $A$ shared"), ("rspw", "reshared, public")]
ADD = [("rca", "RCA"), ("ppa", "PPA"), ("ppa4", "PPA4")]
def load(f):
    d = {}
    p = VD / f
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("pre") and r.get("online") and float(r["online"]) > 0:
                d.setdefault(r["name"].split("_", 1)[1], []).append(r)
    return d
res = {(p, k): load(f"res_{p}_{'fin2' if k == 'fin' else k}.csv") for p in ("fp", "ag") for k in ("conf", "fin")}
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
for r in csv.DictReader(open(VD / "comm_fp.csv")):
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
                g, o = comm[f"conf_{n}"], comm[f"fin2_{n}"]
                f.write(f"{y} {{{name}, {an}}} {g['trip'] + g['pass']:.1f} {o['trip'] + o['pass']:.1f} "
                        f"{g['trip']:.1f} {o['trip']:.1f} {g['online']:.1f} {o['online']:.1f}\n")
                y += 1
print("wrote", [f"triad_comm_c{c}.dat" for c in "01"])
