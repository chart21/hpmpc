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
