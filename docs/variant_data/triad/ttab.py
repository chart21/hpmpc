#!/usr/bin/env python3
"""ttab.py DIR: tables of the triad all_opt configs (ImageNet, dummy weights) from res_{fp,ag}_{conf,opt,fin,fin64}.csv."""
import csv, os, statistics as st, sys
D = sys.argv[1]
CONF = [("a2b", "A2bits"), ("a2bk0", "A2bits, A not known"), ("a2bpw", "A2bits, public"),
        ("rs", "reshared"), ("rsk0", "reshared, A not known"), ("rspw", "reshared, public")]
ADD = ["rca", "ppa", "ppa4"]
def load(f):
    d = {}
    p = os.path.join(D, f)
    if not os.path.exists(p): return d
    for r in csv.DictReader(open(p)):
        if r.get("pre") and r.get("online") and float(r["online"]) > 0:
            d.setdefault(r["name"].split("_", 1)[1], []).append(r)
    return d
res = {(pair, k): load(f"res_{pair}_{k}.csv") for pair in ("fp", "ag") for k in ("conf", "opt", "fin", "fin64", "fin2")}
def med(pair, k, n, col):
    rows = res[(pair, k)].get(n)
    return st.median(float(r[col]) for r in rows) if rows else None
def f(v, d=2): return "-" if v is None else f"{v:.{d}f}"
out = []
for c in ("0", "1"):
    out.append(f"\n### COMPRESS={c}\n")
    out.append("| config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |")
    out.append("|---|---|" + "---|" * 10)
    for key, name in CONF:
        for a in ADD:
            n = f"{key}_{a}_c{c}"
            vals = [med("fp", "conf", n, "pre"), med("fp", "fin2", n, "pre"), med("fp", "fin64", n, "pre"),
                    med("ag", "conf", n, "pre"), med("ag", "fin2", n, "pre"), med("ag", "fin64", n, "pre"),
                    med("fp", "conf", n, "online"), med("fp", "fin2", n, "online"), med("ag", "conf", n, "online"), med("ag", "fin2", n, "online")]
            out.append(f"| {name} | {a.upper()} | " + " | ".join(f(v, 2 if i < 6 else 3) for i, v in enumerate(vals)) + " |")
print("\n".join(out))
# summary ranges
for pair in ("fp", "ag"):
    for c in ("0", "1"):
        sp, so, fo, fp_, f64 = [], [], [], [], []
        for key, _ in CONF:
            for a in ADD:
                n = f"{key}_{a}_c{c}"
                cp, fpre = med(pair, "conf", n, "pre"), med(pair, "fin2", n, "pre")
                co, fon = med(pair, "conf", n, "online"), med(pair, "fin2", n, "online")
                op = med(pair, "opt", n, "online")
                p64 = med(pair, "fin64", n, "pre")
                if cp and fpre: sp.append(cp / fpre)
                if co and fon: so.append(co / fon)
                if op and fon: fo.append(op / fon)
                if fpre: fp_.append(fpre)
                if p64 and fpre: f64.append(p64 / fpre)
        if sp:
            print(f"{pair} c{c}: pre conf/fin {min(sp):.1f}-{max(sp):.1f}x, online conf/fin {min(so):.2f}-{max(so):.2f}x, "
                  f"online opt/fin {min(fo):.2f}-{max(fo):.2f}x" if fo else "", f"fin pre {min(fp_):.2f}-{max(fp_):.2f} s",
                  f"fin64/fin pre {min(f64):.2f}-{max(f64):.2f}" if f64 else "")
