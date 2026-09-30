#!/usr/bin/env python3
"""ttab.py DIR: tables of the triad all_opt configs (ImageNet, dummy weights) from res_{fp,ag}_{conf,opt,fin,fin64}.csv."""
import csv, os, statistics as st, sys
D = sys.argv[1]
# use cases: UC1 weights known to none (A_KNOWN=0), UC2 known to one (A_KNOWN=1, MWK), UC3 known to all (public)
CONF = [("a2bk0", "UC1, A2bits"), ("rsk0", "UC1, reshared"), ("a2b", "UC2, A2bits"), ("rs", "UC2, reshared"),
        ("a2bpw", "UC3, A2bits"), ("rspw", "UC3, reshared")]
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
# UC3: five interleaved runs of each build (res_*_ab.csv) replace the earlier conf / fin2 runs
for pair in ("fp", "ag"):
    ab = load(f"res_{pair}_ab.csv")
    for k, prefix in (("conf", "conf_"), ("fin2", "fin2_")):
        res[(pair, k)].update({n: [r for r in rows if r["name"].startswith(prefix)] for n, rows in ab.items()
                               if any(r["name"].startswith(prefix) for r in rows)})
def med(pair, k, n, col):
    rows = res[(pair, k)].get(n)
    return st.median(float(r[col]) for r in rows) if rows else None
def f(v, d=2): return "-" if v is None else f"{v:.{d}f}"
out = []
for c in ("0", "1"):
    out.append(f"\n### COMPRESS={c}\n")
    out.append("| use case, config | adder | pre conf fp | pre fin fp | pre fin64 fp | pre conf ag | pre fin ag | pre fin64 ag | online conf fp | online fin fp | online conf ag | online fin ag |")
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

# communication (MiB sent + received by P0) from comm_fp.csv; network counters are 10^6 B, triple counters MiB
MIB = 1e6 / 2**20
comm = {}
for r in csv.DictReader(open(os.path.join(D, "comm_fp.csv"))):
    v = {k: float(x) for k, x in r.items() if k != "tag"}
    comm[r["tag"].rsplit("_r", 1)[0]] = (v["trip_sent"] + v["trip_recv"] + v["keyex_sent"] + v["keyex_recv"],
                                         (v["pre_sent"] + v["pre_recv"]) * MIB, (v["on_sent"] + v["on_recv"]) * MIB)
out = []
for c in ("0", "1"):
    out += [f"\n### Communication, COMPRESS={c}\n",
            "| use case, config | adder | triples conf | triples fin | pre conf | pre fin | pre factor | online conf | online fin | online factor |",
            "|---|---|" + "---|" * 8]
    for key, name in CONF:
        for a in ADD:
            n = f"{key}_{a}_c{c}"
            (tc, pc, oc), (tf, pf, of) = comm[f"conf_{n}"], comm[f"fin2_{n}"]
            out.append(f"| {name} | {a.upper()} | {tc:.0f} | {tf:.0f} | {tc + pc:.0f} | {tf + pf:.0f} | {(tc + pc) / (tf + pf):.2f} | "
                       f"{oc:.1f} | {of:.1f} | {oc / of:.2f} |")
print("\n".join(out))
