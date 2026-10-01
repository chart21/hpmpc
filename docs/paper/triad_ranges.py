#!/usr/bin/env python3
"""triad_ranges.py: the ranges quoted in triad.tex's Results paragraph (as given vs final, per pair and use case),
from docs/variant_data/triad/res_{fp,ag}_final4.csv (as given) and the newest final rerun (final8, else final7, final6,
final5), and the newest rerun against the one before it."""
import csv, statistics as st
from pathlib import Path
VD = Path(__file__).parent / "../variant_data/triad"
def load(f, prefix):
    d = {}
    for r in csv.DictReader(open(VD / f)):
        if r["name"].startswith(prefix + "_") and r.get("pre") and r.get("online") and float(r["online"]) > 0:
            d.setdefault(r["name"].split("_", 1)[1], []).append((float(r["pre"]), float(r["online"])))
    return {k: (st.median(x[0] for x in v), st.median(x[1] for x in v)) for k, v in d.items()}
for p, pn in (("fp", "Zen 4"), ("ag", "Zen 3")):
    conf = load(f"res_{p}_final4.csv", "conf")
    for f, pre in (("final9", "fin9"), ("final8", "fin8"), ("final7", "fin7"), ("final6", "fin6"), ("final5", "fin4")):
        if (VD / f"res_{p}_{f}.csv").exists():
            fin = load(f"res_{p}_{f}.csv", pre); break
    for uc, keys in (("UC1+UC2", ("a2bk0", "rsk0", "a2b", "rs")), ("UC2", ("a2b", "rs")), ("UC3", ("a2bpw", "rspw"))):
        names = [n for n in fin if n.split("_")[0] in keys]
        def rng(v): return f"{min(v):.2f}-{max(v):.2f}"
        print(f"{pn} {uc} ({f}, {len(names)} builds): pre {rng([conf[n][0] for n in names])} -> {rng([fin[n][0] for n in names])} "
              f"({min(conf[n][0] / fin[n][0] for n in names):.1f}-{max(conf[n][0] / fin[n][0] for n in names):.1f}x); "
              f"online {rng([conf[n][1] for n in names])} -> {rng([fin[n][1] for n in names])} "
              f"({min(conf[n][1] / fin[n][1] for n in names):.1f}-{max(conf[n][1] / fin[n][1] for n in names):.1f}x)")
    if f in ("final8", "final9"):  # the newest rerun against the one before it, per build
        prev = {"final8": ("final7", "fin7"), "final9": ("final8", "fin8")}[f]
        old = load(f"res_{p}_{prev[0]}.csv", prev[1])
        for i, ph in ((0, "pre"), (1, "online")):
            d = sorted(fin[n][i] - old[n][i] for n in fin if n in old)
            print(f"{pn} {f} - {prev[0]} {ph}: median {st.median(d):+.2f} s, {d[0]:+.2f}..{d[-1]:+.2f} s over {len(d)} builds;"
                  f" worst {max((fin[n][i] - old[n][i], n) for n in fin if n in old)[1]}")
