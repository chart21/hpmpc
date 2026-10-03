#!/usr/bin/env python3
# ana.py RES.csv COMM.csv: medians per build (pre s, online s, rounds) and P0's traffic (MiB, sent + received)
import csv, sys, statistics as st
res, comm = sys.argv[1], sys.argv[2]
d = {}
for r in csv.DictReader(open(res)):
    try:
        pre, onl = float(r['pre']), float(r['online'])
    except (ValueError, KeyError):
        continue
    if pre <= 0:
        continue
    wr = r.get('mb_live') or ''  # vsum.sh prints no mb_live: waited;rounds sits there
    rounds = wr.split(';')[1] if ';' in wr else ''
    d.setdefault(r['name'], []).append((pre, onl, rounds))
c = {}
for r in csv.DictReader(open(comm)):
    c[r['name']] = r
MB = 1.048576
print(f"{'build':28s} {'n':>2s} {'pre s':>7s} {'onl s':>7s} {'rounds':>6s} {'he':>7s} {'ot':>7s} {'pre MiB':>8s} {'onl MiB':>8s}")
for k in sorted(d):
    v = d[k]
    cc = c.get(k, {})
    f = lambda x: f"{float(cc[x]) / MB:8.1f}" if x in cc else f"{'':>8s}"
    print(f"{k:28s} {len(v):2d} {st.median(x[0] for x in v):7.2f} {st.median(x[1] for x in v):7.3f} {v[0][2]:>6s} "
          f"{f('he')}{f('ot')}{f('pre_total')}{f('online')}")
