import csv, statistics as st, collections, sys
# ana2.py COMM RES...: medians per build (pre s, online s, rounds) + P0 traffic (MiB) for the TS1 ImageNet matrix
comm, res = sys.argv[1], sys.argv[2:]
t = collections.defaultdict(list)
for fn in res:
    for r in csv.reader(open(fn)):
        if r and r[0] == 'fp' and r[4] and float(r[4]) > 0:
            t[r[1]].append((float(r[4]), float(r[5]), r[13].split(';')[1] if ';' in r[13] else ''))
c = {}
for r in csv.DictReader(open(comm)):
    c[r['name']] = r
uc = {'a2bk0': 'UC1', 'a2b': 'UC2', 'a2bpw': 'UC3'}
V = [('xl0', 'TS{L} (no dcut)'), ('xl', 'TS{L}'), ('xm', 'TS_Mix'), ('xw', 'TS_Mix+w_t'), ('x1', 'TS1')]
print(f"{'UC':4s} {'adder':5s} {'variant':16s} {'n':>2s} {'pre s':>6s} {'online s':>8s} {'rounds':>6s} {'pre MiB':>8s} {'online MiB':>10s}")
for u in ['a2bk0', 'a2b', 'a2bpw']:
    for a in ['rca', 'ppa', 'ppa4']:
        for v, lab in V:
            n = f'im_{u}_{a}_{v}'
            if n not in t: continue
            pre = st.median(x[0] for x in t[n]); onl = st.median(x[1] for x in t[n])
            cc = c.get(n, {})
            f = lambda k: float(cc[k]) / 1.048576 if k in cc else float('nan')
            print(f"{uc[u]:4s} {a:5s} {lab:16s} {len(t[n]):2d} {pre:6.2f} {onl:8.3f} {t[n][-1][2]:>6s} {f('pre_total'):8.0f} {f('online'):10.1f}")
