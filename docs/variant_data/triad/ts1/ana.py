import csv, statistics as st, collections, sys
res, comm = sys.argv[1], sys.argv[2]
t = collections.defaultdict(list)
for r in csv.reader(open(res)):
    if r and r[0] == 'fp' and r[4]:
        t[r[1]].append((float(r[4]), float(r[5]), r[13].split(';')[1] if ';' in r[13] else ''))
c = {}
for r in csv.DictReader(open(comm)):
    c[r['name']] = r
uc = {'a2bk0': 'UC1', 'a2b': 'UC2', 'a2bpw': 'UC3'}
print(f"{'UC':4s} {'adder':5s} {'variant':10s} {'pre s':>6s} {'online s':>8s} {'rounds':>6s} {'pre MiB':>8s} {'online MiB':>10s}")
for u in ['a2bk0', 'a2b', 'a2bpw']:
    for a in ['rca', 'ppa', 'ppa4']:
        for v, lab in [('tl', 'TS{L}'), ('tn', 'TS{L}+fix'), ('tf', 'TS_Mix')]:
            n = f'im_{u}_{a}_{v}'
            if n not in t: continue
            pre = st.median(x[0] for x in t[n]); onl = st.median(x[1] for x in t[n])
            cc = c.get(n, {})
            f = lambda k: float(cc[k]) / 1.048576 if k in cc else float('nan')
            print(f"{uc[u]:4s} {a:5s} {lab:10s} {pre:6.2f} {onl:8.3f} {t[n][0][2]:>6s} {f('pre_total'):8.0f} {f('online'):10.1f}")
