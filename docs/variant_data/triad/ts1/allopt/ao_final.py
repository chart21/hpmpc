import csv, statistics as st, collections, sys
# ao_final.py: per all_opt config (UC x family x adder x COMPRESS) every truncation variant that builds, the medians
# (pre s, online s), rounds and P0 traffic; '*' marks the least preprocessing / online traffic of the config
W = sys.argv[1]
def load(fns):
    t = collections.defaultdict(list)
    for fn in fns:
        for r in csv.reader(open(f'{W}/{fn}')):
            if r and r[0] == 'fp' and r[4] and float(r[4]) > 0:
                t[r[1]].append((float(r[4]), float(r[5]), r[13].split(';')[1]))
    return t
a = load(['res_ao.csv']); b = load(['res_b2.csv'])
c = {}
for fn in ['comm_ao.csv', 'comm_b2.csv']:
    for r in csv.DictReader(open(f'{W}/{fn}')):
        c.setdefault(r['name'], r)
mib = lambda n, k: float(c[n][k]) / 1.048576
rows = []
for fam, fl in [('a2b', 'A2bits'), ('rs', 'reshared')]:
    for u, ul in [('k0', 'UC1'), ('', 'UC2'), ('pw', 'UC3')]:
        for ad in ['rca', 'ppa', 'ppa4']:
            for cc in ['c0', 'c1']:
                cfg = f"{ul} {fl} {ad.upper()} {'COMPRESS=1' if cc == 'c1' else ''}".strip()
                base = f'ao_{fam}{u}_{ad}_{cc}'
                vs = []
                if fam == 'rs' and u == 'pw' and cc == 'c0':
                    vs = [('TS{L} as before (no delayed cut)', base, b), ('TS{L} + delayed cut', base + '_dc', b)]
                else:
                    vs = [('TS{L}', base, a)]
                if fam == 'a2b' and cc == 'c0':
                    vs += [('TS_Mix', f'ao_{fam}{u}_{ad}_mx', a), ('TS1', f'ao_{fam}{u}_{ad}_t1', a),
                           ('TS1, F=8', f'ao_{fam}{u}_{ad}_t1f8', b)]
                got = [(l, n, t[n]) for l, n, t in vs if n in t]
                bp = min(mib(n, 'pre_total') for _, n, _ in got); bo = min(mib(n, 'online') for _, n, _ in got)
                for i, (l, n, x) in enumerate(got):
                    pm, om = mib(n, 'pre_total'), mib(n, 'online')
                    rows.append(f"| {cfg if i == 0 else ''} | {l} | {st.median(v[0] for v in x):.2f} | {st.median(v[1] for v in x):.3f} | "
                                f"{x[-1][2]} | {pm:.0f}{'*' if pm == bp and len(got) > 1 else ''} | {om:.1f}{'*' if om == bo and len(got) > 1 else ''} |")
print('| config | truncation | pre s | online s | rounds | pre MiB | online MiB |')
print('|---|---|---|---|---|---|---|')
print('\n'.join(rows))
