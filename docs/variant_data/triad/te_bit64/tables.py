#!/usr/bin/env python3
# tables.py: LaTeX rows for the paper from the round CSVs (medians; P0's traffic sent + received, MiB)
import csv, sys, statistics as st
MB = 1.048576
def load(res, comm):
    d = {}
    for r in csv.DictReader(open(res)):
        try:
            pre, onl = float(r['pre']), float(r['online'])
        except (ValueError, KeyError):
            continue
        if pre <= 0:
            continue
        wr = r.get('mb_live') or ''
        d.setdefault(r['name'], []).append((pre, onl, wr.split(';')[1] if ';' in wr else ''))
    c = {r['name']: r for r in csv.DictReader(open(comm))}
    out = {}
    for k, v in d.items():
        cc = c.get(k, {})
        out[k] = dict(pre=st.median(x[0] for x in v), onl=st.median(x[1] for x in v), rounds=v[0][2],
                      pmb=float(cc['pre_total']) / MB if cc else None, omb=float(cc['online']) / MB if cc else None)
    return out
def f(x, p=2):
    return '' if x is None else (f"{x:,.{p}f}".replace(',', '{,}'))
mode = sys.argv[1]
if mode == 'te':
    t = load(sys.argv[2], sys.argv[3])
    UC = [('a2bk0', 'UC1'), ('a2b', 'UC2'), ('a2bpw', 'UC3')]
    for u, ul in UC:
        for ad, al in (('rca', 'RCA'), ('ppa', 'PPA'), ('ppa4', 'PPA4')):
            first = True
            for v, vl in (('xl', 'TS\\{L\\}'), ('x1', 'TS1'), ('te1', 'TE1'), ('te0', 'TE0')):
                r = t.get(f'it_{u}_{ad}_{v}')
                if not r:
                    continue
                lab = f'{ul} {al}' if first else ''
                first = False
                print(f"{lab} & {vl} & {f(r['pmb'],0)} & {r['pre']:.2f} & {f(r['omb'],1)} & {r['onl']:.3f} & {r['rounds']}\\\\")
        if u != 'a2bpw':
            print('\\addlinespace')
elif mode == 'x64':
    t = load(sys.argv[2], sys.argv[3])
    for fam, fl in (('a2b', 'A2bits'), ('rs', 'reshared')):
        for u, ul in (('k0', 'UC1'), ('', 'UC2'), ('pw', 'UC3')):
            for ad, al in (('rca', 'RCA'), ('ppa', 'PPA'), ('ppa4', 'PPA4')):
                a, b = t.get(f'x32_{fam}{u}_{ad}'), t.get(f'x64_{fam}{u}_{ad}')
                if not a or not b:
                    continue
                print(f"{ul} {fl} {al} & {f(a['pmb'],0)} / {f(b['pmb'],0)} & {a['pre']:.2f} / {b['pre']:.2f} & "
                      f"{f(a['omb'],0)} / {f(b['omb'],0)} & {a['onl']:.2f} / {b['onl']:.2f} & {a['rounds']} / {b['rounds']}\\\\")
            if not (fam == 'rs' and u == 'pw'):
                pass
        print('\\addlinespace')
    for u, ul in (('k0', 'UC1'), ('', 'UC2')):
        b = t.get(f'x64r_a2b{u}_rca')
        if b:
            print(f"% repack {ul} A2bits RCA 64: pre MiB {f(b['pmb'],0)} pre s {b['pre']:.2f} he {b}")
