#!/usr/bin/env python3
"""ana10.py: the all_opt campaign at hpmpc 4b43984 (run8.sh, res8.csv / comm8.csv, flare / polynize, medians of 3).

  python3 ana10.py trunc|cut|final|x64|wan

trunc: tab:trunc rows (TS{L}, TS_Mix, TS1, TE1, TE0; A2bits, UC1-3 x RCA / PPA / PPA4).
cut:   the cut variants (no cut, identity cut, narrow cut; reshared: no cut / cut).
final: writes ../res_fp_final10.csv and ../comm_fp10.csv in make_triad.py's formats (the 36 all_opt builds).
x64:   tab:bit64 rows (res9.csv: 64 bits, F = 12, against the 32-bit COMPRESS=0 builds of res8.csv).
wan:   the WAN builds with and without output repacking (res10.csv).

Units: ConvTriple's counters (HE, OT, key exchange) are MiB, hpmpc's (preprocessing pass, online) 10^6 bytes; all
values here are MiB."""
import csv, statistics as st, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
MIB = 1e6 / 2**20


def runs(f):
    d = collections.defaultdict(list)
    for r in csv.reader(open(HERE / f)):
        if not r or r[0] != 'fp':
            continue
        try:
            d[r[1]].append(dict(pre=float(r[4]), onl=float(r[5]), rounds=int(r[13].split(';')[1]), row=r))
        except (ValueError, IndexError):
            pass
    return {k: dict(pre=st.median(x['pre'] for x in v), onl=st.median(x['onl'] for x in v),
                    rounds=int(st.median(x['rounds'] for x in v)), n=len(v), rows=[x['row'] for x in v]) for k, v in d.items()}


def comm(f):
    out = {}
    for r in csv.DictReader(open(HERE / f)):
        he, ot, keys, hp, on = (float(r[k]) for k in ('he', 'ot', 'keys', 'hpmpc_pre', 'online'))
        out[r['name']] = dict(he=he, ot=ot, keys=keys, hp=hp * MIB, pmb=he + ot + keys + hp * MIB, omb=on * MIB)
    return out


def n(x, p):
    return f"{x:,.{p}f}".replace(',', '{,}')


UC = [('1', 'UC1'), ('2', 'UC2'), ('3', 'UC3')]
AD = [('rca', 'RCA'), ('ppa', 'PPA'), ('ppa4', 'PPA4')]


def main():
    mode = sys.argv[1]
    if mode in ('trunc', 'cut', 'final'):
        t, c = runs('res8.csv'), comm('comm8.csv')
    if mode == 'trunc':
        for u, ul in UC:
            for a, al in AD:
                first = True
                for v, vl in (('c0', 'TS\\{L\\}'), ('mix', 'TS\\_Mix'), ('ts1', 'TS1'), ('te1', 'TE1'), ('te0', 'TE0')):
                    k = f'a2{u}_{a}_{v}'
                    r, cc = t[k], c[k]
                    lab = f'{ul} {al}' if first else ''
                    first = False
                    print(f"{lab} & {vl} & {n(cc['pmb'], 0)} & {r['pre']:.2f} & {n(cc['omb'], 1)} & {r['onl']:.3f} & {r['rounds']}\\\\")
            if u != '3':
                print('\\addlinespace')
    elif mode == 'cut':
        for fam, fl, vs in (('a2', 'A2bits', (('noc', 'none'), ('idc', 'identity'), ('c0', 'narrow'))),
                            ('rs', 'reshared', (('noc', 'none'), ('c0', 'identity')))):
            for u, ul in UC:
                for a, al in AD:
                    first = True
                    for v, vl in vs:
                        k = f'{fam}{u}_{a}_{v}'
                        r, cc = t[k], c[k]
                        lab = f'{ul} {fl} {al}' if first else ''
                        first = False
                        print(f"{lab} & {vl} & {n(cc['pmb'], 0)} & {n(cc['ot'], 0)} & {r['pre']:.2f} & {n(cc['omb'], 1)} & {r['onl']:.3f} & {r['rounds']}\\\\")
                if not (fam == 'rs' and u == '3'):
                    pass
            print('\\addlinespace')
    elif mode == 'final':
        key = {('a2', '1'): 'a2bk0', ('rs', '1'): 'rsk0', ('a2', '2'): 'a2b', ('rs', '2'): 'rs', ('a2', '3'): 'a2bpw',
               ('rs', '3'): 'rspw'}
        hdr = "pair,name,tag,wall,pre,online,acc_ok,acc_n,hash,procs,pre_phases_CONV;BN;BOOL;COT;MUX;FC,live_ms_ACT;CONV;BN,mb_pre,mb_live,wait_s;rounds,errors\n"
        with open(HERE / '../res_fp_final10.csv', 'w') as f, open(HERE / '../comm_fp10.csv', 'w') as g:
            f.write(hdr)
            g.write("tag,trip_sent,trip_recv,pre_sent,pre_recv,on_sent,on_recv,rounds,keyex_sent,keyex_recv\n")
            for (fam, u), kk in key.items():
                for a, _ in AD:
                    for cz in ('0', '1'):
                        src = f'{fam}{u}_{a}_c{cz}'
                        dst = f'fin10_{kk}_{a}_c{cz}'
                        for i, row in enumerate(t[src]['rows']):
                            row = list(row)
                            row[1], row[2] = dst, f'{dst}_f10r{i + 1}'
                            f.write(','.join(row) + '\n')
                        cc = c[src]
                        # sums only (sent + received); make_triad.py adds the pairs
                        g.write(f"{dst}_r1,{cc['he'] + cc['ot']:.3f},0,{cc['hp'] / MIB:.3f},0,{cc['omb'] / MIB:.3f},0,"
                                f"{t[src]['rounds']},{cc['keys']:.4f},0\n")
        print('wrote ../res_fp_final10.csv ../comm_fp10.csv')
    elif mode == 'x64':
        t8, c8 = runs('res8.csv'), comm('comm8.csv')
        t9, c9 = runs('res9.csv'), comm('comm9.csv')
        for fam, fl in (('a2', 'A2bits'), ('rs', 'reshared')):
            for u, ul in UC:
                for a, al in AD:
                    x, y = f'{fam}{u}_{a}_c0', f'x{fam}{u}_{a}'
                    A, B, CA, CB = t8[x], t9[y], c8[x], c9[y]
                    print(f"{ul} {fl} {al} & {n(CA['pmb'], 0)} / {n(CB['pmb'], 0)} & {A['pre']:.2f} / {B['pre']:.2f} & "
                          f"{n(CA['omb'], 0)} / {n(CB['omb'], 0)} & {A['onl']:.2f} / {B['onl']:.2f} & {A['rounds']} / {B['rounds']}\\\\")
            print('\\addlinespace')
    elif mode == 'wan':
        t, c = runs('res10.csv'), comm('comm10.csv')
        for fam, fl in (('a2', 'A2bits'), ('rs', 'reshared')):
            for u, ul in UC[:2]:
                for a, al in AD:
                    p, q = f'w{fam}{u}_{a}_n', f'w{fam}{u}_{a}_rp'
                    print(f"{ul} {fl} {al} & {n(c[p]['pmb'], 0)} / {n(c[q]['pmb'], 0)} & {t[p]['pre']:.2f} / {t[q]['pre']:.2f} & "
                          f"{t[p]['onl']:.2f} / {t[q]['onl']:.2f}\\\\")


if __name__ == '__main__':
    main()
