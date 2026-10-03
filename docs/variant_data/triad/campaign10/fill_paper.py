#!/usr/bin/env python3
"""fill_paper.py: the campaign's rows and ranges into ../../../paper (tab:trunc, tab:cutin, the TS_Mix / TE time ranges).
Replaces the %...% markers, or the rows between the table's \\midrule and \\bottomrule when they are filled already."""
import re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ana10

HERE = Path(__file__).parent
PAPER = HERE / '../../../paper'
t, c = ana10.runs('res8.csv'), ana10.comm('comm8.csv')


def rows(mode):
    return subprocess.run([sys.executable, str(HERE / 'ana10.py'), mode], capture_output=True, text=True, check=True).stdout


def put_rows(text, label, body):
    i = text.index(r'\label{' + label + '}')
    a = text.rindex(r'\midrule', 0, i) + len(r'\midrule') + 1
    b = text.rindex(r'\bottomrule', 0, i)
    return text[:a] + body.rstrip('\n') + '\n' + text[b:]


def rng(vals, d=2):
    return f"{min(vals):.{d}f}--{max(vals):.{d}f}\\s"


A = ('rca', 'ppa', 'ppa4')
U = '123'
mix = [t[f'a2{u}_{a}_mix']['pre'] - t[f'a2{u}_{a}_c0']['pre'] for u in U for a in A]
te = [t[f'a2{u}_{a}_te1']['pre'] - t[f'a2{u}_{a}_ts1']['pre'] for u in U for a in A]
on_te1 = [t[f'a2{u}_{a}_te1']['onl'] for u in U for a in A]
on_ts1 = [t[f'a2{u}_{a}_ts1']['onl'] for u in U for a in A]
te0 = [t[f'a2{u}_ppa4_te0']['onl'] - t[f'a2{u}_ppa4_te1']['onl'] for u in U]
print('TS_Mix - TS{L} pre s', [f'{x:+.2f}' for x in mix])
print('TE1 - TS1 pre s', [f'{x:+.2f}' for x in te])
print('TE1 online', rng(on_te1), 'TS1 online', rng(on_ts1), 'TE0 - TE1 PPA4 online', [f'{x:+.3f}' for x in te0])

p = PAPER / 'triad.tex'
s = p.read_text()
trunc = rows('trunc')
s = s.replace('%TRUNCROWS%', '') if '%TRUNCROWS%' in s else s
s = put_rows(s, 'tab:trunc', trunc)
s = s.replace('%TSMIXTIME%', f"{min(mix):.2f}--{max(mix):.2f}\\s")
s = s.replace('%TETIME%', f"{min(te):.1f}--{max(te):.1f}\\s")
repl = f"TE1's online phase takes about as long as TS1's ({rng(on_te1)[:-2]}\nagainst {rng(on_ts1)})"
s = re.sub(r"TE1's online phase takes about as long as TS1's \([0-9.]+--[0-9.]+\n?against [0-9.]+--[0-9.]+\\s\)",
           lambda m: repl, s)
p.write_text(s)

p = PAPER / 'paper.tex'
s = p.read_text()
# keep the UC2 blocks only
out, keep = [], False
for l in rows('cut').splitlines():
    if l.startswith('UC'):
        keep = l.startswith('UC2 ')
    elif l.startswith('\\addlinespace'):
        if out and out[-1] != '\\addlinespace':
            out.append('\\addlinespace')
        keep = False
        continue
    if keep:
        out.append(l.replace('UC2 ', '', 1))
while out and out[-1] == '\\addlinespace':
    out.pop()
s = s.replace('%CUTROWS%', '') if '%CUTROWS%' in s else s
s = put_rows(s, 'tab:cutin', '\n'.join(out))
p.write_text(s)
print('filled tab:trunc, tab:cutin')
