#!/usr/bin/env python3
"""make_phases.py: data/triad_phases.tex, the preprocessing / online phases of six COMPRESS=0 all_opt builds on flare /
polynize at the final code (hpmpc 1421df0; median of two phase-instrumented runs, docs/variant_data/triad/phases2_fp.log).
The conv triples run alongside the preprocessing pass; 'tail' is the part that is left after the pass."""
import re, statistics as st
from pathlib import Path
rows = {}
for l in open("../variant_data/triad/phases2_fp.log"):
    m = re.match(r"pq3_(\S+) r\d P0 (.*)", l)
    if m:
        parts = m.group(2).split("|")
        rows.setdefault(m.group(1), []).append({k: float(v) for k, v in re.findall(r"([A-Za-z_0-9]+)=([0-9.]+)", parts[0] + " " + parts[1])})
SEL = [("a2bk0_rca_c0", "UC1, A2bits, RCA"), ("rsk0_ppa4_c0", "UC1, reshared, PPA4"), ("a2b_rca_c0", "UC2, A2bits, RCA"),
       ("rs_ppa4_c0", "UC2, reshared, PPA4"), ("a2bpw_rca_c0", "UC3, A2bits, RCA"), ("rspw_rca_c0", "UC3, reshared, RCA")]
out = []
for n, lab in SEL:
    g = lambda k: st.median(r.get(k, 0.0) for r in rows[n])
    ps, pe, cj, ce = g("pass_start"), g("pass_end"), g("conv_joined"), g("complete_end")
    tuples = g("BOOL") - g("otsetup") + g("BOOL3") + g("BOOL4") + g("RANDOM_MUL")
    other = g("pre") - g("otsetup") - tuples - g("BOOL_COT_MULT") - (pe - ps) - (cj - pe) - (ce - cj)
    out.append(f"{lab} & {g('pre'):.2f} & {g('otsetup'):.2f} & {tuples:.2f} & {g('BOOL_COT_MULT'):.2f} & {pe - ps:.2f} & "
               f"{cj - pe:.2f} & {ce - cj:.2f} & {other:.2f} & {g('online'):.2f} & {g('rounds'):,.0f} & {g('wait'):.2f}\\\\")
Path("data/triad_phases.tex").write_text("\n".join(out) + "\n\\bottomrule\n")
print("\n".join(out))
# the pass over all 18 builds, for the text
passes = sorted(st.median(r["pass_end"] - r["pass_start"] for r in v) for v in rows.values())
print(f"pass over {len(passes)} builds: {passes[0]:.2f}-{passes[-1]:.2f} s")
