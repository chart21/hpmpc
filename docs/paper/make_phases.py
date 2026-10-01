#!/usr/bin/env python3
"""make_phases.py: data/triad_phases.tex, the preprocessing / online phases of six COMPRESS=0 all_opt builds on flare /
polynize at the final code (round 7: hpmpc 92a2c0e; median of two phase-instrumented runs,
docs/variant_data/triad/phases9_fp.log; round 6: phases8_fp.log, round 3: phases2_fp.log).

Timeline (P0, relative to the preprocessing timer): with secret weights the pass first runs the network over the masks,
then starts the conv triples and runs the OT phase (ot_start .. ot_end) inside the pass, then the real sweep; with
public weights the OT phase runs before the pass, and the A2bits builds' Boolean addition inside it (after the mask-only
forward). 'pass' is the pass's own time (without the OT phase or the Boolean addition inside it), 'tail' the part of
the conv triples left after the pass, 'after' the rest of the preprocessing."""
import re, statistics as st
from pathlib import Path
rows = {}
for l in open("../variant_data/triad/phases9_fp.log"):
    m = re.match(r"pq9_(\S+) r\d P0 (.*)", l)
    if m:
        parts = m.group(2).split("|")
        rows.setdefault(m.group(1), []).append({k: float(v) for k, v in re.findall(r"([A-Za-z_0-9]+)=([0-9.]+)", parts[0] + " " + parts[1])})
SEL = [("a2bk0_rca_c0", "UC1, A2bits, RCA"), ("rsk0_ppa4_c0", "UC1, reshared, PPA4"), ("a2b_rca_c0", "UC2, A2bits, RCA"),
       ("rs_ppa4_c0", "UC2, reshared, PPA4"), ("a2bpw_rca_c0", "UC3, A2bits, RCA"), ("rspw_rca_c0", "UC3, reshared, RCA")]

def phases(n, r):
    g = lambda k: r.get(k, 0.0)
    ps, pe = g("pass_start"), g("pass_end")
    cj = g("conv_joined") if "conv_joined" in r else pe
    ce = g("complete_end")
    tuples = g("BOOL") - g("otsetup") + g("BOOL3") + g("BOOL4") + g("RANDOM_MUL")
    bake = g("BOOL_COT_MULT")
    inner = 0.0  # OT work inside the pass
    if g("ot_start") >= ps:
        inner += g("ot_end") - g("ot_start")
    elif n.startswith("a2bpw"):
        inner += bake  # UC3 A2bits: the Boolean addition waits for the mask-only forward
    pas = pe - ps - inner
    tail, after = cj - pe, ce - cj
    other = g("pre") - g("otsetup") - tuples - bake - pas - tail - after
    return dict(pre=g("pre"), otsetup=g("otsetup"), tuples=tuples, bake=bake, pas=pas, tail=tail, after=after,
                other=other, online=g("online"), rounds=g("rounds"), wait=g("wait"), ot=g("ot_end") - g("ot_start"))

med = {n: {k: st.median(phases(n, r)[k] for r in v) for k in phases(n, v[0])} for n, v in rows.items()}
out = []
for n, lab in SEL:
    m = med[n]
    out.append(f"{lab} & {m['pre']:.2f} & {m['otsetup']:.2f} & {m['tuples']:.2f} & {m['bake']:.2f} & {m['pas']:.2f} & "
               f"{m['tail']:.2f} & {m['after']:.2f} & {m['other']:.2f} & {m['online']:.2f} & {m['rounds']:,.0f} & "
               f"{m['wait']:.2f}\\\\")
Path("data/triad_phases.tex").write_text("\n".join(out) + "\n\\bottomrule\n")
print("\n".join(out))
# ranges over all 18 builds, for the text
for k in ("otsetup", "pas", "tail", "bake", "wait", "ot"):
    v = sorted(m[k] for m in med.values())
    print(f"{k} over {len(v)} builds: {v[0]:.2f}-{v[-1]:.2f} s")
for grp, f in (("RCA", "_rca_"), ("PPA4", "_ppa4_")):
    v = sorted(m["tuples"] for n, m in med.items() if f in n)
    print(f"tuples {grp}: {v[0]:.2f}-{v[-1]:.2f} s")
v = sorted(m["rounds"] for n, m in med.items() if "_rca_" in n)
print(f"RCA rounds: {v[0]:,.0f}-{v[-1]:,.0f}")
