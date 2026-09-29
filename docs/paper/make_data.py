#!/usr/bin/env python3
"""make_data.py: the pgfplots data files of the paper (data/*.dat) from the measurement CSVs.

Sources: docs/variant_data/*.csv (variant study, this repository) and data/e2e_compact.csv (ImageNet preprocessing
steps s0-s10, copied from the ConvTriple optimization report). Run from docs/paper/.
"""
import csv
import statistics
from pathlib import Path

VD = Path("../variant_data")
OUT = Path("data")
ADDERS = ["rca", "ppa", "ppa4"]
FAMILIES = [("plain", "plain"), ("r", "reshare"), ("rs", "reshare+sim"), ("a", "a2b"), ("ak", "a2b+AKTE")]
PRETTY = {"rca": "RCA", "ppa": "PPA", "ppa4": "PPA4"}


def vlabel(a, fname):
    return "{" + PRETTY[a] + " " + fname + "}"


def load(path):
    rows = {}
    for r in csv.reader(open(path)):
        if len(r) > 9 and r[0] in ("fp", "ag") and r[4] and r[4] != "pre":
            rows.setdefault(r[1], []).append(r)
    return rows


def med(rows, i):
    return statistics.median(float(r[i]) for r in rows)


def write(name, header, lines):
    with open(OUT / name, "w") as f:
        f.write(" ".join(header) + "\n")
        for l in lines:
            f.write(" ".join(str(x) for x in l) + "\n")


# ImageNet preprocessing steps (algofi / goracle, 32 threads, one image): stacked components
steps = [("s0", "s0"), ("s1", "s1"), ("s2", "s2v"), ("s3", "s3v"), ("s4", "s4v"), ("s5", "s5v"), ("s6", "s6v"),
         ("s7", "s7"), ("s8", "pl1"), ("s9", "otref13"), ("s10", "fin")]
e2e = {r["key"]: r for r in csv.DictReader(open(OUT / "e2e_compact.csv"))}
lines = []
for i, (label, key) in enumerate(steps):
    r = e2e[key]
    f = lambda k: float(r[k] or 0)
    conv = f("conv s") + f("conv filter NTT s")
    key_setup = f("key setup s")
    boolean = f("bool s")
    muxcot = f("mux s") + f("cot s")
    fc = f("fc s")
    total = f("preprocessing s")
    other = total - conv - key_setup - boolean - muxcot - fc
    traffic = f("triple MiB sent P0") + f("triple MiB recv P0")
    lines.append((i, label, f"{conv:.2f}", f"{key_setup:.2f}", f"{boolean:.2f}", f"{muxcot + fc:.2f}",
                  f"{other:.2f}", f"{total:.2f}", f"{traffic:.0f}"))
write("imagenet_steps.dat", ["x", "step", "conv", "keysetup", "bool", "muxcotfc", "other", "total", "trafficMiB"], lines)

# Variant study, round 4 (+ multi-batch a2b from the 64404c3 builds): per variant pre / online on both pairs
for mode in ("s", "m"):
    fp, ag = load(VD / "res_fp_r4a.csv"), load(VD / "res_ag_r4a.csv")
    lines = []
    x = 0
    for fuse in ("1", "0"):
        for a in ADDERS:
            for f, fname in FAMILIES:
                n = f"{mode}_{a}_{f}_f{fuse}_t2"
                if n not in fp:
                    continue
                lines.append((f"{x:.1f}", vlabel(a, fname), f"{med(fp[n], 4):.3f}", f"{med(ag[n], 4):.3f}",
                              f"{med(fp[n], 5):.3f}", f"{med(ag[n], 5):.3f}"))
                x += 1
            x += 0.6  # gap between adders
        x += 0.8  # gap between fused and unfused
    write(f"matrix_{mode}.dat", ["x", "variant", "prefp", "preag", "onfp", "onag"], lines)

# Machine ratio (algofi/goracle over flare/polynize), per variant
lines = []
for mode in ("s", "m"):
    for r in (l.rsplit(maxsplit=4) for l in open(OUT / f"matrix_{mode}.dat").read().splitlines()[1:]):
        lines.append((mode, r[0].split(maxsplit=1)[1].replace(" ", "_").strip("{}"), f"{float(r[2]) / float(r[1]):.3f}",
                      f"{float(r[4]) / float(r[3]):.3f}"))
write("machine_ratio.dat", ["mode", "variant", "pre", "online"], lines)

# Round 2 -> round 4 (flare / polynize), plain family: the effect of the shared-OT tuples and the batched BN
r2, r4 = load(VD / "res_fp.csv"), load(VD / "res_fp_r4.csv")
lines = []
for mode in ("s", "m"):
    for fuse in ("1", "0"):
        for a in ADDERS:
            n = f"{mode}_{a}_plain_f{fuse}_t2"
            lines.append((f"{mode}-{a}-f{fuse}", f"{med(r2[n], 4):.3f}", f"{med(r4[n], 4):.3f}"))
write("r2_r4.dat", ["variant", "r2", "r4"], lines)

# Accuracy with the AdamW model: 100-image single batch and 192-image multi-batch, per variant
single = {r["name"]: r for r in csv.DictReader(open(VD / "res_fp_wd.csv")) if r.get("pre")}
multi = {r["name"]: r for r in csv.DictReader(open(VD / "res_ag_wda.csv")) if r.get("pre")}
lines = []
x = 0
for fuse in ("1", "0"):
    for a in ADDERS:
        for f, fname in FAMILIES:
            s = single.get(f"a_{a}_{f}_f{fuse}_t2")
            m = multi.get(f"m_{a}_{f}_f{fuse}_t2")
            lines.append((f"{x:.1f}", vlabel(a, fname), f"{100 * int(s['acc_ok']) / int(s['acc_n']):.1f}",
                          f"{100 * int(m['acc_ok']) / int(m['acc_n']):.1f}"))
            x += 1
        x += 0.6
    x += 0.8
write("accuracy.dat", ["x", "variant", "single", "multi"], lines)

# CUT_FRACTIONAL_BITS_OPT: CUT=1 (round 4) vs CUT=0, flare / polynize, plain family
c0 = load(VD / "res_fp_c0.csv")
lines = []
for mode in ("s", "m"):
    for fuse in ("0", "1"):
        for a in ADDERS:
            n1, n0 = f"{mode}_{a}_plain_f{fuse}_t2", f"{mode}_{a}_plain_f{fuse}_c0"
            if n0 in c0:
                lines.append((f"{mode}-{a}-f{fuse}", f"{med(r4[n1], 4):.3f}", f"{med(c0[n0], 4):.3f}",
                              f"{med(r4[n1], 5):.3f}", f"{med(c0[n0], 5):.3f}"))
write("cut.dat", ["variant", "pre1", "pre0", "on1", "on0"], lines)
print("wrote", sorted(p.name for p in OUT.glob("*.dat")))

# Settings: A_KNOWN=1 (round 4) vs A_KNOWN=0 vs public weights (TRUNC_DELAYED=1, BIT_INJECTION_TRUNC_SIM=1),
# all families, flare / polynize; public-weight accuracy with the AdamW model on algofi / goracle
kpw = load(VD / "res_fp_kpw.csv")
pwacc = {r["name"]: r for r in csv.DictReader(open(VD / "res_ag_pwwd.csv")) if r.get("pre")}
for mode in ("s", "m"):
    lines = []
    y = 0
    for fuse in ("1", "0"):
        for a in ADDERS:
            n1, n0, npw = (f"{mode}_{a}_plain_f{fuse}_{s}" for s in ("t2", "k0", "pw"))
            val = lambda rows, n, i: f"{med(rows[n], i):.3f}" if n in rows else "nan"
            lines.append((y, "{" + PRETTY[a] + (", fused" if fuse == "1" else ", unfused") + "}",
                          val(r4, n1, 4), val(kpw, n0, 4), val(kpw, npw, 4),
                          val(r4, n1, 5), val(kpw, n0, 5), val(kpw, npw, 5)))
            y += 1
    write(f"settings_{mode}.dat", ["y", "label", "pre1", "pre0", "prepw", "on1", "on0", "onpw"], lines)
# ranges over all 30 variants per setting, for the text
for mode in ("s", "m"):
    for suf in ("k0", "pw"):
        pre = [med(kpw[n], 4) for n in kpw if n.startswith(mode + "_") and n.endswith("_" + suf)]
        on = [med(kpw[n], 5) for n in kpw if n.startswith(mode + "_") and n.endswith("_" + suf)]
        if pre:
            print(f"{mode} {suf}: {len(pre)} variants, pre {min(pre):.2f}-{max(pre):.2f}, online {min(on):.3f}-{max(on):.3f}")
acc_s = [100 * int(r["acc_ok"]) / int(r["acc_n"]) for n, r in pwacc.items() if n.startswith("a_")]
acc_m = [100 * int(r["acc_ok"]) / int(r["acc_n"]) for n, r in pwacc.items() if n.startswith("m_")]
print(f"public weights, AdamW: single {len(acc_s)} variants {min(acc_s):.0f}-{max(acc_s):.0f}%, "
      f"multi {len(acc_m)} variants {min(acc_m):.1f}-{max(acc_m):.1f}%")
