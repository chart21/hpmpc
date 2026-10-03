#!/usr/bin/env python3
"""BITLENGTH = 64 specializations of the 2PC ROT-preprocessing adders (programs/functions/adders/zero_add_adders).

The circuits come from the circuit generator (generate_hpp.py; LLM_TEST=<its checkout>, default ~/workspace/llm_test,
reproduces the imported 8/16/32-bit files bit for bit with the flags below). RCA and PPA (Sklansky) take its generic
builders; for the 4-way PPA, which it builds by hand per width, this script adds a width-generic tree: slices 1..k-1
(slice 1 the most significant below the MSB slice 0) in groups of 3 (level 0; the last group may have 1 or 2 slices),
then blocks of up to 4 items per level until one is left, the carry into slice 0. At k = 64: 21 groups, blocks of 4
groups (level 1), a superblock of the first 4 blocks and the tail (block 4 + group 20, level 2), one dot gate (level 3)
-- 4 AND levels (the 4-way trees of 32 bits take 3; 3 levels of these gates cover at most 48 slices).

CUT_FRACTIONAL_BITS_OPT at 64 bits (narrow adders): the MSB adder of width 64 - F over slices F..63, for the values of
F in NARROW_F, written to zero_add_adders/narrow64/<family>.hpp (each width under #if FRACTIONAL == F) and
narrow64/widths.h (which F have them). The 32-bit cut identity-substitutes the full-width circuit instead.

Usage: gen_64bit_adders.py [--check]   (rewrites the k = 64 block of every file and the narrow64 files; --check only
compares)
"""
import os
import re
import sys

GEN = os.environ.get("LLM_TEST", os.path.expanduser("~/workspace/llm_test"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ADDERS = os.path.join(REPO, "programs", "functions", "adders", "zero_add_adders")
sys.path.insert(0, GEN)
os.chdir(GEN)

import circuit_builder as cb  # noqa: E402
import generate_hpp as gh  # noqa: E402
from circuit_core import Circuit  # noqa: E402

K = 64
GROUPS = [(3 * j + 1, 3 * j + 2, 3 * j + 3) for j in range(21)]  # slices 1..63 (the a-known 64-bit circuit)
NARROW_F = [8, 10, 12, 14, 16, 18, 20, 24]  # CUT_FRACTIONAL_BITS_OPT at 64 bits: adders of width 64 - F
# ... at 32 bits under the A2B bake for the prefix adders (their identity-substituted cut fails there, see
# docs/BITLENGTH64.md): width 32 - F, a-known / AB PPA and the AB four-way circuit (the a-known one is hand-fixed)
NARROW32_F = list(range(2, 13))
NARROW32 = ["ppa_msb_unsafe_and_a_ab", "ppa_msb_unsafe_and_ab", "ppa_msb_4way_and_ab"]
NARROW32_DIR = os.path.join(ADDERS, "narrow32")
# TE (TRUNC_APPROACH 2 / 3, 2PC): the low carry by an a-known MSB adder of width F + 1 (8 and 16 exist already)
LOW_F = [f for f in range(3, 25) if f + 1 not in (8, 16)]
LOW = {"rca_msb_and_a_ab": "rca_msb_and_a_ab", "ppa_msb_unsafe_and_a_ab": "ppa_msb_unsafe_and_a_ab"}


def _block(ctx, name, items):  # G = G0 ^ P0 G1 ^ P0 P1 G2 ^ P0 P1 P2 G3, P = P0 P1 P2 P3
    g = [x[0] for x in items]
    p = [x[1] for x in items]
    terms = []
    if len(items) > 1:
        terms.append(ctx.dot2(p[0], g[1], f"{name}_t1"))
    if len(items) > 2:
        terms.append(ctx.dot3(p[0], p[1], g[2], f"{name}_t2"))
    if len(items) > 3:
        terms.append(ctx.dot4(p[0], p[1], p[2], g[3], f"{name}_t3"))
    # the dot gates XORed first, then the non-dot G0 (the generator's grouping of dot chains)
    acc = terms[0]
    for i, t in enumerate(terms[1:], 1):
        acc = ctx.xor(acc, t, f"{name}_s{i}")
    gout = ctx.xor(acc, g[0], f"{name}_G")
    pout = None
    if all(x is not None for x in p):
        if len(items) == 4:
            pout = ctx.and4(p[0], p[1], p[2], p[3], f"{name}_P")
        elif len(items) == 3:
            pout = ctx.and3(p[0], p[1], p[2], f"{name}_P")
        else:
            pout = ctx.and2(p[0], p[1], f"{name}_P")
    return gout, pout


def _tree(ctx, gp):
    """Blocks of up to 4 items per level over the level-0 groups gp[j] = (G_j, P_j), ordered from the most significant
    (the last group's P is None, never needed); a single item passes through. Returns the carry into slice 0."""
    level = 1
    while len(gp) > 1:
        nb = (len(gp) + 3) // 4
        out = []
        for i in range(nb):
            items = gp[4 * i:4 * i + 4]
            if len(items) == 1:
                out.append(items[0])
                continue
            name = f"L{level}_B{i}" if level == 1 else f"L{level}_S{i}" if level == 2 else \
                f"L{level}" if nb == 1 else f"L{level}_B{i}"
            out.append(_block(ctx, name, items))
        gp = out
        level += 1
    return gp[0][0]


def build_ppa_msb_4way_generic(k=K):
    c = Circuit("PPA_MSB_4Way", k)
    c.a_known_pairs = True
    c.input_wires = [f"a[{i}]" for i in range(k)] + [f"b[{i}]" for i in range(k)]
    c.output_wires = ["msb"]
    for w in c.input_wires + c.output_wires:
        c.get_or_create_wire(w)
    ctx = cb._4WayCtx(c)

    def gk(prefix, sl):  # B3L1_G: a1 b1 ^ p1 a2 b2 ^ p1 p2 a3 b3 (1 to 3 slices)
        t = [ctx.dot2(f"a[{sl[0]}]", f"b[{sl[0]}]", f"{prefix}_g1")]
        if len(sl) > 1:
            p1 = ctx.make_p(sl[0])
            t.append(ctx.dot3(p1, f"a[{sl[1]}]", f"b[{sl[1]}]", f"{prefix}_t1"))
        if len(sl) > 2:
            p2 = ctx.make_p(sl[1])
            t.append(ctx.dot4(p1, p2, f"a[{sl[2]}]", f"b[{sl[2]}]", f"{prefix}_t2"))
        if len(t) == 1:
            return t[0]
        acc = ctx.xor(t[0], t[1], f"{prefix}_s1" if len(t) > 2 else f"{prefix}_out")
        return ctx.xor(acc, t[2], f"{prefix}_out") if len(t) > 2 else acc

    def pk(prefix, sl):  # B3L1_P
        ps = [ctx.make_p(i) for i in sl]
        if len(ps) == 3:
            return ctx.and3(ps[0], ps[1], ps[2], f"{prefix}_out")
        return ctx.and2(ps[0], ps[1], f"{prefix}_out") if len(ps) == 2 else ps[0]

    slices = list(range(1, k))
    groups = [slices[i:i + 3] for i in range(0, len(slices), 3)]
    gp = []
    for j, sl in enumerate(groups):
        last = j == len(groups) - 1
        tag = f"{sl[0]}_{sl[-1]}"
        g = gk(f"W{len(sl)}L1_{tag}" if last else f"B3G_{tag}", sl)
        gp.append((g, None if last else pk(f"B3P_{tag}", sl)))
    carry = _tree(ctx, gp)
    p0 = ctx.xor("a[0]", "b[0]", "p0")
    ctx.xor(carry, p0, "msb")
    return c


def ppa4_reshared_slices(k):
    """The slices whose input pair the reshared generic 4-way circuit reshares: the first of each group."""
    return [i for i in range(1, k) if i % 3 == 1]


def build_ppa_msb_4way_64_and_a(k=K):
    c = Circuit("PPA_MSB_4Way_A", k)
    c.input_wires = [f"a[{i}]" for i in range(k)] + [f"b[{i}]" for i in range(k)]
    c.output_wires = ["msb"]
    for w in c.input_wires + c.output_wires:
        c.get_or_create_wire(w)
    c.fixed_mask_wires = {f"b[{i}]" for i in range(k)}
    ctx = cb._4WayCtx(c)
    gp = []
    for j, (i1, i2, i3) in enumerate(GROUPS):
        if j < 20:
            g, sh = ctx.b3l1_g_and_a(f"B3G_{i1}_{i3}", i1, i2, i3)
            gp.append((g, ctx.b3l1_p_and_a(f"B3P_{i1}_{i3}", i1, i2, i3, shared=sh)))
        else:
            gp.append((ctx.w3l1_and_a(f"W3L1_{i1}_{i3}", i1, i2, i3), None))
    carry = _tree(ctx, gp)
    p0 = ctx.xor("a[0]", "b[0]", "p0")
    ctx.xor(carry, p0, "msb")
    return c


_base_build = cb.build_circuit


def build_circuit(circuit_type, k):
    if k not in (8, 16, 32) and circuit_type.startswith("ppa_msb_4way"):
        both = {f"a[{i}]" for i in range(k)} | {f"b[{i}]" for i in range(k)}
        if circuit_type == "ppa_msb_4way":
            c = build_ppa_msb_4way_generic(k)
        elif circuit_type == "ppa_msb_4way_and_ab":
            c = build_ppa_msb_4way_generic(k)
            c.name, c.fixed_mask_wires = "PPA_MSB_4Way_AB", both
        elif circuit_type == "ppa_msb_4way_and_a" and k == K:
            c = build_ppa_msb_4way_64_and_a()
        elif circuit_type == "ppa_msb_4way_and_a_ab" and k == K:
            c = build_ppa_msb_4way_64_and_a()
            c.name, c.fixed_mask_wires = "PPA_MSB_4Way_A_AB", both
        else:
            raise ValueError(circuit_type)
        c.explicit_prepare_dot = True  # as the generator's 4-way builders
        return c
    return _base_build(circuit_type, k)


cb.build_circuit = build_circuit
gh.build_circuit = build_circuit

# repo file -> (generator circuit type, reshare mode); the flags reproduce the imported 8/16/32-bit files
FAMILIES = {
    "rca_msb_and_ab": ("rca_msb_and_ab", "off"),
    "rca_msb_and_ab_reshared": ("rca_msb_and_ab", "all"),
    "rca_msb_and_a_ab": ("rca_msb_and_a_ab", "off"),
    "rca_and_ab": ("rca_and_ab", "off"),
    "rca_and_ab_reshared": ("rca_and_ab", "all"),
    "rca_and_a_ab": ("rca_and_a_ab", "off"),
    "ppa_msb_unsafe_and_ab": ("ppa_msb_unsafe_and_ab", "off"),
    "ppa_msb_unsafe_and_ab_reshared": ("ppa_msb_unsafe_and_ab", "all"),
    "ppa_msb_unsafe_and_a_ab": ("ppa_msb_unsafe_and_a_ab", "off"),
    "ppa_msb_4way_and_ab": ("ppa_msb_4way_and_ab", "off"),
    "ppa_msb_4way_and_ab_reshared": ("ppa_msb_4way_and_ab", "all"),
    # ppa_msb_4way_and_a_ab: the generator's a-known four-way circuits need the hand fixes of hpmpc 4ec9292 (operand
    # order, dot-pending products, chain masks); BITLENGTH 64 takes the AB circuit instead (share_conversion.hpp)
}
MARK = "// 64-bit: generated by scripts/circuits/gen_64bit_adders.py"
# the MSB adders, with narrow (64 - F)-bit versions for the cut, and the slices their reshared circuits reshare
NARROW = {
    "rca_msb_and_ab": None,
    "rca_msb_and_ab_reshared": lambda k: [k - 1],
    "rca_msb_and_a_ab": None,
    "ppa_msb_unsafe_and_ab": None,
    "ppa_msb_unsafe_and_ab_reshared": lambda k: list(range(1, k)),
    "ppa_msb_unsafe_and_a_ab": None,
    "ppa_msb_4way_and_ab": None,
    "ppa_msb_4way_and_ab_reshared": ppa4_reshared_slices,
}
NARROW_DIR = os.path.join(ADDERS, "narrow64")


def block_k(ctype, rs, k):
    code, _, _ = gh.generate_hpp_family(ctype, [k], use_rs=rs, group=False)
    m = re.search(rf"// Wire assignment code for \w+ \({k}-bit\)", code)
    block = code[m.start():].rstrip() + "\n"
    # The generator's RESHARE_OPT_SIM branches drop the input re-masking. Keyed on the bake (RESHARE_BAKE_ACTIVE), and
    # as in the 8/16/32-bit files they re-mask with zero_add_local: locally where the input's mask is baked
    # (reshare_sim_on), with the ordinary zero_add for the others (residual sums, comparisons, ...)
    block = block.replace("#if RESHARE_OPT_SIM == 1", "#if RESHARE_BAKE_ACTIVE")
    return re.sub(r"(\s+)([ab]_\d+_p) = ([ab]\[\d+\]);[^\n]*\(reshare opt sim\)\n#else\n(\s+[ab]_\d+_p = [ab]\[\d+\]"
                  r"\.zero_add\(([^\n]*)\);[^\n]*\n)",
                  lambda m: f"{m.group(1)}{m.group(2)} = {m.group(3)}.zero_add_local({m.group(5)});  // mask pre-baked; no "
                            f"communication\n#else\n{m.group(4)}", block)


def block_64(ctype, rs):
    return block_k(ctype, rs, K)


B3_COUNT = {}  # reshared four-way circuit width -> Beaver3TupleCount (the reshare bake's tuple stride, widths.h)
SIM_ZA = {}  # ... -> whether its RESHARE_OPT_SIM branches skip the input zero_adds the bake serves (widths.h)
SIM_BRANCH = re.compile(r"#if RESHARE_BAKE_ACTIVE\n\s+[ab]_\d+_p = [ab]\[\d+\]\.zero_add_local\([^\n]*\n#else\n(\s+[ab]_\d+_p = [ab]\[\d+\]"
                        r"\.zero_add\([^\n]*\n)#endif\n")


def check_ppa4_sim(blk, k):
    """The reshare bake's maps for the reshared four-way circuit of width k (protocols/beaver_triples.hpp): reshared
    slice i (i % 3 == 1) takes random_triples[(i - 1) / 3]; the RESHARE_OPT_SIM branches skip the zero_adds of slice
    i (i % 3 == 2) with 3-tuple 2 (i - 2) / 3 (.b on the a-wire, which P0 prepares as its mask, .c on the b-wire, which
    P1 bakes; party-local 3-tuples). Where the generator skips other zero_adds (k - 1 = 2 mod 3: the b-wires of
    slices i % 3 == 0 with 4-tuple .d fields, which the bake cannot serve), all of them are reverted to ordinary
    zero_adds (their preprocessing deltas are sent; the reshares stay baked). Returns the block."""
    rt = [(int(i), int(j)) for i, j in re.findall(r"\ba\[(\d+)\]\.reshare_a\(random_triples\[(\d+)\]\.a\)", blk)]
    if rt != [(i, (i - 1) // 3) for i in ppa4_reshared_slices(k)]:
        raise SystemExit(f"ppa4 k={k}: reshare order {rt}")
    ok = True
    want = [(i, 2 * (i - 2) // 3) for i in range(2, k) if i % 3 == 2]
    for w, fld in (("a", "b"), ("b", "c")):
        sim = re.findall(rf"#if RESHARE_BAKE_ACTIVE\n\s+{w}_(\d+)_p = {w}\[\d+\]\.zero_add_local\([^\n]*\n#else\n\s+{w}_\d+_p = "
                         rf"{w}\[(\d+)\]\.zero_add\((beaver\d)_tuples\[(\d+)\]\.(\w)\)", blk)
        got = sorted((int(i), int(t)) for i, i2, b, t, f in sim if i == i2 and b == "beaver3" and f == fld)
        ok = ok and got == want and len(sim) == len(got)
    B3_COUNT[k] = int(re.search(r"static constexpr int Beaver3TupleCount = (\d+);", blk).group(1))
    SIM_ZA[k] = ok
    return blk if ok else SIM_BRANCH.sub(lambda m: m.group(1), blk)


def narrow_file(f, ctype, rs):
    out = [f"// {f}: CUT_FRACTIONAL_BITS_OPT at BITLENGTH 64 -- the MSB adder of width 64 - FRACTIONAL over slices",
           "// FRACTIONAL..63 (get_msb_range), generated by scripts/circuits/gen_64bit_adders.py (generate_hpp.py, LLM_TEST)",
           "#pragma once", "#if BITLENGTH == 64"]
    for j, F in enumerate(NARROW_F):
        k = K - F
        blk = block_k(ctype, rs, k)
        want = NARROW[f]
        if want is not None:  # the A2B prepare's reshare positions (aby2_*.hpp) must match the circuit's
            got = sorted({int(x) for x in re.findall(r"\b[ax]\[(\d+)\]\.reshare_a\(", blk)})
            if got != sorted(want(k)):
                raise SystemExit(f"{f} k={k}: reshared slices {got} differ from the A2B prepare's {want(k)}")
        if f == "ppa_msb_4way_and_ab_reshared":
            blk = check_ppa4_sim(blk, k)
        out.append(("#if" if j == 0 else "#elif") + f" FRACTIONAL == {F}")
        out.append(blk.rstrip())
    out += ["#endif", "#endif", ""]
    return "\n".join(out)


def narrow32_file(f, ctype, rs):
    out = [f"// {f}: CUT_FRACTIONAL_BITS_OPT at BITLENGTH 32 under the A2B bake -- the MSB adder of width 32 - FRACTIONAL",
           "// over slices FRACTIONAL..31 (get_msb_range), generated by scripts/circuits/gen_64bit_adders.py (generate_hpp.py)",
           "#pragma once", "#if BITLENGTH == 32"]
    for j, F in enumerate(NARROW32_F):
        out.append(("#if" if j == 0 else "#elif") + f" FRACTIONAL == {F}")
        out.append(block_k(ctype, rs, 32 - F).rstrip())
    out += ["#endif", "#endif", ""]
    return "\n".join(out)


def low_file(f, ctype):
    out = [f"// {f}: the TE low carry (TRUNC_APPROACH 2 / 3, 2PC, Ts1Range::te) -- the MSB adder of width FRACTIONAL + 1,",
           "// generated by scripts/circuits/gen_64bit_adders.py (generate_hpp.py, LLM_TEST); widths 8 and 16 are in ../" + f + ".hpp",
           "#pragma once"]
    for j, F in enumerate(LOW_F):
        out.append(("#if" if j == 0 else "#elif") + f" FRACTIONAL == {F}")
        out.append(block_k(ctype, "off", F + 1).rstrip())
    out += ["#endif", ""]
    return "\n".join(out)


def widths_header():
    cond = " || ".join(f"(F) == {F}" for F in NARROW_F)
    cases = " ".join(f"case {k}: return {n};" for k, n in sorted(B3_COUNT.items(), reverse=True))
    za = " || ".join(f"k == {k}" for k, v in sorted(SIM_ZA.items(), reverse=True) if v)
    return ("// generated by scripts/circuits/gen_64bit_adders.py: the values of FRACTIONAL with narrow MSB adders at\n"
            "// BITLENGTH 64 (narrow64/*.hpp), which CUT_FRACTIONAL_BITS_OPT needs there\n"
            "#pragma once\n"
            f"#define CUT_FRAC_NARROW64_HAVE(F) ({cond})\n"
            "// ... at BITLENGTH 32 (narrow32/: the prefix adders under the A2B bake)\n"
            f"#define CUT_FRAC_NARROW32_HAVE(F) ((F) >= {NARROW32_F[0]} && (F) <= {NARROW32_F[-1]})\n"
            "// Beaver3TupleCount of the reshared four-way circuits of 64 and 64 - F bits (the reshare bake's tuple stride)\n"
            f"constexpr int ppa4_reshared_b3_count_wide(int k)\n{{\n    switch (k)\n    {{\n        {cases}\n"
            "        default: return -1;\n    }\n}\n"
            "// ... and whether their RESHARE_OPT_SIM branches skip the input zero_adds of slices i % 3 == 2 (3-tuple\n"
            "// 2 (i - 2) / 3); the others zero_add normally (gen_64bit_adders.py: check_ppa4_sim)\n"
            f"constexpr bool ppa4_reshared_sim_za_wide(int k)\n{{\n    return {za};\n}}\n")


def write(path, new, check, name):
    old = open(path).read() if os.path.exists(path) else None
    if check:
        print(f"{name}: {'same' if new == old else 'differs'}")
        return new != old
    if new != old:
        open(path, "w").write(new)
        print(f"{name}: written")
    else:
        print(f"{name}: unchanged")
    return False


def main():
    check = "--check" in sys.argv
    bad = 0
    for f, (ctype, rs) in FAMILIES.items():
        path = os.path.join(ADDERS, f + ".hpp")
        src = open(path).read()
        i = src.find(MARK)
        if i < 0:  # the first blocks were appended by hand with a shorter note
            i = src.find("// 64-bit: generated by generate_hpp.py")
        head = (src[:i] if i >= 0 else src).rstrip() + "\n"
        note = ("; the cut runs the narrow\n// adders of narrow64/ instead of identity-substituting this circuit.\n" if f in NARROW else
                ".\n")
        new = head + "\n" + MARK + " (generate_hpp.py, LLM_TEST)" + note + block_64(ctype, rs)
        if f == "ppa_msb_4way_and_ab_reshared" and not check_ppa4_sim(block_64(ctype, rs), K) == block_64(ctype, rs):
            raise SystemExit("ppa4 k=64: the SIM zero_adds differ from the bake's maps")
        if f in NARROW:
            new += f'\n#include "narrow64/{f}.hpp"  // CUT_FRACTIONAL_BITS_OPT at 64 bits\n'
            bad += write(os.path.join(NARROW_DIR, f + ".hpp"), narrow_file(f, ctype, rs), check, "narrow64/" + f)
        if f in NARROW32:
            new += f'#include "narrow32/{f}.hpp"  // ... at 32 bits under the A2B bake\n'
            os.makedirs(NARROW32_DIR, exist_ok=True)
            bad += write(os.path.join(NARROW32_DIR, f + ".hpp"), narrow32_file(f, ctype, rs), check, "narrow32/" + f)
        bad += write(path, new, check, f)
    bad += write(os.path.join(NARROW_DIR, "widths.h"), widths_header(), check, "narrow64/widths.h")
    os.makedirs(os.path.join(ADDERS, "low"), exist_ok=True)
    for f, ctype in LOW.items():
        bad += write(os.path.join(ADDERS, "low", f + ".hpp"), low_file(f, ctype), check, "low/" + f)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
