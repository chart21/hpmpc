#!/usr/bin/env python3
"""BITLENGTH = 64 specializations of the 2PC ROT-preprocessing adders (programs/functions/adders/zero_add_adders).

The circuits come from the circuit generator (generate_hpp.py; LLM_TEST=<its checkout>, default ~/workspace/llm_test,
reproduces the imported 8/16/32-bit files bit for bit with the flags below). RCA and PPA (Sklansky) take its generic
builders; for the 4-way PPA, which it builds by hand per width, this script adds the 64-bit trees: slices 1..63 (slice
1 the most significant below the MSB slice 0) in 21 groups of 3 (level 0), blocks of 4 groups (level 1), a superblock
of the first 4 blocks and the tail (block 4 + group 20, level 2), and one dot gate (level 3) -- 4 AND levels (the
4-way trees of 32 bits take 3; 3 levels of these gates cover at most 48 slices).

Usage: gen_64bit_adders.py [--check]   (rewrites the k = 64 block of every file; --check only compares)
The 32-bit-only hand additions of the files (CUT_FRACTIONAL_BITS_OPT, RESHARE_OPT_SIM) have no 64-bit counterpart.
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
GROUPS = [(3 * j + 1, 3 * j + 2, 3 * j + 3) for j in range(21)]  # slices 1..63


def _tree(ctx, gp):
    """Levels 1-3 over the 21 level-0 groups gp[j] = (G_j, P_j) (P_20 unused); returns the carry into slice 0."""
    def block(name, items):  # G = G0 ^ P0 G1 ^ P0 P1 G2 ^ P0 P1 P2 G3, P = P0 P1 P2 P3
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

    # level 1: blocks of 4 groups (block 4 = groups 16..19 keeps its P for the tail)
    b = [block(f"L1_B{i}", gp[4 * i:4 * i + 4]) for i in range(5)]
    # level 2: superblock of blocks 0..3; tail = block 4 then group 20 (its P is not needed)
    s0 = block("L2_S0", b[0:4])
    s1g = ctx.xor(ctx.dot2(b[4][1], gp[20][0], "L2_S1_t1"), b[4][0], "L2_S1_G")
    # level 3: the carry into slice 0
    return ctx.xor(ctx.dot2(s0[1], s1g, "L3_t1"), s0[0], "L3_G")


def build_ppa_msb_4way_64(k=K):
    c = Circuit("PPA_MSB_4Way", k)
    c.a_known_pairs = True
    c.input_wires = [f"a[{i}]" for i in range(k)] + [f"b[{i}]" for i in range(k)]
    c.output_wires = ["msb"]
    for w in c.input_wires + c.output_wires:
        c.get_or_create_wire(w)
    ctx = cb._4WayCtx(c)

    def g3(prefix, i1, i2, i3):  # B3L1_G: a1 b1 ^ p1 a2 b2 ^ p1 p2 a3 b3
        t0 = ctx.dot2(f"a[{i1}]", f"b[{i1}]", f"{prefix}_g1")
        p1 = ctx.make_p(i1)
        t1 = ctx.dot3(p1, f"a[{i2}]", f"b[{i2}]", f"{prefix}_t1")
        p2 = ctx.make_p(i2)
        t2 = ctx.dot4(p1, p2, f"a[{i3}]", f"b[{i3}]", f"{prefix}_t2")
        return ctx.xor(ctx.xor(t0, t1, f"{prefix}_s1"), t2, f"{prefix}_out")

    def p3(prefix, i1, i2, i3):  # B3L1_P
        return ctx.and3(ctx.make_p(i1), ctx.make_p(i2), ctx.make_p(i3), f"{prefix}_out")

    gp = []
    for j, (i1, i2, i3) in enumerate(GROUPS):
        g = g3(f"B3G_{i1}_{i3}" if j < 20 else f"W3L1_{i1}_{i3}", i1, i2, i3)
        gp.append((g, p3(f"B3P_{i1}_{i3}", i1, i2, i3) if j < 20 else None))
    carry = _tree(ctx, gp)
    p0 = ctx.xor("a[0]", "b[0]", "p0")
    ctx.xor(carry, p0, "msb")
    return c


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
    if k == K and circuit_type.startswith("ppa_msb_4way"):
        both = {f"a[{i}]" for i in range(K)} | {f"b[{i}]" for i in range(K)}
        if circuit_type == "ppa_msb_4way":
            c = build_ppa_msb_4way_64()
        elif circuit_type == "ppa_msb_4way_and_ab":
            c = build_ppa_msb_4way_64()
            c.name, c.fixed_mask_wires = "PPA_MSB_4Way_AB", both
        elif circuit_type == "ppa_msb_4way_and_a":
            c = build_ppa_msb_4way_64_and_a()
        elif circuit_type == "ppa_msb_4way_and_a_ab":
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


def block_64(ctype, rs):
    code, _, _ = gh.generate_hpp_family(ctype, [K], use_rs=rs, group=False)
    m = re.search(r"// Wire assignment code for \w+ \(64-bit\)", code)
    block = code[m.start():].rstrip() + "\n"
    # The generator's RESHARE_OPT_SIM branches drop the input re-masking: valid only under the reshare bake, which is
    # 32-bit only (RESHARE_BAKE_ACTIVE); key them on it so the 64-bit circuits re-mask
    return block.replace("#if RESHARE_OPT_SIM == 1", "#if RESHARE_BAKE_ACTIVE")


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
        new = head + "\n" + MARK + " (generate_hpp.py, LLM_TEST); no CUT_FRACTIONAL_BITS_OPT\n// shortcuts (32-bit only).\n" + block_64(ctype, rs)
        if check:
            same = new == src
            bad += not same
            print(f"{f}: {'same' if same else 'differs'}")
        elif new != src:
            open(path, "w").write(new)
            print(f"{f}: written")
        else:
            print(f"{f}: unchanged")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
