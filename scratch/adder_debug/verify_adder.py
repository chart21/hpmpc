#!/usr/bin/env python3
import re, sys

# Mask = frozenset of atomic terms; XOR = symmetric difference.
def XOR(*ms):
    r = frozenset()
    for m in ms:
        r = r ^ m
    return r

def parse_assign(expr, reshare_map):
    """Return mask (frozenset of atoms) for an 'assign'/output-mask expression."""
    expr = expr.strip()
    # FUNC_XOR(x, y)
    m = re.fullmatch(r'FUNC_XOR\((.*),(.*)\)', expr)
    if m:
        return XOR(parse_assign(m.group(1), reshare_map), parse_assign(m.group(2), reshare_map))
    # triples[j].a / .b / .c
    m = re.fullmatch(r'triples\[(\d+)\]\.([abc])', expr)
    if m:
        return frozenset({f"T{m.group(1)}{m.group(2)}"})
    # random_triples[k].a/.b
    m = re.fullmatch(r'random_triples\[(\d+)\]\.([ab])', expr)
    if m:
        return frozenset({f"R{m.group(1)}{m.group(2)}"})
    # rNN constant
    m = re.fullmatch(r'r\d+', expr)
    if m:
        return frozenset({expr})
    raise ValueError(f"can't parse assign: {expr!r}")

def verify(path, reshared):
    text = open(path).read()
    # build reshare map a[i]/b[i] -> random_triples[k] from the whole file's matching class — caller passes step only,
    # but reshare happens in ctor. We instead infer from the convention a[i] uses random_triples[i-1] (verified separately).
    mask = {}      # wire name -> frozenset
    errors = []
    # For reshared inputs, a[i] carries atom 'A{i}' (its reshare lambda), b[i] carries 'B{i}'.
    def input_mask(tok):
        m = re.fullmatch(r'a\[(\d+)\]', tok)
        if m: return frozenset({f"Ain{m.group(1)}"})
        m = re.fullmatch(r'b\[(\d+)\]', tok)
        if m: return frozenset({f"Bin{m.group(1)}"})
        if tok in mask: return mask[tok]
        raise ValueError(f"unknown wire {tok}")

    for raw in text.splitlines():
        line = raw.split('//')[0].strip().rstrip(';')
        if not line or '=' not in line: continue
        lhs, rhs = line.split('=', 1)
        lhs = lhs.strip(); rhs = rhs.strip()
        if not re.fullmatch(r'[A-Za-z_]\w*', lhs):  # skip a[i].x = ... or msb etc handled below
            pass
        # XOR combination:  w = X ^ Y
        m = re.fullmatch(r'(\S+)\s*\^\s*(\S+)', rhs)
        if m:
            mask[lhs] = XOR(input_mask(m.group(1)), input_mask(m.group(2)))
            continue
        # zero_add:  w = X.zero_add(EXPR)
        m = re.fullmatch(r'(\S+)\.zero_add\((.*)\)', rhs)
        if m:
            mask[lhs] = parse_assign(m.group(2), None)
            continue
        # prepare_and:  w = X.prepare_and(Y, ASSIGN, triples[j].c)
        m = re.fullmatch(r'(\S+)\.prepare_and\((\S+),\s*(.*),\s*triples\[(\d+)\]\.c\)', rhs)
        if m:
            X, Y, assign, j = m.group(1), m.group(2), m.group(3), int(m.group(4))
            mx, my = input_mask(X), input_mask(Y)
            need_a, need_b = frozenset({f"T{j}a"}), frozenset({f"T{j}b"})
            if mx != need_a or my != need_b:
                errors.append(f"AND triples[{j}]: X({X}) mask={set(mx)} expect {set(need_a)}; "
                              f"Y({Y}) mask={set(my)} expect {set(need_b)}")
            mask[lhs] = parse_assign(assign, None)
            continue
        # prepare_and_reshared: w = a[i].prepare_and_reshared(b[i], ASSIGN, random_triples[k].b)
        m = re.fullmatch(r'a\[(\d+)\]\.prepare_and_reshared\(b\[(\d+)\],\s*(.*),\s*random_triples\[(\d+)\]\.b\)', rhs)
        if m:
            i1, i2, assign, k = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            # a[i],b[i] reshared with random_triples[i-1]; the AND must use that same k
            if i1 != i2:
                errors.append(f"reshared AND a[{i1}] vs b[{i2}] mismatch")
            if k != i1 - 1:
                errors.append(f"reshared AND a[{i1}]&b[{i2}] uses random_triples[{k}].b but inputs reshared with random_triples[{i1-1}]")
            mask[lhs] = parse_assign(assign, None)
            continue
        # complete_and / other -> ignore
    return errors

errs = verify(sys.argv[1], True)
if errs:
    print(f"FOUND {len(errs)} inconsistencies:")
    for e in errs: print("  ", e)
else:
    print("All Beaver AND gates consistent.")
