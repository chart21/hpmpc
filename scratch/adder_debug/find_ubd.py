#!/usr/bin/env python3
import re, sys

# Detect use-before-definition of prefix wires within each adder class (cases run in order 0..N).
# A wire (g_*, pg_*, p_*) used on the RHS must have been assigned on an earlier line.
# a[i]/b[i] are inputs (always available). complete_and() lines neither def nor use for this purpose.

WIRE = re.compile(r'\b(p_\w+|g_\w+|pg_\w+)\b')

def scan_class(lines, cls_name):
    defined = set()
    viol = []
    for lineno, raw in lines:
        line = raw.split('//')[0].strip().rstrip(';')
        if '=' not in line: continue
        if '==' in line or 'level' in line: continue
        m = re.match(r'([A-Za-z_]\w*)\s*=\s*(.*)', line)
        if not m: continue
        lhs, rhs = m.group(1), m.group(2)
        # uses on rhs
        for w in WIRE.findall(rhs):
            if w not in defined:
                viol.append((lineno, lhs, w))
        # now define lhs (only if lhs is a wire name we track or any plain identifier)
        defined.add(lhs)
    return viol

text = open(sys.argv[1]).read().splitlines()
# split into classes by "k == N"
classes = []
cur = None
for i, l in enumerate(text):
    m = re.search(r'k == (\d+)', l)
    if m:
        if cur: classes.append(cur)
        cur = (f"k=={m.group(1)} (line {i+1})", [])
    if cur:
        cur[1].append((i+1, l))
if cur: classes.append(cur)

total = 0
for name, lines in classes:
    v = scan_class(lines, name)
    if v:
        print(f"### {name}: {len(v)} use-before-def")
        for lineno, lhs, w in v:
            print(f"    line {lineno}: {lhs} = ... uses {w} (not yet defined)")
        total += len(v)
if total == 0:
    print("No use-before-def violations.")
