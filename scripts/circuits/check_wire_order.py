#!/usr/bin/env python3
"""check_wire_order.py FILE [k]: in each specialization of a zero_add_adders circuit, every wire must be complete
(assigned, or its prepare_* / mask_and_send_* finished by complete_*()) before it is read. Dot terms (prepare_dot*,
local a-known products) are summed locally before their chain is sent. Exit 1 on a problem."""
import re, sys
# prepare_* finished by complete_*()) before it is read; reports reads of pending or later-assigned wires
def specs(src):
    for m in re.finditer(r'enable_if<\(k == (\d+)\)>::type>\s*\{', src):
        k = int(m.group(1)); start = m.end(); depth = 1; i = start
        while depth: 
            depth += {'{':1,'}':-1}.get(src[i],0); i += 1
        yield k, src[start:i]
ident = re.compile(r'\b([A-Za-z_]\w*)\b(?!\s*\()')
def check(body, k):
    st = body.find('void step()')
    if st < 0: return ['no step()']
    step = body[st:]
    text = re.sub(r'//[^\n]*', '', step)
    text = re.sub(r'\bcase\s+\d+\s*:', ';', text)
    stmts = []
    for raw in text.split(';'):
        line = ' '.join(raw.split())
        line = re.sub(r'^(void step\(\)\s*\{|switch\s*\(level\)\s*\{|\{|\})\s*', '', line).strip()
        line = re.sub(r'^(break|level\+\+|\})\s*', '', line).strip()
        if not line or line in ('break', 'level++') or line.startswith(('if (', 'else', 'for (', 'return')): continue
        stmts.append(line + ';')
    assigned = set()
    for s in stmts:
        m = re.match(r'(\w+)\s*=(?!=)', s)
        if m: assigned.add(m.group(1))
    asynch = set(re.findall(r'(\w+)\.complete_\w+\(\)', ' '.join(stmts)))
    complete = set(); pending = set(); errs = []
    for s in stmts:
        m = re.match(r'(\w+)\.complete_\w+\(\)\s*;', s)
        if m:
            v = m.group(1)
            if v not in pending: errs.append(f'complete of non-pending {v}: {s}')
            pending.discard(v); complete.add(v); continue
        m = re.match(r'(\w+)\.mask_and_send\w*\((.*)\)\s*;', s)
        if m: # V.mask_and_send_dot...(): the local dot sum V goes out, V is pending until complete_*()
            v = m.group(1)
            if v not in complete: errs.append(f'{v} sent before complete: {s}')
            pending.add(v); complete.discard(v); continue
        m = re.match(r'(\w+)\.prepare_\w+\((.*)\)\s*;', s)
        if m: # V.prepare_remask(...): V's own value is complete, it is pending until complete_*()
            v = m.group(1)
            for u in re.findall(r'(?<![.\w])([A-Za-z_]\w*)\b(?!\s*\()', m.group(2)) + [v]:
                if u in assigned and u not in complete:
                    errs.append(f'{u} read before complete: {s}')
            pending.add(v); complete.discard(v); continue
        m = re.match(r'(\w+)\s*=(?!=)\s*(.*);', s)
        if not m:
            # statements like x.prepare_...(...) without assignment
            rhs = s; lhs = None
        else:
            lhs, rhs = m.group(1), m.group(2)
        # names used: identifiers not followed by '(' and not after '.' (members)
        used = set()
        for mm in re.finditer(r'(?<![.\w])([A-Za-z_]\w*)\b(?!\s*\()', rhs):
            used.add(mm.group(1))
        for u in used:
            if u in assigned and u not in complete:
                errs.append(f'{u} read before complete: {s}')
        if lhs:
            local = re.search(r'\.(prepare_dot\w*|mult_a_known_to_evaluators)\(', rhs)
            if lhs in asynch and not local and re.search(r'\.(prepare|remask|and_a|reshare)\w*\(', rhs): pending.add(lhs); complete.discard(lhs)
            else: complete.add(lhs)
    if pending: errs.append(f'never completed: {sorted(pending)[:5]}')
    return errs
src = open(sys.argv[1]).read()
only = int(sys.argv[2]) if len(sys.argv) > 2 else None
bad = 0
for k, body in specs(src):
    if only and k != only: continue
    e = check(body, k)
    print(f'k={k}: {len(e)} problems'); bad += len(e)
    for x in e[:6]: print('   ', x[:160])
sys.exit(1 if bad else 0)
