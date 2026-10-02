import csv, sys
# acc_tab.py RES: correct of 256 per UC x variant x F (CIFAR-10, AdamW ResNet50, plaintext 189)
acc = {}
for r in csv.reader(open(sys.argv[1])):
    if r and r[0] == 'fp' and r[6]:
        acc[r[1]] = int(r[6])
V = [('tl0', 'TS{L}, no A2B_DELAYED_CUT'), ('tl', 'TS{L}'), ('mx', 'TS_Mix'), ('mxw', 'TS_Mix + w_t'), ('t1', 'TS1'), ('t1w', 'TS1 + w_t')]
print('| UC | variant | F = 5 | F = 8 | F = 10 |')
print('|---|---|---|---|---|')
for u, lab in [('c1', 'UC1'), ('c2', 'UC2'), ('c3', 'UC3')]:
    for v, vl in V:
        cells = []
        for f in ['', 'f8', 'f10']:
            n = f'a_{u}{f}_{v}'
            cells.append(str(acc[n]) if n in acc else '')
        if any(cells):
            print(f'| {lab} | {vl} | ' + ' | '.join(cells) + ' |')
