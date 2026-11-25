"""consistency (a)/(c): which map is the 'fastest control' and the 'fastest spectral variant' in (M6)
(M9_summary.json, part A, 'compare'), and the ratios quoted in the abstract, (M6) and Figure 4(b). Read-only.
Usage (from project root): python3 paper/audit/checks/consistency_m6_controls.py
"""
import json
m9 = json.load(open('muon/results/M9_summary.json'))
A = m9['A']
print('A keys:', list(A.keys())[:20])
first = True
for inst, blk in A.items():
    if not isinstance(blk, dict):
        continue
    cells = blk.get('cells', blk)
    for b, cell in cells.items():
        if not isinstance(cell, dict) or 'compare' not in cell:
            continue
        cmp_ = cell['compare']
        if first:
            print('compare keys:', list(cmp_.keys())); first = False
        def short(x):
            if isinstance(x, dict):
                return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if not isinstance(v, (dict, list))}
            return x
        print(f"{inst:9s} b={b:>3s}")
        for k, v in cmp_.items():
            print('      ', k, ':', str(short(v))[:230])
