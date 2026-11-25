"""consistency (a): numbers of (M7) / Figure 4(c) / abstract / Table 2 from the raw summaries.
Read-only. Usage (from project root): python3 paper/audit/checks/consistency_m7_order.py
"""
import json, math
m9 = json.load(open('muon/results/M9_summary.json'))
m9b = json.load(open('muon/results/M9b_summary.json'))
blocks = {'soft-mem': m9['D']['soft-mem'], 'ls-mem': m9['D']['ls-mem'], 'soft-cond(control)': m9['D']['soft-cond'], 'model': m9b['D']}
first = True
tab = {}
for inst, blk in blocks.items():
    for b in sorted(blk['cells'], key=int):
        cell = blk['cells'][b]
        for scheme, sc in cell['schemes'].items():
            if first:
                print('scheme keys:', list(sc.keys())); print('fastest keys:', list(sc['fastest'].keys()) if isinstance(sc.get('fastest'), dict) else sc.get('fastest')); first = False
            f = sc.get('fastest')
            tab[(inst, int(b), scheme)] = (sc.get('rho_star'), f)
            fs = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()} if isinstance(f, dict) else f
            print(f"{inst:19s} b={b:>2s} {scheme:9s} seeds={cell.get('seeds')} rho_star={sc.get('rho_star')} fastest={fs}")
print()
print('definitions (M9):'); [print('  ', k, ':', str(v)[:400]) for k, v in m9['definitions'].items() if any(s in k.lower() for s in ('rho', 'fast', 'track', 'follow', 'star'))]
print('definitions (M9b):'); [print('  ', k, ':', str(v)[:400]) for k, v in m9b['definitions'].items()]
print()
print('savings of the permutation, passes to 1e-9 at the fastest cooldown (median):')
for inst in ('soft-mem', 'ls-mem', 'model'):
    for b in (1, 5, 20, 50):
        fi, fr = tab[(inst, b, 'iid')][1], tab[(inst, b, 'reshuffle')][1]
        if isinstance(fi, dict) and isinstance(fr, dict):
            print(f"  {inst:9s} b={b:2d}: iid {fi['median']:.2f} -> perm {fr['median']:.2f}: saving {100*(1-fr['median']/fi['median']):.1f}%  ratio perm/iid {fr['median']/fi['median']:.2f}")
print('instance / model ratios:')
for inst in ('soft-mem', 'ls-mem'):
    for b in (1, 5, 20, 50):
        for sch in ('iid', 'reshuffle'):
            a, m = tab[(inst, b, sch)][1], tab[('model', b, sch)][1]
            print(f"  {inst:9s} b={b:2d} {sch:9s}: {a['median']/m['median']:.2f}")
