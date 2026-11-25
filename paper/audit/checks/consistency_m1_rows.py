"""consistency (a): passes per family and instance in muon/results/M1_summary.json (head-to-head, batch one),
to compare the numbers quoted in (M4), (M3), (M5), (M2). Read-only.
Usage (from project root): python3 paper/audit/checks/consistency_m1_rows.py
"""
import json
m1 = json.load(open('muon/results/M1_summary.json'))
for inst, blk in m1.items():
    sp = blk['spec']; info = blk['info']
    print(f"== {inst}: n={sp.get('n')} K={sp.get('K')} d={sp.get('d')} kappa={sp.get('kappa')} tol={blk['tol']} "
          f"info={ {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if not isinstance(v, (dict, list))} }")
    rows = blk['rows']
    if rows and inst == 'soft-mem':
        print('   row keys:', list(rows[0].keys()))
    for r in rows:
        S = r.get('summary') or {}
        out = []
        for tol in ('1e-06', '1e-09'):
            c = S.get(tol)
            if c:
                out.append(f"{tol}: {c['median']} [{c['min']}, {c['max']}] {c['reached']}/{c['runs']}")
        print(f"   {r['family']:34s} {r['group']:9s} {' | '.join(out)}  best={r.get('best')}")
