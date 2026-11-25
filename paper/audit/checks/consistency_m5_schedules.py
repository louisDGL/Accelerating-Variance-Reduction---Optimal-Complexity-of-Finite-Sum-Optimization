"""consistency (a)/(c): which schedules are compared in (M5) / Figure 4(a) "with a table" vs "without a table"
(M9_summary.json, part C). Read-only. Usage (from project root): python3 paper/audit/checks/consistency_m5_schedules.py
"""
import json
m9 = json.load(open('muon/results/M9_summary.json'))
C = m9['C']
for inst, blk in C.items():
    print('==', inst, 'passes', blk['passes'], 'seeds', blk['seeds'])
    print('   points:', json.dumps(blk['points']))
    print('   controls_schedule:', json.dumps(blk['controls_schedule']))
    for sect in [k for k in blk.keys() if k not in ('spec', 'passes', 'seeds', 'norms', 'code', 'points', 'controls_schedule')]:
        sec = blk[sect]
        if not isinstance(sec, dict):
            print('  ', sect, ':', str(sec)[:200]); continue
        for name, r in sec.items():
            if not isinstance(r, dict):
                print('  ', sect, '|', name, ':', str(r)[:200]); continue
            cfg = r.get('cfg', {})
            runs = r.get('runs', [])
            mins = sorted(x.get('min_rel') for x in runs if isinstance(x, dict) and 'min_rel' in x)
            fin = sorted(x.get('final_rel') for x in runs if isinstance(x, dict) and 'final_rel' in x)
            ons = sum(1 for x in runs if isinstance(x, dict) and x.get('on_schedule'))
            cf = {k: cfg.get(k) for k in ('estimator', 'batch', 'orth', 'lr_decay', 'lr', 'momentum', 'nesterov', 'ns_eps', 'order') if k in cfg}
            med = lambda a: (a[len(a)//2] if a else None)
            print(f"   {sect:9s}| {name:44s} cfg={cf} on_schedule={ons}/{len(runs)} min_rel median={med(mins)} final_rel median={med(fin)}")
