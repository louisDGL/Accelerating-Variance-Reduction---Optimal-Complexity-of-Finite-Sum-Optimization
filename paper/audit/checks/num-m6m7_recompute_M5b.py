"""Audit num-m6m7: recompute the M5b ratios quoted in (M6) (nuclear-scaled SAGA-Muon over the best Euclidean family)
from the rows of M5b_summary.json: fine grid, shifted grid, factor-2 control and M5, data seeds 0, 1, 2.
Run from the project root: python3 paper/audit/checks/num-m6m7_recompute_M5b.py
"""
import json, math
M5b = json.load(open('muon/results/M5b_summary.json'))
TOL = '1e-06'
def ratio(rows):
    def ok(r):
        s = r['summary'][TOL]; return s['reached'] == s['runs']
    eu = [rows[f]['summary'][TOL]['median'] for f in ('SAGA', 'B-SAGA') if f in rows and ok(rows[f])]
    m = rows['SAGA-Muon']
    if not eu or not ok(m): return None
    return m['summary'][TOL]['median'] / min(eu)
def f(x): return '  -  ' if x is None else '%.3f' % x
for inst, by_seed in M5b['instances'].items():
    for ds, c in by_seed.items():
        for b in sorted(c['batches'], key=int):
            cell = c['batches'][b]
            out = {}
            for part in ('fine', 'shifted', 'factor2', 'm5'):
                if part in cell and cell[part] and 'rows' in cell[part]:
                    out[part] = ratio(cell[part]['rows'])
            rows = cell['fine']['rows'] if 'fine' in cell else {}
            det = ''
            if rows:
                det = ' | fine: ' + ' ; '.join('%s %.2f (%d/%d)' % (k, r['summary'][TOL]['median'], r['summary'][TOL]['reached'], r['summary'][TOL]['runs']) for k, r in rows.items())
            boot = cell.get('fine', {}).get('boot')
            print('%-9s data seed %s b=%-2s fine %s shifted %s factor2 %s M5 %s  boot %s%s' % (
                inst, ds, b, f(out.get('fine')), f(out.get('shifted')), f(out.get('factor2')), f(out.get('m5')),
                json.dumps(boot)[:80] if boot else '-', det))
print()
print('Paper (M6): softmax b=4,16: 0.74, 0.75 ; least squares b=4,16: 0.74, 0.55 ; b=1: 0.91, 1.04 ; ls-orth b=1,4: 1.30, 1.11 ; factor 2 at b=16: 1.7-2.7')
