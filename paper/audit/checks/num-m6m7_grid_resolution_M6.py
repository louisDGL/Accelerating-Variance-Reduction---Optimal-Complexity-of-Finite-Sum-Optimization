"""Audit num-m6m7: resolution of the (q, lr_0) grid of the recipe families of M9 part A.
Prints the tuning scores (passes to 1e-6, tuning seeds) over the whole (q, lr_0) grid for the fastest spectral
variant and the fastest control of each minibatch cell, and the cost in passes of one step of the lr_0 grid
(half a decade) and of the q grid at the selected point.
Run from the project root: python3 paper/audit/checks/num-m6m7_grid_resolution_M6.py
"""
import json, math
M9 = json.load(open('muon/results/M9_summary.json'))
A = M9['A']
TOL = '1e-06'

def table(row):
    pts = {}
    for t in row['tune']:
        p = t['point']
        s = t['score_refined'] if t.get('score_refined') is not None else t['score']
        pts[(p['lr_decay'], p['lr'])] = (t['score'], t.get('score_refined'), t['kept'])
    qs = sorted({k[0] for k in pts}); lrs = sorted({k[1] for k in pts})
    return pts, qs, lrs

def fmt(x):
    if x is None: return '   . '
    if not math.isfinite(x): return '  inf'
    return '%5.1f' % x

summary = []
for inst in A:
    for b in ('16', '32'):
        cell = A[inst]['cells'][b]
        byv = {r['variant']: r for r in cell['rows'].values()}
        cmp_ = cell['compare']
        sv = cmp_['spectral']['variant']; cv = cmp_['control']['variant']
        print('=== %s b=%s : fastest spectral %s, fastest control %s, ratio %.3f' % (inst, b, sv, cv, cmp_['ratios']['spectral/control']))
        for v in (sv, cv):
            row = byv[v]
            pts, qs, lrs = table(row)
            bq, bl = row['best']['lr_decay'], row['best']['lr']
            print('  %s: selected q=%g lr0=%g, confirmation median %.2f ; tuning score (seeds 0-1; refined on seeds 2-4 in parentheses)' % (v, bq, bl, row['summary'][TOL]['median']))
            print('     lr0 \\ q ' + ' '.join('%13g' % q for q in qs))
            for l in lrs:
                print('     %7g  ' % l + ' '.join('%s (%s)' % (fmt(pts[(q, l)][0]), fmt(pts[(q, l)][1])) if (q, l) in pts else '      -      ' for q in qs))
            # cost of one lr0 grid step at the selected q, predicted by the schedule law: ln(sqrt(10)) / ln(1/q)
            pred = math.log(10 ** 0.5) / math.log(1 / bq)
            up = pts.get((bq, min([l for l in lrs if l > bl], default=None)))
            dn = pts.get((bq, max([l for l in lrs if l < bl], default=None)))
            here = pts[(bq, bl)]
            print('     one half-decade of lr0 at q=%g: predicted +%.1f passes (%.0f%% of the median); measured (tuning seeds): lr0 up %s, here %s, lr0 down %s' % (
                bq, pred, 100 * pred / row['summary'][TOL]['median'], fmt(up[0]) if up else '-', fmt(here[0]), fmt(dn[0]) if dn else '-'))
            summary.append((inst, b, v, bq, bl, row['summary'][TOL]['median'], pred, 100 * pred / row['summary'][TOL]['median']))
        print()
print('SUMMARY: cost of one step of the lr_0 grid (x sqrt(10)) at the selected q, relative to the confirmation median')
for s in summary:
    print('  %-9s b=%-2s %-7s q=%g lr0=%-5g median %.1f : one lr0 step = %.1f passes = %.0f%%' % s)
