"""Audit num-m6m7: does the ratio 'fastest spectral variant / fastest control' of (M6) survive a finer (q, lr_0) grid?
Small reproduction with the code of the repository (muon/experiments/common.run_one, families.recipe_family), nothing
is written to the repository caches. For one cell (instance, b) and the two variants compared by the paper in that
cell, the (q, lr_0) grid of M9 part A (q in steps of 0.1, lr_0 in half-decades) is refined around the selected
points: q in steps of 0.025-0.05, lr_0 in steps of 10^(1/8). Selection on tuning seeds, report on the confirmation
seeds 10..19 (the seeds of the paper).
Usage (from the project root):
  python3 paper/audit/checks/num-m6m7_finer_grid_M6.py <instance> <b> [--check] [--small]
Runs of the audit: soft-cond 32, ls-cond 32, soft-cond 16, ls-cond 16 (full refinement); ls-orth 16 --small.
"""
import json, math, os, sys, time
import numpy as np
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'muon'))
sys.path.insert(0, os.path.join(ROOT, 'muon', 'experiments'))
import common
from families import recipe_family

M9 = json.load(open('muon/results/M9_summary.json'))
inst, b = sys.argv[1], sys.argv[2]
A = M9['A'][inst]
spec = A['spec']; TOL = A['tol']
cell = A['cells'][b]
byv = {r['variant']: r for r in cell['rows'].values()}
ORTH = {'polar': ('svd', False), 'polar+N': ('svd', True), 'ns5': ('ns5', False), 'ns5+N': ('ns5', True),
        'frob': ('frob', False), 'frob+N': ('frob', True), 'sign': ('sign', False), 'sign+N': ('sign', True)}
BUDGET = 90.0
SMALL = '--small' in sys.argv   # reduced refinement for the n = 500 instances (CPU limit of the audit)
TUNE = (0, 1) if SMALL else (0, 1, 2); CONF = tuple(range(10, 20)); NCONF = 5 if SMALL else 9

def run(variant, q, lr, seed):
    orth, nes = ORTH[variant]
    fam = recipe_family(orth, nes, int(b), [q], [lr])
    pts, tasks = common.make_tasks(spec, fam, BUDGET, [seed], stop_rel=TOL / 3)
    r = common.run_one(tasks[0])
    return r['passes_to']['%g' % TOL]

def med(v):
    v = sorted(v); return float(np.median(v))

t0 = time.time()
if '--check' in sys.argv:   # reproduce the confirmation median of the paper at the selected point
    for v in (cell['compare']['spectral']['variant'], cell['compare']['control']['variant']):
        bq, bl = byv[v]['best']['lr_decay'], byv[v]['best']['lr']
        ps = [run(v, bq, bl, s) for s in CONF]
        print('%s %s b=%s %s q=%g lr0=%g: recomputed median %.2f [%.2f, %.2f] ; M9_summary %.2f  (%.1fs)' % (
            inst, spec['loss'], b, v, bq, bl, med(ps), min(ps), max(ps), byv[v]['summary']['%g' % TOL]['median'], time.time() - t0))
    sys.exit(0)

out = {}; out_any = {}
for v in (cell['compare']['spectral']['variant'], cell['compare']['control']['variant']):
    bq, bl = byv[v]['best']['lr_decay'], byv[v]['best']['lr']
    qs = sorted({round(bq - d, 4) for d in ((0.05, 0.025, 0.0) if SMALL else (0.1, 0.075, 0.05, 0.025, 0.0))})
    lrs = [bl * 10 ** (-k / 8.0) for k in (range(0, 5) if SMALL else range(-1, 6))]
    best = None; tab = {}
    for q in qs:
        for lr in lrs:
            ps = [run(v, q, lr, s) for s in TUNE]
            sc = med(ps) if all(math.isfinite(p) for p in ps) else float('inf')
            tab[(q, lr)] = sc
            if best is None or sc < best[0]:
                best = (sc, q, lr)
    print('--- %s b=%s %s: paper point q=%g lr0=%g (confirmation median %.2f). Tuning medians (seeds %s), inf = a seed misses within %g passes' % (
        inst, b, v, bq, bl, byv[v]['summary']['%g' % TOL]['median'], TUNE, BUDGET))
    print('      lr0/lr0_paper ' + ' '.join('%7.3f' % (lr / bl) for lr in lrs))
    for q in qs:
        print('      q=%-6g      ' % q + ' '.join(('%7.1f' % tab[(q, lr)]) if math.isfinite(tab[(q, lr)]) else '    inf' for lr in lrs))
    # confirm the NCONF best tuning points on seeds 10..19; keep the best one whose 10 seeds reach the target
    # ('10/10 rule') and the smallest finite median whatever the seeds lost (the rule of the 'compare' block of M9)
    cands = sorted(tab.items(), key=lambda kv: kv[1])[:NCONF]
    conf = []
    for (q, lr), sc in cands:
        if not math.isfinite(sc): continue
        ps = [run(v, q, lr, s) for s in CONF]
        conf.append((med(ps), q, lr, sum(math.isfinite(p) for p in ps), min(ps), max(ps)))
        print('      confirm q=%g lr0=%.4g: median %.2f [%.2f, %.2f] reached %d/10' % (q, lr, med(ps), min(ps), max(ps), conf[-1][3]))
    full = [c for c in conf if c[3] == 10]
    anyr = [c for c in conf if math.isfinite(c[0])]
    out[v] = min(full) if full else None
    out_any[v] = min(anyr) if anyr else None
    print('      elapsed %.1fs' % (time.time() - t0))
sv, cv = cell['compare']['spectral']['variant'], cell['compare']['control']['variant']
print('RESULT %s b=%s: paper %s %.2f / %s %.2f = %.3f' % (inst, b, sv, byv[sv]['summary']['%g' % TOL]['median'], cv, byv[cv]['summary']['%g' % TOL]['median'], cell['compare']['ratios']['spectral/control']))
for lab, o in (('finer grid, 10/10 rule', out), ('finer grid, smallest median (seeds may be lost)', out_any)):
    if o.get(sv) and o.get(cv):
        print('RESULT %s b=%s: %s: %s %.2f (q=%g, lr0=%.4g, %d/10) / %s %.2f (q=%g, lr0=%.4g, %d/10) = %.3f' % (
            inst, b, lab, sv, o[sv][0], o[sv][1], o[sv][2], o[sv][3], cv, o[cv][0], o[cv][1], o[cv][2], o[cv][3], o[sv][0] / o[cv][0]))
