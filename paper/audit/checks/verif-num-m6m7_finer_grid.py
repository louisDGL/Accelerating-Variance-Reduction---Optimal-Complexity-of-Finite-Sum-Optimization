"""Counter-check of finding F2 of audit num-m6m7 (independent of num-m6m7_finer_grid_M6.py).

Question: does the ratio 'fastest spectral variant / fastest control' of (M6) move when the (q, lr_0) grid of M9
part A is refined, at EQUAL protocol for the two terms?

Differences with the auditor's script, on purpose:
  * the SAME grid for the two variants of a cell, symmetric around the points selected in the repository
    (q above and below, lr_0 above and below), so that neither term is refined more than the other;
  * selection on the TUNING seeds only, with the rule of muon/experiments/common.tune_and_confirm (geometric mean of
    common.score over seeds 0-1, the three best points re-run on seeds 2-4, selected point = smallest geometric mean
    over the five seeds), budget and stop levels of the repository (300 passes, TUNE_STOP, CONFIRM_STOP); the
    screening stage (a cost-saving device) is skipped: every grid point is run at full budget;
  * the confirmation seeds 10..19 are used ONCE, at the selected point (no choice of the best point on them).
Nothing is written to the repository; runs go through common.run_one (no cache).

Usage (project root): python3 paper/audit/checks/verif-num-m6m7_finer_grid.py <instance> <b> <variant> [--check]
       [--q 0.75,0.775,...] [--k -4,4] [--lr-ref 3.0]
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'muon'))
sys.path.insert(0, os.path.join(ROOT, 'muon', 'experiments'))
import common as C  # noqa: E402
from families import recipe_family  # noqa: E402

ORTH = {'polar': ('svd', False), 'polar+N': ('svd', True), 'ns5': ('ns5', False), 'ns5+N': ('ns5', True),
        'frob': ('frob', False), 'frob+N': ('frob', True), 'sign': ('sign', False), 'sign+N': ('sign', True)}


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


inst, b, variant = sys.argv[1], sys.argv[2], sys.argv[3]
M9 = json.load(open('muon/results/M9_summary.json'))
A = M9['A'][inst]
spec, TOL, BUDGET = A['spec'], A['tol'], A['passes']
row = {r['variant']: r for r in A['cells'][b]['rows'].values()}[variant]
tk = '%g' % TOL


def run(q, lr, seed, stop):
    orth, nes = ORTH[variant]
    fam = recipe_family(orth, nes, int(b), [q], [lr])
    _, tasks = C.make_tasks(spec, fam, BUDGET, [seed], stop_rel=stop * TOL)
    return C.run_one(tasks[0])


def confirm(q, lr):
    ps = [run(q, lr, s, C.CONFIRM_STOP)['passes_to'][tk] for s in C.CONFIRM_SEEDS]
    return float(np.median(ps)), min(ps), max(ps), int(np.isfinite(ps).sum())


t0 = time.process_time()
bq, bl = row['best']['lr_decay'], row['best']['lr']
if '--check' in sys.argv:
    m, lo, hi, k = confirm(bq, bl)
    print('CHECK %s b=%s %s at the repository point q=%g lr0=%g: median %.2f [%.2f, %.2f] %d/10 ; M9_summary %.2f  '
          '(cpu %.1fs)' % (inst, b, variant, bq, bl, m, lo, hi, k, row['summary'][tk]['median'],
                           time.process_time() - t0))
    sys.exit(0)

qs = [float(x) for x in opt('--q').split(',')]
klo, khi = (int(x) for x in opt('--k', '-4,4').split(','))
ref = float(opt('--lr-ref', str(bl)))
lrs = [float('%.6g' % (ref * 10 ** (k / 8.0))) for k in range(klo, khi + 1)]
runs = {}
for q in qs:
    for lr in lrs:
        runs[(q, lr)] = {s: C.score(run(q, lr, s, C.TUNE_STOP), TOL, BUDGET) for s in C.TUNE_SEEDS}
score2 = lambda p: C.geo_mean([runs[p][s] for s in C.TUNE_SEEDS])        # noqa: E731
score_all = lambda p: C.geo_mean(list(runs[p].values()))                   # noqa: E731
print('--- %s b=%s %s: repository point q=%g lr0=%g (confirmation median %.2f, %d/10). Tuning score (geometric mean, '
      'seeds 0-1; >= %g: a seed misses)' % (inst, b, variant, bq, bl, row['summary'][tk]['median'],
                                           row['summary'][tk]['reached'], BUDGET))
print('      lr0      ' + ' '.join('%7.3g' % lr for lr in lrs))
for q in qs:
    print('      q=%-6g  ' % q + ' '.join(('%7.1f' % score2((q, lr))) if score2((q, lr)) < BUDGET else '      -'
                                           for lr in lrs))
ranked = sorted(runs, key=lambda p: (score2(p), p))
fin = [p for p in ranked[:C.REFINE_TOP] if math.isfinite(score2(p))]
for p in fin:
    for s in C.REFINE_SEEDS:
        runs[p][s] = C.score(run(p[0], p[1], s, C.TUNE_STOP), TOL, BUDGET)
best = min(fin, key=lambda p: (score_all(p), score2(p), p))
for p in fin:
    print('      finalist q=%g lr0=%g: score on 5 tuning seeds %.2f%s' % (p[0], p[1], score_all(p),
                                                                         '  <- selected' if p == best else ''))
edge = [n for n, v, g in (('q', best[0], qs), ('lr0', best[1], lrs)) if v in (g[0], g[-1])]
m, lo, hi, k = confirm(*best)
print('RESULT %s b=%s %s: finer grid, selected on tuning seeds: q=%g lr0=%g%s -> confirmation median %.2f [%.2f, %.2f] '
      '%d/10 (repository grid: %.2f)   cpu %.1fs' % (inst, b, variant, best[0], best[1],
                                                     ' (edge: %s)' % ','.join(edge) if edge else '', m, lo, hi, k,
                                                     row['summary'][tk]['median'], time.process_time() - t0))
