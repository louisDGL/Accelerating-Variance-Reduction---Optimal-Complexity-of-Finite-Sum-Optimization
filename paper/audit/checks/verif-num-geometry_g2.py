"""Counter-check of finding G2 (num-geometry): SAGA passes and the ratio 0.20--0.37 of (M3).

Read-only. Reads muon/results/M1_summary.json, M5b_summary.json, M8_summary.json and recomputes every
median from the per-seed confirm_passes (no summary field is trusted).

Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python3 paper/audit/checks/verif-num-geometry_g2.py
"""
import json
import math
import os
import statistics as st
import sys

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RES = os.path.join(ROOT, 'muon', 'results')


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def med(xs):
    xs = [float(x) for x in xs]
    return st.median(xs) if xs else float('nan')


def fmt_pt(p):
    out = []
    for k, v in p.items():
        if isinstance(v, dict) and 'c' in v:
            out.append('%s=%.4g/%s' % (k, v['c'], v['over']))
        else:
            out.append('%s=%s' % (k, v))
    return ', '.join(out)


M1 = load('M1_summary.json')
M5b = load('M5b_summary.json')
M8 = load('M8_summary.json')

POLAR_RECOMP = ['SAGA-Muon (joint)', 'soft SAGA-Muon', 'IS soft SAGA-Muon', 'split SAGA-Muon']
EUCL_NOPREC = ['B-SAGA', 'IS-B-SAGA']

print('=' * 100)
print('A. M1 (data draw 0), SAGA: selected step, confirmation median, tuning curve')
for inst in ('soft-cond', 'ls-cond', 'ls-orth'):
    rows = {r['family']: r for r in M1[inst]['rows']}
    r = rows['SAGA']
    cp = r['confirm_passes']
    print('%-9s SAGA best %s  median %.2f  [%.1f, %.1f]  reached %d/%d' % (
        inst, fmt_pt(r['best']), med(cp), min(cp), max(cp), sum(math.isfinite(x) for x in cp), len(cp)))
    print('          finalists:', [fmt_pt(p) for p in r['finalists']])
    for t in r['tune']:
        if t['kept'] or math.isfinite(t['score']):
            print('            tune %-18s score %-8s refined %s' % (
                fmt_pt(t['point']), '%.1f' % t['score'], t['score_refined']))
        else:
            print('            tune %-18s score inf (screen_rel %.3g)' % (fmt_pt(t['point']), t['screen_rel']))

print('=' * 100)
print('B. M5b (data draw 0), b = 1: factor-2 control, fine grid 2^(1/4), shifted grid')
m5b_fine = {}
for inst in ('soft-cond', 'ls-cond', 'ls-orth'):
    cell = M5b['instances'][inst]['0']['batches']['1']
    for part in ('m5', 'factor2', 'fine', 'shifted'):
        if part not in cell:
            continue
        rows = cell[part]['rows'] if 'rows' in cell[part] else cell[part]
        for fam in ('SAGA', 'B-SAGA', 'SAGA-Muon'):
            if fam not in rows:
                continue
            r = rows[fam]
            cp = r.get('confirm_passes')
            if cp is None:
                print('%-9s %-8s %-9s (no per-seed passes) %s' % (inst, part, fam, str(r)[:150]))
                continue
            reached = sum(math.isfinite(x) for x in cp)
            print('%-9s %-8s %-9s %-34s median %.2f [%.1f, %.1f] %d/%d edge=%s' % (
                inst, part, fam, fmt_pt(r['best']), med(cp), min(cp), max(cp), reached, len(cp), r.get('edge')))
            if part == 'fine':
                m5b_fine[(inst, fam)] = med(cp)
    w = cell.get('window', {})
    if 'SAGA' in w:
        print('          window SAGA (passes at fine selection x 2^(j/8)):')
        ws = w['SAGA']
        for j in sorted(ws, key=lambda s: int(s)):
            v = ws[j]
            if isinstance(v, dict):
                v = v.get('passes', v.get('confirm_passes', v))
            try:
                print('             j=%3s  factor %.3f  median %.2f' % (j, 2 ** (int(j) / 8.0), med(v)))
            except Exception:
                print('             j=%3s  %s' % (j, str(v)[:120]))

print('=' * 100)
print('C. Three draws (M1 = draw 0, M8 = draws 1, 2): SAGA and best polar-recomputing family (factor-2 grids)')
ratios_saga, ratios_eucl = {}, {}
table = {}
for inst in ('soft-cond', 'ls-cond'):
    for draw in (0, 1, 2):
        rows = M1[inst]['rows'] if draw == 0 else M8[inst][str(draw)]['rows']
        rows = {r['family']: r for r in rows}
        saga = med(rows['SAGA']['confirm_passes'])
        cands = {}
        for f in POLAR_RECOMP:
            cp = rows[f]['confirm_passes']
            if all(math.isfinite(x) for x in cp):
                cands[f] = med(cp)
        bf = min(cands, key=cands.get)
        eu = {f: med(rows[f]['confirm_passes']) for f in EUCL_NOPREC}
        be = min(eu, key=eu.get)
        ratios_saga[(inst, draw)] = cands[bf] / saga
        ratios_eucl[(inst, draw)] = cands[bf] / eu[be]
        table[(inst, draw)] = (saga, rows['SAGA']['best'], bf, cands[bf], be, eu[be])
        print('%-9s draw %d  SAGA %-14s %7.2f | best polar %-20s %6.2f (%s) | ratio/SAGA %.3f | '
              'best Eucl. no prec. %-10s %6.2f ratio %.3f' % (
                  inst, draw, fmt_pt(rows['SAGA']['best']), saga, bf, cands[bf], fmt_pt(rows[bf]['best']),
                  ratios_saga[(inst, draw)], be, eu[be], ratios_eucl[(inst, draw)]))
        if draw > 0:
            t = rows['SAGA'].get('tune')
            if t:
                print('            SAGA tune:', [(fmt_pt(x['point']), round(x['score'], 1)) for x in t
                                                 if math.isfinite(x['score'])])
print('range ratio / SAGA (paper: 0.20--0.37): %.3f -- %.3f' % (min(ratios_saga.values()), max(ratios_saga.values())))
print('range ratio / best Eucl. no prec. (paper: 0.78--1.08): %.3f -- %.3f' % (
    min(ratios_eucl.values()), max(ratios_eucl.values())))

print('=' * 100)
print('D. Draw 0, same confirmation seeds: ratio against SAGA at a resolved step (M5b fine, b = 1)')
for inst in ('soft-cond', 'ls-cond'):
    saga2, _, bf, bp, _, _ = table[(inst, 0)]
    sf = m5b_fine[(inst, 'SAGA')]
    mf = m5b_fine[(inst, 'SAGA-Muon')]
    rows = {r['family']: r for r in M1[inst]['rows']}
    joint = med(rows['SAGA-Muon (joint)']['confirm_passes'])
    print('%-9s SAGA factor-2 %.2f -> fine %.2f (%.1f %%)' % (inst, saga2, sf, 100 * (sf / saga2 - 1)))
    print('          best polar-recomputing family on factor-2 grid (%s, %.2f) / SAGA fine = %.3f   [paper-style '
          'ratio: %.3f]' % (bf, bp, bp / sf, bp / saga2))
    print('          SAGA-Muon joint: factor-2 %.2f / SAGA factor-2 = %.3f ; fine %.2f / SAGA fine = %.3f' % (
        joint, joint / saga2, mf, mf / sf))
print()
print('E. Sensitivity of the lower end 0.20: draw 2 softmax, SAGA selected at 2/R2 (136 passes) while draws 0, 1 use '
      '4/R2 (about 68--70 passes).')
for draw in (0, 1, 2):
    saga, best, bf, bp, _, _ = table[('soft-cond', draw)]
    print('   draw %d: SAGA %s, %.2f passes ; best polar %.2f ; ratio %.3f' % (draw, fmt_pt(best), saga, bp, bp / saga))
