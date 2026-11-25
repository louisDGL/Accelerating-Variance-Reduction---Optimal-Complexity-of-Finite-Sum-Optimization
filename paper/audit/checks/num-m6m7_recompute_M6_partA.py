"""Audit num-m6m7: recompute the (M6) ratios of M9 part A from the per-row summaries of M9_summary.json
(not from the 'compare' block), with selected points, seeds reached, and a bootstrap over confirmation seeds.
Run from the project root: python3 paper/audit/checks/num-m6m7_recompute_M6_partA.py
"""
import json, math
import numpy as np

M9 = json.load(open('muon/results/M9_summary.json'))
A = M9['A']
TOL = '1e-06'
rng = np.random.default_rng(0)

def med(row):
    return row['summary'][TOL]['median']

def boot_ratio(num, den, nb=4000):
    a = np.array(num, float); b = np.array(den, float)
    out = []
    for _ in range(nb):
        ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
        ma = np.median(a[ia]); mb = np.median(b[ib])
        out.append(ma / mb if np.isfinite(mb) and mb > 0 else np.nan)
    out = np.array(out)
    return np.nanpercentile(out, 5), np.nanpercentile(out, 95)

allr = {'ctrl': {}, 'eucl': {}, 'nuc': {}, 'polar_frob': {}}
for inst in A:
    for b in sorted(A[inst]['cells'], key=int):
        cell = A[inst]['cells'][b]
        rows = cell['rows']
        byv = {r['variant']: r for r in rows.values()}
        spec = {v: byv[v] for v in ('polar', 'polar+N', 'ns5', 'ns5+N')}
        ctrl = {v: byv[v] for v in ('frob', 'frob+N', 'sign', 'sign+N')}
        # fastest = smallest median (as the summary does)
        s = min(spec, key=lambda v: med(spec[v]))
        c = min(ctrl, key=lambda v: med(ctrl[v]))
        # fastest among variants whose 10 confirmation seeds all reach the target
        sfull = [v for v in spec if spec[v]['summary'][TOL]['reached'] == spec[v]['summary'][TOL]['runs']]
        s10 = min(sfull, key=lambda v: med(spec[v])) if sfull else None
        cfull = [v for v in ctrl if ctrl[v]['summary'][TOL]['reached'] == ctrl[v]['summary'][TOL]['runs']]
        c10 = min(cfull, key=lambda v: med(ctrl[v])) if cfull else None
        def desc(r):
            sm = r['summary'][TOL]
            bst = r['best']
            if 'lr_decay' in bst:
                p = 'q=%g lr0=%g' % (bst['lr_decay'], bst['lr'])
            else:
                p = ' '.join('%s=%s' % (k, (('%.4g/%s' % (v['c'], v['over'])) if isinstance(v, dict) else v)) for k, v in bst.items())
            return '%-18s med %.2f [%.2f, %.2f] %d/%d (%s) edge=%s' % (r['variant'], sm['median'], sm['min'], sm['max'], sm['reached'], sm['runs'], p, r.get('edge'))
        print('=== %s b=%s' % (inst, b))
        for v in ('polar', 'polar+N', 'ns5', 'ns5+N', 'frob', 'frob+N', 'sign', 'sign+N', 'SAGA', 'B-SAGA', 'SAGA-Muon (joint)'):
            if v in byv:
                print('   ', desc(byv[v]))
        r_sc = med(spec[s]) / med(ctrl[c])
        lo, hi = boot_ratio(spec[s]['confirm_passes'], ctrl[c]['confirm_passes'])
        print('  fastest spectral %s / fastest control %s = %.3f  boot90 [%.2f, %.2f]   (compare block: %s)' % (
            s, c, r_sc, lo, hi, cell['compare']['ratios'].get('spectral/control')))
        if s10 != s or c10 != c:
            print('  restricted to 10/10 rows: %s / %s = %.3f' % (s10, c10, med(spec[s10]) / med(ctrl[c10])))
        allr['ctrl'][(inst, b)] = r_sc
        # same-momentum comparisons
        print('  no-momentum: polar/frob = %.3f, ns5/frob = %.3f ; with momentum: polar+N/frob+N = %.3f, ns5+N/frob+N = %.3f' % (
            med(byv['polar']) / med(byv['frob']) if np.isfinite(med(byv['frob'])) else float('nan'),
            med(byv['ns5']) / med(byv['frob']) if np.isfinite(med(byv['frob'])) else float('nan'),
            med(byv['polar+N']) / med(byv['frob+N']), med(byv['ns5+N']) / med(byv['frob+N'])))
        allr['polar_frob'][(inst, b)] = med(byv['polar']) / med(byv['frob']) if np.isfinite(med(byv['frob'])) else float('nan')
        if 'SAGA' in byv:
            e = min(('SAGA', 'B-SAGA'), key=lambda v: med(byv[v]))
            r_e = med(spec[s]) / med(byv[e]); r_n = med(spec[s]) / med(byv['SAGA-Muon (joint)'])
            lo, hi = boot_ratio(spec[s]['confirm_passes'], byv[e]['confirm_passes'])
            lo2, hi2 = boot_ratio(spec[s]['confirm_passes'], byv['SAGA-Muon (joint)']['confirm_passes'])
            print('  / best of SAGA,B-SAGA (%s) = %.3f boot90 [%.2f, %.2f] ; / nuclear-scaled SAGA-Muon = %.3f boot90 [%.2f, %.2f]' % (e, r_e, lo, hi, r_n, lo2, hi2))
            allr['eucl'][(inst, b)] = r_e; allr['nuc'][(inst, b)] = r_n
            print('  nuclear-scaled SAGA-Muon / best Euclid = %.3f' % (med(byv['SAGA-Muon (joint)']) / med(byv[e])))
        print('  lr grids:', {k: v for k, v in cell.get('lr_grids', {}).items()})
        if 'part_B' in cell:
            print('  part_B block:', json.dumps(cell['part_B'])[:900])

def rng_(d, bs):
    v = [x for (i, b), x in d.items() if b in bs and np.isfinite(x)]
    return '%.3f - %.3f (n=%d)' % (min(v), max(v), len(v))
print()
print('RANGE b=1   spectral/control      :', rng_(allr['ctrl'], ('1',)), '   paper: 0.92-1.06')
print('RANGE b=16,32 spectral/control    :', rng_(allr['ctrl'], ('16', '32')), '   paper: 0.58-0.93')
print('RANGE b=16,32 spectral/Euclid     :', rng_(allr['eucl'], ('16', '32')), '   paper: 0.41-0.89')
print('RANGE b=16,32 spectral/nuclear    :', rng_(allr['nuc'], ('16', '32')), '   paper: 0.81-1.04')
print('RANGE b=1 polar/frob (part A rows):', rng_(allr['polar_frob'], ('1',)), '   paper: 1.16-1.36')
# with the polar row of part B (finer q grid) over the frob row of part A
B = M9['B']
print()
for inst in A:
    rb = [r for r in B[inst]['rows'].values() if r.get('variant') == 'polar' or 'polar]' in r.get('family', '')]
    fro = [r for r in A[inst]['cells']['1']['rows'].values() if r['variant'] == 'frob'][0]
    pa = [r for r in A[inst]['cells']['1']['rows'].values() if r['variant'] == 'polar'][0]
    for r in rb:
        print('%-9s partB %s (q=%g lr0=%g) med %.2f ; partA polar (q=%g lr0=%g) med %.2f ; partA frob (q=%g lr0=%g) med %.2f ; B/A-frob = %.3f ; A/A-frob = %.3f' % (
            inst, r['family'], r['best']['lr_decay'], r['best']['lr'], med(r), pa['best']['lr_decay'], pa['best']['lr'], med(pa),
            fro['best']['lr_decay'], fro['best']['lr'], med(fro), med(r) / med(fro), med(pa) / med(fro)))
    print('   q grid part A:', A[inst]['q_grid'], ' q grid part B:', B[inst].get('q_grid'), ' lr grid part B:', B[inst].get('lr_grid'))
