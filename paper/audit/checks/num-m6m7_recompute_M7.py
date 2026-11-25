"""Audit num-m6m7: recompute the (M7) numbers from part D of M9_summary.json and from M9b_summary.json:
rho_star (at least 4 of 5 seeds on schedule), fastest rho where every seed reaches 1e-9, passes, savings of a
fresh permutation per pass, ratios instances / model, x_star of part L.
Run from the project root: python3 paper/audit/checks/num-m6m7_recompute_M7.py
"""
import json, math, statistics as st
M9 = json.load(open('muon/results/M9_summary.json'))
M9b = json.load(open('muon/results/M9b_summary.json'))

def rho_star(by_rho, quorum=0.8):
    ok = [e['rho'] for e in by_rho if e['on_schedule'] >= quorum * e['runs']]
    return min(ok) if ok else None
def rho_star_all(by_rho):
    ok = [e['rho'] for e in by_rho if e['on_schedule'] == e['runs']]
    return min(ok) if ok else None
def fastest(by_rho):
    best = None
    for e in by_rho:
        if e['reached'] == e['runs']:
            m = st.median(e['passes_to'])
            if best is None or m < best[1]:
                best = (e['rho'], m, min(e['passes_to']), max(e['passes_to']), e['on_schedule'])
    return best

res = {}
for inst in ('soft-mem', 'ls-mem', 'soft-cond'):
    D = M9['D'][inst]
    print('=== %s (%s) seeds per cell: %s ; rho grid %s' % (inst, D['role'], D['cells'][list(D['cells'])[0]]['seeds'], D['rho_grid']))
    for b in sorted(D['cells'], key=int):
        sch = D['cells'][b]['schemes']
        row = {}
        for name in ('iid', 'reshuffle', 'shuffle_once'):
            s = sch[name]
            rs = rho_star(s['by_rho']); ra = rho_star_all(s['by_rho']); f = fastest(s['by_rho'])
            row[name] = (rs, ra, f)
            assert rs == s['rho_star'], (rs, s['rho_star'])
            assert abs(f[1] - s['fastest']['median']) < 1e-9 and f[0] == s['fastest']['rho'], (f, s['fastest'])
            print('  b=%-2s %-12s memory factor %s rho_star(4/5) %s  rho_star(5/5) %s  monotone %s | fastest rho %s: %.1f [%.1f, %.1f] passes (seeds on schedule there: %d/5)' % (
                b, name, ('%.3f' % s['memory_factor']) if s.get('memory_factor') else '  -  ', rs, ra, s['rho_star_monotone'], f[0], f[1], f[2], f[3], f[4]))
        i_, r_ = row['iid'][2], row['reshuffle'][2]
        print('      permutation vs independent: passes %.1f -> %.1f : saving %.1f %% ; rho_star %s -> %s' % (i_[1], r_[1], 100 * (1 - r_[1] / i_[1]), row['iid'][0], row['reshuffle'][0]))
        res[(inst, b)] = (i_[1], r_[1], row['iid'][0], row['reshuffle'][0])
print()
for bs, lab in ((('20', '50'), 'b = 20, 50'), (('5', '20', '50'), 'b = 5..50'), (('1', '5'), 'b = 1, 5'), (('1', '5', '20', '50'), 'all b')):
    sel = [(k, v) for k, v in res.items() if k[0] in ('soft-mem', 'ls-mem') and k[1] in bs]
    sav = [100 * (1 - v[1] / v[0]) for k, v in sel]
    print('%-10s memory-limited: rho_star iid %.3g-%.3g ; rho_star permutation %.3g-%.3g ; passes iid %.1f-%.1f ; permutation %.1f-%.1f ; saving %.1f-%.1f %%' % (
        lab, min(v[2] for k, v in sel), max(v[2] for k, v in sel), min(v[3] for k, v in sel), max(v[3] for k, v in sel),
        min(v[0] for k, v in sel), max(v[0] for k, v in sel), min(v[1] for k, v in sel), max(v[1] for k, v in sel), min(sav), max(sav)))
print('Paper: rho_star 0.27-0.43 (b = 5..50, independent) ; 0.10-0.18 (b = 20, 50, permutation) ; passes 16.7-20.0 -> 9.1-12.2 ; 36-49 %')

print()
print('=== M9b part D (model, n = %s, K = %s)' % (M9b['D']['spec'].get('n'), M9b['D']['spec'].get('K')))
Dm = M9b['D']['cells']
for b in sorted(Dm, key=int):
    sch = Dm[b]['schemes']
    out = {}
    for name in ('iid', 'reshuffle'):
        s = sch[name]
        f = fastest(s['by_rho'])
        out[name] = (rho_star(s['by_rho']), f)
        print('  model b=%-2s %-10s rho_star %s | fastest rho %s: %.1f [%.1f, %.1f] passes' % (b, name, out[name][0], f[0], f[1], f[2], f[3]))
    for inst in ('soft-mem', 'ls-mem'):
        i_, r_ = res[(inst, b)][0], res[(inst, b)][1]
        print('      %-8s / model: independent %.2f ; permutation %.2f' % (inst, i_ / out['iid'][1][1], r_ / out['reshuffle'][1][1]))
    print('      model: permutation vs independent: passes %.1f -> %.1f (%+.0f %%), rho_star %s -> %s' % (
        out['iid'][1][1], out['reshuffle'][1][1], 100 * (out['reshuffle'][1][1] / out['iid'][1][1] - 1), out['iid'][0], out['reshuffle'][0]))
print('Paper: instances need 1.7-2.7 x the passes of the model (independent), 1.0-1.3 x with a permutation at b = 20, 50')

print()
L = M9b['L']
print('=== M9b part L: K=%s passes=%s runs=%s x_grid=%s x_ref=%.3f' % (L['K'], L['passes'], L['runs'], L['x_grid'], L['x_ref']))
for nk, cell in L['cells'].items():
    print(' ', nk, 'keys:', list(cell.keys()))
    for order, o in cell['orders'].items():
        print('    %-28s' % order, {k: (v if not isinstance(v, (dict, list)) else '...') for k, v in o.items()})
        for hk in ('level_half', 'level_end'):
            if hk in o:
                h = o[hk]
                print('        %s: %s' % (hk, {k: (v if not isinstance(v, (dict, list)) else (v if len(str(v)) < 200 else '...')) for k, v in h.items()}))
    print('    linear_saga:', cell.get('linear_saga'))
print('Paper: recipe follows 1.2-1.3 nats (independent) and 1.7-1.9 (permutation); linear SAGA single runs about 1.6 (n = 1000)')
