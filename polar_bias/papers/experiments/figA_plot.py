"""FIG A: max(K,d)(a1+a2) against max(K,d) for three aspect ratios, proved envelope, Marchenko-Pastur limits."""
from common import *
from figA_compute import SIZES, TABLE, kappa
plt = style()
# assemble from the per-size cache written by figA_compute.py (sizes not finished there are simply absent)
rows = {(r['K'], r['d']): r for r in map(json.loads, open(os.path.join(DATA, 'fig_constants_rows.jsonl')))}
D = dict(series={g: [rows[s] for s in v if s in rows] for g, v in SIZES.items()},
         limits={g: dict(gamma=1 / float(g), kappa=kappa(1 / float(g))[0], six_kappa=6 * kappa(1 / float(g))[0], mp_norm=kappa(1 / float(g))[1]) for g in SIZES})
dump(D, 'fig_constants.json')
sig = lambda x: float(f"{x:.6g}")
tab = [{k: (sig(v) if isinstance(v, float) else v) for k, v in rows[s].items() if k in ('K', 'd', 'q', 'a1', 'a2', 'a1_plus_a2', 'lower', 'upper', 'scaled')}
       for s in TABLE if s in rows]
json.dump(tab, open(os.path.join(FIG, 'table_constants.json'), 'w'), indent=1)
print('table sizes missing:', [s for s in TABLE if s not in rows], ' limits:', {g: round(v['six_kappa'], 6) for g, v in D['limits'].items()})
fig, ax = plt.subplots(figsize=(3.25, 2.4))
lab = {'1': r'$d=K$', '2': r'$d=2K$', '4': r'$d=4K$'}; col = {'1': OI['blue'], '2': OI['verm'], '4': OI['green']}
mk = {'1': 'o', '2': 's', '4': '^'}
for g in ('1', '2', '4'):
    r = D['series'][g]; x = np.array([max(a['K'], a['d']) for a in r], float)
    lo = np.array([a['scaled_lower'] for a in r]); hi = np.array([a['scaled_upper'] for a in r])
    if g == '1':
        ax.fill_between(x, lo, hi, color=col[g], alpha=0.13, lw=0, label='proved bounds ($d=K$)')
    ax.plot(x, [a['scaled'] for a in r], color=col[g], marker=mk[g], label=lab[g], zorder=3)
    ax.axhline(D['limits'][g]['six_kappa'], color=col[g], ls=':', lw=0.8, zorder=1)
    ax.text(330, D['limits'][g]['six_kappa'], ('$12/5$' if g == '1' else rf"$6\kappa(1/{g})$"), color=col[g], fontsize=7, va='center', ha='left')
xu = np.geomspace(2, 256, 100); ax.plot(xu, 3 * xu / (xu + 2), color='0.45', lw=0.6, zorder=1, label=r'upper bound (all $K,d$)')
ax.set_xscale('log', base=2); ax.set_xlim(1.7, 300)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128', '256'])
ax.set_xlabel(r'$\max(K,d)$'); ax.set_ylabel(r'$\max(K,d)\,(a_1+a_2)$')
ax.legend(loc='lower right', handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
save(fig, 'fig_constants')
