"""FIG C (final): digits least-squares instance, no fitted parameter.  Combines the runs of digits_sim.py (b <= 512, both beta),
the large-batch runs of digits_state.py (beta = 0, multinomial sampling) and the frozen-noise zeros of digits_frozen.py."""
from common import *
plt = style()
D = load('fig_digits.json')
rd = lambda f: [json.loads(l) for l in open(os.path.join(DATA, f))]
fro = sorted(rd('digits_frozen.jsonl'), key=lambda r: r['b']); fro = [r for r in fro if r['T'] >= 40000 or r['b'] == 128]
sta = sorted(rd('digits_state.jsonl'), key=lambda r: r['b']); sta = [r for r in sta if r['b'] > 512]
pE = D['pred_excess_coef']
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.25), gridspec_kw=dict(wspace=0.36))
r0 = [x for x in D['rows'] if x['beta'] == 0.0]; r9 = [x for x in D['rows'] if x['beta'] == 0.9]
ax = axs[0]
ax.errorbar([r['b'] for r in fro], [r['proj_mean'] for r in fro], [r['proj_se'] for r in fro], color=OI['green'], marker='^', label='frozen noise', elinewidth=0.7)
ax.errorbar([x['b_eff'] for x in r0] + [r['b'] for r in sta], [x['proj'] for x in r0] + [r['all']['proj'] for r in sta],
            [x['proj_se'] for x in r0] + [r['proj_se'] for r in sta], color=OI['blue'], marker='o', label=r'iteration, $\beta=0$', elinewidth=0.7)
ax.errorbar([x['b_eff'] for x in r9], [x['proj'] for x in r9], [x['proj_se'] for x in r9], color=OI['verm'], marker='s', label=r'iteration, $\beta=0.9$', elinewidth=0.7)
ax.axhline(1, color='k', ls='--', lw=0.8, zorder=0)
ax.set_ylabel(r'$b_{\mathrm{eff}}\,\langle\nabla F(W_b),g_*\rangle/\|g_*\|_F^2$'); ax.set_ylim(0, 2.45)
ax.legend(loc='upper right', handlelength=1.6, labelspacing=0.25)
ax = axs[1]
ax.plot([x['b_eff'] for x in r0] + [r['b'] for r in sta], [x['excess'] for x in r0] + [r['all']['loss_ratio'] * pE / r['b'] ** 2 for r in sta], color=OI['blue'], marker='o')
ax.plot([x['b_eff'] for x in r9], [x['excess'] for x in r9], color=OI['verm'], marker='s')
xx = np.array([3, 5e4]); ax.plot(xx, pE / xx ** 2, color='k', ls='--', lw=0.8, zorder=0, label=r'$\langle g_*,H^{-1}g_*\rangle/(2b_{\mathrm{eff}}^2)$')
ax.set_yscale('log'); ax.set_ylabel(r'$F(W_b)-F_*$'); ax.legend(loc='upper right', handlelength=1.6)
ax = axs[2]
ax.plot([r['b'] for r in fro], [r['cos'] for r in fro], color=OI['green'], marker='^')
ax.plot([x['b_eff'] for x in r0] + [r['b'] for r in sta], [x['cos'] for x in r0] + [r['all']['cos_W'] for r in sta], color=OI['blue'], marker='o')
ax.plot([x['b_eff'] for x in r9], [x['cos'] for x in r9], color=OI['verm'], marker='s')
ax.axhline(1, color='k', ls='--', lw=0.8, zorder=0); ax.set_ylabel('cosine with the predicted direction'); ax.set_ylim(0.8, 1.02)
for a, t in zip(axs, ['(a)', '(b)', '(c)']):
    a.set_xscale('log'); a.set_xlabel(r'$b_{\mathrm{eff}}$'); a.set_xlim(3, 5e4)
tag(axs[0], '(a)'); tag(axs[1], '(b)', y=0.12); tag(axs[2], '(c)', y=0.12)
save(fig, 'fig_digits')
dump(dict(frozen=fro, state=sta), 'fig_digits_largeb.json')
for r in fro: print('frozen', r['b'], round(r['proj_mean'], 3), round(r['proj_se'], 3), round(r['cos'], 4))
for r in sta: print('state', r['b'], {k: round(v, 3) for k, v in r['all'].items()}, round(r['proj_se'], 3))
