"""FIG B: (a) first-order law across noise laws, (b) momentum effective batch, (c) sign-blind law: polar vs sign descent."""
from common import *
plt = style()
D = load('fig_law_pred.json'); raw = {}
for l in open(os.path.join(DATA, 'fig_law_blocks.jsonl')):                # per-block results written by figB_compute.py
    r = json.loads(l); raw.setdefault(r['key'], {})[r['blk']] = r         # (a block recomputed after an interruption counts once)
raw = {k: list(v.values()) for k, v in raw.items()}
gst = {k: np.array(v) for k, v in D['gstar'].items()}
ms = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))))
out = dict(consts=D['consts'], gstar=D['gstar'], kron_gstar_se_fro=D['kron_gstar_se'], block=D['block'], law={}, momentum=[], signblind=[])

fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.2), gridspec_kw=dict(wspace=0.36))
# ---- (a) ----------------------------------------------------------------------------------------------------------
ax = axs[0]
LAWS = [('iid', 'i.i.d. entries', 'o'), ('coskew', 'sign-blind co-skew', 's'), ('factor', 'one skewed factor', '^'),
        ('rank1', 'rank-one, isotropic', 'D'), ('kron', 'rank-one, Kronecker', 'v')]
for (name, lab, mk), c in zip(LAWS, COLS):
    g = gst[name].ravel(); rows = []
    for key in sorted([k for k in raw if k.startswith(f'law|{name}|')], key=lambda k: int(k.split('|')[2])):
        b = int(key.split('|')[2]); V = np.array([r['v'] for r in raw[key]])
        m, se = ms(V @ g / (g @ g))
        cosv = float(V.mean(0) @ g / np.linalg.norm(V.mean(0)) / np.linalg.norm(g))
        rows.append(dict(b=b, ratio=m, se=se, blocks=len(V), cos=cosv, bG=V.mean(0).tolist(), max_resid=max(r['resid'] for r in raw[key])))
    out['law'][name] = rows
    ax.errorbar([r['b'] for r in rows], [r['ratio'] for r in rows], [r['se'] for r in rows], color=c, marker=mk, label=lab, elinewidth=0.7)
ax.axhline(1, color='k', ls='--', lw=0.8, zorder=0)
ax.set_xscale('log', base=4); ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
ax.set_xlabel('$b$'); ax.set_ylabel(r'$\langle b\,G_b,\,g_*\rangle/\|g_*\|_F^2$')
ax.set_ylim(-0.85, 1.3); ax.set_yticks([0, 0.5, 1.0]); ax.legend(loc='lower right', handlelength=1.5, labelspacing=0.25, borderaxespad=0.1)
tag(ax, '(a)')
# ---- (b) ----------------------------------------------------------------------------------------------------------
ax = axs[1]; gper = float(gst['iid'][0, 0]); betas = [0.0, 0.5, 0.9, 0.95]
for j, (b, c, mk) in enumerate(zip([1, 4, 16], COLS, ['o', 's', '^'])):
    xs, f, fe, h, he = [], [], [], [], []
    for i, beta in enumerate(betas):
        key = f'mom|{beta}|{b}'
        if key not in raw:
            continue
        m, se = ms(np.array([r['v'] for r in raw[key]]).mean(1))
        beff = b * (1 + beta + beta ** 2) / (1 - beta ** 2); bvar = b * (1 + beta) / (1 - beta)
        out['momentum'].append(dict(beta=beta, b=b, b_eff=beff, b_var=bvar, G_mean_entry=m, se=se, ratio_beff=m * beff / gper, ratio_beff_se=se * beff / gper,
                                    ratio_bvar=m * bvar / gper, ratio_bvar_se=se * bvar / gper, blocks=len(raw[key]), samples=len(raw[key]) * 1_000_000))
        xs.append(i + 0.18 * (j - 1)); f.append(m * beff / gper); fe.append(se * beff / gper); h.append(m * bvar / gper); he.append(se * bvar / gper)
    ax.errorbar(xs, f, fe, color=c, marker=mk, ls='none', label=f'$b={b}$', elinewidth=0.7)
    ax.errorbar(xs[1:], h[1:], he[1:], color=c, marker=mk, ls='none', mfc='white', mew=0.8, elinewidth=0.7)
bb = np.linspace(0, 0.95, 200); xx = np.interp(bb, betas, range(4))
ax.plot(xx, (1 + bb) ** 2 / (1 + bb + bb ** 2), color='0.55', lw=0.7, ls=':', zorder=0)
ax.axhline(1, color='k', ls='--', lw=0.8, zorder=0)
ax.set_xticks(range(4)); ax.set_xticklabels(['0', '0.5', '0.9', '0.95']); ax.set_xlim(-0.5, 3.5)
ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\bar b\;G_{b,\beta}\,/\,g_*$')
ax.set_ylim(0.8, 1.52)
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.5), handlelength=1.0, labelspacing=0.25, borderaxespad=0.1, handletextpad=0.3)
ax.text(3.45, 0.855, r'filled: $\bar b=b_{\mathrm{eff}}$', fontsize=7, ha='right', va='bottom')
ax.text(3.45, 1.42, r'hollow: $\bar b=b\,(1+\beta)/(1-\beta)$', fontsize=7, ha='right', va='bottom')
tag(ax, '(b)')
# ---- (c) ----------------------------------------------------------------------------------------------------------
ax = axs[2]; rows = []
for key in sorted([k for k in raw if k.startswith('law|coskew|')], key=lambda k: int(k.split('|')[2])):
    b = int(key.split('|')[2]); P = np.array([r['v'][0] for r in raw[key]]); S = np.array([r['sign'][0] for r in raw[key]])
    rows.append(dict(b=b, polar=ms(P)[0], polar_se=ms(P)[1], sign=ms(S)[0], sign_se=ms(S)[1]))
out['signblind'] = rows; g11 = float(gst['coskew'][0, 0]); out['signblind_pred'] = g11
bs = [r['b'] for r in rows]
ax.errorbar(bs, [r['polar'] for r in rows], [r['polar_se'] for r in rows], color=OI['blue'], marker='o', label='polar step', elinewidth=0.7)
ax.errorbar(bs, [r['sign'] for r in rows], [r['sign_se'] for r in rows], color=OI['verm'], marker='s', label='sign step', elinewidth=0.7)
ax.axhline(g11, color='k', ls='--', lw=0.8, zorder=0); ax.axhline(0, color='k', ls='--', lw=0.8, zorder=0)
ax.set_xscale('log', base=4); ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
ax.set_xlabel('$b$'); ax.set_ylabel(r'$b\,(G_b)_{11}$')
ax.set_ylim(-0.068, 0.108); ax.text(1100, g11 + 0.003, r'$(g_*)_{11}$', fontsize=7, ha='right', va='bottom')
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.6), handlelength=1.5, labelspacing=0.25)
tag(ax, '(c)')
save(fig, 'fig_law'); dump(out, 'fig_law.json')
for n in out['law']:
    print(n, [(r['b'], round(r['ratio'], 4), round(r['se'], 4), r['blocks']) for r in out['law'][n]])
for r in out['momentum']:
    print(r['beta'], r['b'], f"{r['ratio_beff']:.4f}+-{r['ratio_beff_se']:.4f}  var-norm {r['ratio_bvar']:.4f}")
for r in rows:
    print(r)
print('max resid', max(x['resid'] for v in raw.values() for x in v))
