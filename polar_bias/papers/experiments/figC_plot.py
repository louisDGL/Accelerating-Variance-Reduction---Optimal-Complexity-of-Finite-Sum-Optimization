"""FIG C: real-data check of prediction (5) on the digits least-squares instance (no fitted parameter)."""
from common import *
from digits_problem import *
import glob
plt = style()
p = problem(); K, d, n, Hx, Ws = p['K'], p['d'], p['n'], p['Hx'], p['Ws']; Hi = np.linalg.inv(Hx)
Z = np.load(os.path.join(DATA, 'digits_gstar.npz')); gs = Z['gs']; g = gs.mean(0)
nucf = lambda M: float(np.linalg.svd(M, compute_uv=False).sum())
sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
pred_nuc = nucf(g); pred_nuc_se = sem([nucf(x) for x in gs])
pred_E = 0.5 * float((g @ Hi * g).sum()); pred_E_se = sem([0.5 * (x @ Hi * x).sum() for x in gs])
D0 = g @ Hi                                                       # predicted b (W_b - W_*)
V = per_sample_grads(p); Sig = V.T @ V / n
out = dict(K=K, d=d, n=n, lam=LAM, pred_nuc=pred_nuc, pred_nuc_se=pred_nuc_se, pred_fro=float(np.linalg.norm(g)),
           pred_fro_seed_sd=float(np.std([np.linalg.norm(x) for x in gs], ddof=1)), pred_nuc_seed_sd=float(np.std([nucf(x) for x in gs], ddof=1)),
           pred_excess_coef=pred_E, pred_excess_coef_se=pred_E_se, gstar_S=int(Z['S']), gstar_seeds=len(gs),
           cov_eig_min=float(Z['cov_eigs'][0]), cov_eig_max=float(Z['cov_eigs'][-1]), mahalanobis=float(g.ravel() @ np.linalg.solve(Sig, g.ravel())),
           Fstar=float(F(p, Ws)), F0=float(F(p, 0 * Ws)), Hx_eigs=np.linalg.eigvalsh(Hx).tolist(), gstar=g.tolist(), rows=[])


def stats(Dl, beff):
    """Dl: (R,K,d) per-run tail-averaged W - W_*.  Returns the plotted statistics of the run-averaged point."""
    m = Dl.mean(0); G = m @ Hx
    return np.array([beff * nucf(G), 0.5 * float((G * m).sum()), float((m * D0).sum() / np.linalg.norm(m) / np.linalg.norm(D0)),
                     beff * float((G * g).sum() / (g * g).sum())])


def jack(Dl, beff):
    R = len(Dl); full = stats(Dl, beff)
    js = np.array([stats(np.delete(Dl, r, 0), beff) for r in range(R)])
    return full, np.sqrt((R - 1) / R * ((js - js.mean(0)) ** 2).sum(0))


for f in sorted(glob.glob(os.path.join(DATA, 'digits_runs', 'main_*.npz')), key=lambda f: (float(np.load(f)['beta']), int(np.load(f)['b']))):
    z = np.load(f); beta, b, R = float(z['beta']), int(z['b']), int(z['R'])
    beff = b * (1 + beta + beta ** 2) / (1 - beta ** 2)
    D1, D2 = z['W1'] - Ws, z['W2'] - Ws; Dl = (D1 + D2) / 2
    full, se = jack(Dl, beff); h1 = stats(D1, beff); h2 = stats(D2, beff)
    tot = Dl.sum(0); ustat = 0.5 * (float((tot @ Hx * tot).sum()) - float(((Dl @ Hx) * Dl).sum())) / (R * (R - 1))   # unbiased for the limit point
    out['rows'].append(dict(beta=beta, b=b, b_eff=beff, R=R, T=int(z['T']), eta0=float(z['eta0']), t0=float(z['t0']), eta_end=float(z['eta_end']),
                            seconds=float(z['seconds']), scaled_nuc=full[0], scaled_nuc_se=se[0], excess=full[1], excess_se=se[1], cos=full[2], cos_se=se[2],
                            proj=full[3], proj_se=se[3], excess_debiased=ustat, excess_pred=pred_E / beff ** 2,
                            half1=dict(scaled_nuc=h1[0], excess=h1[1], cos=h1[2]), half2=dict(scaled_nuc=h2[0], excess=h2[1], cos=h2[2]),
                            half_shift_over_se=float(abs(h1[0] - h2[0]) / max(se[0], 1e-300)),
                            mahalanobis_bias=out['mahalanobis'] * (b * (1 + beta) / (1 - beta)) / beff ** 2))
# step-size sensitivity (not plotted): same runs with eta0 divided by 4
out['eta_check'] = []
for f in sorted(glob.glob(os.path.join(DATA, 'digits_runs', 'etaquarter_*.npz'))):
    z = np.load(f); beta, b = float(z['beta']), int(z['b']); beff = b * (1 + beta + beta ** 2) / (1 - beta ** 2)
    full, se = jack((z['W1'] + z['W2']) / 2 - Ws, beff)
    ref = [a for a in out['rows'] if a['beta'] == beta and a['b'] == b][0]
    out['eta_check'].append(dict(beta=beta, b=b, eta0=float(z['eta0']), eta_end=float(z['eta_end']), scaled_nuc=full[0], scaled_nuc_se=se[0], excess=full[1],
                                 cos=full[2], scaled_nuc_main=ref['scaled_nuc'], scaled_nuc_main_se=ref['scaled_nuc_se'], excess_main=ref['excess']))
    print('eta check', out['eta_check'][-1])
dump(out, 'fig_digits.json')

fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.3), gridspec_kw=dict(wspace=0.38))
sty = {0.0: (OI['blue'], 'o', r'$\beta=0$'), 0.9: (OI['verm'], 's', r'$\beta=0.9$')}
for beta, (c, mk, lab) in sty.items():
    r = [x for x in out['rows'] if x['beta'] == beta]
    if not r:
        continue
    x = [a['b_eff'] for a in r]
    axs[0].errorbar(x, [a['scaled_nuc'] for a in r], [a['scaled_nuc_se'] for a in r], color=c, marker=mk, label=lab, elinewidth=0.7)
    axs[1].errorbar(x, [a['excess'] for a in r], [a['excess_se'] for a in r], color=c, marker=mk, label=lab, elinewidth=0.7)
    axs[2].errorbar(x, [a['cos'] for a in r], [a['cos_se'] for a in r], color=c, marker=mk, label=lab, elinewidth=0.7)
xx = np.array([3, 2600.0])
axs[0].axhline(pred_nuc, color='k', ls='--', lw=0.8, zorder=0, label=r'$\|g_*\|_*$')
axs[0].set_ylabel(r'$b_{\mathrm{eff}}\,\|\nabla F(W_b)\|_*$'); axs[0].set_ylim(0, None)
axs[1].plot(xx, pred_E / xx ** 2, color='k', ls='--', lw=0.8, zorder=0, label=r'$\langle g_*,H^{-1}g_*\rangle/(2b_{\mathrm{eff}}^2)$')
axs[1].set_yscale('log'); axs[1].set_ylabel(r'$F(W_b)-F_*$')
axs[2].axhline(1, color='k', ls='--', lw=0.8, zorder=0); axs[2].set_ylabel(r'$\cos\angle\,(W_b-W_*,\;H^{-1}g_*)$'); axs[2].set_ylim(0.8, 1.02)
for ax, t in zip(axs, ['(a)', '(b)', '(c)']):
    ax.set_xscale('log'); ax.set_xlabel(r'$b_{\mathrm{eff}}$'); ax.set_xlim(3, 2600)
axs[0].legend(loc='lower right', handlelength=1.6, labelspacing=0.25); axs[1].legend(loc='upper right', handlelength=1.6, labelspacing=0.25)
axs[2].legend(loc='lower right', handlelength=1.6, labelspacing=0.25)
tag(axs[0], '(a)'); tag(axs[1], '(b)', x=0.03, y=0.12); tag(axs[2], '(c)', x=0.03, y=0.12)
save(fig, 'fig_digits')
print(f"pred ||g*||_* = {pred_nuc:.4f} +- {pred_nuc_se:.4f} (seed sd {out['pred_nuc_seed_sd']:.4f}); |g*|_F = {out['pred_fro']:.4f} (seed sd {out['pred_fro_seed_sd']:.4f}); "
      f"<g*,H^-1 g*>/2 = {pred_E:.4f} +- {pred_E_se:.4f}; g*'Sig^-1 g* = {out['mahalanobis']:.2f}")
for a in out['rows']:
    print(f"beta={a['beta']} b={a['b']:3d} beff={a['b_eff']:7.1f}  beff|G|_*={a['scaled_nuc']:.3f}+-{a['scaled_nuc_se']:.3f} (halves {a['half1']['scaled_nuc']:.3f}/{a['half2']['scaled_nuc']:.3f})"
          f"  proj={a['proj']:.3f}+-{a['proj_se']:.3f}  excess={a['excess']:.3e}+-{a['excess_se']:.1e} (debiased {a['excess_debiased']:.3e}; pred {a['excess_pred']:.3e};"
          f" halves {a['half1']['excess']:.3e}/{a['half2']['excess']:.3e})  cos={a['cos']:.3f}+-{a['cos_se']:.3f}  eta_end={a['eta_end']:.1e} {a['seconds']:.0f}s")
