"""Write results/bias_law/NUMBERS_twomoment.md from the JSON files behind fig_constants / fig_law / fig_digits."""
from common import *
A = load('fig_constants.json'); B = load('fig_law.json'); C = load('fig_digits.json')
tab = json.load(open(os.path.join(DATA, 'table_constants.json')))
L = []; w = L.append
w('# Numbers behind fig_constants, fig_law, fig_digits (two-moment paper)\n')
w('All "+-" are one Monte-Carlo standard error unless stated. Scripts: `experiments/bias_law/paper_figures/` (figA_*, figB_*, digits_*, figC_plot, make_numbers).\n')
w('## FIG A - constants (exact, de Bruijn Pfaffian formula of experiments/bias_law/a1/wish.py; precision raised until E tr W^2 = Kd(K+d+1) holds to 1e-25)\n')
w(f"Cross-checks: q_(2,2) = {tab[0]['q']:.6f} = (3 sqrt2 - 2)/7 = {(3*2**.5-2)/7:.6f} (16 digits agree; MC 0.3204); q_(3,5) = {[r for r in tab if (r['K'],r['d'])==(3,5)][0]['q']:.6f} (MC 0.8425); "
  "all 15 Monte-Carlo values of results/c02_constants.json agree to their 3-4 digits.\n")
w('| K | d | q | a1 | a2 | a1+a2 | lower 3/(K+d+1) | upper 3/(max+2) | max*(a1+a2) |\n|---|---|---|---|---|---|---|---|---|')
for r in tab:
    w(f"| {r['K']} | {r['d']} | {r['q']:.6g} | {r['a1']:.6g} | {r['a2']:.6g} | {r['a1_plus_a2']:.6g} | {r['lower']:.6g} | {r['upper']:.6g} | {r['scaled']:.6g} |")
w('\n(128,128) NOT computed (interrupted; a single size needs > 10 min). Largest sizes done: square 96x96, d=2K 64x128, d=4K 64x256.\n')
n = sum(len(v) for v in A['series'].values())
ok = all(r['lower'] < r['a1_plus_a2'] < r['upper'] and r['a1'] < 0 < r['a2'] for v in A['series'].values() for r in v)
w(f"a1 < 0 < a2 and 3/(K+d+1) < a1+a2 < 3/(max+2): {'hold' if ok else 'VIOLATED'} at all {n} plotted sizes.\n")
w('| ratio | limit 6 kappa(gamma) | largest size | max*(a1+a2) there | gap to limit |\n|---|---|---|---|---|')
for g, v in A['series'].items():
    r = v[-1]; lim = A['limits'][g]['six_kappa']
    w(f"| d={g}K | {lim:.6f} | {r['K']}x{r['d']} | {r['scaled']:.5f} | {lim - r['scaled']:.4f} |")
w('\nMarchenko-Pastur limits by numerical quadrature (density normalisation checked: ' + ', '.join(f"{v['mp_norm']:.8f}" for v in A['limits'].values()) + '); 12/5 = 2.4 reproduced to 1e-6. Approach to the limit is from below and slow (gap ~ O(1/max)).\n')

w('## FIG B(a) - law across noise laws, 2x3: ratio <b G_b, g*>/|g*|_F^2 (zero of the frozen mean field)\n')
w(f"Blocks of {B['block']:.0e} batch means (1e6 for rank-one laws at b >= 256), zero solved per block, se = block-to-block. Predictions use EXACT cumulants and exact q_(2,3) = {B['consts']['q']:.6f}, "
  f"a1 = {B['consts']['a1']:.5f}, a2 = {B['consts']['a2']:.5f}; Kronecker law: J, T by Monte Carlo (4 x 4e6), |se(g*)|_F = {B['kron_gstar_se_fro']:.4f} on |g*|_F = {np.linalg.norm(B['gstar']['kron']):.3f}.\n")
w('| law | |g*|_F | ' + ' | '.join(f'b={b}' for b in (1, 4, 16, 64, 256, 1024)) + ' | pooled b>=64 |\n|---|---|---|---|---|---|---|---|---|')
for name, rows in B['law'].items():
    d = {r['b']: r for r in rows}
    big = [r for r in rows if r['b'] >= 64]; ww = np.array([1 / r['se'] ** 2 for r in big]); pm = float((ww * [r['ratio'] for r in big]).sum() / ww.sum())
    w(f"| {name} | {np.linalg.norm(B['gstar'][name]):.4f} | " + ' | '.join(f"{d[b]['ratio']:.3f}+-{d[b]['se']:.3f}" if b in d else '-' for b in (1, 4, 16, 64, 256, 1024))
      + f" | {pm:.3f}+-{ww.sum() ** -0.5:.3f} |")
w('\nCosine between b G_b and g* at b = 1024: ' + ', '.join(f"{k} {v[-1]['cos']:.4f}" for k, v in B['law'].items()) + '.\n')
w('Not as predicted / caveats: (i) rank-one laws converge slowly (0.69-0.73 at b=64, 0.89-0.90 at b=256, 0.96-0.99 at b=1024) and the isotropic rank-one law has the WRONG SIGN at b=4 (-0.021+-0.0004); '
  '(ii) "one skewed factor" plateaus at 0.94 (pooled b>=16: 0.939+-0.017, about 3.5 se below 1) with block-to-block scatter across b much smaller than the quoted se (chi2 = 0.02 on 3 dof) - unresolved; '
  '(iii) i.i.d. law sits at 0.990+-0.003 for b>=64 (3 se below 1). (ii)-(iii) are percent-level and may be statistical, but they were not tuned away.\n')
w('## FIG B(b) - momentum (frozen stationary filtered noise, i.i.d. Exp(1)-1 entries, 2x3), g* per entry = %.5f\n' % B['gstar']['iid'][0][0])
w('| beta | b | b_eff | G b_eff/g* (filled) | G b(1+beta)/(1-beta)/g* (hollow) | samples |\n|---|---|---|---|---|---|')
for r in sorted(B['momentum'], key=lambda r: (r['beta'], r['b'])):
    w(f"| {r['beta']} | {r['b']} | {r['b_eff']:.2f} | {r['ratio_beff']:.4f}+-{r['ratio_beff_se']:.4f} | {r['ratio_bvar']:.4f}+-{r['ratio_bvar_se']:.4f} | {r['samples']:.0e} |")
w('\nSkewness-effective normalisation gives 1 within ~1% for b >= 4 (0.92-0.99 at b = 1, a finite-b effect). The variance-effective normalisation gives (1+beta)^2/(1+beta+beta^2) = 1.286 (0.5), 1.332 (0.9), 1.333 (0.95), '
  'i.e. ABOVE 1 (the brief expected "0.75-0.78": that is the reciprocal, prediction/measurement). Consistent with results/c04_frozen.json (0.917, 0.997, 0.974, 1.017, 1.008).\n')
w('## FIG B(c) - sign-blind co-skewness law: b (G_b)_11, prediction polar %.5f, sign descent 0\n' % B['signblind_pred'])
w('| b | polar | sign (-b median) |\n|---|---|---|')
for r in B['signblind']:
    w(f"| {r['b']} | {r['polar']:.4f}+-{r['polar_se']:.4f} | {r['sign']:.4f}+-{r['sign_se']:.4f} |")
w('\nSign step: -0.057 at b=1, -0.021 at b=4, -0.0044+-0.0009 at b=16 (4.7 se: a higher-order residue decaying faster than 1/b), compatible with 0 for b >= 64; polar step within 1-1.6 se of the prediction for b >= 16.\n')

w('## FIG C - digits least squares (real data), prediction (5), no fitted parameter\n')
w(f"Instance: sklearn load_digits, n = {C['n']}, d = {C['d']} leading principal components of standardised pixels (centred, not whitened), K = {C['K']} fixed random orthonormal projections "
  f"(default_rng(0)) of the centred one-hot labels, lam = {C['lam']}; N = {C['K']*C['d']}. F(0)-F* = {C['F0']-C['Fstar']:.4f}. Hessian eigenvalues {min(C['Hx_eigs']):.2f}..{max(C['Hx_eigs']):.2f}.")
w('WHY NOT one-hot: K=10 gives a singular gradient covariance (16-dim kernel, min eig -2e-16); K=9 (drop a class) is nondegenerate but eig ratio 1.3e-4 and g*^T Sigma^-1 g* ~ 90 (pilot, 4e5 samples), '
  'so b <= 128 is far from first order; the K=4 projection was chosen BEFORE any simulation on the criterion g*^T Sigma^-1 g* (26 here).')
w(f"Gradient covariance at W*: eig min/max = {C['cov_eig_min']:.3e}/{C['cov_eig_max']:.3f} = {C['cov_eig_min']/C['cov_eig_max']:.2e}.")
w(f"Prediction ({C['gstar_seeds']} independent MC evaluations of J, T with {C['gstar_S']:.0e} Gaussian samples each, linear control variate): ||g*||_* = {C['pred_nuc']:.4f} +- {C['pred_nuc_se']:.4f} "
  f"(sd over seeds {C['pred_nuc_seed_sd']:.4f} = {100*C['pred_nuc_seed_sd']/C['pred_nuc']:.2f}%); |g*|_F = {C['pred_fro']:.4f} (seed sd {C['pred_fro_seed_sd']:.4f}); "
  f"<g*,H^-1 g*>/2 = {C['pred_excess_coef']:.4f} +- {C['pred_excess_coef_se']:.4f} (i.e. <g*,H^-1 g*> = {2*C['pred_excess_coef']:.4f} +- {2*C['pred_excess_coef_se']:.4f}); g*^T Sigma^-1 g* = {C['mahalanobis']:.1f}.\n")
r0 = C['rows'][0]
w(f"Runs: batches with replacement, exact polar (batched SVD), eta_t = {r0['eta0']} (1+t/{r0['t0']:.0f})^-0.75, start W = 0, tail = last third split in two halves; R and T in the table. se = jackknife over runs.\n")
w('| beta | b | b_eff | R | T | b_eff ||grad F||_* | ratio to ||g*||_* | halves (1st/2nd) | excess loss | predicted | ratio | halves ratio 2nd/1st | cos | <b_eff G, g*>/|g*|^2 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in C['rows']:
    w(f"| {a['beta']} | {a['b']} | {a['b_eff']:.1f} | {a['R']} | {a['T']:.0e} | {a['scaled_nuc']:.3f}+-{a['scaled_nuc_se']:.3f} | {a['scaled_nuc']/C['pred_nuc']:.3f} | {a['half1']['scaled_nuc']:.3f}/{a['half2']['scaled_nuc']:.3f} | "
      f"{a['excess']:.3e}+-{a['excess_se']:.1e} | {a['excess_pred']:.3e} | {a['excess']/a['excess_pred']:.3f} | {a['half2']['excess']/a['half1']['excess']:.3f} | {a['cos']:.4f}+-{a['cos_se']:.4f} | {a['proj']:.3f}+-{a['proj_se']:.3f} |")
w('\nEquilibration: first and second half of the tail agree within the jackknife se for beta = 0 (differences <= 0.8% in the norm, <= 1.6% in the loss) and within ~2 se for beta = 0.9 '
  '(largest: b=64, 4.82 vs 4.56, 5.5% in norm, 12% in loss, se of each half ~0.09) - no systematic drift direction. Noise-debiased excess loss (U-statistic over runs) differs from the plain one by < 0.4% everywhere.')
for e in C['eta_check']:
    w(f"Step-size check (eta0/4 = {e['eta0']}, same batch seeds): beta={e['beta']}, b={e['b']}: b_eff||grad||_* = {e['scaled_nuc']:.3f}+-{e['scaled_nuc_se']:.3f} vs {e['scaled_nuc_main']:.3f}+-{e['scaled_nuc_main_se']:.3f}; excess {e['excess']:.3e} vs {e['excess_main']:.3e}.")
w('\nNOT AS PREDICTED: the direction is right (cos 0.89 -> 0.983 for beta=0, 0.986 -> 0.996 for beta=0.9) but the magnitude has not converged. beta = 0: b ||grad F||_* rises from 0.45 to 1.94 times ||g*||_* at b = 128 '
  '(7.36 vs 3.80) and comes back to 1.47 at b = 512; excess loss is 0.16x (b=4) to 4.8x (b=128) the prediction, 2.6x at b=512. beta = 0.9: 0.95, 1.12, 1.26, 1.29, 1.23, 1.15 times ||g*||_* for b = 4..128 '
  '(non-monotone, peak at b_eff ~ 456, decreasing towards the prediction); excess loss 0.93x-1.86x. The two beta do not collapse on b_eff at fixed b_eff (7.36 vs 4.26 near b_eff ~ 120), so the excess is a higher-order (finite-b) '
  'effect with a different momentum dependence, not a wrong b_eff; it is equilibrated and for beta=0 insensitive to the step size (b=128: identical at eta0/4). For beta=0.9 there is a residual step-size effect: at b=16, eta0/4 lowers the value by 3% (4.63 vs 4.77; same random batches, so the difference is real), i.e. towards the prediction but far from closing the 26% gap. '
  'A b = 2048 run was interrupted and is not available, so convergence of the beta = 0 curve to ||g*||_* is NOT demonstrated, only its turn-over.\n')
w('## Not done / compute notes\n')
w('(128,128) constants and sizes with min(K,d) > 96 (square) not computed; digits b = 2048 (beta=0) interrupted at 50%; no b > 128 for beta = 0.9. '
  'Total compute exceeded the 45-minute budget (two machine crashes, shared machine); the last sequential law job peaked at ~5 GB RSS.\n')
open(os.path.join(RES, 'NUMBERS_twomoment.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
