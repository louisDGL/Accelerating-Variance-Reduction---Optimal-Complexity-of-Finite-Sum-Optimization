# Novelty ledger

Date: 2026-11-25. Sources: the manuscript `paper/paper_saga_to_muon.tex`, the archived notes
`archive/muon/notes_spectral_theory.tex`, `muon/notes_theory.md`, `muon/literature_check.md`, and the targeted search
recorded in `polar_bias/literature_agent_report.md` (23 queries, about thirty sources opened, the five 2026 arXiv papers closest
to the topic read as PDFs). Not finding a result is not a proof that it does not exist; the limits are listed at the end.

## 1. Already in the manuscript or the repository (starting point, not claimed)

| Item | Where | Status there |
|---|---|---|
| Stochastic polar steps = stochastic mirror descent on the smoothed nuclear norm; limit gradient in minus the nuclear median | manuscript, Theorem 2 | proved (additive i.i.d. noise, no momentum) |
| Offset for i.i.d. entries: `kappa_3/(6 sigma^2 b) * a/c * 11^T` | manuscript, Prop. 4 | formal |
| Operator form `G_b = -(1/b) J^{-1} T`, `T = (1/6) E[polar(Z) <kappa_3, He_3(Z)>]`, general covariance | archived notes, Prop. "size of the offset" (b) | formal |
| `a/c = 3/(d+2)` for K = 1; `a = -E[||Z||_* He_4(Z_11)]` | archived notes | computed |
| Table of `a/c` (11 sizes), "d a/c between 1 and 2.8" | archived notes, Table | Monte Carlo |
| Momentum factors `(1-beta^2)/(1+beta+beta^2)` and the Nesterov variant | archived notes, same Prop. (c) | scalar, frozen iterate, formal |
| Plateau `b ||grad F(W_b)||_* ~ 0.3` on a finite-sum least-squares problem | manuscript, Fig. 3(c) | measured, unexplained |

## 2. Closest published results

| Reference | Assumptions | Conclusion | What it leaves open |
|---|---|---|---|
| Xie, arXiv:2609.32861 (Sec. 5, App. A, B read) | Gaussian noise | linear response `c = E||Z||_*/(Kd)`; scalar Hermite weights `w_l` of the polar map (`w_3` not computed); bound on the third derivative | no non-Gaussian input, no degree-3 tensor, no bias |
| Chen, Li, Wang, Lu, arXiv:2610.31728 | isotropic Gaussian; two-task model | same linear response; `O(1/B_eff)` displacement of a stationary drift with the variance-based `B_eff = B(1+beta)/(1-beta)` (curvature x variance) | no skewness, no third cumulant |
| Do, Dereich, Jentzen, arXiv:2608.04607 (Prop. 6.2, 6.11, 6.13) | dimension one, quadratic, Lipschitz Newton-Schulz map, EMA momentum, convergence assumed | every L1 limit is a zero of the mean field of the stationary filtered noise; leading asymptotics built from second and third moments of the filter | no convergence proof, no exact sign/polar map (the offset has another order for it), no matrices, no effective batch |
| Schwank, Drton, arXiv:2508.12926 (Thm 3.5) | Euclidean norm, M-dependent coordinates, dimension p -> infinity | geometric median `m_i = mu_i - alpha_i/p + o(1/p)`, `alpha_i = (1/(2 sigma^2)) sum_j E[X_i X_j^2]`; remark: 1/K scaling for medians of means | fixed dimension, other norms, the exact factor `1/(p+2)` |
| Minsker, Strawn, arXiv:2307.03111 | Euclidean | geometric median of the law of a sample mean: `O(n^{-1/2})` bound | no expansion |
| Cornish-Fisher (1938) | scalar | median of a mean `= -kappa_3/(6 sigma^2 b)` | - |
| Paquette et al., arXiv:2605.09552 | - | variance-based effective batch with momentum | third-cumulant effective batch |
| Huang, Pu, Johansson, arXiv:2611.18124 | shuffling, momentum | stationarity rates; failure for fixed momentum on some finite sum | limit point |

## 3. What this note adds (see `paper/polar_bias.pdf`)

| Result | Status | Why it is not a corollary of the above |
|---|---|---|
| Thm 4: `Med_nu(xi_b) = -g*/b + o(1/b)`, `g* = -(1/6) J^{-1} T[K_3]`, any norm, any noise law under Cramer | proved (modulo Bhattacharya-Rao Thm 20.1 as quoted) | the repository has the formula only formally; the literature has the Euclidean case in the regime p -> infinity only. The localization argument (convexity + uniform-in-shift Edgeworth) is new here. Taken alone this is a rigorous version of a known formal statement. |
| **Thm 6: two-moment theorem** (isotropic covariance): reduction to `E[z||z||_F^2]`, `E[z z^T z]`; `a_1, a_2` as rational functions of one number `q_{K,d}`; `m_3 = (K+d-q) m_1`; `a/c = 3(3+2q)/((K+2)(d+2))`; bounds `9/((K+2)(d+2)) < a/c < 3/(max+2)`; `r a_1 + a_2 > 0`; negative fourth derivative of the smoothed nuclear norm; Marchenko-Pastur limit `theta(gamma) in [12/5, 3]` | proved | not found anywhere; replaces a Monte-Carlo table by an identity and proves the "factor of order max(K,d)" suppression with an explicit constant |
| Cor. 9: orthogonalization multiplies the offset of Frobenius normalization by `(3+2q)(Kd+2)/((K+2)(d+2))` in (1, r) for rank-one noise | proved | new prediction |
| Cor. 10: no domination between sign descent and polar steps; explicit sign-blind and polar-blind noise | proved (a_1 != 0 needed for the polar-blind example) | new |
| Thm 12: state-dependent noise, all (approximate) zeros of the mean field at `W* + H^{-1} g*/b + o(1/b)` | proved under a uniform-expansion assumption, verified for models linear in the noise (least squares) | not found |
| Prediction of the plateau of Fig. 3(c): 0.314 against 0.308 measured at b = 32 | numerical, no fitted parameter | new |
| Thm 15: almost-sure convergence of polar steps on an EMA with fixed beta and Robbins-Monro lengths to minus the nuclear median of the filtered noise | proved (small-ball assumption on the noise, exponent theta in (1/2, 1]) | Do et al. assume convergence, in dimension one, for a Lipschitz map |
| Thm 16: bias of a filtered noise = bias at batch `b rho_2/rho_3`; EMA: `b (1+beta+beta^2)/(1-beta^2)` | proved for b -> infinity, fixed filter; the factor itself is in the archived notes (scalar, formal) | the theorem (matrices, any norm, any filter, with proof) and its contrast with the variance-based effective batch |
| Prop. 18: complement duality under sampling without replacement, zero offset at b = n/2, sign change | exact part proved; the factor `(n-2b)/(n-2)` is formal | not found |

## 4. Not established (do not cite as results)

- `a_1 < 0` for every size (verified on 15 sizes and in the proportional limit).
- The finite-sum (discrete noise) case of Thm 4: Cramer's condition fails for a finite data set; weak-Cramer expansions
  (Angst-Poly, Song) may apply but were not checked.
- The regime beta -> 1 at fixed b.
- The exact wording of Bhattacharya-Rao Theorem 20.1 was not read in the book (the non-i.i.d. Theorem 20.6 was read as
  quoted by Angst-Poly); Li-Sun (2002) was read as quoted in arXiv:2507.14940; the Bhattacharya-Ghosh argument for Cramer's
  condition of analytic statistics is cited from memory.

## 5. Limits of the search

No citation-graph search; several sign-momentum papers (Cutkosky-Mehta, Crawshaw et al., Sun et al.) were not opened;
Gotze-Hipp was not opened; the search was run on one day with web search only.
