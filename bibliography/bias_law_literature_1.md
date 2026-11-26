# Novelty check of candidate results C1-C5 (literature agent report, 2026-11-25)

Scope: targeted search for prior work on C1-C5, plus two tool checks (Bhattacharya-Rao Theorem 20.1; half-integer
Wishart / Marchenko-Pastur moments). Read-only on the repo; this file is the only file created.

Reading depth is stated per source with these labels:
- **[PDF-read]**: PDF downloaded from arxiv.org in this session, converted with `pdftotext`, statement read by me.
- **[PDF-scan]**: same download, but only keyword scan plus the passages around the hits.
- **[repo]**: taken from the repo's earlier records, not re-opened.
- **[search]**: seen only in a search-engine summary; not opened. Do not cite on this basis.
- **[memory]**: recalled, not verified in this session.

All five 2026 arXiv identifiers named in the task (2608.04607, 2609.32861, 2610.31728, 2605.09552, 2611.18124) were
fetched successfully as PDFs.

---

## 0. Summary of verdicts

| | Verdict | Most dangerous overlap |
|---|---|---|
| C1 general law, any norm | **PARTIALLY ANTICIPATED** (Euclidean case, in another asymptotic regime; scalar case classical) | Schwank-Drton arXiv:2508.12926 Thm 3.5 + remark on median of means; Minsker-Strawn arXiv:2307.03111 (object `m_n` and O(n^-1/2) bound) |
| C2 spectral closed form | **NOT FOUND** (its Euclidean K=1 shadow, the moment E[xi \|xi\|^2], is in Schwank-Drton) | Schwank-Drton coefficient alpha_i; Chen et al. eq. (111) for the Gaussian linear-response operator |
| C3 momentum | **PARTIALLY ANTICIPATED** | Do-Dereich-Jentzen arXiv:2608.04607 Prop. 6.2 (mean field of the stationary filtered noise, with momentum, 1-D) and Prop. 6.11/6.13 (leading order in M with the filter's 2nd and 3rd moments) |
| C4 without replacement | factor: **FOUND as classical** (finite-population third moment); its use for a sign change of the optimizer bias: **NOT FOUND** | classical survey-sampling formula (not opened in a primary source; verified by enumeration) |
| C5 state-dependent noise | **PARTIALLY ANTICIPATED** (zeroth order and one model-specific 1/B_eff expansion) | Chen-Li-Wang-Lu arXiv:2610.31728 Thm E.1 and App. C two-task expansion of the zero of the stationary drift |

Nothing found states C1 in the generality "any norm, arbitrary third-cumulant tensor, fixed dimension, b -> infinity",
and nothing found gives the spectral closed form of C2.

---

## 1. What the repo records say about the closest papers (and what I re-read)

Repo sources: `paper/paper_saga_to_muon.bib`, `muon/literature_check.md` (lines 218, 254, 379, 381, 1214, 1216),
`paper/audit/evidence/cite-stat.md`, `cite-muon.md`, `novelty.md`.

**Do, Dereich, Jentzen, arXiv:2608.04607v1** ("On MUON optimization: From non-convergence to an error analysis with
Polar Express and the Newton-Schulz polynomial from implementations", 82 p.).
- Repo: Thm 1.2 error bound `E||Theta_n - theta||^2 <= C(1/M + gamma_n)` for matrix parameters with momentum and
  generalized Newton-Schulz (NS); Thm 1.3 / 6.16 non-convergence in dimension one "for almost all M" when
  `E[(X - EX)^3] != 0`; Prop. 6.2: any L^1 limit is a zero of the Muon vector field.
- Re-read by me **[PDF-read]**: Thm 1.3, Prop. 6.2, Prop. 6.13, Cor. 6.14, proof of Prop. 6.11.
  - **Momentum: yes.** Thm 1.3 has `m_n = alpha m_{n-1} + (1-alpha) (1/M) sum_m grad L(Theta_{n-1}, X_{n,m})`,
    `Theta_n = Theta_{n-1} - gamma_n NS(m_n)`, alpha in (0,1), `sum gamma_n = infinity`.
  - **Limits as zeros of a mean field with momentum: yes, in 1-D.** Prop. 6.2, for `L(theta,x) = lambda|theta-x|^2`,
    bounded i.i.d. scalar data, Phi bounded and globally Lipschitz: if `E|Theta_n - vartheta| -> 0` then
    `f(vartheta) = 0` with `f(theta) = E[Phi((1-alpha) sum_{k>=0} alpha^k (1/M) sum_m grad L(theta, X_{k+1,m}))]`
    (eq. 6.7). This is exactly the mean field of the stationary EMA-filtered noise. It is a necessary condition
    (convergence is assumed, not proved), Phi must be Lipschitz (NS polynomial maps; the exact sign is excluded),
    scalar parameter, quadratic loss.
  - **Expansion in M: leading order of the field at the minimizer, not of its zero.** Prop. 6.13:
    `liminf_M M^{3/2} E[NS(-2 lambda (1-alpha) sum_k alpha^k Xbar_k^M)] > 0` for centred data with `E X^3 != 0`,
    alpha in [0,1). Prop. 6.11 proves `lim M^{3/2} E[Z_M |Z_M|]` exists and is nonzero through a characteristic-function
    computation whose limit is `integral of -(A/6) exp(-B t^2/2) dt` with A, B the third and second moments of the
    per-sample filtered variable `Y_1 = sum_k alpha^k X_{k+1,1}` (as I read eqs. 6.159-6.183; the defining display
    6.151 was not read line by line). No expansion of the zero, no effective-batch formula, no statement in terms
    of a median. "Almost all M" comes from this asymptotic sign argument.
  - Remark (my inference, not in the paper): their map is NS applied after the normalisation
    `theta/(b_00|theta| + b_01)` with `b_01 > 0`, hence linear near 0; the field at the minimizer is of order
    M^{-3/2} with O(1) slope, so the offset of the zero would be O(M^{-3/2}), not the O(1/M) of the exact polar/sign
    map. The exact-sign regime of C1-C3 is not covered by their computation.

**Xie, arXiv:2609.32861v1** ("Muon Under Gradient Noise and the Limits of Orthogonalization Near Optima").
- Repo: Prop. 3.1, `Psi_sigma(Y) = E P(Y + sigma Z)` is smooth, `D Psi_sigma(0) = (c/sigma) Id`,
  `c = E||Z||_*/(mn)`; Gaussian noise only.
- Re-read **[PDF-scan]**: keyword scan for skew / third / cumulant / median / Edgeworth: no hit on any of them in a
  relevant sense. The paper states "Real gradient noise is anisotropic, non-Gaussian, and has a nonzero mean" only
  as a caveat of the experiments (Sec. 7.4). Momentum is treated through a "stationary noise buffer" driven by
  Gaussian innovations (Thm 5.1, 6.1, 6.2): covariance and loss-floor results, no bias. One sentence notes "At
  nonzero y the common stationary mean may differ from the minimizer" without quantifying it. Third derivative of
  the Gaussian-smoothed map bounded (App. A), used for the cubic remainder only.

**Chen, Li, Wang, Lu, arXiv:2610.31728v1** ("When and How Does Muon Precondition? The Role of Structured Gradient
Noise").
- Repo: Prop. 3.1 `E[msgn(G + sigma Xi)] = (c_d/sigma) G + O(||G||_F^3/sigma^3)` (isotropic Gaussian, square);
  Thm 4.1 two-task batch-size threshold.
- Re-read **[PDF-scan + passages read]**:
  - No third-cumulant, skewness, Edgeworth or median result (keyword scan: zero relevant hits).
  - App. E.3, Thm E.1 (local sampling limit): with `W_B = W* + A/sqrt(B)`,
    `sqrt(B) G_B(W_B) -> H*[A] + Z` in law and `lim_B E msgn(G_B(W_B)) = E msgn(H*[A] + Z)`, Z centred Gaussian with
    covariance H*. Eq. (111): `DF_Sigma(0)[A] = E[msgn(Z) <Sigma^{-1} vec A, vec Z>]` for anisotropic Gaussian Z,
    "does not simplify to a scalar closed form". So the **anisotropic Gaussian linear-response operator (the
    J_Sigma = Hessian of psi_Sigma at 0 of C1/C5) is published**; the next order is not.
  - App. C (two-task associative memory, momentum): stationary "frozen-momentum mean drift"
    `psi_{B,beta}(gamma) = E phi(gamma; Z/p1, (1-Z)/p2)`, unique zero `gamma*_{B,beta}`, and the expansion
    `psi_{B,beta}(0) = (p1 - p2)/(2 B_eff p1 p2) + o(1/B_eff)`, `psi'(0) = -1 + o(1)`, hence an explicit
    O(1/B_eff) displacement of the zero, with `B_eff = B(1+beta)/(1-beta)` (variance-equivalent batch). This is a
    **1/B expansion of the zero of a stationary Muon mean drift with momentum**, but in one specific categorical
    model and driven by curvature-times-variance (`h''(p1) Var(Z)/2`), not by a third cumulant; the effective batch
    is the second-moment one, not the `(1+beta+beta^2)/(1-beta^2)` of C3.

**Paquette, Marshall, Benigni, Wang, Agarwala, Paquette, arXiv:2605.09552v2** ("Phases of Muon").
- Repo: high-dimensional least squares, rank min(N_in, N_out, B), `B_eff = B(1+beta)/(1-beta)`, transition at
  B_eff ~ N.
- Re-read **[PDF-scan]**: "bias" appears as a property of the risk recursion ("sets the bias of MUON to
  1/(2 sqrt(pi))") and as "finite-N drift bias of the polar factor"; one supplementary lemma notes that a third
  cumulant is O(1/k) inside an entrywise `E sign(g + x) = sqrt(2/pi) x + O(|x|^3)` argument. No limit-gradient
  offset, no median, no skewness law. Variance-based effective batch only.

**Huang, Pu, Johansson, arXiv:2611.18124v1** ("Muon Attains O(eps^-3) Iteration Complexity under Shuffling").
- Repo: Cor. 3.1, Thm 3.2 (for every fixed beta a 2x2 finite sum with `||grad f(W_t)||_* >= 1/N` under any
  permutation order, constant steps).
- Re-read **[PDF-scan]**: no skewness / cumulant / median content. Their lower bound is a without-replacement
  non-convergence statement at constant step size and is not a b-dependence law; it does not touch C4.

**Others in the repo records**: Chen et al. 2020 (arXiv:1906.01736) Prop. 1 and Thm 3 (signSGD follows the median;
non-convergence set by the median-mean gap); Bernstein et al. 2019 Assumption 4 (unimodal symmetric noise);
Karimireddy et al. 2019 Thm I; Cardot-Cenac-Zitt 2013 (normalized Robbins-Monro steps converge to the geometric
median: the Euclidean K=1 case of the paper's Theorem (i)); Minsker 2015 (definition of the geometric median in a
Banach space); Cornish-Fisher via a secondary source and a numerical check. `novelty.md` records that an arXiv API
query "abs:muon AND (median OR mirror descent OR nuclear norm OR biased OR skewed)" returned 28 entries with no
median statement for polar steps.

---

## 2. C1 - general 1/b law for any norm (expansion of the norm-median of the law of a sample mean)

**Verdict: PARTIALLY ANTICIPATED.** Not found as stated. The Euclidean instance exists in a different asymptotic
regime (dimension -> infinity), the object itself (geometric median of the law of a sample mean, as the bias of a
median-of-means) is studied with an upper bound only, and the scalar case is classical.

Closest works.

1. **R. Schwank, M. Drton, "On the distance between mean and geometric median in high dimensions",
   arXiv:2508.12926v3 (6 Feb 2026).** [PDF-read: Thm 3.4, Thm 3.5, Section 5]
   - Assumptions: p-variate X in a class D(M, q, c, sigma_min, C): M-dependent components (each component depends
     on at most a fixed number of neighbours), q-th moments, variances bounded below and above.
   - Thm 3.4 (q >= 3): `||m - mu||_inf <= B_3/p`.
   - Thm 3.5 (q > 3, `(1/p) sum Var X_i -> sigmabar^2`): `m_{p,i} = mu_i - alpha_i/p + o(1/p)` with
     `alpha_i = (1/(2 sigmabar^2)) sum_{|j-i|<=M} E[(X_i - mu_i)(X_j - mu_j)^2]`; for independent components this
     "reduces to the skewness of X_i".
   - Section 5: "Theorem 3.5 can also be applied to the geometric median of means [...] alpha_i = alpha_i^(1)/K.
     In other words, the first order bias of the population geometric median of means scales exactly as 1/K with
     the block size K."
   - Relation to C1: this is the Euclidean (K=1 in the project's matrix notation) case with the same driving
     moment `E[xi_i ||xi||^2]` and the same 1/b scaling, but the expansion parameter is the dimension p (b fixed),
     under M-dependence. C1 is at fixed dimension with b -> infinity and any covariance / cumulant tensor.
     Consistency check (mine): for i.i.d. entries the project's K=1 constant gives an offset
     `-kappa_3/(2 sigma^2 b (d+2))`, Schwank-Drton give `-kappa_3/(2 sigma^2 b p)`; they agree to leading order in d.
     **This paper must be cited next to C1 and next to the existing a/c = 3/(d+2) statement.**
2. **S. Minsker, N. Strawn, "The Geometric Median and Applications to Robust Mean Estimation",
   arXiv:2307.03111 (SIAM J. Math. Data Sci. per the arXiv listing: venue not verified).** [PDF-read: abstract,
   Section 2 around eqs. 2.3-2.4, Thm 3.3]
   - Defines `m_n` = "the geometric median of the distribution P^(n) of the average (1/n) sum Y_j" and calls
     `||m_n - mu||` "the main contribution to the bias" of the geometric median of means; dimension-free upper
     bound of the form `sqrt(||Sigma_Y|| k/N)`, i.e. O(n^{-1/2}) in the block size. Thm 3.3: for absolutely
     continuous Y, `||m|| <= min( sqrt(tr Sigma), sqrt(||Sigma||) E^{1/2}||Y-m||^{-2} / E||Y-m||^{-1} )`.
   - Relation to C1: exactly C1's object for the Euclidean norm, with an upper bound of order n^{-1/2} and no
     expansion. C1 sharpens it to an exact n^{-1} first-order term.
3. **Scalar case: Cornish-Fisher** (median of a mean = mean - kappa_3/(6 sigma^2 b) + o(1/b)). Already cited in the
   paper; verified in the repo through a secondary source and numerics. A modern statement of the same first-order
   median expansion through a one-term Edgeworth expansion: **P. Orenstein, "Robust Mean Estimation with the
   Bayesian Median of Means", arXiv:1906.01204**, Prop. 22 [PDF-read: statement and first lines of proof]:
   `median(Y | theta_hat) = thetabar - (1/3) sqrt(s^2/(n alpha + 2)) skew_hat + o(1/(n alpha))` for Dirichlet-weighted
   means (Edgeworth expansion of Weng 1989).
4. **Chen, Chen, Sun, Wu, Hong, arXiv:1906.01736 (NeurIPS 2020).** [PDF-read: Thm 3, Thm 4 with eqs. 12-16]
   Thm 4: for fixed numbers u_1..u_{2n+1} perturbed by symmetric unimodal noise of scale b,
   `E[median(u_i + b xi_i)] = mean(u) + O(max|u_i - u_j|^2 / b)`, and the asymmetric part of the median's density
   has L^1 mass `O(max|ubar - u_i|^2 / b^2)`. A rate for the median-mean gap under added symmetric noise across
   workers: a different mechanism (noise scale, not batch size; no cumulant formula), coordinatewise only.
5. **Sign / normalized / nonlinear SGD theory under asymmetric noise.**
   - Armacki, Yu, Sharma, Pang, Bajovic, Jakovetic, Kar, "Nonlinear Stochastic Gradient Descent and Heavy-tailed
     Noise: A Unified Framework and High-probability Guarantees", arXiv:2410.13954. [PDF-read: Assumption 5,
     Thm 3, Remarks 15-19] Noise = mixture `(1-lambda) P_1 + lambda P_2` with P_1 symmetric; convergence to a
     neighbourhood of stationarity of size governed by `lambda/(1-lambda)`; Remark 16 says the non-symmetric part
     shrinks with the batch size by a generalized CLT, without a rate or a constant. General nonlinearities
     (sign, normalization, clipping), but an upper bound only, and not an LMO-of-a-norm statement.
   - Armacki, Bajovic, Jakovetic, Kar, arXiv:2507.09093 [PDF-scan]: handles non-symmetric noise by symmetrizing
     the estimator (changing the algorithm), no bias formula.
   - Bernstein et al. (arXiv:1810.05291), Karimireddy et al. (arXiv:1901.09847) [repo]: symmetry assumption and
     counterexamples only. Jiang et al., arXiv:2507.12091 [PDF-scan] and Sun et al. ICML 2023 [search]: sign
     momentum rates without symmetry; no limit-point characterization.
   - Tao, Yu, Zhang, arXiv:2605.06615 [PDF-scan]: "skewed" refers to noise concentrated on few coordinates, not
     to third moments. Wang et al., arXiv:2611.16560, App. C.5 [PDF-read: first page of C.5]: non-Gaussian
     minibatch noise in a Gaussian regression model, where the noise at zero error is a symmetric scale mixture;
     no bias. Compagnoni et al., arXiv:2411.15958 [PDF-scan]: zero hits for skew/asymmetric.
6. **Muon-specific**: none of 2608.04607, 2609.32861, 2610.31728, 2605.09552, 2611.18124, 2606.08388
   (Beneventano et al., [PDF-scan]: "bias" means spectral/implicit bias) contains an Edgeworth term or a
   third-cumulant law.

What remains open / new in C1: (a) the fixed-dimension, b -> infinity expansion with an arbitrary third-cumulant
tensor and covariance; (b) any non-Euclidean norm (l_infinity dual / sign with correlated noise, spectral); (c) the
representation `-(1/6b) [Hess psi_Sigma(0)]^{-1} <D^4 psi_Sigma(0), K_3>`; (d) rigour under Cramer's condition.
I found no "Edgeworth expansion of the spatial median of a sample mean" in the statistics literature, but that
literature is large and I searched it with four queries only (see Section 9): treat (a) for the Euclidean norm as
"not found", not as "proved absent". Zou-Peng-Feng-Wang (Biometrika 2014), reported by a search summary to contain
an expansion of the spatial median under spherical symmetry in high dimension, was **not opened**.

Technical caution for the rigorous claim (see Section 7): the Bhattacharya-Rao remainder contains the Gaussian
average of the oscillation of f at radius ~ e^{-dn}; for `f = LMO(G sqrt(b) + .)` this needs a bound on the
Gaussian measure of a shrinking neighbourhood of the discontinuity set of the LMO, uniform in the shift G. This
is easy for the polar map (rank-deficient matrices) and for the Euclidean sign, but it should be written, and for
polyhedral norms (sign descent) the discontinuity set is a union of hyperplanes and the same remark applies.

---

## 3. C2 - spectral closed form for isotropic covariance

**Verdict: NOT FOUND.**

- No paper found reduces a third-order bias of the polar map to the two O(K) x O(d)-equivariant cubic moments
  `E[xi ||xi||_F^2]` and `E[xi xi^T xi]`, nor gives coefficients in terms of `E sum s_i` and `E sum s_i^3` of a
  Gaussian matrix.
- Shadows: (i) Schwank-Drton's `alpha_i` is the K=1 instance where the two invariants coincide
  (`xi xi^T xi = xi ||xi||^2` for a vector); (ii) the denominator `c_{K,d} = E||Z||_*/(Kd)` is published (Xie
  Prop. 3.1; Chen et al. Prop. 3.1, both re-checked in the repo's audit); (iii) Chen et al. eq. (111) gives the
  anisotropic Gaussian response operator without closed form; (iv) Xie, App. A, bounds the third derivative of
  the Gaussian-smoothed polar map by `sqrt(6r)/sigma^3` through a degree-three Wick product, i.e. the same
  Hermite-3 object as `a_{K,d}`, but only as a bound for a remainder, with no non-Gaussian input.
- Xie's main text expands the polar map's fluctuations "degree by degree in the Hermite expansion" (Thm 5.1,
  Cor. 5.2) for covariance purposes [PDF-scan: contribution list only]. I did not read whether explicit Hermite
  weights of degree 3 of the polar map are tabulated there; if they are, `alpha_{K,d}, beta_{K,d}` may be
  expressible through them. **Uncertain item: check Xie Section 5 / App. B before claiming the coefficients are
  new as numbers.**
- Marchenko-Pastur asymptotics of the coefficients: not found anywhere in this context.

---

## 4. C3 - momentum (median of the stationary filtered noise; third-cumulant effective batch)

**Verdict: PARTIALLY ANTICIPATED.**

- **Do-Dereich-Jentzen, Prop. 6.2** (Section 1 above) already identifies, in dimension one and for Lipschitz
  NS-type maps, the limit of Muon with fixed EMA momentum and `sum gamma_n = infinity` as a zero of the mean field
  evaluated on the stationary filtered noise `(1-alpha) sum alpha^k (.)`. They call the filtered variables
  "stationary momentum random variables (SMRVs)" (Section 6.2). Differences with C3: they assume convergence
  (necessary condition) instead of proving almost-sure convergence; scalar parameter and quadratic loss; Lipschitz
  map (no exact polar/sign); no interpretation as a (nuclear) median; not matrix-valued.
- **Do-Dereich-Jentzen, Prop. 6.11 and 6.13**: leading-order asymptotics in M of `E[Z_M |Z_M|]` and of the NS field
  at the minimizer for the filtered sum, with the constant built from the second and third moments of the
  per-sample filtered variable. The ratio (sum w_k^3)/(sum w_k^2) is therefore implicit in their constant, but
  they state no effective batch size and no expansion of the zero. C3's
  `b_eff = b (1+beta+beta^2)/(1-beta^2)` was **not found** in any source.
- **Variance-based effective batch `B(1+beta)/(1-beta)`** is published twice in the Muon literature (Paquette et
  al. 2605.09552; Chen et al. 2610.31728 eq. before (24)), and Chen et al. use it for the O(1/B_eff) displacement
  of the zero of their two-task stationary drift (Section 1 above). A reader will expect that formula; C3 should
  state explicitly that the skewness-driven bias follows a *different* effective batch and why
  (ratio of sum w^2 to sum w^3 instead of 1/sum w^2), and cite both.
- **Xie 2609.32861**, Thm 5.1 / 6.1 / 6.2: stationary EMA buffer with Gaussian innovations; covariance and loss
  floors only.
- Sign momentum literature (Signum in Bernstein et al. 2018; Sun et al. 2023 [search]; Jiang et al. 2507.12091
  [PDF-scan]; Cutkosky-Mehta and Crawshaw et al., not opened): rates, no limit-point law under skewed noise.
- Almost-sure convergence of exact-polar Muon with fixed beta and Robbins-Monro steps to a characterized point:
  not found (search summaries of 2601.19156, 2609.33152, 2505.23737, 2507.01598, 2606.14560 mention only
  expected nuclear-norm stationarity rates; these were not opened).

What remains new: the almost-sure convergence theorem with exact polar and matrix parameters; the identification
with the nuclear median of N_beta; the closed-form b_eff (and the Nesterov variant).

---

## 5. C4 - sampling without replacement

**Verdict: the cumulant factor is classical (FOUND, but not opened in a primary source); the optimization
consequence (sign change of the first-order bias at b = n/2) NOT FOUND.**

- The identity `E[(ybar_b - Ybar)^3] = (n-b)(n-2b) / (b^2 (n-1)(n-2)) * mu_3` (mu_3 = population third central
  moment with divisor n) is the classical third moment of the mean under simple random sampling without
  replacement, i.e. the with-replacement value `mu_3/b^2` times `(n-b)(n-2b)/((n-1)(n-2))`. I verified it by
  exact enumeration (n = 9, b = 2..6, agreement to 1e-15). I could not open a primary reference: search summaries
  point to Cochran, *Sampling Techniques* (1977, ch. 2), Skellam (1949), Wishart (1952), and B. O'Neill, "Some
  Useful Moment Results in Sampling Problems", Amer. Statist. 68(4):282-296 (2014) [all search-only]. Attribute it
  as classical and cite one of these after checking the page.
- The sign change at b = n/2 follows from the complement symmetry (the mean of a sample and of its complement are
  affinely anti-related), also classical in spirit. No optimization paper found states it for sign / normalized /
  polar steps. Huang-Pu-Johansson (2611.18124) and Li-Tang-Zou (2602.11557) treat reshuffling but not this effect.
- Caution: the first-order bias also needs the variance factor `(n-b)/(n-1)`, so the ratio kappa_3/sigma^2 picks up
  `(n-2b)/(n-2)`; and when b/n is not small the Edgeworth regime "b -> infinity at fixed n" is not the i.i.d. one
  (finite-population Edgeworth expansions: Robinson 1978, Babu-Singh 1985, Bloznelis-Gotze 2001 [memory, not
  verified here]). A rigorous C4 should cite a finite-population expansion rather than Bhattacharya-Rao.

---

## 6. C5 - state-dependent noise, displacement of the zero of the mean field

**Verdict: PARTIALLY ANTICIPATED** (structure known; the third-cumulant formula not found).

- **Chen et al. 2610.31728, Thm E.1 + eq. (111)** [PDF-read]: at `W = W* + A/sqrt(B)` the mean polar step converges
  to `E msgn(H*[A] + Z)`, Z ~ N(0, Sigma*) with the local covariance; the anisotropic Gaussian response
  `DF_Sigma(0)` is written. This is the zeroth-order (Gaussian, symmetric) version of C5's mean field at the
  1/sqrt(B) scale; by symmetry its zero is A = 0. C5 is the next order.
- **Chen et al., App. C** [PDF-read]: explicit `O(1/B_eff)` displacement of the zero of a stationary mean drift in
  the two-task model (curvature x variance mechanism, parameter-dependent noise). Same type of statement as C5, in
  one model, with a different mechanism.
- **Do-Dereich-Jentzen** [PDF-read]: zero of the mean field differs from the minimizer for almost all M (1-D,
  additive data noise); error bound C(1/M + gamma_n) for matrix parameters (Thm 1.2) which is consistent with an
  O(1/M) or smaller displacement but gives no constant.
- **Chen et al. 2020 (1906.01736), Thm 3** [PDF-read]: signSGD converges to a point whose gradient is set by the
  median-mean gap (existence of an instance; no expansion).
- **Singh, Mishra, Raut, arXiv:2607.12360** [PDF-scan, Lemma 1 and Remark 5 read]: multiplicative scale-family
  noise; sign reliability `P(Z > -1/sigma_1)` independent of |x|; "a Z with sufficiently negative median makes the
  normalized drift point outward". State-dependent noise and a median appear together, but for floors at constant
  step, not for a displaced zero.
- No source gives `W_b - W* = -(1/(6b)) H^{-1} J^{-1} T[K_3*] + o(1/b)`.

---

## 7. Item 3 - Bhattacharya & Rao, Theorem 20.1 and relatives

I could not open the book. What I could verify from papers that quote it:

- **Non-i.i.d. version, Theorem 20.6**, quoted as Theorem 4.1 in J. Angst, G. Poly, "A weak Cramer condition and
  application to Edgeworth expansions", arXiv:1511.08750 (Electron. J. Probab. 22 (2017), paper 59) [PDF-read:
  the statement], who write "as it is stated in Theorem 20.6 of [BR86]":
  Let (X_n) be independent random vectors in R^k with zero means such that (1) the smallest eigenvalue of the
  average covariance V_n is bounded away from zero uniformly in n; (2) for some integer s > 2, the average s-th
  absolute moment rho_s(n) is bounded uniformly in n and, for all eps > 0,
  `(1/n) sum_j E[ 1{||X_j|| > eps sqrt(n)} ||X_j||^s ] -> 0`; (3) uniformly in n large enough the characteristic
  functions satisfy Cramer's condition, `for all R > 0, limsup_n sup_{||t|| > R} |phi_{X_n}(t)| < 1`. Then for every
  real Borel function f on R^k with `M_{s'}(f) < infinity` for some `0 <= s' <= s`,

      | int f dQ_n - int f dQtilde_n | <= M_{s'}(f) delta_1(n) + c(s,k) omegabar_f(2 e^{-dn} : Phi),

  with `delta_1(n) = o(n^{-(s-2)/2})`, d > 0 and c(s,k) explicit constants,
  `M_s(f) = sup_x |f(x)|/(1 + ||x||^s)`, `omega_f(x : lambda) = sup_{B(x,lambda)} f - inf_{B(x,lambda)} f`,
  `omegabar_f(2 eps : Phi) = int omega_f(x : eps) dPhi(x)` (notation as printed by Angst-Poly), Q_n the law of
  the normalized sum and Qtilde_n the Edgeworth signed measure with terms up to n^{-(s-2)/2}.
- **Theorem 20.1** is the i.i.d. case of the same statement: i.i.d. vectors with zero mean, nonsingular covariance,
  finite s-th absolute moment (integer s >= 3), Cramer's condition `limsup_{||t|| -> infinity} |Qhat_1(t)| < 1`;
  same conclusion with the signed measure `sum_{r=0}^{s-2} n^{-r/2} P_r(-Phi_{0,V} : {chi_nu})`. This wording is
  **[memory]** plus the indirect evidence that Angst-Poly (Section 6.2) and K. Song, "A Uniform-in-P Edgeworth
  Expansion under Weak Cramer Conditions", arXiv:1806.01431 [PDF-read: introduction and Thm 2.1], both say they
  follow "the proofs of Theorems 20.1 and 20.6" with the Cramer condition used only once, in the bound of the
  integral I_1 of eq. (20.36), p. 211. Song adds that the proof of 20.1 assumes i.i.d. summands. Page numbers
  cited by Song for the 2010 SIAM edition: (20.8) p. 208, (20.12) p. 209, (20.17)-(20.19) p. 210, Corollary 20.5
  p. 215 (convex sets), Corollary 3.2 p. 24 (`omegabar_{1_A}(eps : Phi) <= c(s,d) eps` for convex A).
  **Check the exact wording of 20.1 in the book before quoting it.**
- **Weak-Cramer versions usable for discrete, non-lattice summands** (relevant for finite sums / minibatches of a
  finite dataset, where the classical Cramer condition fails): Angst-Poly Thm 4.2 (same conclusion under a "weak
  mean Cramer condition" `|phi(t)| <= 1 - C/||t||^b`); Song Thm 2.1 [PDF-read]: under a uniform s-th moment bound
  and the mean weak Cramer condition with parameters (b, c, R), `0 < b < 2/max(s-3, 1)`, for all Borel f and
  n >= n_0, `| int f d(Q_n - Qtilde_{n,s}) | <= C n^{-(s-2)/2} (1 + M_s(f)) + C omegabar_f(n^{-(s-2)/2} ; Phi)`. Note
  the oscillation radius is polynomial here, not exponential, and the remainder is O, not o: with s = 3 it is of
  the same order n^{-1/2} as the first Edgeworth term, so a one-term expansion with a smaller remainder needs
  s >= 4 moments in Song's form (my reading of the bound, not a statement of the paper).
- **Smooth functions without Cramer**: F. Gotze, C. Hipp, "Asymptotic expansions in the central limit theorem
  under moment conditions", Z. Wahrsch. verw. Gebiete 42 (1978), 67-87. Not opened (Springer). Two secondary
  descriptions [search]: arXiv:2511.01603 says they "derived an asymptotic expansion of the expectation of a
  smooth function of the sample mean under the moment condition", and Barbour (1986) says that for expectations
  of smooth functions "the natural moment conditions are all that is required". The LMO is not smooth, so this
  route would apply only after using the Gaussian part of the noise, or to smoothed maps (Newton-Schulz
  polynomials: there it applies directly and no Cramer condition is needed). Partial-Cramer conditions for
  functions of sample means: Bai-Rao (1991) and arXiv:2511.01603 [PDF-scan].

---

## 8. Item 4 - half-integer Wishart moments and Marchenko-Pastur moments

- **Finite K, d, real Gaussian (Laguerre orthogonal ensemble).** No simple closed form for `E||Z||_*` or
  `E sum s_i^3` was found for general (K, d). For K = 1 it is the chi mean,
  `sqrt(2) Gamma((d+1)/2)/Gamma(d/2)`, and `E s^3 = 2^{3/2} Gamma((d+3)/2)/Gamma(d/2)` [elementary].
  The relevant structural reference is **F. D. Cunden, F. Mezzadri, N. O'Connell, N. Simm, "Moments of Random
  Matrices and Hypergeometric Orthogonal Polynomials", arXiv:1805.08760 (Commun. Math. Phys. 2019)** [PDF-read:
  eqs. 4.10-4.13, Thm 4.4, Thm 8.3, Thm 8.9, Cor. 8.10, opening of Section 8.3]:
  - Complex Wishart (LUE), `Q_k^C(m,n) = E Tr X^k` for complex k: `Q_k^C(m,n)/(k+alpha)! = (mn/(1+alpha)!)
    3F2(1-k, 2+k, 1-n; 2, 2+alpha; 1)`, extended to complex k by Carlson's theorem (Thm 4.4: a continuous dual Hahn
    polynomial in k). So half-integer moments (k = 1/2, 3/2) of the complex Gaussian case are finite terminating
    sums.
  - Real Wishart (LOE): three-term combinations of consecutive moments have hypergeometric-polynomial factors
    (Thm 8.3); for **odd size** there is a duality with the symplectic ensemble valid "for all k in C"
    (Thm 8.9, eq. 8.34: `Q_k^R(2m+1, 2n+1) = 2^{k+1} Q_k^H(m,n) + 2^k Gamma(k + m - n + 1/2) f_k(m,n)`); the
    even-size case "has a different analytic structure ... somewhat mysterious to us" and is treated only for the
    Gaussian ensemble. Their Q^R normalisation and the definition of f_k were not transcribed; re-read before use.
  - Consequence: explicit `alpha_{K,d}, beta_{K,d}` are available as finite sums when min(K,d) is odd (through
    this duality), and otherwise through the LOE one-point density (Laguerre polynomials plus an erf/incomplete
    gamma correction term) integrated against x^{1/2}, x^{3/2}; I found no paper that tabulates them.
- **Marchenko-Pastur moments.** With MP_gamma normalised to mean one, ratio gamma in (0, 1], the integer moments are
  Narayana polynomials `2F1(1-k, -k; 2; gamma)`, and the analytic continuation in k is
  `int x^s dMP_gamma = 2F1(-s, 1-s; 2; gamma)`. I verified this numerically for s = 1/2 and 3/2 at
  gamma = 0.25, 0.5, 0.9 (agreement to 7 digits), and at gamma = 1 against the quarter-circle values
  `int x^{1/2} = 8/(3 pi)`, `int x^{3/2} = 64/(15 pi)` (Gauss summation). Hence
  `E||Z||_* ~ K sqrt(d) 2F1(-1/2, 1/2; 2; K/d)` and `E sum s_i^3 ~ K d^{3/2} 2F1(-3/2, -1/2; 2; K/d)` for K <= d;
  both reduce to complete elliptic integrals K(gamma), E(gamma). The continuation formula itself is my derivation
  (standard), checked numerically; Cunden et al. Section 5.2, eq. (5.7), give an equivalent 2F1 expression for the
  Mellin transform `zeta_inf(s) = int x^{-s} rho_inf(x) dx` of the MP density in their normalisation (displayed
  formula only partly legible in the text extraction: not transcribed).

---

## 9. Scope and limits

**Repo files read**: `paper/paper_saga_to_muon.bib` (entry list), `paper/audit/evidence/cite-stat.md` and
`cite-muon.md` (in full), `muon/literature_check.md` (grep hits and lines 218, 254, 379, 381; headings),
`paper/audit/evidence/novelty.md` (grep hits), `paper/paper_saga_to_muon.tex` (grep hits around l. 160-164,
418-466).

**PDFs downloaded and read at the depth stated above** (arxiv.org, `pdftotext`): 2608.04607, 2609.32861,
2610.31728, 2605.09552, 2611.18124, 2606.08388, 2607.12360, 2605.06615, 2611.16560, 1906.01204, 1906.01736,
2610.16868 (scan only, nothing relevant), 2508.12926, 2307.03111, 2511.01603, 1806.01431, 2411.15958, 1511.08750,
1805.08760, 2410.13954, 2507.09093, 2507.12091.

**Web searches run (23; "extended" unless marked std)**:
1. signSGD bias skewed asymmetric gradient noise converges to median not mean, batch size 1/b bias, Edgeworth OR skewness
2. asymptotic expansion of the spatial (geometric) median of the distribution of a sample mean, skewness third cumulant, Edgeworth OR Cornish-Fisher multivariate median
3. Muon optimizer skewed gradient noise bias non-convergence third moment batch size momentum orthogonalized
4. normalized SGD OR sign SGD with momentum asymmetric noise bias stationary point median of noise "effective batch size" skewness
5. "spatial median" OR "geometric median" of a skewed multivariate distribution: asymptotic expansion ... "mean-median" multivariate inequality small skewness
6. "Theorem 20.1" Bhattacharya Rao ... Edgeworth expansion expectation of bounded measurable function Cramer condition oscillation
7. (std) third cumulant of sample mean sampling without replacement finite population (N-n)(N-2n)/((N-1)(N-2)) skewness
8. expected nuclear norm of Gaussian random matrix closed form; half-integer moments real Wishart singular values; Marchenko-Pastur moment of order 1/2 and 3/2
9. stochastic steepest descent arbitrary norm LMO stochastic gradient bias fixed point ... Scion bias
10. signSGD SDE drift erf non-Gaussian skewed noise stationary distribution bias ... Edgeworth correction 1/sqrt(B)
11. Bhattacharya Ranga Rao Theorem 20.1 statement "Cramer's condition" ... "2e^{-dn}"
12. (std) Gotze Hipp 1978 "Asymptotic expansions in the central limit theorem under moment conditions" smooth functions without Cramer condition
13. (std) Angst Poly "A weak Cramer condition and application to Edgeworth expansions" arXiv Theorem 20.1
14. (std) "Theorem 20.6" OR "Theorem 20.8" Bhattacharya Rao Edgeworth expansion independent non-identically distributed
15. signSGD momentum Signum theory asymmetric noise bias; "sign of momentum" skewed noise convergence to median; normalized SGD with momentum non-symmetric noise limit point
16. moments of sum of singular values Gaussian matrix ... Cunden Mezzadri O'Connell Simm
17. nonlinear SGD (sign, normalization, clipping) heavy-tailed noise "symmetric" assumption relaxed asymmetric noise bias neighbourhood
18. (std) Cochran Sampling Techniques skewness of sample mean simple random sampling without replacement third moment formula
19. Muon OR "spectral descent" OR "matrix sign" stochastic gradient non-Gaussian noise third-order correction expected polar factor Edgeworth skewness
20. sign-based optimizer stationary point shift state-dependent multiplicative gradient noise skewness ... expansion of the fixed point "mean field" zero
21. (std) Zou Peng Feng Wang 2014 sphericity ... expansion spatial median; "Bahadur representation" spatial median bias
22. Muon momentum "effective batch size" third moment OR skewness EMA filtered noise "1-beta^2" "1-beta^3"
23. Muon convergence theory 2026 nuclear norm stationarity stochastic "almost sure" convergence momentum Robbins-Monro ... "median"

**Not opened / could not be opened (do not cite on the basis of this report)**:
- Bhattacharya & Rao, *Normal Approximation and Asymptotic Expansions* (book): Theorem 20.1 wording from memory
  and from the quotation of Theorem 20.6 by Angst-Poly.
- Gotze-Hipp 1978 (Springer); Cochran 1977, Skellam 1949, Wishart 1952, O'Neill 2014 (C4 formula checked by
  enumeration only); Zou-Peng-Feng-Wang, Biometrika 2014; Mottonen arXiv:2308.06264; Brown 1983.
- Named in the task and not opened in this session: Safaryan-Richtarik (arXiv:1905.12938), Crawshaw et al.,
  Cutkosky-Mehta, Kornilov et al. (arXiv:2502.07923 appeared in results), Pethick et al. (arXiv:2502.07338; the
  repo records a title screen of its 228 citing papers), Sun et al. ICML 2023, Bernstein et al. 2018/2019 and
  Karimireddy et al. 2019 (repo audit quotes used instead). These are rate / counterexample papers as far as the
  search summaries and the repo records show; I did not verify the absence of a bias expansion in them.
- 2026 Muon convergence papers seen only in search summaries: 2601.19156, 2609.33152, 2505.23737, 2507.01598,
  2606.14560, 2603.15059, 2605.18528, 2606.21514, 2602.07425, 2605.16311, 2607.29674, 2510.09827, 2506.15054,
  2610.38773 (Rennala NSGD), 2506.06606 (Stacey), 2609.35297, 2602.16340.

**Limits of the method**: keyword scans can miss a result stated in other words (e.g. "asymmetry", "third-order
Hermite coefficient", "odd cumulant"); I scanned for skew / third / cumulant / median / Edgeworth / asymmetr /
bias. No citation-graph search (papers citing Schwank-Drton, Minsker-Strawn, Do-Dereich-Jentzen) was run: this is
the most useful next step, since a follow-up of Schwank-Drton at fixed dimension would be the direct competitor of
C1. No OpenReview search. Search engine is US-only and its summaries were not trusted for content: every claim
above that is not marked [search] or [memory] comes from text I read.

---

## 10. Follow-up (requested by the coordinator as "8. Follow-up"; numbered 10 because sections 8 and 9 already exist)

Added 2026-11-25. Labels as in the header ([PDF-read], [search], [memory]).

### 10.1 Xie arXiv:2609.32861: is the degree-3 Hermite component of the polar map given?

Read in full [PDF-read]: Section 5 (Thm 5.1, Cor. 5.2, "The reference implementation"), Appendix A (all), Appendix B
(B.1-B.4, all). **Answer: no.** The paper contains only the scalar *weights* (squared L2 norms) of the Hermite
degrees, without any formula for the degree-3 one, and nothing equivalent to `E[polar(Z) (x) He_3(Z)]`.

What is there, quoted:
- Section 5: "At Y = 0 the polar map is odd, so even degrees vanish and the residual starts at degree three.
  Orthogonal equivariance makes each degree covariance a multiple of the identity, C_l(0) = w_l I_d [...] with
  Hermite weights w_l >= 0, w_1 = c^2, and sum_{l odd} w_l = b." (b = r/d, r = min(m,n), d = mn.)
- App. B.1, eq. (34): `E[f_l(z) f_l(z)^T] = w_l I_d, w_l >= 0, sum_{l odd} w_l = r/d = b`, proved by "Independent
  row and column sign changes force distinct matrix coordinates to have zero covariance; row and column
  permutations equate their variances." Only the *second-moment matrix of each degree component* is reduced by
  symmetry (to one scalar). The degree-3 coefficient tensor itself is never written, and w_3 is never computed:
  only the totals `sum_{l>=3} w_l = b - c^2` and bounds through `q_3` are used.
- The only closed forms: scalar case through `E[sign(Z) sign(Z')] = (2/pi) arcsin(rho)` (App. B.2), and the
  shape limits of c in App. A (`c = sqrt(2/pi)` for 1x1, `c = E chi_d / d` for 1 x d, quarter-circle limit
  `E||Z||_* / N^{3/2} -> 8/(3 pi)` for N x N, `R_0 -> 9 pi^2/64`). NS5 weights are "estimate[d] numerically".
- App. A: `||D^3 Psi_sigma(Y)[A,B,C]||_F <= (sqrt(6r)/sigma^3) ||A||_F ||B||_F ||C||_F`, from "the degree-three
  Wick product of <Z,A>, <Z,B>, <Z,C>; its variance is at most 3! ||A||^2||B||^2||C||^2". An upper bound on the
  third derivative of the Gaussian-smoothed polar map, no value.
- Not present anywhere (keyword and full reading of the sections above): `E sum s_i^3`, `E sum_{i!=j} s_i s_j/(s_i+s_j)`,
  any decomposition of an O(K) x O(d)-equivariant 4-tensor into two invariants, any non-Gaussian input.

Two side remarks for C2/C3:
- The equivariance argument used by Xie (sign flips and permutations, then full O(m) x O(n)) is the same tool as
  in C2, applied to a 2-tensor (one invariant). C2's 4-tensor reduction with two invariants is not in the paper.
- Cor. 5.2 / eq. (38): the degree-l component of the filtered stream is attenuated by
  `q_l(beta) = (1-beta)(1+beta^l) / ((1+beta)(1-beta^l))`, with `q_3 = (1-beta+beta^2)/(1+beta+beta^2)`; Prop. B.1
  gives the Nesterov filter weights `a_0 = 1-beta^2, a_j = (1-beta) beta^{j+1}`, `v = sum a_j^2 =
  (1-beta)(1+2beta-2beta^3)/(1+beta)`. These are lag-covariance factors of a *Gaussian* buffer (how the cubic
  Hermite part of the output decorrelates), not the third-cumulant factor `sum w^3 / sum w^2` of a non-Gaussian
  buffer behind C3's `b_eff`. Similar-looking cubic-in-beta expressions, different quantity; worth one sentence
  in the paper to prevent confusion. The Nesterov weights and v can be cited from Prop. B.1.

### 10.2 Schwank-Drton arXiv:2508.12926: precise statement and scope

Read in full [PDF-read]: Sections 1-5 (the main text is 6 pages); appendix headings and the proof of Thm 3.5
scanned.

- Setting (Section 2): "We denote the Euclidean norm by || . ||"; `D(m) = E||X - m||`. Def. 3.1: X is
  M-dependent and belongs to D(M, q, c, sigma_min, C) "if all components X_i admit a Lebesgue density bounded by
  c, with E|X_i|^q <= C and sd[X_i] >= sigma_min > 0".
- **Theorem 3.5** (verbatim up to typography): "Let (X_i)_{i in N} in D(M, q, c, sigma_min, C) with q > 3 and
  assume (1/p) sum_{i=1}^p Var[X_i] -> sigmabar^2. Fix i in N. Then, m_{p,i} = mu_i - alpha_i/p + o(1/p), where
  alpha_i := (1/(2 sigmabar^2)) sum_{|j-i|<=M} E[(X_i - mu_i)(X_j - mu_j)^2]."
  Followed by: "When M = 0, i.e., (X_i)_i are independent, the sum reduces to the skewness of X_i."
  (Thm 3.4, q >= 3: `||m - mu||_inf <= B_3/p`; Lemma 3.3, q > 2: `||m - mu||_inf <= B_1`.)
- **Fixed dimension with the number of averaged samples -> infinity: not treated.** The only statement about
  averaging is the Section 5 remark: with `X_i = (1/K) sum_k X_i^(k)`, "alpha_i = alpha_i^(1)/K. In other words,
  the first order bias of the population geometric median of means scales exactly as 1/K with the block size
  K." This is Thm 3.5 applied at fixed K with p -> infinity; the o(1/p) remainder is not uniform in K and no
  K -> infinity statement at fixed p is made. The symbol n in the paper is the sample size of the *sample*
  median in the simulations (n := p^3), unrelated.
- **Other norms: none.** No remark on general norms, matrix norms, nuclear norm or Banach spaces; "Banach"
  occurs only in the title of the Minsker (2015) reference; `|| . ||_inf` is used only to measure the distance
  m - mu. Keyword scan for nuclear / matrix / Banach / spectral: no hit in the text.
- **No fixed-p formula.** Nothing like 1/(p+2) appears. Their starting point is the exact fixed-p identity (1),
  `m - mu = E[1/||X-m||]^{-1} E[(X-mu)/||X-m||]` (valid for p large enough, Lemma A.2), and identity (3), but
  both are implicit in m and are only used for the p -> infinity limit. The project's fixed-d constant
  (3/(d+2) for i.i.d. entries, K = 1) is therefore not in this paper; it is consistent with their limit
  (factor 3/(d+2) * 1/6 = 1/(2(d+2)) vs their 1/(2p)).
- Their assumptions (bounded densities, M-dependence, q > 3) and C1's (Cramer condition, fixed dimension,
  arbitrary dependence between coordinates) do not contain one another.

### 10.3 Perturbation bound for the polar factor of full-rank rectangular matrices

**What I read** [PDF-read]: Z. Xu, T. Zhang, "Sharp Frobenius-norm perturbation bounds for subunitary and positive
polar factors", arXiv:2507.14940v2, Introduction, which states as "Theorem 1.1 (Li-Sun)", attributed to their
ref. [23] = W. Li and W. W. Sun, "Perturbation bounds of unitary and subunitary polar factors", SIAM J. Matrix
Anal. Appl. 23 (2002), no. 4, 1183-1193, doi:10.1137/S0895479801394623:

> Let A, Atilde = A + E in C^{m x n}_r [same rank r] have the generalized polar decompositions A = QH,
> Atilde = Qtilde Htilde. Then `||Q - Qtilde||_F <= (2/(sigma_r + sigmatilde_r)) ||E||_F`.

Here Q is the canonical subunitary factor (`Q = U_1 V_1^*` from the compact SVD, `R(Q^*) = R(H)`), and
sigma_r, sigmatilde_r are the smallest *positive* singular values of A and Atilde. Xu-Zhang call it "the best
known previous perturbation bound" for equal ranks and say Li-Sun "noted that their bound is sharp in certain
cases".

Consequences for the project's use:
- **Yes**: for m x n matrices of full column rank n (same m, n; r = n), the canonical subunitary factor is the
  polar factor U V^T, so `||Q - Qtilde||_F <= 2 ||A - Atilde||_F / (sigma_n(A) + sigma_n(Atilde))`, constant
  exactly 2, Frobenius norm on both sides, no smallness condition on E beyond both matrices having rank n. The
  statement is for complex matrices; real matrices are a special case. Full row rank (K < d) follows by
  transposition (or directly, since the theorem only needs equal rank r = min(m,n)).
- The Frobenius norm cannot be replaced by an arbitrary unitarily invariant norm with the same constant: Xu-Zhang
  give a 3 x 3 rank-2 counterexample in the spectral norm (`||Q - Qtilde||_2 > ||A - Atilde||_2` with
  `2/(sigma_2 + sigmatilde_2) = 1`), disproving a conjecture of W. Li (2008). A search summary of W. Li, Linear
  Algebra Appl. 429 (2008) reports the constant 3 for any unitarily invariant norm:
  `||Qtilde - Q|| <= (3/(sigma_r + sigmatilde_r)) ||E||` [search only].
- If the two matrices have different ranks the bound fails by a rank-counting argument (Xu-Zhang, via Mirsky):
  relevant only at rank-deficient points, which have probability zero under noise with a density.

**What I could not open**: R.-C. Li, "New perturbation bounds for the unitary polar factor", SIAM J. Matrix Anal.
Appl. 16 (1995), 327-332 (paywalled). The abstract of its technical-report version (UC Berkeley, 1994;
fetched through the summarizing fetch tool) says only: A is m x n, m >= n, full column rank; for m = n the bound
is "valid for any unitarily invariant norm, is sharper and simpler than Mathias's"; for m > n "we also establish
a bound for unitarily invariant norm". No constants in the abstract. Higham, *Functions of Matrices*, Thm 8.10:
not opened.

**[memory], not verified here**: R.-C. Li (1995) proved `||Q - Qtilde|| <= (2/(sigma_n + sigmatilde_n)) ||A - Atilde||`
for square nonsingular matrices in every unitarily invariant norm, and a weaker constant for m > n (an extra
term of order 1/max(sigma_n, sigmatilde_n)); Li-Sun (2002) removed the extra term in the Frobenius norm, which
is how Higham's Thm 8.10 states it (full rank m x n, Frobenius norm, constant 2). Safe attribution for the
rectangular Frobenius bound with constant 2: **Li and Sun (2002)**, optionally "see also Higham (2008, Thm 8.10)"
after checking the book; cite R.-C. Li (1995) for the square case only.
