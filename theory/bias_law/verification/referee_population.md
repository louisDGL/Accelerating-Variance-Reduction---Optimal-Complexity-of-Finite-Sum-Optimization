# Referee report: Section 3, Theorem "Finite-population bias" (sampling without replacement)

Note checked: `theory/bias_law/verification/orbital-bias_source.tex`, lines 184-257 (Section 3), plus what it uses
from Section 2 (lines 147-182) and the Appendix (lines 343-352). Theorem "Invariant norms and arbitrary weights"
and estimate (eq:cubes) for independent sums were taken as given, as instructed. Line numbers refer to the `.tex`.

Scripts and logs (all in `theory/bias_law/verification/population/`):

| script | content | log |
|---|---|---|
| `exact_checks.py` | items 1-5: exact enumeration, rational arithmetic, sympy, deterministic quadrature | `exact_checks.log` |
| `median_r3.py` | item 6: coefficient by quadrature, exact law (DP) and Monte Carlo for geometric medians in R^3, scalar sanity check | `median_r3.log` |
| `mc_validation.py` | validation of the Monte Carlo pipeline against the exact law | `mc_validation.log` |
| `residue_classes.py` | exact medians along residue classes of b mod 3 (size of the lattice effect) | `residue_classes.log` |

## Overall verdict

**The theorem and its proof are correct.** I found no wrong step and no real gap. Every ingredient that the
Section 2 / Appendix argument needs for the conditional law is re-established in the proof: (a) an O(1) bound on
the L^1(Q_j) norm of the conditional characteristic function for every cube, (b) an integrated O(eps^3)
comparison with a Gaussian-polynomial transform on Q_0, (c) total mass one and evenness of the order-eps^2 term,
(d) uniform sub-Gaussian tails, (e) uniform polynomial-Gaussian majorants and a uniform lower bound for J.
Only cosmetic fixes are suggested (listed at the end). The numerical test in R^3 confirms the coefficient, the
sign reversal and the scaling; it also shows that on a lattice population the true error in `b m` is of order
1/v and oscillates with the residue class of b, far below the proved bound v^(-gamma).

| item | verdict |
|---|---|
| 1. conditioning identity, formula for Psi_0 | CORRECT |
| 2. large frequencies | CORRECT |
| 3. small frequencies and conditioning | CORRECT (with minor fix: state Q_n, see below) |
| 4. transfer, scalings, exact cumulants | CORRECT (with minor fixes of wording) |
| 5. duality, example, half sampling | CORRECT |
| 6. numerical test | CONSISTENT with the theorem (details and caveats below) |

---

## Item 1. Conditioning identity (lines 202-217) — CORRECT

* Line 206. With I_i i.i.d. Bernoulli(p), sum_i (I_i - p) = |S| - np = |S| - b. Given |S| = b the set S is uniform among
  b-subsets, and T = h^{-1}(sum_{i in S} z_i - p sum_i z_i) = b xi_{n,b}/h because sum_i z_i = 0. Correct.
* Lines 211-216. h C_0 = |S| - b is integer valued **because np = b is an integer** (otherwise {C_0 = 0} is empty).
  Hence int_{-pi h}^{pi h} e^{iuC_0} du = 2 pi h 1{C_0 = 0}, so
  int H_0(t,u) du = 2 pi h E[e^{i<t,T>}; C_0 = 0], and the ratio is the conditional characteristic function.
  The denominator equals 2 pi h P(Bin(n,p) = b).
* Numerical confirmation (`exact_checks.log`, ITEM 1): for six random populations (n = 6..10, N = 1,2,3) the formula
  (periodic trapezoid rule, exact for trigonometric polynomials) agrees with full enumeration of subsets to 5e-16, and
  the denominator equals 2 pi h P(Bin = b) to 12 digits. The denominator divided by sqrt(2 pi) equals
  1 - (1-r)/(12 v) + O(v^-2), uniformly in p (e.g. n = 40000, b = 40: 0.99791888 vs 0.99791666), confirming
  "tends uniformly to sqrt(2 pi)" (line 224) whenever v -> infinity.

## Item 2. Large frequencies (lines 219-224) — CORRECT

* |E e^{is(I-p)}|^2 = |1 - p + p e^{is}|^2 = 1 - 2r(1 - cos s): verified symbolically.
* |H_0| <= D **exactly**: sqrt(1-x) <= e^{-x/2} for x in [0,1] is 1 - x <= e^{-x}; here x = 2r(1 - cos s) <= 4r <= 1.
  2000 random (population, t, u): no violation, max |H_0|/D = 0.999996.
* D is a characteristic function: exp(r(cos x - 1)) = E exp(ix(P_1 - P_2)) with P_1, P_2 i.i.d. Poisson(r/2) (checked
  symbolically), so D(t,u) is the c.f. of sum_i (P_{i,1} - P_{i,2}) (z_i, 1)/h, a symmetric probability law on R^{N+1}.
  The translated-tent inequality of the Appendix (line 346-350) only needs: nonnegative function that is the Fourier
  transform of a positive finite measure. It therefore applies to D directly in N+1 dimensions; no splitting into
  blocks A, B and no AM-GM step is needed here, as the note says.
* r sum_i s_i^2 = t' Sigma_n t + u^2 is **exact**: r/h^2 = 1/n, and the cross term 2u n^{-1} sum_i a_i(t) vanishes because
  sum z_i = 0 (numerically 9e-16). With Sigma_n >= lambda Id this gives D <= exp(-c''(|t|^2 + u^2)) on the centred cube where
  all |s_i| <= (M|t| + |u|)/h are small, uniformly in p.
* Geometry: the tent of support side 4a = ch is >= 2^{-(N+1)} on the concentric cube of side 2a = ch/2 (consistent with
  "side ch" / "side ch/2" in line 224). Q_j x [-pi h, pi h] has sides (c' h, ..., c' h, 2 pi h), so it is covered by
  at most (ceil(2c'/c))^N ceil(4 pi/c) such cubes, a number independent of n, p, j.
* What is needed downstream is int_{Q_j} |Psi_0(t)| dt <= C for j != 0 (Appendix, last sentence of line 350, used
  in the derivation of (eq:cubes)). This is what is obtained:
  int_{Q_j} |Psi_0| <= (int_{Q_j x [-pi h, pi h]} D) / |denominator| <= C / (sqrt(2 pi)(1 - o(1))).
* Remark (not an error): the bound cannot be improved to "small". For the lattice population {-1,2}
  (n = 30, b = 10) there are resonances inside the strip, e.g. (t,u) = (2 pi h/3)(1,1), where D = |H_0| = 1 and
  the shifted tent integral equals the centred one (1.17412 both; `exact_checks.log`). Only boundedness on each cube is
  available, and only boundedness is used.

## Item 3. Small frequencies and conditioning (lines 226-245) — CORRECT

* Cumulants of I - p (sympy): kappa_2 = r, kappa_3 = r(1-2p) = r d, kappa_4 = r(1-6r) = r d_4. Correct.
* Uniform remainder (line 232). f(p,z) = log(1 + p(e^z - 1)) - pz is jointly analytic for |z| <= rho (|e^z - 1| <= 1/2)
  and p in a complex neighbourhood of [0,1]; f(0,z) = 0 and f(1,z) = log e^z - z = 0. Hence g = f/(p(1-p)) is jointly
  analytic and bounded on [0,1] x {|z| <= rho} (its boundary values are e^z - 1 - z at p = 0 and e^{-z} - 1 + z at p = 1),
  and Cauchy's estimate bounds the order-5 Taylor remainder of g(p, .) uniformly in p. The argument is valid.
  Numerically sup_p sup_s |remainder|/(r|s|^5) = 0.00833 = 1/120 for |s| <= 1 on a grid of p reaching 1e-6 and 1 - 1e-6.
* Bookkeeping: r n h^{-j} = eps^{j-2}, so the summed logarithm is
  -(t' Sigma_n t + u^2)/2 + eps d A_3/6 + eps^2 d_4 A_4/24 + O(eps^3 (M|t| + |u|)^5); exponentiation gives the displayed
  polynomial (line 235). |d| <= 1 and |d_4| <= 1, so all constants are uniform in p. The quadratic form is
  jointly nondegenerate (>= min(lambda,1)(|t|^2 + u^2)), so the Appendix Taylor estimate applies.
* Region c_2 h <= |u| <= pi h, |t| <= c_1 h (line 238): |a_i(t)|/h <= M c_1 < c_2/2, hence
  c_2/2 < |s_i| < pi + c_2/2 and dist(s_i, 2 pi Z) >= min(c_2/2, pi - c_2/2) = c_2/2. So 1 - cos s_i >= c and
  |H_0| <= D <= exp(-c r n) = exp(-c h^2). Correct (the upper limit |u| <= pi h is what keeps s_i away from 2 pi).
  Its contribution, integrated over Q_0, is <= C h^{N+1} e^{-c h^2} = O(eps^3).
* u-integration (sympy, `exact_checks.log`): with m_k(t) = n^{-1} sum_l a_l(t)^k,
  E_u A_3 = -i(m_3 + 3 m_1) = n^{-1} sum (i a_l)^3 because m_1 = 0 (the surviving u^2 term is exactly the one killed by
  sum_i a_i = 0; odd powers of u vanish). So the first-order term is eps d K_{3,n}[(it)^3]/6. Correct.
* The order-eps^2 term, which the note does not write out, is
  R(t) = d_4 (m_4 + 6 m_2 + 3)/24 - d^2 (m_3^2 + 9 m_2^2 + 18 m_2 + 15)/72,
  real, even, of degree 6, with coefficients bounded in terms of M. R(0) = d_4/8 - 5d^2/24 = -(1-r)/12, which
  reproduces the local-limit expansion of the denominator found in item 1. After division,
  **Q_n(t) = R(t) - R(0) = d_4 m_4/24 - d^2 m_3^2/72 - d^2 m_2^2/8 - (r/2) m_2**, so Q_n is even and Q_n(0) = 0. Correct.
  Consistency check: the quadratic part -(r eps^2/2) t' Sigma_n t = -(1/(2n)) t' Sigma_n t says that the covariance of the
  conditional law is (1 + 1/n) Sigma_n + O(n^-2), in agreement with the exact n/(n-1) Sigma_n of line 252.
* Whole statement tested numerically (scalar populations, exact Psi_0): h^3 sup_{|t| <= min(6, h/2)} |Psi_0 - Gamma_n| with the
  explicit Q_n above stays bounded as n = 48 -> 3072 (lattice population, p = 1/4: 0.028, 0.057, 0.054, 0.054;
  p = 1/2: 0.009, 0.006, 0.003, 0.001; see the log for p = 3/4, 1/12 and a non-lattice population), while
  h^2 sup |Psi_0 - first-order transform| converges to a nonzero constant (0.153). So the expansion, the first-order
  coefficient eps d K_{3,n}/6 and the second-order term are all right.
* Line 245, "integrated error on Q_0 remains O(eps^3)": numerator = sqrt(2 pi) e^{-m_2/2}(1 + eps c_3 + eps^2 R) + E with
  int_{Q_0} |E| = O(eps^3), denominator = sqrt(2 pi)(1 + eps^2 R(0)) + O(eps^3). Correct.

## Item 4. Transfer to the median (lines 247-256) — CORRECT

* Maclaurin (line 248). E exp(s sum_{i in S} y_i) = e_m(x)/C(n,m) with x_i = e^{s y_i} > 0 is the definition of the normalized
  elementary symmetric mean. Maclaurin's inequality says (e_k/C(n,k))^{1/k} is nonincreasing in k, so
  e_m/C(n,m) <= (e_1/n)^m: the direction is the right one. Then n^{-1} sum e^{s y_i} <= e^{M^2 s^2/2} by Hoeffding's lemma
  (centred, |y_i| <= M). 400 random cases by enumeration: no violation. m_0 = min(b, n-b) <= 2h^2 because
  max(p, 1-p) >= 1/2. The complement trick is legitimate because sum_{S} z_i = -sum_{S^c} z_i. So the coordinates of
  T | C_0 = 0 are sub-Gaussian with a constant depending only on M. (This is Hoeffding's 1963 comparison; fine.)
* Uniformity. lambda Id <= Sigma_n <= M^2 Id; |K_{3,n}| <= M^3; the constants of Lemma (laplace) depend only on nu and G;
  J(Sigma) is continuous and positive definite on the compact set of admissible covariances, so its smallest eigenvalue
  is bounded below; D^4P(0) is bounded there. All constants are uniform as claimed.
* The potential P_w is P + eps d D^3P[K_{3,n}]/6 + eps^2 (even terms). In the sphere argument of Section 2 (line 179) only
  three facts about the eps^2 term are used: bounded second derivatives (for P_w * h_s = P_w + O(delta^2) and the Hessian
  lower bound), evenness (gradient O(eps^2 |x_0|) = O(eps^3) at x_0), and a polynomial-Gaussian majorant for |p_w|.
  All hold for Q_n. Hence gradient P_w(x_0) = O(eps^3) at x_0 = eps d g_*^{(n)} (this is tau_3 -> eps d in line 179).
* Scalings and sign. H(x) = E nu(x + T) is minimised near x_0, and medians of T are m_T = -x, so
  m_T = -eps d g_* + O(d_eps). T = (b/h) xi, hence m_T = (b/h) m and
  **b m = -h eps d g_* + O(h d_eps) = -(1-2p) g_*^{(n)} + O(h d_eps)**. Sign: positive skewness and p < 1/2 give a
  median below the mean, as for independent sampling; it flips for p > 1/2. Correct.
* Exponent. h d_eps = eps^{q-1} L^{(N+1)/4} with q - 1 = kappa/(N - kappa), eps = v^{-1/2}, so the rate is
  v^{-kappa/(2(N-kappa))} = v^{-gamma}; L = log(e sqrt v) <= log(ev). kappa = 1/2, N = k^2 gives 1/(4k^2 - 2). Correct.
* (n-2b)/(n-2) - (1-2p) = 2(n-2b)/(n(n-2)) = O(1/n) and 1/n <= 1/(4v) <= v^{-gamma}: absorbed. Correct.
  (The theorem therefore cannot distinguish the two coefficients; the note says so honestly.)
* Exact conditional cumulants (lines 251-255): verified in exact rational arithmetic by full enumeration for
  n = 6, b = 2 (scalar and 2-dimensional), n = 6, b = 4, n = 7, b = 3 (2-dim) and n = 9, b = 5 (3-dim), all entries of the
  covariance and of the third-moment tensor. Both identities hold exactly. The ratio of multipliers is
  (n-2b)/(h(n-2)) (sympy). Since g_* is homogeneous of degree (1, -1) in (K_3, Sigma)
  (P_{c Sigma}(x) = sqrt c P_Sigma(x/sqrt c)), the coefficient computed from the exact cumulants is
  ((n-2b)/(h(n-2))) g_*, i.e. after multiplication by h exactly the (n-2b)/(n-2) of (eq:fp). The remark is right.

## Item 5. Exact identities and the example (line 199) — CORRECT

* [pb, Proposition 18] is indeed the complement-duality proposition (counted in `polar_bias.tex`, line 600).
* Duality Med(xi_{n,n-b}) = -(b/(n-b)) Med(xi_{n,b}): (n-b) m' = -b m, and the prediction satisfies it identically,
  -((n - 2(n-b))/(n-2)) g_* = +((n-2b)/(n-2)) g_*. Verified by enumeration (n = 10, b = 3, 4, Euclidean median in R^2) and
  on the exact R^3 numbers of item 6 (n = 108: 0.007344 vs 0.007344; n = 216: 0.028799 vs 0.028799).
* Half sampling: "symmetry makes zero a median exactly" is right for an **arbitrary** centred population, not just a
  symmetric one. The symmetry in question is complementation: xi_{n,n/2} and -xi_{n,n/2} have the same law, so the convex
  objective is even and 0 minimises it. (Checked on a generic asymmetric population, n = 10, b = 5: gradient at 0 is
  1e-17 and the objective at 0 equals the minimum.) It is the b = n/2 case of the duality; the wording "symmetry" is
  slightly elliptic but correct.
* Example: entries -1, 2 with probabilities 2/3, 1/3 are centred; a product law on k^2 entries has probabilities with
  denominator 3^{k^2}, so populations of size a multiple of 3^{k^2} realise it exactly with Sigma_n, K_{3,n} equal to the
  moments of the law; the factor 4 makes n/4 and 3n/4 integers. (n-2b)/(n-2) -> 1/2 and -1/2, so the scaled **negative**
  medians tend to +g_*/2 and -g_*/2. Correct, including signs.

## Item 6. Numerical test (Euclidean norm on R^3)

Setting: N = 3, G = SO(3); the orbital integral is sin(u)/u, so kappa = 1/2 is admissible and gamma = 1/10.
The scalar case N = 1 is **not** covered by the theorem (N >= 3 is required); it is included only as a sanity check.

**Coefficient.** J and D^4P(0)[K_3, .] were computed by Gaussian integration by parts in polar coordinates
(exact radial chi moments, Gauss-Legendre x trapezoid on the sphere; two different Hermite forms agree to all printed digits,
and J agrees with E[(I - uu')/|Z|] to 6e-15). For Sigma = sigma^2 Id the closed form of [pb] (Remark after the
first-order law), g_* = E[zeta |zeta|^2]/(2 sigma^2 (N+2)), is **verified** (generic K_3, sigma^2 = 0.5 and 3: agreement to
all printed digits; also analytically from E|x+Z| = sqrt(2/pi)(2 + rho^2/3 - rho^4/60 + ...) for N = 3).

**B. Lattice population, exact.** Product law on {-1,2}^3 realised by repetition (n a multiple of 108),
Sigma_n = 2 Id, g_* = 0.1 (1,1,1). The law of the batch sum is computed exactly by dynamic programming over the 8
categories (normalisation checked against C(n,b)), and the geometric median by Newton (checked against Nelder-Mead and a
line scan). By exchangeability the three coordinates are equal.

| n | b | p | v | b m (exact) | -((n-2b)/(n-2)) g_* | diff | diff x v |
|---|---|---|---|---|---|---|---|
| 108 | 9 | 1/12 | 8.2 | 0 (pinned at the atom 0) | -0.08491 | +0.0849 | 0.70 |
| 108 | 27 | 1/4 | 20.2 | -0.00734 | -0.05094 | +0.0436 | 0.88 |
| 108 | 54 | 1/2 | 27 | 0 | 0 | 0 | |
| 108 | 81 | 3/4 | 20.2 | +0.00734 | +0.05094 | -0.0436 | -0.88 |
| 216 | 18 | 1/12 | 16.5 | -0.03274 | -0.08411 | +0.0514 | 0.85 |
| 216 | 54 | 1/4 | 40.5 | -0.02880 | -0.05047 | +0.0217 | 0.88 |
| 216 | 81 | 3/8 | 50.6 | -0.00734 | -0.02523 | +0.0179 | 0.91 |
| 216 | 108 | 1/2 | 54 | 0 | 0 | 0 | |
| 216 | 162 | 3/4 | 40.5 | +0.02880 | +0.05047 | -0.0217 | -0.88 |
| 432 | 36 | 1/12 | 33 | -0.05829 | -0.08372 | +0.0254 | 0.84 |
| 432 | 108 | 1/4 | 81 | -0.03943 | -0.05023 | +0.0108 | 0.88 |

(Curiosity, unexplained: the exact values at (n,b) = (108,27) and (216,81) agree to six decimals, -0.0073437 and
-0.0073439; both were confirmed by two optimisers and a line scan. I did not pursue it.)

The sign reversal at p = 3/4 and the exact zero at p = 1/2 are reproduced, and b m converges to the predicted value
with error ~ 0.88/v along b in 3Z. The error is a **lattice effect of order 1/v**: the batch sum lives on
-b(1,1,1) + 3Z^3, and when 3 | b the origin is an atom of mass ~ v^{-3/2} which pulls the median towards 0 (it pins it
exactly at 0 for n = 108, b = 9). Along the residue classes of b mod 3 (`residue_classes.log`, exact):

| n | b | b mod 3 | v | b m (exact) | prediction | diff | diff x v |
|---|---|---|---|---|---|---|---|
| 216 | 52 | 1 | 39.5 | -0.05730 | -0.05234 | -0.00496 | -0.196 |
| 216 | 53 | 2 | 40.0 | -0.04804 | -0.05140 | +0.00336 | +0.134 |
| 216 | 54 | 0 | 40.5 | -0.02880 | -0.05047 | +0.02167 | +0.878 |
| 216 | 55 | 1 | 41.0 | -0.05427 | -0.04953 | -0.00474 | -0.194 |
| 216 | 56 | 2 | 41.5 | -0.04532 | -0.04860 | +0.00328 | +0.136 |
| 432 | 106 | 1 | 80.0 | -0.05360 | -0.05116 | -0.00244 | -0.195 |
| 432 | 107 | 2 | 80.5 | -0.04902 | -0.05070 | +0.00168 | +0.135 |
| 432 | 108 | 0 | 81.0 | -0.03943 | -0.05023 | +0.01080 | +0.875 |
| 432 | 109 | 1 | 81.5 | -0.05215 | -0.04977 | -0.00238 | -0.194 |
| 432 | 110 | 2 | 82.0 | -0.04764 | -0.04930 | +0.00166 | +0.136 |

So v (b m - prediction) oscillates between about -0.19, +0.13 and +0.88 according to b mod 3. This is the
three-residue-class oscillation familiar from the scalar lattice case, but here it sits at relative order 1/v instead
of order 1, as the theorem asserts for N >= 3 (the proved bound v^{-1/10} polylog is far from sharp in this geometry;
atoms of mass eps^N move the T-median by O(eps^3) when N = 3).

**C. Lattice population, larger n, Monte Carlo** (4e7 batches, multivariate-hypergeometric counts; the pipeline
reproduces the exact value at n = 216, b = 54 over four seeds with |z| <= 1.6, and at n = 108, b = 27 with |z| <= 2.1, see
`mc_validation.log`; it is unreliable only when the median is within noise of an atom, e.g. n = 216, b = 81, a case not
used below). Mean over the three exchangeable coordinates:

| n | b | p | v | b m (MC) | se | predicted | 0.88/v expected from B |
|---|---|---|---|---|---|---|---|
| 864 | 216 | 1/4 | 162 | -0.0472 | 0.0017 | -0.0501 | +0.0054 |
| 864 | 432 | 1/2 | 216 | +0.0005 | 0.0018 | 0 | 0 |
| 864 | 648 | 3/4 | 162 | +0.0460 | 0.0017 | +0.0501 | -0.0054 |
| 1728 | 432 | 1/4 | 324 | -0.0474 | 0.0024 | -0.0501 | +0.0027 |
| 1728 | 1296 | 3/4 | 324 | +0.0484 | 0.0024 | +0.0501 | -0.0027 |
| 3456 | 864 | 1/4 | 648 | -0.0532 | 0.0035 | -0.0500 | +0.0014 |
| 1728 | 144 | 1/12 | 132 | -0.0748 | 0.0016 | -0.0834 | +0.0064 |

All within about two standard errors of prediction + 0.88/v (0.85/v for p = 1/12).

**D. Non-isotropic, non-product population** (12 distinct atoms in R^3, eigenvalues of Sigma_n 0.23, 0.89, 4.02;
g_*^{(n)} = (0.1040, 0.0190, 0.3011) by quadrature; 1.2e7 batches). b m versus -((n-2b)/(n-2)) g_*^{(n)}:

| n | b | p | v | b m (MC) | predicted | (b m - pred)/se per coordinate |
|---|---|---|---|---|---|---|
| 120 | 30 | 1/4 | 22.5 | (-0.0510, -0.0080, -0.1533) | (-0.0529, -0.0097, -0.1531) | +1.0, +1.2, -0.1 |
| 120 | 60 | 1/2 | 30 | (-0.0042, -0.0018, -0.0046) | 0 | -1.9, -1.2, -1.5 |
| 120 | 90 | 3/4 | 22.5 | (+0.0526, +0.0109, +0.1504) | (+0.0529, +0.0097, +0.1531) | -0.1, +0.9, -1.0 |
| 480 | 48 | 1/10 | 43.2 | (-0.0818, -0.0136, -0.2442) | (-0.0835, -0.0153, -0.2419) | +0.6, +0.9, -0.6 |
| 480 | 120 | 1/4 | 90 | (-0.0522, -0.0099, -0.1508) | (-0.0522, -0.0096, -0.1512) | 0.0, -0.1, +0.1 |
| 480 | 360 | 3/4 | 90 | (+0.0498, +0.0076, +0.1546) | (+0.0522, +0.0096, +0.1512) | -0.6, -0.7, +0.7 |
| 1920 | 480 | 1/4 | 360 | (-0.0562, -0.0123, -0.1580) | (-0.0520, -0.0095, -0.1507) | -0.5, -0.5, -0.7 |
| 1920 | 1440 | 3/4 | 360 | (+0.0600, +0.0117, +0.1641) | (+0.0520, +0.0095, +0.1507) | +1.0, +0.4, +1.3 |

Standard errors per coordinate are 0.001-0.003 (n = 120), 0.003-0.005 (n = 480), 0.005-0.010 (n = 1920). The prediction,
including the direction of g_*^{(n)} for a non-isotropic covariance and the sign reversal, is confirmed at the 2-5%
level of |g_*|. (At p = 1/2 the exact median is 0; the MC value is 1.2-1.9 se away on each coordinate, i.e. a
moderately unlucky draw, chi-square 7.2 on 3 degrees of freedom, p about 0.07; I report it as it came out.)

**Honest limits of the test.** (i) The numerics cannot discriminate (n-2b)/(n-2) from 1-2p: the difference is
O(g_*/n) < 0.001, below both the 1/v lattice term and the MC error. (ii) They say nothing about the exponent gamma;
the observed error is ~1/v. (iii) The Euclidean norm is the easiest invariant geometry; the nuclear-norm case was not
simulated.

**E. Scalar case (not covered).** For the scalar lattice population {-1,2} the true median of the batch sum is a
lattice point: b x median = 0, -1, +1 for b = 1728, 1729, 1730 (n = 6912), no convergence, as expected and as the
N >= 3 hypothesis reflects. The mid-point-interpolated median reproduces the formal coefficient with
g_* = kappa_3/(6 sigma^2) = 1/6: -0.0833 against -(n-2b)/(n-2)/6 = -0.0834 at n = 6912, b = 1728, and +0.0831 against
+0.0834 at n = 1728, b = 1296. So the coefficient and the sign reversal are right at the level of the smooth part.

---

## Suggested minor fixes (none affects validity)

1. Line 189-196: state n >= 3 and that b = np is an integer with 1 <= b <= n-1 (both implicit; "v sufficiently large"
   forces n >= 4v).
2. Line 240-244: write Q_n explicitly,
   Q_n = d_4 m_4/24 - d^2 m_3^2/72 - d^2 m_2^2/8 - (r/2) m_2, m_k = n^{-1} sum_l a_l(t)^k; note that it is **not** of the form
   (eq:P) (it has a second-order term -(1/(2n)) t' Sigma_n t, the n/(n-1) covariance inflation), which is why line 250
   correctly appeals only to evenness and bounded derivatives.
3. Line 238: say explicitly that the bound |u| <= pi h is used for dist(s_i, 2 pi Z) >= c_2/2, and that the neglected
   region contributes <= C h^{N+1} e^{-c_4 h^2} after integration over Q_0.
4. Line 248: the constant is C = 1/2 (Hoeffding's lemma); the population coordinate must be centred (it is).
5. Line 199: replace "symmetry" by "complement symmetry (the law of xi_{n,n/2} is symmetric for every centred
   population)".
6. Line 224: mention that Q_0 is chosen inside {|t| <= c_1 h} (Euclidean ball), i.e. a sqrt(N) <= c_1 h.
7. Optional remark: for the Euclidean norm in R^3 the lattice contribution to b m is O(1/v) and depends on the residue
   class of b (table above); the rate v^{-gamma} is an upper bound only, as the note already says for (eq:square).
