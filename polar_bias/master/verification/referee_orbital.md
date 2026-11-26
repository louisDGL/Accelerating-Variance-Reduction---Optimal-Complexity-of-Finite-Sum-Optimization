# Referee report: `orbital-bias_source.tex`, Sections 1-2 and Appendix

Scope: Theorem 1 (square-matrix bias), Theorem 2 (invariant norms, arbitrary weights), proof of the orbital
assertions, SO(3) sharpness paragraph, Lemma 3 (`lem:laplace`), comparison of convex potentials, localisation,
Appendix (Fourier cube estimate). Sections 3-4 were not read.

Scripts (all in `polar_bias/master/verification/orbital/`, run with `polar_bias/.venv/bin/python`):
`t1_algebra.py`, `t2_so3.py`, `t3_exponents.py`, `t6_median.py` (output saved in `t6_output.txt`).

## Overall verdict

I tried to break every step and found **no error that affects a stated result** in the scope. Every exponent
was re-derived and matches. What I found are minor imprecisions (list at the end), one citation I could not
verify, and one harmless non-optimality (the power of the logarithm). The independent Monte Carlo test of the
3x3 theorem supports the statement in all three residue classes, with slow convergence and genuine locking at
0 for small b in the class b = 0 mod 3.

| Item | Verdict |
|---|---|
| 1. Orbital oscillation (l.113-120), sharpness (l.122) | CORRECT WITH MINOR FIX |
| 2. Lemma `laplace` (l.126-143) | CORRECT |
| 3. Comparison, sandwich, subgradients, localisation, exponents (l.145-182) | CORRECT WITH MINOR FIX (cosmetic) |
| 4. Appendix (l.343-352) | CORRECT |
| 5. Scope / example formula (l.58, l.61-66) | CORRECT |
| 6. Numerical test of the theorem | SUPPORTS the theorem; class 0 mod 3 converges slowly, locked at 0 for b = 12 |

---

## 1. Orbital oscillation — CORRECT WITH MINOR FIX

**Reduction "nonzero derivative of order <= s" => (1+u)^{-1/s} (l.115).** Correct. This is exactly
Stein, *Harmonic Analysis*, Ch. VIII, 2.2, Prop. 5 (finite-type phases), made uniform in parameters:

- Polarisation: homogeneous polynomials of degree j are spanned by j-th powers of linear forms, so a nonzero
  j-th differential has a nonzero pure directional derivative. OK. (The statement "not all derivatives of
  orders 1..s vanish" is chart independent, so "in local coordinates" is harmless.)
- Uniformity: the phase `<e, g y>` is jointly smooth in (e,y,g) on the compact set S x S x G, so nonvanishing
  of the selected directional derivative is an open condition and a finite cover exists. OK.
- j >= 2: one-dimensional van der Corput on line segments of a convex coordinate box, with amplitude
  (cutoff times smooth Haar density) handled by the standard corollary
  `|int e^{iu phi} psi| <= c_j u^{-1/j} (|psi(b)| + int |psi'|)`; no monotonicity is needed for j >= 2. OK.
- j = 1: van der Corput needs monotone phi', which is not available; the note correctly replaces it by one
  integration by parts using bounded second derivatives (l.115, parenthesis). OK.
- Since j <= s and the integral is trivially <= 1, `(1+u)^{-1/j} <= (1+u)^{-1/s}`. OK.

Minor fixes:
- (l.115) The partition of unity must live on the product S x S x G (functions psi_a(e,y,g)), not on G alone,
  since the cover of G that is admissible depends on (e,y). With that reading all amplitude bounds are
  uniform. The sentence "a smooth partition subordinate to it" is compatible with this but should say so.
- (l.115, l.338) Citation. The tool is the classical one-dimensional van der Corput lemma plus Stein's
  Prop. 5; the multidimensional Carbery-Christ-Wright theory is not needed. I could not check offline that
  "Lemma 1.1" of [ccw] is the classical lemma with amplitude; cite Stein VIII.1.2 (Prop. 2 and Corollary) and
  VIII.2.2 (Prop. 5) instead or in addition.

**Irreducible case (l.117).** Correct. `V_{j+1} = V_1 + g V_j = V_j + g V_j` (as V_1 is contained in V_j).
If `V_{j+1} = V_j` then V_j is g-invariant, hence G-invariant (connectedness), nonzero, hence E. So the
dimension increases strictly and `V_N = E` (uses N >= 2 for "proper invariant line"). With left-invariant
fields, `X_{A_1}...X_{A_i} phi(g) = <e, g A_1...A_i y>`, so the values over i <= N span `<g^T e, E>`, nonzero
for e != 0. An operator `X_{A_1}...X_{A_i}` has order i and no zeroth-order term, so some coordinate
derivative of order in [1, i] is nonzero at g. Gives kappa = 1/N. OK.

**Matrices (l.119).** Algebra verified by hand and numerically (`t1_algebra.py`):
- `A_{ih}A_{hj} = E_{ij}` for distinct i,h,j; `-A_{ij}^2 = E_{ii}+E_{jj}`; with three indices these give every
  `E_{ii}`. Span of products of two skew s x s matrices: dimension 1 for s = 2 (only multiples of Id), s^2 for
  s = 3,...,6.
- In singular-vector bases CM gives all units with column index <= r, MD all with row index <= r, and
  `A_{pi} M A_{iq'} = s_i E_{pq'}` for p,q' > r >= i (checked: `A_{pi}M = s_i E_{pi} - s_p E_{ip}` and
  s_p = 0). The orthogonal change of basis may have determinant -1; conjugation still preserves so(k). OK.
- The mixed derivative is genuinely needed: without it the span at a rank-one 3x3 M has dimension 5 of 9
  (rank two: 8 of 9); with it, 9 of 9 at every rank, including equal singular values. Same for 3x4, 4x4, 5x3,
  5x5 at every rank.
- Rectangular k != l: not needed for Theorem 1, but the claim of l.108 is true. In fact it is true in more
  cases than stated: the span is full for 3x2, 4x2, 2x5, 3x1 as well, i.e. whenever max(k,l) >= 3 (the note
  only claims k,l >= 3; not an error).
- Size 2, explicit failure: y = Id/sqrt2, e = diag(1,-1)/sqrt2. Then `U y V^T` is a rotation matrix divided
  by sqrt2 and `<e, U y V^T> = 0` for all U,V in SO(2) (numerically 1e-16). The orbital integral is
  identically 1: no decay at all. Algebraically the derivative span at M = Id is span{J, Id}, dimension 2 of
  4. So the hypothesis (eq. `orbits`) fails for every kappa > 0 in size 2, consistent with the theorem's
  restriction k >= 3.

**Numerics for SO(3) x SO(3) (`t2_so3.py`).** By bi-invariance e and y can be taken signed-diagonal, so the
sup is over (tau, sigma) in S^2 x S^2 with phase `sum tau_i sigma_j U_ij V_ij`. Search over 3289 pairs
(structured: identity-like, rank one, rank two, signed, permuted; plus 3000 random) with 2e5 Haar samples,
local refinement, and re-evaluation of the best candidates on a fresh sample of 2e6 (noise 7e-4):

| u | sup | sup * sqrt(u) | identity sqrt(3/(pi u)) | rank one Si(u)/u |
|---|---|---|---|---|
| 5 | 0.408 | 0.91 | 0.437 | 0.310 |
| 10 | 0.301 | 0.95 | 0.309 | 0.166 |
| 20 | 0.219 | 0.98 | 0.219 | 0.077 |
| 40 | 0.157 | 0.99 | 0.155 | 0.040 |
| 80 | 0.110 | 0.98 | 0.109 | 0.020 |
| 160 | 0.078 | 0.99 | 0.077 | 0.010 |
| 320 | 0.055 | 0.98 | 0.055 | 0.005 |

Fitted exponent for u >= 20: -0.5005. The maximiser found is always e = y = +-Id/sqrt3 (up to signs), and
`sup * sqrt(u) -> sqrt(3/pi) = 0.977`. Rank-one pairs decay like 1/u (exactly `Si(u)/u`, since U_11 is
uniform on [-1,1]). Fully consistent with u^{-1/2} and with its sharpness.

**Sharpness paragraph (l.122).** Correct. `tr U = 4s^2 - 1` (checked to 1e-15 on 2e6 samples), density
`(2/pi) sqrt(1-s^2)` (histogram), and by high-precision quadrature
`E e^{iu tr U} * sqrt(pi u) * e^{iu - i pi/4} = 1.0082, 1.0030, 0.9995, 1.0002, 1.000001` at
u = 10, 40, 160, 640, 2560. So the constant `e^{-iu+i pi/4}/sqrt(pi u)` is right. The endpoint contributions
are O(u^{-3/2}) (amplitude vanishes like a square root) and the integration by parts away from 0 is valid.
Remark: for unit e = y = Id/sqrt3 the frequency is u/3, which the text covers by "up to a constant rescaling".

## 2. Lemma `laplace` (l.126-143) — CORRECT

- `mu = Delta nu >= 0` locally finite (convex implies subharmonic). Degree-one homogeneity of nu makes mu
  homogeneous of degree -1 as a distribution, so `mu(B(0,s)) = s^{N-1} mu(B(0,1))`. OK.
- `lambda = Delta(chi nu) = chi mu + 2 grad chi . grad nu + nu Delta chi` is valid (nu Lipschitz). It is
  G-invariant because chi is radial, nu is invariant and the Laplacian commutes with O(E).
  `|lambda|(B(0,s)) <= C s^{N-1}` for s <= 2: for s <= 1 only mu contributes; for 1 < s <= 2 bound by a constant. OK.
- Averaging: `hat lambda(t) = int int_G e^{-i<t,gz>} dg d lambda(z)` by invariance of the signed measure and
  Fubini (|lambda| finite). The inner integral is bounded by `C(1+|t||z|)^{-kappa}` for every z, because the
  hypothesis is a sup over all unit y; nothing is required of the orbit dimension. Then take absolute values
  against |lambda|. Valid for signed measures. OK.
- Dyadic sum: for |t| >= 1, annuli |z| ~ 2^{-k} >= 1/|t| contribute `|t|^{-kappa} 2^{-k(N-1-kappa)}`
  (geometric because N-1 > kappa), the rest contributes `|t|^{-(N-1)} <= |t|^{-kappa}`. OK.
- `-|t|^2 hat F_1 = hat lambda`, `|hat F_1| <= ||F_1||_1`, so `|hat F_1(t)| <= C(1+|t|)^{-2-kappa}`.
  `F_R(z) = R F_1(z/R)` gives `hat F_R(t) = R^{N+1} hat F_1(Rt)`. OK.

## 3. Comparison, sandwich, subgradients, localisation — CORRECT (cosmetic remarks)

Exponents (hand and `t3_exponents.py`):
- On Q_j, j != 0: `|t|_inf >= a(2|j|_inf - 1) >= a|j|_inf`, so `|t| ~ |j|/eps`. OK.
- `R^{N+1}(R|j|/eps)^{-2-kappa} = R^{N-1-kappa} eps^{2+kappa} |j|^{-2-kappa}`. OK.
- With T = eps/s >= 1: `sum_{j != 0} |j|^{-2-kappa}(1+|j|/T)^{-D} <= C T^{N-2-kappa}` (needs N > 2+kappa and
  D > N-2-kappa; both hold). Total `eps^{2+kappa}(eps/delta)^{N-2-kappa} = eps^N delta^{-(N-2-kappa)}`. OK.
- `eps^N = delta^{N/q} = delta^{N-kappa}`, so the sum is `C R^{N-1-kappa} delta^2`. OK.
- `||f||_1 <= ||F_R||_1 <= C R^{N+1}`. OK.
- Truncation: for |x| <= 2 and s <= 2 delta <= 1 the two functions differ only if `|X| > R-3`, by at most
  `C(1+|X|)`; coordinatewise Hoeffding with sum a_i^2 = 1 gives `e^{-cR^2}`, and `R = C_0 sqrt(L)` gives
  `(eps/e)^{c C_0^2} <= eps^4`. Same for |p_w| (coefficients tau_3 <= eps, tau_4 <= eps^2 bounded). OK.
- `(P_w * h_s) = P_w + O(s^2)` by evenness and bounded Hessian. OK.
- Total: `C(eps^3 R^{N+1} + R^{N-1-kappa} delta^2 + eps^4 + delta^2) <= C R^{N+1} delta^2` because
  `eps^3 <= eps^{2q}` iff `2q <= 3` iff `N >= 3 kappa`. True. `R^{N+1} delta^2 = C eps^{2q} L^{(N+1)/2} = C d_eps^2`. OK.
- Edgeworth operator (eq. `P`): checked against the cumulant expansion of the characteristic function;
  coefficients tau_3/6, tau_4/24, tau_3^2/72 are right.
- Sandwich (l.170-175): `2 nu(z-u) <= nu(z) + nu(z-2u)` integrated against h_delta gives
  `2 nu*h_delta <= nu + nu*h_{2 delta}` (change of variables w = 2u is right); Jensen gives the upper bound.
  Then `H >= 2P_w - P_w - 3C d^2`, `H <= P_w + C d^2`. OK. Numerically the two slacks are >= 0 at regular and
  singular points.
- Values to subgradients (l.177): `d <v,u> <= H(x+du) - H(x) <= d <grad P_w(x),u> + C d^2`, and the same with
  -u. Error O(d). OK. The expectation of a measurable selection of the sample subgradient lies in the
  subdifferential of H, so the remark at l.108 holds.
- J > 0 (l.179): correct for every norm (if d_vv nu = 0 then nu is affine along v, contradicting
  nu(tv) = |t| nu(v)); pairing with the positive Gaussian density is right.
- `grad P_w(x_0) = O(eps^3)`: P is even, so `D^3 P(0) = D^5 P(0) = 0`; `grad P(x_0) = J x_0 + O(|x_0|^3)`;
  the cubic term gives `(tau_3/6) D^4P(0)[K_3,.] + O(tau_3 |x_0|^2) = -tau_3 J g_* + O(eps^3)`; the remaining
  two are `O(eps^2 |x_0|)`. They cancel to O(eps^3). OK.
- Sphere localisation: `<v,u> >= -C eps^3 + (cA - C) d_eps > 0` since `eps^3 << d_eps` (q < 3/2); exterior
  minimiser contradicts the subgradient inequality. `sqrt(rho_2) tau_3 = rho_3/rho_2`. Sign: H is a function of
  the shift x and the median is -x, giving `m + (rho_3/rho_2) g_*`. OK.
- Equal weights: `sqrt(rho_2) d_eps = b^{-(1+q)/2} L^{(N+1)/4}` and for kappa = 1/2,
  `(1+q)/2 = 1 + 1/(4N-2)` (sympy: identity). `L = log(e sqrt b) <= log(eb)`. Theorem 1 follows. OK.

Cosmetic:
- (l.163) "|X| > R-4" can be "R-3"; harmless.
- (l.168, l.88) The logarithm is not optimal. The main term carries `R^{N-1-kappa}` and the `eps^3 R^{N+1}`
  term is polynomially smaller, so `d_eps` could be `eps^q L^{(N-1-kappa)/4}`. The stated bound is true.
- (l.70) The restriction kappa <= 1/2 is only used through `N > 2+kappa` and `N > 3 kappa`.

## 4. Appendix (l.343-352) — CORRECT

- Blocks: filling A until its sum first reaches 1/4 gives a sum below `1/4 + eps^2 <= 1/2`, so B has at
  least 1/2. `|Psi| = sqrt(g_A g_B) <= (g_A+g_B)/2`. OK.
- Tent: support `[-2a,2a]^N`, inverse transform is a Fejer kernel >= 0;
  `int g_A(t) D(t-t_0) dt = c E[e^{i<t_0,Y>} Dcheck(Y)] <= c E Dcheck(Y)`. On the support
  `|a_i t| <= 2 sqrt(N) c_0 <= u_0`, so `g_A <= e^{-c|t|^2}`. Shifted tent >= 2^{-N} on `t_0 + [-a,a]^N`. OK.
  This agrees with [db] Lemma 1 line by line (different normalisation of the tent).
- Near zone: remainder `sum a_i^5 |t|^5 <= eps^3 |t|^5`; cubic and quartic `<= C(eps|t|^3 + eps^2|t|^4)`;
  with `eps|t| <= sqrt(N) c_0` the nonquadratic exponent is `<= lambda_min |t|^2/4`. Dropped terms: r, cq, q^2,
  r-cross terms and `|w|^3 e^{|w|}`; the largest power is `(eps^3|t|^5)^3 = eps^9 |t|^15`, so `1+|t|^15`
  is a valid (generous) majorant; [db] Lemma 2 has |t|^9 using `eps|t| <= C`. OK.
- Outside Q_0 the Gaussian-polynomial transform has mass `e^{-c/eps^2}`. Fourier inversion is legitimate
  (hat f integrable). OK.
- Countable weights: the product converges since sum a_i^2 = 1 and the mean is zero; the block construction
  and Hoeffding extend. OK.

## 5. Scope and the example — CORRECT

- Nothing in the proof of Theorem 2 is specific to the nuclear norm: Lemma 3 uses only convexity,
  homogeneity and invariance; the sandwich uses only convexity; J > 0 is proved for every norm; P is smooth
  for every norm because Sigma > 0. So `g_*` in (eq. `coefficient`) is well defined for operator and Ky Fan
  norms, which are invariant under O(k) x O(k), hence under SO(k) x SO(k). Only the side remark
  "grad P = E polar" (l.48) is nuclear-specific, and it is labelled as such.
- Example (l.61-66): entries -1, 2 with probabilities 2/3, 1/3 have mean 0, sigma^2 = 2, kappa_3 = 2.
  [pb] Theorem `thm:two`(a),(c) with `A = B = kappa_3 11^T` ([pb] Corollary `cor:dim`) gives
  `-b Med = (a_1+a_2) kappa_3/(6 sigma^2) 11^T = (3+2q_{k,k})/(2(k+2)^2) 11^T`. Matches. Monte Carlo:
  `q_{3,3} = 0.68691 +- 0.00005` (cross-check `m_3/m_1 = 6 - q` holds to 4e-5), so `g_* = 0.08748` per entry.
- Sign descent values "0, 1, -1" (l.66): nearest point of `b + 3Z` to 1/6; matches [db] `thm:sign`(b)(ii).
- The theorem/corollary numbers quoted from [pb] and [db] ("Corollary 8", "Theorem 12", "Conjecture 18",
  "Theorem 8", "Lemmas 1-3") are consistent with counting the shared counters in the sources; I did not
  compile them.

## 6. Independent numerical test of Theorem 1 (`t6_median.py`, `t6_output.txt`)

Setup: 3x3, `S_b = -b + 3 Bin(b,1/3)` entrywise, `h_b(m) = E || m 11^T + S_b ||_*`. The restriction to the
direction 11^T is exact for existence: the law of S_b is invariant under row and column permutations and H is
convex, so the average of a minimiser over S_3 x S_3 is a minimiser of the form m 11^T. (Non-uniqueness of
the full argmin set is not probed.) The theorem predicts `argmin -> g_* = 0.08748`. Estimator: minimiser of
the sample average with a mean-zero linear control variate, 16 batches of 4e5 (6.4e6 samples per b),
standard error from batches.

| b | b mod 3 | argmin m | ratio to g_* | P(det S_b = 0) |
|---|---|---|---|---|
| 12 | 0 | 0.0000 +- 0.0000 | 0.000 | 0.144 |
| 13 | 1 | 0.0864 +- 0.0006 | 0.987 +- 0.007 | 0.051 |
| 14 | 2 | 0.0787 +- 0.0008 | 0.900 +- 0.010 | 0.047 |
| 30 | 0 | 0.0247 +- 0.0009 | 0.283 +- 0.011 | 0.052 |
| 31 | 1 | 0.0895 +- 0.0011 | 1.023 +- 0.013 | 0.018 |
| 32 | 2 | 0.0836 +- 0.0010 | 0.956 +- 0.011 | 0.018 |
| 99 | 0 | 0.0693 +- 0.0010 | 0.792 +- 0.012 | 0.012 |
| 100 | 1 | 0.0871 +- 0.0014 | 0.996 +- 0.016 | 0.0043 |
| 101 | 2 | 0.0870 +- 0.0012 | 0.995 +- 0.014 | 0.0042 |
| 300 | 0 | 0.0821 +- 0.0018 | 0.939 +- 0.021 | 0.0030 |
| 301 | 1 | 0.0838 +- 0.0025 | 0.958 +- 0.028 | 0.0010 |
| 302 | 2 | 0.0850 +- 0.0023 | 0.971 +- 0.026 | 0.0010 |
| 999 | 0 | 0.0896 +- 0.0047 | 1.024 +- 0.053 | 0.0006 |
| 1000 | 1 | 0.0934 +- 0.0053 | 1.068 +- 0.061 | 0.0002 |
| 1001 | 2 | 0.0840 +- 0.0046 | 0.960 +- 0.053 | 0.0002 |

Honest reading:
- Classes 1 and 2 mod 3: within a few percent of g_* already at b about 30-100, and statistically consistent
  with g_* at b = 100, 300, 1000 (2 to 6 percent resolution).
- Class 0 mod 3: the lattice contains singular matrices with large probability and h_b has a real kink at
  m = 0. Exact one-sided derivatives at m = 0: b = 12: [-0.185, +0.034] contains 0, so the minimiser is
  **locked at exactly 0** (all 16 batches return 0). b = 30: [-0.087, -0.009], not locked but the minimiser is
  only 0.28 g_*. b = 99: 0.79 g_*; b = 300: 0.94 g_*; b = 999: 1.02 +- 0.05 g_*. The deficit shrinks roughly
  like 1/b (0.72, 0.21, 0.06), in line with P(det S_b = 0), which decays roughly like b^{-1.2}.
- So the data support convergence to the same limit in all three classes, as Theorem 1 states, and contradict
  nothing. They do not test the rate; the proved rate b^{-1/34} is far slower than what is observed. For
  finite b in the class 0 mod 3 the asymptotic value is a poor approximation below b of a few hundred, and at
  b = 12 the bias is exactly zero. The sentence at l.66 is asymptotic and correct, but a reader should not
  apply it at small b.

## Minor fixes, collected

1. l.115: state that the partition of unity is on S x S x G; cite Stein VIII.1.2 and VIII.2.2 Prop. 5
   (the [ccw] "Lemma 1.1" pointer is unverified and the multidimensional theory is not used).
2. l.108: the matrix claim holds for max(k,l) >= 3, not only k,l >= 3 (optional strengthening).
3. l.119: say explicitly that the mixed derivative is indispensable for rank-deficient M (span is 5 of 9
   without it at rank one in size 3) and that derivative sizes are those of the k x k and l x l skew algebras.
4. l.163: "R-4" may read "R-3".
5. l.88/l.168: the logarithmic power (N+1)/4 can be lowered to (N-1-kappa)/4; current statement is true.
6. l.66: add a caveat that for b = 0 mod 3 the convergence is slow and the minimiser is exactly 0 for small b
   (b = 12 in size 3).

Not verified: the exact numbering of Lemma 1.1 in [ccw]; uniqueness of minimisers off the 11^T line in the
numerical test; anything in Sections 3-4.
