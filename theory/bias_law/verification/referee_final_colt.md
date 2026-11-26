# Referee report: "Edgeworth without Cramér: the bias of orthogonalized stochastic steps under discrete noise"

Files read in full: `paper/bias_law/colt/discrete.tex`, `theory/bias_law/shared/statements.tex`, `theory/bias_law/shared/macros.tex`,
`theory/bias_law/proofs/p_{gauss,fourier,codim,laplace,nuclear,sign,wor,lattice}.tex`, `results/bias_law/NUMBERS_discrete.md`,
`results/bias_law/figure_data/table_exact.json`, `paper/bias_law/colt/tab_exact.tex`, `discrete.bbl`, the three figures used by the paper.
Nothing heavy was run: one two-second quadrature (the 2x2 value of g_*) and hand algebra. No file other than this
report was edited.

Overall verdict: the logical chain is sound. I found no error that invalidates a theorem. There are four small
mathematical slips (one wrong formula in a statement, one wrong sign, one borderline hypothesis, one mis-stated
justification), a handful of sentences in the main text that say more than is proved or more than the data show, one
caption that contradicts the data, and heavy notation overloading.

Note on labels: in the compiled PDF the assumptions are lettered A (bounded noise, `ass:B`), B (singular set, `ass:R`),
C (decay, `ass:L`). Below I use the source labels (B, R, L).

---------------------------------------------------------------------------------------------------------------------

## 1. Mathematical correctness, item by item

### Definition of the class X(eps), (E1)-(E3) -- CORRECT (presentation fixes)
- Consistent with every use. The correspondence `p-hat` <-> `p` is right: `int e^{i<t,y>}(-d)^alpha phi = (it)^alpha phi-hat`,
  `He_3 phi = -D^3 phi`.
- `He_3^Sigma` is used here (main text, Section 2) but defined only in Lemma gauss (Appendix A). The norm `|K|` of a
  tensor is never specified. "even polynomial of degree at most 6 with r-hat(0)=0" is what Lemma locate needs (it uses
  that even-order derivatives of the odd field vanish at 0); fine.
- The text after the definition calls (E2) "a one-term Edgeworth expansion ... on the cube of side 1/eps". It is a
  two-term expansion (the `eps^2` term `r-hat` is included, error `eps^3`) on a cube of side `2c_*/eps`.

### Lemma gauss -- CORRECT
All six items check, including the independence of `s` and `Z'` in (4) and the scaling in (5)
(`g_*(c Sigma, c K_3) = g_*(Sigma, K_3)`).

### Lemma test -- CORRECT
Inversion formula, the use of (E2) on `Q_0`, the Gaussian tail of `p-hat` outside `Q_0` (absorbed into `eps^3`), (E1) on
each `Q_j`, `|t|_inf >= a(2|j|_inf - 1) >= a|j|_inf` on `Q_j`, and the `m`-fold integration by parts are all right.

### Lemma ball -- CORRECT
(i) Fejér kernel bound, covering by at most `(1+1/c_*)^N` cubes for `r >= eps`. (ii) packing argument; uses `0 in V`,
`rho <= R`, homogeneity of the cone. Both fine.

### Lemma locate -- CORRECT
`grad Psi = Phi + (1/6) D^3 Phi[K] + sum c_alpha d^alpha Phi`; each remainder is `O(eps^3)` on `|x| <= A eps`.

### Lemma cube -- CORRECT
Checked every constant: `(2M)^4/24 = (2/3)M^4`; `u_0`; the split (sum of A lies in `[rho_2/4, rho_2/2)`); AM-GM;
positivity of the Fejér transform; `|w_k s| <= u_0` on the support of the tent; `c_1 = u_0/(2 sqrt N)`;
`C_1 = 2^N (8 pi/lambda)^{N/2}`. The main-text "Idea of proof" describes it faithfully.

### Lemma near -- CORRECT (one undefined symbol)
The cumulant expansion, `rho_p rho_2^{-p/2} <= eps_w^{p-2}`, and the expansion of `e^A` are right.
`K_4` is never defined (it must be the fourth cumulant tensor, not `E zeta^{(x)4}`). Implicitly `c_2 <= 1/(M sqrt N)`.

### Prop sums -- CORRECT
(E1) from Lemma cube, (E2) by integrating Lemma near, (E3) by Hoeffding, `Lambda = M^2`. Constants depend only on
`N, M, lambda`, so the "uniformly over weight sequences" claims downstream are justified.

### Lemma examples -- CORRECT
- (a),(b) fine (c >= 2 needs every block of size >= 2).
- (c) The word count is right: `k + l = 2j+1` letters, `2(j+1)` half-resolvents, each letter `M`/`M^T` absorbs a distinct
  adjacent half, `k+1` halves remain, so `|D^k G_tau| <= C (sigma_K^2+tau^2)^{-(k+1)/2}` and the tau-integral gives
  `sigma_K^{-k}` for `k >= 1`.
- Volume bound: `sigma_K >= K^{-1/2} min_j dist(row_j, span of the others)` is right, and the span has codimension
  at least `d-K+1`.
- The last sentence (c = 1 only for l_1 and square nuclear) is asserted, not proved; harmless.

### Prop field -- CORRECT
Decomposition `F_1+F_2+F_3`, dyadic layers (convergent because `m = N+1 > c`), total `(eps^c + eps^3) R^N`, `R ~ sqrt(L_eps)`.

### Proof of Theorem codim -- CORRECT
Monotonicity argument is right; `rho <= eps` for small eps because `min(c,3) >= 2`. Scaling:
`psi_{xi_w}(G) = sqrt(rho_2) H(G/sqrt(rho_2))`, `x-hat = eps_3 g_*`, `sqrt(rho_2) eps_3 = rho_3/rho_2`. Batch mean:
`b^{-3/2}` (c=2), `b^{-2}` (c>=3), each with `log^{N/2}`. Typo: display (codimX) ends with a comma before "In particular".

### Lemma lapmeasure -- CORRECT
Scaling `mu(B(0,s)) = s^{N-1} mu(B(0,1))`, product rule, `F_R(z) = R F_1(z/R)`, `|F_R-hat(t)| <= C R^{N+1}(1+R|t|)^{-2-kappa}`.

### Prop value -- CORRECT
- `delta = eps^q`, `q = N/(N-kappa)`: `eps^N = delta^{N-kappa}`, so `eps^N s^{-(N-2-kappa)} <= delta^2` for `s in {delta, 2 delta}`.
- Lattice sum: `sum_{0<|j|<=eps/s} |j|^{-2-kappa} ~ (eps/s)^{N-2-kappa}` needs exactly `2+kappa < N`; the tail needs
  `D > N-2-kappa`. (When `eps/(2 delta) < 1` the first sum is empty and the bound still holds with a constant.)
- `eps^3 <= delta^2` iff `2q <= 3` iff `N >= 3 kappa`: true for `N >= 3`, `kappa <= 1`.
- Sandwich (both inequalities) is right; `int nu(z-2u) h_delta(u) du = (nu*h_{2 delta})(z)`.
- Result `C eps^{2q} L^{(N+1)/2}`.

### Lemma upgrade -- CORRECT
Constant `2 + C_2/2`; needs `eta <= 1` so that `x + sqrt(eta) u` stays in `B(0,2)`.

### Proof of Theorem laplace -- CORRECT
`d_eps = eps^q L^{(N+1)/4}`, `eps^3 <= d_eps <= eps`. Cosmetic: the line `<v,u> >= lambda_min rho - 2 C_1 d_eps` silently
absorbs the `C_A eps^3` error of Lemma locate into `C_1 d_eps` (take `C_1 >= C_A`). Batch mean exponent
`1 + kappa/(2(N-kappa))` is right. The "every measurable selection" sentence is justified.

### Lemma orbit -- CORRECT WITH FIX
The dyadic sum `sum_i (1+|t|2^{-i-1})^{-kappa} 2^{-i(N-1)}` is `O(|t|^{-kappa})` only if `N-1 > kappa`. With `N-1 = kappa`
(i.e. `N=2, kappa=1`, allowed by the hypotheses "N >= 2, 0 < kappa <= 1") it is `|t|^{-1} log|t|`.
Fix: assume `N >= 3` (or `kappa < N-1`) and write "using `N-1 > kappa`". No application is affected (N >= 4, kappa = 1/2).

### Lemma matrices -- CORRECT WITH FIX (two slips, argument stands)
- Sign: `A_{ih}A_{hj} = +E_{ij}`, not `-E_{ij}` (expand `(E_ih - E_hi)(E_hj - E_jh)`). Irrelevant for the span.
- The second derivatives of the phase in the exponential chart are the *symmetrized* products
  `(1/2)(A_1A_2+A_2A_1)y`, which give only symmetric matrices times `y`; the text says the derivatives range over the
  span of unsymmetrized `A_1A_2 y`. The claim survives because first derivatives supply the skew part and symmetrized
  products of skew matrices span all symmetric matrices in size >= 3 (`E_ij+E_ji`, and `E_ii` from the three
  `E_ii+E_jj`). One sentence should say this.
- Mixed term `A_{pi} y A_{iq} = s_i E_{pq}` is right. The compactness/van der Corput step is right, and the quoted
  result (Stein, Ch. VIII, 2.2, Prop. 5: decay `lambda^{-1/k}` with a constant controlled by the `C^{k+1}` norm) is what
  is needed.
- Sharpness remark (k=l=3, `tr U = 1+2 cos theta`): right; the critical point at `theta = pi` carries nonzero density.

### Lemma twobytwo -- CORRECT (all parts)
- Isometry, `det M = (|y_+|^2-|y_-|^2)/2`, `||M||_*^2 = ||M||_F^2 + 2|det M| = 2 max(|y_+|^2,|y_-|^2)`: right.
- Laplacian: `Delta(sqrt2 |y_+|) = sqrt2/|y_+|` in two variables, and `sqrt2/|y_+| <= 2/|y|` on `Omega_+`: right.
- Jump term: gradient jump `sqrt2 (y_+-hat, -y_--hat)`, normal `(y_+,-y_-)/|y|`, inner product
  `sqrt2 (|y_+|+|y_-|)/|y| = 2` on `V`: the coefficient `2 dS_V` is right. No atom at 0 (homogeneity -1, N = 4).
- Bessel reduction and `|J_0(u)| <= C(1+u)^{-1/2}`: right.
- Four contributions: bounded density (`int r^{-1/2} r dr`), `|y_{s'}|^{-1}` (integrable in the other plane),
  `|y_s|^{-1}` (`int r^{-3/2} r dr`), surface (`int rho^{-1/2} rho^2 d rho`): each `O(|t|^{-1/2})`. `2 + 1/2 < 4`.
- Independent check: the resulting 2x2 constant gives `g_* = (3+2q_{2,2})/32` with `q_{2,2} = 0.32036`, i.e. `0.113773`,
  matching the quoted `0.113774`.

### Proof of Theorem nuclear -- CORRECT
- Exponents: `K != d`: `c = |K-d|+1`, `2 gamma = min(c,3)-1`. `K = d`: `q-1 = kappa/(N-kappa) = 1/(2K^2-1) = 2 gamma`,
  so `gamma = 1/(4K^2-2)`, `1/14` for K = 2.
- Log powers: Theorem nuclear states `log^N`; Theorems codim and laplace deliver `log^{N/2}` and `log^{(N+1)/4}`. The
  stated power is weaker, hence consistent.
- Weighted form: `sqrt(rho_2) eps_w^{1+2 gamma} = (max_k w_k) eps_w^{2 gamma}`: follows. Uniformity follows from Prop sums.
- Assumption L is met: `N = 4, 2.5 < 4` (2x2); `N = K^2 >= 9` (K >= 3).
- Typo: the statement writes `Med_*`, undefined (should be `Med_nu`). "Let Assumption B hold. Then for every bounded law
  with nondegenerate covariance" says the hypothesis twice; the dependence of the `O` constant on `(K,d,M,lambda)`
  should be stated.

### Corollary mom -- CORRECT
`rho_p = (1-beta)^p b^{1-p}/(1-beta^p)`; `rho_3/rho_2 = 1/b_eff`; `eps_w^2 = (1-beta^2)/b = theta`;
`max_k w_k = (1-beta)/b <= 3/b_eff` is equivalent to `1+beta+beta^2 <= 3(1+beta)`, true. Uniformity in `(b,beta)` holds.
The factor 4/3 between the two effective batch sizes as `beta -> 1` is right.

### Corollary fs -- CORRECT (scope caveat, see Section 2)
Covariance positive definite iff the gradients affinely span; for `n = N+1` the differences form a basis, so the law
is lattice. The statement is pointwise in `W`.

### Proof of Theorem sign -- CORRECT WITH FIX
- (a), (b), (i), (iii) and the last sentence are right. `g_* = 1/6`, `h = 3`, values `0, 1, -1` for the {-1,2} law
  are right.
- (ii) as stated is wrong in its formula: for `b = j mod q` the coset is `Lambda_b = -(j p h/q + hZ)`, not
  `-(jh/q + hZ)`. Counterexample: the paper's own example, `a = -1, h = 3, p/q = -1/3`, `b = 1 mod 3`: the stated coset is
  `-(1+3Z)`, nearest point to 1/6 is `-1`; the true value (and the one quoted in Section 3) is `+1`.
  Fix: replace `\frac{jh}q` by `\frac{jph}q` (or write "the coset `Lambda_b`").

### Prop worclass -- CORRECT
- Conditioning identity: `chi_T(t) = int H_0(t,u)du / int H_0(0,u)du` over `[-pi h, pi h]`; denominator
  `2 pi h P(Bin(n,p)=b) = sqrt(2 pi)(1+O(1/v))`. Right.
- Majorant: `|phi_p(s)|^2 = 1 - 2r(1-cos s)`, `D = exp(-r sum(1-cos s_i))` is the characteristic function of the
  symmetrized Poisson vector. Right.
- (E1) in dimension N+1: `|s_i| <= 1` on the support of the tent, `r sum s_i^2 = <t,Sigma_n t> + u^2` (uses
  `sum z_i = 0`, `h^2 = nr`), bounded covering number. Right.
- (E2) uniformly in p: Bernoulli cumulants `r, r(1-2p), r(1-6r)`; division by `p(1-p)` is legitimate; for
  `c_2 h <= |u| <= pi h` each `s_i` is at distance `>= c_2/2` from `2 pi Z`; the odd-in-u terms and `sum a_l = 0` kill the
  cross terms of `A_3`. I checked `q_2(0) = -(1-r)/12`, which is the known second Stirling term for the binomial
  point probability, so numerator and denominator are consistent.
- (E3): complement trick (`m_0 <= 2h^2`), Hoeffding (1963) Theorem 4, variance proxy `2M^2`. Right.

### Proof of Theorem wor -- CORRECT (one mis-stated justification in the main text)
- Scaling to `b m`: `argmin psi = (h/b) argmin_x E nu(x+T)`, so `|b m + (1-2p) g_*^{(n)}| <= C h^{-2 gamma} log = C v^{-gamma} log`.
- `1-2p` versus `(n-2b)/(n-2)`: difference `2(n-2b)/(n(n-2)) = O(1/n) = O(1/v)`, below `v^{-gamma}` since `gamma <= 1`. So the
  theorem is true as stated, but it is equally true with `(n-2b)/n`: the proof does not distinguish them.
- The exact cumulants in the parenthesis are right (`Cov T = n/(n-1) Sigma_n`,
  `E T^{(x)3} = (1-2p) n^2/((n-1)(n-2)) K_{3,n}/h`), and inserting them gives exactly `(n-2b)/(n-2)`.
- The duality `Med(xi_{n,n-b}) = -(b/(n-b)) Med(xi_{n,b})` is right, but it does NOT single out `(n-2b)/(n-2)`: any
  coefficient antisymmetric under `b -> n-b`, `1-2p` included, is consistent with it. The main text (Section 6, idea of
  proof) says the exact form "is the one consistent with the exact identity"; that is the wrong reason. The right reason
  is the exact third cumulant. See required fix R4.

### Prop lattice2 and Appendix G -- NOT A THEOREM
Stated as a "Proposition" with a formal derivation and no proof (the existence of the limit defining the lattice sum
is not shown either). The algebra of the derivation is self-consistent and the three predicted coefficients match the
data. It should not carry the name Proposition (R6).

### Cross-statement consistency
- Log powers: consistent (see Theorem nuclear).
- Weighted-sum forms: follow in all three theorems.
- "Uniformly" claims: justified (Prop sums and Prop worclass give class constants depending only on `N, M, lambda`).
- Assumption L (`N >= 3`, `2+kappa < N`): met in every application.
- Main-text Lemma sym-main(a) omits the hypothesis `0 < kappa <= 1` of Lemma orbit. Without it the statement is not
  proved (the bound saturates at `|t|^{-(N-1)}` because `lambda_nu` has a `|z|^{-1}` singularity) (R3).

---------------------------------------------------------------------------------------------------------------------

## 2. Main text versus appendix

Supported as written: Theorem nuclear, Lemma cube-main and its idea of proof, the idea of proof of Theorem codim, the
three steps of Theorem laplace, Remarks (1) and (3), Corollary mom and the 4/3 remark, further results (i)-(iii).

Overclaims or misdescriptions:
1. Abstract: "b times the median locks onto a lattice and has no limit". Theorem sign gives a limit when `a in hZ`
   (`q = 1`); the introduction correctly says "no limit in general".
2. Introduction, (iv): "the rates of mechanism I are attained". Only the case `c = 2` in the plane, and only by a formal
   computation plus numerics; the paper's own 2x3 example (`c = 2`) shows slope -1, i.e. the rate is not attained there.
3. Introduction: remainder "`b^{-(1+min(c,3))/2}`" and "to order `b^{-N/(N-kappa)}`" omit the logarithms.
4. Introduction: a lattice law arises "whenever features and labels are quantized". Not true in general (for real
   parameters the gradients lie in a finitely generated group, which need not be discrete).
5. Introduction: "the 1/b law is a theorem for finite sums (Corollary fs)" and Discussion: "the bias of Muon-type methods
   on finite data sets is described by the two-moment formula ... whatever the arithmetic". What is proved is a
   statement about the medians of the noise at a fixed `W`. The passage to the limit point of the iteration with
   parameter-dependent noise is listed as open problem (5), and the two-moment formula needs isotropic covariance.
6. Discussion: "polyhedral norms ... lattice noise is visible at first order". Proved for l_1 only; open problem (3) says
   the others "should fail".
7. Section 6: the duality sentence (see Theorem wor above).
8. Introduction: "It is the gradient of the convex function psi_xi". For discrete noise it is a subgradient.
9. Section 5: "which is sharp for k=l=3" / Discussion: "sharp for the orbital integrals": shown for `k=l=3` only.
10. Remark (2) says Figure 1(d),(e) "suggests faster convergence", while Section 7 says "No rate can be read off".
11. Abstract, first sentence, and the first paragraph of the introduction state the convergence result of the
    companion paper without its hypotheses in the abstract (strong convexity, i.i.d. additive noise). The paper depends
    on an anonymous companion for its motivation, the first-order law and the explicit `g_*`; a reader cannot check these.

---------------------------------------------------------------------------------------------------------------------

## 3. Numbers (Section 7, Appendix H, Table 1, captions)

Matching the data: truncation `< 10^{-8}` (max 9.9e-9) and `< 10^{-10}` for vectors; sign values up to 3002; all 18
table entries and the six `g_*`; 2x2 lock up to 513 and `0.031, 0.051, 0.067, 0.074`; 3x3 at `b ~ 3000`; `1/14`, `1/34`;
lattice coefficients `-1.012, +0.333, -0.178` against `-1.028, +0.337, -0.179`; rms 0.02 (0.0192); momentum `0.0868 +- 0.0013`,
`0.0875`, variance rule `0.066-0.068`, "within 5%" (largest 4.7%); sign with momentum `0.168-0.171`; `6.4e7` samples;
`81 x 8 = 648`, `81 x 128 = 10368`; `4e5` steps, 64-256 runs.

Mismatches or descriptions more favourable than the data:
1. Figure 3 caption: "for `b in {1,2,4,8,16}`". The data and the plot contain `b = 3, 12` (nuclear, beta = 0) and
   `b = 3, 6, 12` (sign); the main text itself quotes `b = 6,...,16`. MISMATCH.
2. Slope for 1x2 quoted as `-0.51`; the three fitted slopes are `-0.522, -0.521, -0.509`. The quoted one is the class
   closest to `-1/2`.
3. Figure 2(b), n = 648: the text says only "the sign reversal is visible at both population sizes". The data: every
   `b = 0 mod 3` is exactly 0, all three classes are locked at 0 for `0.42 <= b/n <= 0.58`, classes 1 and 2 are off the line
   by up to 0.63 (rms 0.28-0.29). The line is not followed at n = 648 and the text does not say so.
4. Figure 2(b), n = 10368: "classes 1, 2 follow the line": rms 0.07, with deviations of 6-9 standard errors at
   `b/n = 0.02, 0.98`. Not mentioned. Also not mentioned: the run was interrupted (64 batches up to b = 7775, then 33/16).
5. Figure 2(a): rms 0.02 is right, but the points at small and large fractions (b = 8, 25, 375, 392) are 3 to 8 standard
   errors off the line, and `g_*` carries about 1% Monte-Carlo error. "Changes sign at b = n/2": the point at b = 200 is
   `+0.012 +- 0.014`.
6. 3x3: only `b ~ 3000` is quoted. At `b = 999-1001` all three classes are about 2 standard errors below `g_*`, and class 0
   at `b = 300` is 3.5 standard errors below.
7. Sign with momentum: "for beta in {0.5, 0.9} it agrees with `g_*/b_eff`". For small `b_eff` it does not (0.437 at b = 1,
   0.154 at b = 2, beta = 0.5), and the quoted `0.168-0.171` values are several standard errors above 1/6.
8. Figure 1 caption: "transient locking at 0 for square matrices". Class 0 is also locked for 1x2 (b <= 66) and 2x3
   (b = 3, 6, 9), as panels (b), (c) show. "Convergence" for 2x2 class 0 is not seen numerically (0.074 against 0.114
   at the largest b); it is guaranteed by the theorem.
9. "exactly 0 up to b = 513": for every *computed* multiple of 3 up to 513 (the release happens between 513 and 768).
10. 2x3: the constant on class 0 is 8-10 times larger than on the other classes and the range stops at b = 102; the
    slope statement rests on that short range.

---------------------------------------------------------------------------------------------------------------------

## 4. Citations

I can vouch for: Esseen (1945); Gnedenko-Kolmogorov (lattice and non-lattice one-term expansions; I cannot confirm
the section number "43"); Bhattacharya-Rao; Götze-Hipp (1978, smooth functions without Cramér); Breuillard (2005);
Angst-Poly (2017); Song (2018); Buhler-Gamst-Graham-Hales; Dolgopyat-Fernando; Fernando-Liverani; Hoeffding (1963)
Theorem 4 (convex functions, with versus without replacement); Erdős-Rényi (1959) and Babu-Singh (1985) for the
conditioning device; Stein, Ch. VIII, 2.2, Prop. 5; Kaas-Buhrman (1980) and Hamza (1995) (results on mean-median
distance for the binomial; "pinned to integers" is a fair paraphrase of Kaas-Buhrman); Cornish-Fisher (1938);
Carlson-Cevher-Carin (2015); the sign-descent and geometric-median references.

Cannot vouch for:
- `xie2026muon` (arXiv:2609.32861), `chen2026when` (arXiv:2610.31728), `do2026muon` (arXiv:2608.04607),
  `paquette2026phases` (arXiv:2605.09552): I could not verify existence, identifiers or the characterization "these works
  assume smooth or Gaussian-like noise or bound the bias from above". Check each identifier and that sentence.
- `schwank` (arXiv:2508.12926): not verified.
- `gz` (Götze-Zaitsev 2014): the paper is about explicit rates for quadratic forms; describing it as "the observation
  that quadratic and other curved statistics of lattice sums have better expansions than linear ones" is a loose
  attribution. Rephrase or cite the precise statement.
- `companionI`: anonymous and unverifiable; three load-bearing facts are attributed to it.

---------------------------------------------------------------------------------------------------------------------

## 5. Presentation

Page limit: the main text ends on page 12 (References start on page 12). Respected. Anonymity: `anon` option, empty PDF
metadata, companion cited as Anonymous; no author string in the sources read. Check the supplementary scripts for user
paths. `[12pt]` is an unused class option (log warning).

Notation clashes (each symbol has at least two meanings):
- `chi`: cut-off (Assumption L, Prop field) and characteristic function `chi_X`.
- `p`: comparison density (macro `\psie` prints `p`), sampling fraction `b/n`, numerator in `p/q` (Theorem sign), matrix
  entry (Lemma twobytwo), polynomials `p_1, p_2`, index in `rho_p`.
- `d`: matrix dimension, `1-2p` (Prop worclass), `sqrt(eta)` (Lemma upgrade), `d_eps`, `d_4`.
- `h`: mollifier, lattice span, `sqrt v`, increment in Lemma gauss.
- `q`: exponent `N/(N-kappa)`, denominator in Theorem sign, matrix entry, `q_2`.
- `r`: `p(1-p)`, radius, rank, matrix entry, the polynomial `r`.
- `s`: mollifier scale, frequency, singular values, matrix entry, `(sigma_K^2+tau^2)^{1/2}`.
- `Delta`: Laplacian and the tent function (both within a few lines in Sections 2 and 5).
- `T`: the operator `T_Sigma` and the random vector `T = b xi/h` (Prop worclass). `C'`: a constant and the random variable
  `sum(I_i - p)`. `c_1, c_2`: different roles in Lemma cube/near and in Prop worclass.
- `Phi, phi`: Gaussian mean field/density, standard normal cdf/density (Theorem sign), characteristic function of
  `zeta`, the phase in Lemma matrices.
- `lambda`/`Lambda`: covariance bounds, the measure `lambda_nu`, the coset `Lambda_b`. `kappa` and `kappa_3`. `delta` (mollifier
  scale; tolerance in Theorem sign). `theta` (momentum parameter; angle). `N` and `N_beta`. `K` (rows), `K_3`, `cal K`.
  `R` (radius; remainder). `A` (index set; constant; matrix; exponent). `H` (potential; `H_0`; matrix direction; Hessian).
  `M` (bound; matrix; momentum buffer). `V` (cone; rotation). `a` (half-side; lattice offset; `a_i(t)`).
Undefined: `K_4`; `He_3^Sigma` in the main text; `Med_*`; the tensor norm; `|g_*|` versus `||g_*||` (Figure 2 axis).

---------------------------------------------------------------------------------------------------------------------

## REQUIRED fixes

R1. `theory/bias_law/shared/statements.tex`, `\StmtSign`, item (ii): replace
    `a midpoint of $-(\frac{jh}q+h\Z)$` by `a midpoint of $-(\frac{jph}q+h\Z)$` (wrong coset as written).

R2. `theory/bias_law/proofs/p_laplace.tex`, Lemma orbit: replace `Let $N\ge2$` by `Let $N\ge3$` and `using $N-1\ge\kappa$` by
    `using $N-1>\kappa$`.

R3. `paper/bias_law/colt/discrete.tex`, Lemma sym-main (a): insert the hypothesis, e.g. `for some $0<\kappa\le1$` after
    "orbital integrals satisfy".

R4. `paper/bias_law/colt/discrete.tex`, Section 6, idea of proof of Theorem wor: replace
    "the exact form $(n-2b)/(n-2)$ is the one consistent with the exact identity ... which holds because the complement
    of a sample is a sample" by a sentence saying that `(n-2b)/(n-2)` is what the exact covariance `n/(n-1) Sigma_n` and
    third cumulant give, that it differs from `1-2p` by `O(1/n)`, below the remainder, and that both are consistent with
    the complement identity.

R5. `theory/bias_law/proofs/p_laplace.tex`, Lemma matrices: replace `A_{ih}A_{hj}=-E_{ij}` by `A_{ih}A_{hj}=E_{ij}`, and add one
    sentence: the second derivatives are the symmetrized products, which span the symmetric matrices, the first
    derivatives supply the skew ones.

R6. `paper/bias_law/colt/discrete.tex`, `\begin{proposition}[formal; confirmed numerically]`: rename to a conjecture or "formal
    claim"; and in the introduction, (iv), replace "the rates of mechanism I are attained" by "the rate of mechanism I
    for $c=2$ appears to be attained in the plane".

R7. `paper/bias_law/colt/discrete.tex`, abstract: "locks onto a lattice and has no limit" -> "... and in general has no limit".

R8. `paper/bias_law/colt/discrete.tex`, Figure 3 caption: replace "for $b\in\{1,2,4,8,16\}$" by the batch sizes actually
    plotted (`b <= 16`, including 3, 6, 12 where shown).

R9. `paper/bias_law/colt/discrete.tex`, Section 7, "Without replacement": state that at `n = 648` the line is not followed
    (class 0 identically 0, all classes locked near `b = n/2`, deviations up to 0.6), and that at `n = 10368` the
    deviations at the extreme fractions are significant. Section 7, "Lattice noise": replace "fitted slopes $-0.51$" by
    "$-0.51$ to $-0.52$".

R10. `paper/bias_law/colt/discrete.tex`, Discussion: replace "(the $\ell_1$ norm, and more generally polyhedral norms, whose
     direction maps are locally constant) lattice noise is visible at first order" by a statement for `l_1` only, with
     polyhedral norms as a conjecture; and qualify "the bias of Muon-type methods on finite data sets is described by"
     (fixed-point noise, isotropic covariance for the two-moment formula). Introduction: "whenever features and labels are
     quantized" -> "and can occur when ...".

R11. `theory/bias_law/proofs/p_fourier.tex`, Lemma near: define `K_4` (fourth cumulant tensor of `zeta`).
     `theory/bias_law/shared/statements.tex`, `\StmtNuclear`: `\Med_\ast` -> `\Med_\nu`; `\StmtCodim`: the comma ending (codimX) -> period.

R12. Verify the four 2026 arXiv references and the sentence characterizing them; rephrase the attribution to `gz`.

## Optional suggestions

O1. Rename to remove the worst clashes: tent function (`Delta`), `T` and `C'` in Prop worclass, `d = 1-2p`, `chi` cut-off,
    `h = sqrt v`, and print the comparison density with a letter other than `p`.
O2. Say "two-term" for (E2) and give the side of the cube as `2c_*/eps`.
O3. Carry "up to logarithms" in the introduction's statements of the remainders.
O4. In Section 7 mention the 3x3 values at `b ~ 1000` (2 standard errors low), the small-`b_eff` sign-momentum values, and
    that locking at small `b` occurs also for 1x2 and 2x3; write "every computed multiple of 3 up to 513".
O5. Introduction: "a subgradient of the convex function psi_xi"; say that the companion's convergence theorem assumes
    strong convexity and i.i.d. additive noise where it is first used in the abstract.
O6. Theorem nuclear: drop the duplicated hypothesis and state what the `O` constant depends on.
O7. Proof of Theorem laplace: note `C_1 >= C_A` so that `-2C_1 d_eps` is literally correct.
O8. Remark (2) and Section 7 should agree on whether a rate can be read off the square case.
O9. Remove the unused `[12pt]` option; define `He_3^Sigma` and the tensor norm in Section 2.
