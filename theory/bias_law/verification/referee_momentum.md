# Referee report: Theorem "Convergence with atomic innovations" and Corollary "Discrete square polar descent with momentum"

Source checked: `theory/bias_law/verification/orbital-bias_source.tex`, Section 4 (lines 258-332) and the total-variation
paragraph of Section 5 (line 340). Line numbers refer to that file. Scripts and raw outputs: `theory/bias_law/verification/momentum/`.

## Overall verdict

I tried to break the theorem and could not. Every step of the proof re-derives; the constants quoted are right
(sometimes not sharp, never too small); the sign conventions are consistent; the corollary's algebra is exact.
What remains are **minor clarifications** (listed per item), none of which changes a statement. The simulations agree
with the theorem within error bars, including the case where the median set is an interval and the limit is
genuinely random.

Not checked (out of scope, taken as given): bound (eq:median) of Theorem `th:group` and its claimed uniformity in the
weights. The corollary's uniformity in (b, beta) is inherited entirely from that claim.

| # | Item | Verdict |
|---|------|---------|
| 1 | error recursion, sum eta_t e_t | CORRECT WITH MINOR FIX (wording) |
| 2 | inequality (eq:path), sign of h | CORRECT |
| 3 | Poisson equation, coupling, Hoelder bound | CORRECT WITH MINOR FIX (wording) |
| 4 | martingale part | CORRECT WITH MINOR FIX (state the filtration assumption on the selection) |
| 5 | summation by parts | CORRECT |
| 6 | Lyapunov step, cluster point, single limit | CORRECT |
| 7 | theta = 1, hypotheses on F | CORRECT |
| 8 | Corollary | CORRECT (conditional on th:group as stated) |
| 9 | Simulations | CONSISTENT with the theorem |
| - | Section 5 TV paragraph | CORRECT (citation content not fetched) |

---

## 1. Error recursion (lines 273-277) -- CORRECT WITH MINOR FIX

* Recursion. With N_t = beta N_{t-1} + (1-beta) xi_t and M_t = beta M_{t-1} + (1-beta)(G_t + xi_t),
  e_t = M_t - G_t - N_t = beta(M_{t-1} - N_{t-1}) - beta G_t = beta e_{t-1} + beta(G_{t-1} - G_t). Exact.
  Numerically the identity holds to 5e-16 (scalar script).
* |G_t - G_{t-1}| <= L |W_t - W_{t-1}| = L eta_{t-1} |O_{t-1}| <= L K eta_{t-1}, deterministic. (|O| <= K holds because a
  subgradient of a K-Lipschitz function has Euclidean norm <= K.)
* Unrolling: e_t = beta^t e_0 + sum_{s=1}^t beta^{t-s+1}(G_{s-1} - G_s). Then
  sum_t eta_t |e_t| <= eta_0 |e_0|/(1-beta) + L K sum_s eta_{s-1} sum_{t>=s} eta_t beta^{t-s+1}
  <= eta_0|e_0|/(1-beta) + L K beta/(1-beta) sum_s eta_{s-1}^2, using eta_t <= eta_{s-1} for t >= s.
  So "by monotonicity" is enough, and theta > 1/2 gives finiteness.
* Is the bound deterministic? The double sum is deterministic. The initial term is **not**: e_0 = M_0 - G_0 - N_0 and
  N_0 is the stationary (random, square-integrable) buffer built from the negative-time innovations. So the bound is
  "a.s. finite, with finite second moment", exactly as line 277 says ("almost surely"). Nothing downstream needs more.
* Minor fix: say explicitly that e_0 is random through N_0 and that the time index of the first step / of M_0 is fixed
  (the proof writes "for t >= 1" but the theorem does not fix the starting index).

## 2. Inequality (eq:path) (lines 279-284) -- CORRECT

Subgradient inequality at M_t tested at y = a + N_t: nu(a+N_t) >= nu(M_t) + <O_t, a + N_t - M_t>, and
M_t - a - N_t = (G_t - a) + e_t. Hence
<O_t, G_t - a> >= nu(G_t + N_t + e_t) - nu(a + N_t) - <O_t, e_t> >= [nu(G_t+N_t) - K|e_t|] - nu(a+N_t) - K|e_t|.
That is f_{G_t}(N_t) - 2K|e_t|: the constant 2K is right (one K for the value, one for the linear term) and cannot be
improved in general. The inequality holds for every a, not only minimisers. Numerically the slack was >= 0 at every
step in all runs (min slack 1.4e-12 scalar, 2e-4 matrix with K = sqrt 2).

Sign conventions. Med_nu(Y) = argmin_m E nu(Y - m) (line 39), so -Med_nu(N) = argmin_g E nu(g + N). With a in -Med,
h(g) = E nu(g+N) - E nu(a+N) >= 0 with equality iff g in -Med. Consistent throughout, and consistent with the older
`thm:momconv` (argmin psi_{N_beta} = -Med(N_beta)).

"f_g is 2K-Lipschitz in n uniformly over g and a": yes (difference of two K-Lipschitz functions).

## 3. Poisson equation (lines 286-295) -- CORRECT WITH MINOR FIX

* Coupling. Q^j f_g(n) = E f_g(beta^j n + R_j), h(g) = E f_g(beta^j N + R_j) with N ~ pi independent of R_j. So
  |Q^j f_g(n) - h(g)| <= 2K beta^j E|n - N| <= 2K beta^j (|n| + E|N|). Only E|N| < infinity is needed; the bound is
  **uniform in g and a** because g enters only through a function whose Lipschitz constant in n does not depend on g.
* Change under g -> g'. f_g - f_{g'} = nu(g + .) - nu(g' + .) is bounded by K|g-g'| in sup norm, so the centered
  summand changes by at most 2K|g-g'|. It is also bounded by 2K beta^j E|n-N| (apply the coupling to f_g - f_{g'},
  itself 2K-Lipschitz in n). So the change of the j-th summand is <= min(2K|g-g'|, C beta^j (1+|n|)).
* Sum: sum_j min(A, B beta^j) <= A^alpha B^{1-alpha} / (1 - beta^{(1-alpha)}), giving
  |U_g(n) - U_{g'}(n)| <= C_alpha |g-g'|^alpha (1+|n|)^{1-alpha}, C_alpha = (2K)^alpha C^{1-alpha}/(1-beta^{1-alpha}).
  (Checked numerically on 20000 random (A, B, beta, alpha): worst ratio 0.992.) Applying Q:
  E(1 + |beta n + (1-beta) xi|)^{1-alpha} <= C(1+|n|)^{1-alpha} by concavity. The stated bound for V follows.
  Remark: the sharper bound |g-g'| (1 + log_+((1+|n|)/|g-g'|)) is also available; the Hoelder form suffices.
* Poisson equation U_g - Q U_g = f_g - h(g): Q passes through the series by dominated convergence (majorant
  C beta^j (1 + |n| + E|xi|)). Correct.
* Measurability. n -> U_g(n) is 2K/(1-beta)-Lipschitz (each Q^j f_g is 2K beta^j-Lipschitz) and g -> U_g(n) is locally
  Hoelder, so (g, n) -> U_g(n), V_g(n) are jointly continuous, hence Borel. Composition with random (G_t, N_t) is
  measurable and E[U_{G_t}(N_t) | F_{t-1}] = V_{G_t}(N_{t-1}) by the freezing lemma (integrability from the linear
  growth bound). The paper does not say this; one sentence would close it.
* Minor fix (wording, line 290): "the minimum of the two bounds for the change" -- the second bound for the *change* is
  2C beta^j (1+|n|), not the bound on the summand itself; say so. beta = 0 is covered (0^0 = 1, U_g = f_g - h(g)).

## 4. Martingale part (lines 297-302) -- CORRECT WITH MINOR FIX

* Predictability. With F_t = sigma(xi_s : s <= t) (two-sided), M_t and O_t are F_t-measurable, W_{t+1} is F_t-measurable,
  so G_t = grad F(W_t) is F_{t-1}-measurable. N_{t-1} is F_{t-1}-measurable and xi_t is independent of F_{t-1}.
  Hence E[U_{G_t}(N_t) | F_{t-1}] = (Q U_g)(N_{t-1}) at g = G_t = V_{G_t}(N_{t-1}): a martingale difference.
* Second moments. |U_g(n)| + |V_g(n)| <= C(1+|n|) with C independent of g (item 3). So
  E|U_{G_t}(N_t) - V_{G_t}(N_{t-1})|^2 <= 4 C^2 E(1+|N|)^2 < infinity uniformly in t, with no a priori bound on G_t.
  The claim "even for unbounded G_t" is right.
* sum eta_t^2 < infinity (theta > 1/2) gives an L^2-bounded martingale, which converges a.s.
* Minor fix. "O_t is selected measurably from the observations through time t" must exclude dependence on future
  innovations; if the tie-break uses auxiliary randomness, the filtration has to be enlarged by randomness independent
  of (xi_s)_{s>t}. For the deterministic partial polar factor of the corollary there is nothing to add.

## 5. Summation by parts (lines 302-306) -- CORRECT

sum_{t=1}^T eta_t [V_{G_t}(N_{t-1}) - V_{G_t}(N_t)]
 = eta_1 V_{G_1}(N_0) - eta_T V_{G_T}(N_T) + sum_{t=1}^{T-1} [eta_{t+1} V_{G_{t+1}}(N_t) - eta_t V_{G_t}(N_t)].

* Initial boundary: a.s. finite. Terminal boundary: eta_T |V_{G_T}(N_T)| <= C eta_T (1+|N_T|) -> 0 a.s. because
  sum_T eta_T^2 E(1+|N_T|)^2 < infinity.
* Internal differences = (eta_{t+1} - eta_t) V_{G_t}(N_t) + eta_{t+1}[V_{G_{t+1}}(N_t) - V_{G_t}(N_t)].
  First: sum |eta_{t+1} - eta_t| C E(1+|N|) = C E(1+|N|)(eta_1 - lim eta) < infinity (telescoping, monotone steps).
  Second: <= eta_{t+1} C_alpha (L K eta_t)^alpha (1+|N_t|)^{1-alpha}; expectation <= C eta_{t+1} eta_t^alpha
  ~ t^{-theta(1+alpha)}, summable iff theta(1+alpha) > 1; since theta > 1/2, some alpha < 1 works.
  Here |G_{t+1} - G_t| <= L K eta_t is deterministic, which is what is needed.
* Summable absolute expectations give a.s. absolute convergence. Claim proved.

## 6. Lyapunov step and identification of the limit (lines 308-314) -- CORRECT

* Descent. Phi(w) = F(w) - <a, w> is L-smooth with grad Phi(W_t) = G_t - a, and B_t(a) = Phi(W_t) - Phi(W^a) = D_F(W_t, W^a).
  Phi(W_{t+1}) <= Phi(W_t) - eta_t <G_t - a, O_t> + (L/2) eta_t^2 |O_t|^2, and (eq:path) gives the display at line 311
  with C = L K^2 / 2.
* Robbins-Siegmund by hand. R_t = sum_{s<t}[-eta_s (f_{G_s}(N_s) - h(G_s)) + 2K eta_s |e_s| + C eta_s^2] converges a.s.
  (items 1, 4, 5). S_t = B_t(a) + sum_{s<t} eta_s h(G_s) - R_t is nonincreasing and bounded below (B >= 0, h >= 0, R
  bounded). So S_t converges; the monotone series sum eta_s h(G_s) is bounded hence convergent; hence B_t(a) converges.
* Boundedness. B_t(a) >= (mu/2)|W_t - W^a|^2, so W_t and G_t are bounded (pathwise).
* Cluster point. sum eta_t = infinity (theta <= 1) and sum eta_t h(G_t) < infinity give liminf h(G_t) = 0; h is K-Lipschitz
  and G_t bounded, so some cluster point G-hat has h(G-hat) = 0, i.e. G-hat in -Med.
* -Med is nonempty, compact, convex for **every** integrable law: g -> E nu(g+N) is finite, convex, and coercive since
  E nu(g+N) >= nu(g) - E nu(N). This uses nothing about the support, so laws on proper subspaces are covered and the
  argmin is never unbounded. (Answer to the question asked: yes, always coercive.)
* Countable dense set. The a.s. statements (convergence of R_t, which depends on a through f_g) hold simultaneously
  for a in a countable dense D of -Med. B_t(a) = F(W_t) + F^*(a) - <a, W_t>, with gradient in a equal to W^a - W_t,
  which is bounded on bounded iterates and compact -Med ((grad F)^{-1} is (1/mu)-Lipschitz). So a -> B_t(a) is Lipschitz
  uniformly in t, pathwise, and B_t(G-hat) converges for the (random) G-hat. Along the subsequence,
  W_{t_k} = (grad F)^{-1}(G_{t_k}) -> W^{G-hat}, so B_{t_k}(G-hat) -> 0, hence B_t(G-hat) -> 0 and, by strong convexity,
  the whole sequence converges. The Opial-type argument is complete and uses G-hat only pathwise; no measurability of
  G-hat is needed before the conclusion (afterwards G_inf = lim G_t is measurable).
* Random limit. When -Med is a singleton the limit is deterministic. When it is not, the limit is genuinely random
  (simulation C below: -Med = [-1/3, 1/3] and runs spread over the whole interval). The theorem's wording "a single
  random point" is accurate. The corollary's "deterministic bound on that limit" is legitimate because (eq:median)
  is a sup over all of Med.

## 7. theta = 1 and the hypotheses on F -- CORRECT

* theta = 1: sum eta = infinity, sum eta^2 < infinity, theta(1+alpha) > 1 for every alpha > 0. Nothing in the proof needs
  theta < 1. What is actually used: eta nonincreasing, sum eta = infinity, and sum eta_{t+1} eta_t^alpha < infinity for
  some alpha < 1. (Simulation with theta = 1 converges as predicted.)
* (grad F)^{-1} is defined on all of E: for every a, F - <a, .> is strongly convex and continuous, hence has a unique
  minimiser W^a; equivalently grad F is a continuous strongly monotone map, hence a bijection with (1/mu)-Lipschitz
  inverse. L-smoothness is used only in the descent lemma and in |G_t - G_{t-1}| <= L K eta_{t-1}.
* The innovations are not assumed centered in the theorem; nothing requires it.
* Relation to earlier results: `thm:momconv` (`theory/bias_law/notes/polar_bias.tex`, l. 490) reaches the same conclusion under
  Assumption M, including a small-ball condition on the innovations, via a lag-m decoupling of polar factors;
  `thm:smd` (`paper/paper_saga_to_muon.tex`, l. 395) is the beta = 0 case. The new proof replaces the comparison of
  polar factors by a comparison of loss values and needs no small-ball condition, arbitrary norm, arbitrary
  subgradient selection. It does supersede both for the a.s. convergence statement (not for the rate in thm:smd(a)).

## 8. Corollary (lines 317-332) -- CORRECT (given th:group)

Weights w = (1-beta) beta^j / b, each b times, j >= 0; sum = 1.
* rho_2 = (1-beta)/(b(1+beta)), rho_3 = (1-beta)^2/(b^2(1+beta+beta^2)), so rho_3/rho_2 = (1-beta^2)/(b(1+beta+beta^2)) = 1/B.
* eps^2 = (max w)^2/rho_2 = (1-beta^2)/b. sqrt(rho_2) = eps/(1+beta). B eps^2 = 1+beta+beta^2 in [1, 3).
  All four identities checked numerically over 2000 random (b, beta): relative error 1.6e-15 (`check_algebra.py`).
* Exponent: kappa = 1/2, q = N/(N - 1/2), sqrt(rho_2) d_eps <= eps^{1+q} L^{(N+1)/4}, eps^2 in [1/B, 3/B],
  (1+q)/2 = 1 + 1/(4N-2) exactly; L = log(e/eps) <= C log(eB). The stated bound follows.
* Sign: G_inf in -Med(N_beta), and every m in Med satisfies |m + g_*/B| <= ..., so |G_inf - g_*/B| <= ... . Positive sign
  is right and matches the beta = 0 statement of th:square.
* Applicability: countable weights are allowed in th:group ("finite or countable"); samples are bounded, centered,
  Sigma > 0; k >= 3 gives kappa = 1/2; "B large" is equivalent to "eps small". The exact (partial) polar factor
  M (M^T M)^{+1/2} is a Borel selection of the nuclear subdifferential, so th:momentum applies.
* Uniformity in (b, beta): comes solely from "uniformly over the weights" in th:group, taken as given. The
  beta-dependent constants in the proof of th:momentum do not enter, because only the location of the limit is bounded.
  The order of limits (iteration limit first, then B -> infinity) is stated correctly.
* Remark only: the last sentence of the proof ("The uniform estimate concerns the location of the limit gradient")
  should say outright that no rate of convergence in t uniform in beta is claimed.

## 9. Simulations

Scripts: `momentum/scalar_sign.py`, `momentum/matrix_nuclear.py`, `momentum/check_algebra.py`; raw outputs `momentum/out_*.txt`.
All runs: eta_t = (t+1)^{-theta}, M_0 = 0, N_0 drawn from the stationary law. SE = standard error over independent runs.

### 9a. Scalar, xi in {-1, 2} w.p. 2/3, 1/3, beta = 1/2, F = w^2/2

Here N_beta = -1 + 3U where U has i.i.d. Bernoulli(1/3) binary digits: a singular continuous law with full support
and strictly increasing cdf. So **the median is unique** (not an interval), computed exactly by the digit recursion:
Med = -0.189323332733, target G_inf = +0.189323332733 (Monte Carlo median from 2e6 samples: -0.18852).

256 runs, half started at W_0 = +2 and half at -2.

| theta | t | mean(G_t) - target (SE) | sd across runs | max abs deviation |
|---|---|---|---|---|
| 0.75 | 2.5e5 | -1.5e-3 (6.7e-4) | 1.08e-2 | 3.6e-2 |
| 0.75 | 1e6 | -9.8e-4 (3.9e-4) | 6.3e-3 | 2.0e-2 |
| 0.75 | 4e6 | -7.4e-4 (2.3e-4) | 3.7e-3 | 1.3e-2 |
| 0.75 | 1.6e7 | -4.9e-4 (1.3e-4) | 2.1e-3 | 6.2e-3 |
| 1.0 | 2.5e5 | -1.3e-3 (2.0e-4) | 3.1e-3 | 9.9e-3 |
| 1.0 | 1.6e7 | -1.1e-4 (2.4e-5) | 3.9e-4 | 1.3e-3 |

The spread shrinks roughly like eta_t^{1/2} and every run approaches the same point, independent of the start.
Honest caveat: at the last checkpoint the run-mean still differs from the target by about 4 SE (theta = 0.75) and
5 SE (theta = 1). This residual shrinks with t (monotonically for theta = 1, roughly like t^{-1/2}; for theta = 0.75 it goes
-1.5e-3, -9.8e-4, -7.4e-4, -3.9e-4, -4.9e-4 at t = 2.5e5, 1e6, 4e6, 8e6, 1.6e7) and is far below the per-run spread.
I read it as a finite-time bias, not evidence of a different limit; but the data cannot distinguish "limit = target" from
"limit within about 5e-4 of target".

### 9b. Scalar, interval of medians: xi = +-1, beta = 1/3 (added by the referee)

N_beta lives on a Cantor set with |N_beta| >= 1/3, so Med = [-1/3, 1/3]. 64 runs for each start, theta = 0.75, T = 1.6e7.

| W_0 | min G_T | max G_T | sd of G_T | 10/50/90 % quantiles |
|---|---|---|---|---|
| -2 | -0.322 | +0.321 | 0.209 | -0.303 / -0.017 / +0.257 |
| 0 | -0.329 | +0.262 | 0.184 | -0.301 / -0.092 / +0.213 |
| 0.2 | -0.317 | +0.324 | 0.198 | -0.254 / +0.042 / +0.265 |
| 2 | -0.320 | +0.326 | 0.202 | -0.268 / +0.036 / +0.284 |

All 256 terminal values lie inside [-1/3, 1/3] (no excess at all), and different runs end at different points
spread over the whole interval: the limit is random, as the theorem allows. Caveat: inside the interval the iterate is
essentially the martingale -sum eta_t xi_t, whose tail has sd ~ T^{-1/4}; max |G_T - G_{T/2}| is still about 3e-2 at
T = 1.6e7, matching that prediction. So convergence to a point is slow here and is supported by the predicted decay,
not observed to high precision.

### 9c. 2x2 nuclear norm, discrete noise

xi_t = mean of b matrices with i.i.d. entries in {-1, 2} (2/3, 1/3); F(W) = (1/2) sum A_ij (W_ij - W*_ij)^2 with
A = (1, 2, 0.5, 1.5), a deliberately anisotropic choice; exact polar factor; theta = 0.75; 128 runs; T = 8e6.
Reference: argmin_g E||g + N_beta||_* by L-BFGS on a sample average with common random numbers, 16 independent batches
of 2e6 samples, all four coordinates free (row/column exchange symmetry predicts equal entries).

| (b, beta) | reference argmin, pooled entry (SE) | iteration, mean G_T per entry (SE) | sd across runs |
|---|---|---|---|
| (4, 0.5) | 0.01281 (0.00004); entries 0.01282, 0.01274, 0.01292, 0.01275 (SE ~1e-4) | 0.01287 (1.4e-4), 0.01262 (2.0e-4), 0.01295 (1.1e-4), 0.01297 (1.9e-4) | 1.2e-3 to 2.3e-3 |
| (1, 0.8) | 0.01858 (0.00005); entries 0.01866, 0.01866, 0.01854, 0.01846 (SE ~1e-4) | 0.01852 (2.9e-4), 0.01846 (3.8e-4), 0.01839 (2.1e-4), 0.01894 (3.5e-4) | 2.3e-3 to 4.3e-3 |

Agreement within about 1 SE on every entry, a positive offset as the sign convention predicts, equal entries despite
the anisotropic F (the limit gradient does not depend on F). The per-run spread decreases with t (e.g. (4, 0.5):
8.6e-3 at t = 1.25e5 down to 1.6e-3 at 8e6), consistent with a.s. convergence. Caveat: a single run at T = 8e6 is
only accurate to ~2e-3, i.e. ~15% of the offset; the comparison is statistically meaningful only for the run-mean.
k = 2 is outside the corollary (k >= 3), so this tests the theorem, not the B^{-1} expansion.

## Section 5, total-variation paragraph (line 340) -- CORRECT

beta = 1/2, xi = +-1 equiprobable: N_beta = sum_{j>=0} 2^{-(j+1)} xi_j = 2U - 1 with U uniform on [0,1] (binary expansion),
so pi is uniform on [-1, 1] (KS test on 1e6 samples: p = 0.90). From a fixed point the j-step law has at most 2^j atoms,
so its total-variation distance to pi equals 1 for every finite j; the same holds in Hellinger distance. The sentence
is right; "can fail at every finite time" would read better as "the total-variation distance to stationarity equals 1
at every finite time". The attribution to Duchi et al. (mixing measured in TV/Hellinger) matches my recollection of
that paper but was not re-read for this report.

## Suggested edits (all optional, none changes a statement)

1. l. 273-277: say e_0 is random via N_0; fix the starting index.
2. l. 290: "minimum of 2K|g-g'| and 2C beta^j (1+|n|)" for the change.
3. l. 295 or 302: one sentence on joint continuity of (g, n) -> U_g(n) (measurability, freezing lemma).
4. l. 269: require the selection O_t to be adapted to a filtration for which xi_{t+1} is independent of the past.
5. l. 314: state coercivity E nu(g+N) >= nu(g) - E nu(N) to justify "compact minimizer set".
6. l. 331: say explicitly that no rate in t uniform in beta is claimed.
