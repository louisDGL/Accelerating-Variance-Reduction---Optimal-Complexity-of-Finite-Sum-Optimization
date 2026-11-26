# Adversarial audit of `polar_bias/paper/polar_bias.tex`

Date: 2026-11-25. Scope: the five targeted items below. Line numbers refer to `polar_bias/paper/polar_bias.tex`.
Numerical checks: `polar_bias/note2/audit_checks.py` (run with `polar_bias/.venv/bin/python`, about 3 minutes); outputs quoted in each item.
No existing file was modified.

## Summary of verdicts

| Item | Statement | Verdict |
|---|---|---|
| 1 | Thm `thm:momconv` (convergence with momentum) | CORRECT WITH MINOR FIX (four one-line additions; no counterexample found) |
| 1' | Remark `rem:mom`(i) (combining the two momentum theorems) | GAP, fixable (small-ball hypothesis on `xi_b` is not implied by Assumption N) |
| 2 | Thm `thm:mombias` (first-order bias with a filter) | CORRECT; hypotheses can be weakened and the proof replaced by three lines |
| 3 | `cor:dim` | CORRECT WITH MINOR FIX (Assumption N not stated; abstract's lower bound 12/5 is numerical) |
| 3 | `cor:rank1` | CORRECT WITH MINOR FIX (K,d >= 2 not stated) |
| 3 | `cor:compare` | CORRECT WITH MINOR FIX (standing hypotheses not stated; strictness in (iv)) |
| 3 | Remark `rem:a1` | CORRECT (as a statement of what is and is not proved) |
| 3 | `prop:lin` | CORRECT WITH MINOR FIX (Assumption O missing; the least-squares `chi` as written has a degenerate covariance) |
| 3 | `prop:wor` | CORRECT |
| 4 | `lem:osc`, `lem:edge` | CORRECT (conditional on the quoted Bhattacharya-Rao Theorem 20.1, which I also could not check against the book) |
| 5 | `thm:state` | CORRECT; existence is claimed only under continuity, uniqueness is not claimed, no gradient structure is used |

Nothing was found to be WRONG. I could not construct a counterexample to any statement.

---

## 1. Theorem `thm:momconv` (lines 490-545)

### Line-by-line

**Measurability (496).** With `W_1`, `M_0` deterministic, `W_t` and `G_t` are `F_{t-1}`-measurable, `M_t`, `O_t` are `F_t`-measurable. OK.

**Three-point identity (497-502).** Verified: `V_{t+1}-V_t = D_F(W_{t+1},W_t) + <G_t-G^dag, W_{t+1}-W_t>`. `D_F(W_{t+1},W_t) <= (L_op/2) eta_t^2` because `||polar||_op <= 1`. `||G-G^dag||_*^2 <= 2 L_op D_{F*}(G^dag,G)` is the `1/L_op`-strong convexity of `F*` for the nuclear norm. OK.

**Decomposition (504-510).** `M_t = bar G_t + N_t` is exact. `bar G_t - G_t = (1-beta) sum_{k<t} beta^k (G_{t-k}-G_t) + beta^t (M_0-G_t)`. The first part is `<= C eta_t` deterministically. The second involves `G_t`, which is random; it is nevertheless bounded deterministically by `||G_1||_* + L_op sum_{u<t} eta_u = O(t^{1-theta})` or `O(log t)`, so `beta^t ||M_0-G_t|| <= C eta_t`. *Minor fix: say this.* The lag split `N_t = N_t^{(m)} + beta^m N_{t-m}` is exact, `N_t^{(m)}` depends on `xi_{t-m+1..t}` only, `G_{t-m}` is `F_{t-m-1}`-measurable, `rho_t` is `F_{t-m}`-measurable. OK. (Norms are mixed between Frobenius and nuclear at 505/509; constants only.)

**Lagged drift (512-515).** `c_t = <Phi_{N^{(m)}}(G_{t-m}), G_{t-m}-G^dag>` is legitimate because `G_{t-m}` is `F_{t-m}`-measurable and `N_t^{(m)}` is independent of `F_{t-m}`. Subgradient inequality in the right direction. Coupling `N_beta = N^{(m)} + beta^m N'` (independent) gives the `2 beta^m E||N_beta||_*` comparison. The law of `N^{(m_t)}` depends on `t` through `m_t`; harmless. OK.

**Perturbation (517-524).** Li-Sun needs both matrices of full rank; if `M_t` is rank deficient then `sigma_r(Y_t) <= rho_t` and the bound reduces to the trivial `2 sqrt r`. OK. The conditional small-ball step is correct: conditioning on `F_{t-m}` and on `xi_{t-m+1..t-1}`, `xi_t` is independent of everything else in `Y_t`, and `rho_t` is in the conditioning. Needs `m_t >= 1`. The layer-cake integral gives `C rho log(e/rho) <= C_kappa rho^kappa`. The split `X_t - tilde X_t = <O_t - tilde O_t, G_{t-m}-G^dag> + <O_t, G_t - G_{t-m}>` is what line 522 uses. OK.

**Moment bound (525-528).** `|V_{t+1}-V_t| <= C eta_t (1+V_t)` and `1+V_{t-m} <= 2(1+V_t)` for large `t` are pathwise and deterministic in their constants. Cauchy-Schwarz needs `sup_t E||N_t||^{2 kappa} < inf`, given by `E||xi||_F^2 < inf`. Summability: `eta_t r_t^kappa = O(t^{-theta(1+kappa)} log^kappa t)`, summable iff `theta(1+kappa) > 1`, possible with `kappa<1` iff `theta > 1/2`. `eta_t beta^{m kappa} <= t^{-theta-2kappa}`: summable since `theta+2kappa > 1` follows from `theta(1+kappa)>1`. `eta_t m_t eta_{t-m}` is `O(t^{-2theta} log t)`. Discrete Gronwall gives `sup v_t < inf` and `sum eta_t E p_t < inf`. OK.

**Series (2): are the `D_{t,j}` martingale differences? (531-536).** Yes, and this is the point where the momentum buffer could have broken the argument. It does not, because the decomposition is applied to `tilde X_t`, not to `X_t`: `tilde X_t` is a function of `G_{t-m}` (in `F_{t-m-1}`) and `xi_{t-m+1..t}` only; the buffer's dependence on older noises and iterates was moved into `Delta_t` and paid for in series (1). `D_{t,j}` is `F_{t-j}`-measurable with `E[D_{t,j} | F_{t-j-1}] = 0`, `m_t` is deterministic, so for fixed `j` the sum over `t` is a martingale for `(F_{t-j})_t`. `{m_t > j} = {t > e^{j/c_beta}}`. The Doob-Minkowski bound is `C sum_j (sum_{t>max(T_0,e^{j/c})} eta_t^2)^{1/2} = O(T_0^{1/2-theta} log T_0) -> 0`. *Minor fix: add the sentence that passes from "sup of the tail tends to 0 in L^2" to almost sure convergence (the oscillation `sup_{T,T'>=T_0}` is monotone in `T_0` and tends to 0 in probability, hence a.s.).*

**Robbins-Siegmund step (536-539).** `V_{T+1} + sum eta_t c_t` converges a.s.; adding the summable `2 eta_t beta^{m_t} E||N_beta||` makes the summands nonnegative; `V >= 0` gives both conclusions. OK.

**Identification of the limit (541-544).** OK. *Minor fix: the continuity of `G^dag -> D_{F*}(G^dag,G)` must be uniform in `G` on bounded sets (it is: locally Lipschitz with constant `<= C(1+||G||)/mu_op`), since the cluster point `hat G` is random and the convergence is known only on a countable set of `G^dag`.*

**Parenthetical at 492-493** ("`xi` symmetric with 0 in its support"): one needs 0 in the support of `N_beta`; true (finitely many terms small with positive probability, tail small in probability). OK.

### Attempt at a counterexample (dimension 1, polar = sign)
`F(w)=w^2/2`, `xi = Exp(1)-1` (bounded density, so Assumption M holds), 2000 runs, `T = 2*10^5`:

| beta | theta | `-median(N_beta)` (4e6 stationary samples) | tail-averaged mean of `G_t` |
|---|---|---|---|
| 0.5 | 0.75 | 0.12677 +- 0.00036 | 0.12737 +- 0.00011 |
| 0.9 | 0.75 | 0.02289 +- 0.00014 | 0.02280 +- 0.00014 |
| 0.9 | 1.0 | 0.02256 +- 0.00014 | 0.02291 +- 0.00011 |
| 0.9 | 0.55 | 0.02288 +- 0.00014 | 0.02254 +- 0.00016 |

No discrepancy beyond about 2 standard errors. The Li-Sun inequality was also tested on 80000 random pairs (including nearly singular ones, sizes 2x2 to 4x7): the largest ratio LHS/RHS was 0.999, so the inequality held and the constant 2 is essentially attained.

### Gap in Remark `rem:mom`(i) (584-585) — fixable
The remark combines `thm:momconv` (which needs the small-ball bound `sup_Y P(sigma_r(Y+xi) <= t) <= Ct` for `xi = xi_b`) with `thm:mombias` (Assumption N). Cramér's condition does not imply the small-ball bound (it holds for some singular laws), and a bounded density alone does not either (the set `{sigma_r <= t}` has infinite Lebesgue measure). Fix: assume in the remark that `zeta` has conditional densities of each row given the others bounded by a constant. This passes to `xi_b` for each fixed `b` (condition on `zeta_2..zeta_b` and on the other rows of `zeta_1`; constant `O(b)`), and implies Cramér.

**Verdict item 1: CORRECT WITH MINOR FIX** (the three additions marked above, plus the hypothesis in Remark `rem:mom`(i)).

---

## 2. Theorem `thm:mombias` (lines 547-581)

### The regrouping argument works and gives a shorter proof
`N_w = sum_k w_k xi_b^{(k)} = (1/b) sum_{i<=b} Y_i`, `Y_i = sum_k w_k zeta_{k,i}`, i.i.d. in `i`. Check of Assumption N for `Y`:

- Convergence and moments: `||Y||_{L^3} <= sum_k |w_k| ||zeta||_{L^3} < inf`.
- Cumulants are additive and homogeneous: `Cov(Y) = rho_2 Sigma` (positive definite since `rho_2 > 0`), `K_3(Y) = rho_3 K_3`.
- Cramér: `|phi_Y(t)| = prod_k |phi_zeta(w_k t)| <= |phi_zeta(w_{k_0} t)|` for any `k_0` with `w_{k_0} != 0`, so `limsup < 1`.

Theorem `thm:cf` applied to `Y` gives `-Med(N_w) = -(1/6b) J_{rho_2 Sigma}^{-1} T_{rho_2 Sigma}[rho_3 K_3] + o(1/b) = rho_2^{1/2} rho_2^{-3/2} rho_3 g_*/b = (rho_3/rho_2) g_*/b`, using the scalings of `lem:gauss` (`J_{c Sigma} = c^{-1/2} J`, `T_{c Sigma} = c^{-3/2} T`, both correct by 0-homogeneity of `o`). This is exactly the stated conclusion.

### Consequences
- The stated theorem is **correct**. I also checked the note's own proof (conditioning on `R`, Lindeberg replacement, the factor `w_0^3 + sum_{k>=1} w_k^3 = rho_3`): it is correct, only longer than necessary.
- **`E|zeta|^4 < inf` is superfluous**: three moments suffice.
- **`w_0 != 0` can be weakened** to "some `w_k != 0`".
- `sum |w_k| < inf` can be weakened to `sum w_k^2 < inf` (a.s. convergence by Kolmogorov, third moment by Rosenthal); I did not write this out in detail.
- **General filter weights are covered**, including signed ones. Then `rho_3` can be zero or negative, the formula still holds, but the reading "effective batch size `b rho_2/rho_3`" (554) is meaningful only for `rho_3 > 0`.
- Special cases verified: EMA `rho_2/rho_3 = (1+beta+beta^2)/(1-beta^2)`; Nesterov `rho_p` formula (weights sum to 1).
- The limitation in Remark `rem:mom`(ii) (`beta -> 1` at fixed `b`) remains as stated: regrouping does not help there.

### Numerical check (dimension 1, `zeta = Exp(1)-1`, `g_* = 1/3`, 4e6 samples)
| weights | predicted `(rho_3/rho_2)/3` | `-b * median`, b = 8 / 32 / 128 |
|---|---|---|
| (1) | 0.3333 | 0.3279 / 0.3268 / 0.3350 |
| (1, -0.5) | 0.2333 | 0.2353 / 0.2396 / 0.2520 |
| (0.6, 0.3, 0.1) | 0.1768 | 0.1738 / 0.1765 / 0.1699 |
| (1, -1) | 0 | 0.0010 / -0.0015 / -0.0081 |
| (0.5, -0.9) | -0.1899 | -0.1900 / -0.1896 / -0.1961 |

Standard errors are about 0.002, 0.004, 0.008 for the three batch sizes; everything agrees within about 2.5 standard errors, including the negative `rho_3` case.

**Verdict item 2: CORRECT. Recommended: replace the proof by the regrouping argument, drop the fourth moment, replace `w_0 != 0` by `w != 0`.**

---

## 3. Corollaries of the two-moment theorem, `prop:lin`, `prop:wor`

### `cor:dim` (349-360) — CORRECT WITH MINOR FIX
Hypotheses used: Assumption N (three moments, Cramér) for the entry law, `Sigma = sigma^2 I` (automatic), `K,d >= 2` for the strict bounds, `0 < q < (r-1)/2`.
- `cA = cB = kappa_3 11^T`: verified by independence.
- Upper bound: `3(3+2q) < 3(r+2)` gives `3/(max(K,d)+2)`. Verified.
- **Not stated:** Cramér's condition for the entry law. It is necessary: for lattice entries the median of the mean oscillates (the note says so in `rem:cf`(iii)). Entrywise Cramér does imply joint Cramér for independent entries. Fix: "for i.i.d. entries satisfying Assumption N".
- **Abstract, line 67:** "`theta(gamma) in [12/5, 3]`" — the lower bound rests on the monotonicity of `theta`, which the corollary itself labels numerical. My check: `theta = 6 kappa` is strictly decreasing on a 2001-point grid of (0,1], `theta(1) = 2.4`, `theta(1/4) = 2.8981`, `theta(10^-4) = 2.99996`. The abstract should say the lower bound is numerical.
- The hypergeometric form of `mu_s` matches the Marchenko-Pastur integral to 12 digits at `gamma = 0.3`.

### `cor:rank1` (362-371) — CORRECT WITH MINOR FIX
Hypotheses used: Assumption N, `Sigma = sigma^2 I`, rank one a.s., and **`K,d >= 2`** (not stated; for `r = 1` the ratio is exactly 1 and the interval `(1,r)` is empty).
- Can a rank-one law satisfy Cramér with a nondegenerate covariance? **Yes.** The rank-one variety has dimension `K+d-1 < Kd` but spans `E`. For `zeta = u v^T` with `u, v` independent isotropic, `Cov = sigma_u^2 sigma_v^2 I`. If `(u,v)` has a density, Cramér holds by the Bhattacharya-Ghosh lemma, because the monomials `u_i v_j` together with 1 are linearly independent analytic functions. The class is therefore non-empty (e.g. Gaussian `u, v`); it would be worth adding this sufficient condition next to "Cramér is not automatic".
- The identity `zeta zeta^T zeta = ||zeta||_F^2 zeta`, the factor, both bounds, `1.37` for 2x2 and `2 kappa r`: all verified.

### `cor:compare` (373-397) — CORRECT WITH MINOR FIX
The corollary states no standing hypothesis; every item uses Assumption N with `Sigma = sigma^2 I` and `K,d >= 2` (through `a_1, a_2`).
- (i) The sign-descent claim uses that `Sigma` is diagonal (true here); more simply, coordinate medians depend on marginals only.
- (ii) Verified in full: for the explicit law, `Cov = I`, the only nonzero third cumulants are the permutations of `((11),(12),(12))` with value `sqrt2 rho`, `cA = cB = tau e_1 e_1^T`; the law has a density, so Cramér holds. Needs `d >= 2`.
- (iii) Verified: `cA_11 = 1 + c`, `cB_11 = 1`, `cB_22 = K_3^{(22)^3} = 0`, all other entries zero, so `a_1 cA + a_2 cB = 0`. The existence argument (open mapping near the Gaussian) is only sketched; an explicit law is available by the construction of (ii): `zeta_11 = rho (zeta_22^2 - 1)/sqrt2 + sqrt(1-rho^2) e` with the skewness of `e` chosen so that `kappa_111 = sqrt2 rho / c`.
- (iv) Needs `kappa_3(epsilon) != 0` and `M != 0` for the strict inequality, three moments for `zeta'`, and Cramér for `zeta` (sufficient: Cramér for `zeta'`, since `|phi_zeta| <= |phi_zeta'|`).

### Remark `rem:a1` (399-410) — CORRECT
The equivalence, the threshold expansion `(K-1)/2 - (K-1)(K+2)/(2d)`, the 2x64 value (0.49609), and the singular-value form are verified. In the proportional limit, `kappa(gamma) - 1/(2(1+gamma)) > 0` on my grid (minimum 4e-5 at `gamma = 10^-4`, consistent with equality at 0+). The remark is honest that there is no proof for all sizes; the abstract ("on every size computed") is consistent.

### `prop:lin` (452-466) — CORRECT WITH MINOR FIX
Hypotheses used in the proof: Assumption N for `chi` in `R^p`, continuity and surjectivity of `L_W`, **and Assumption O** (to bound the oscillation term), which is not in the statement.
- The oscillation transfer, the compactness of `{Sigma(W)}`, and the push-forward of the Edgeworth term are correct.
- **Least-squares example (455-456):** as written, `chi = (xx^T - E xx^T, eps x^T)` with the first block in `R^{d x d}` has a degenerate covariance (symmetric matrices), so Assumption N fails literally. Fix: take the first block in `Sym(d)`, `p = d(d+1)/2 + Kd`; `L_W` stays onto through the second block. One also needs `E|x|^6 < inf`, `E|eps|^3|x|^3 < inf`, and a density for `(x, eps)` (nondegeneracy and Cramér, `rem` (iii) at 473-475).

### `prop:wor` (600-614) — CORRECT
Only `sum v_i = 0` and `1 <= b <= n-1` are used. The complement identity, homogeneity and oddness of the median, and symmetry at `b = n/2` are right. The finite-population formulas for the covariance and the third cumulant are the classical ones; the formal factor `(n-2b)/(n-2)` follows from them with the scalings of `lem:gauss` and is consistent with the duality (checked algebraically). It is correctly labelled formal.

---

## 4. Lemma `lem:osc` (166-179) and Lemma `lem:edge` (183-198)

- **Row-distance inequality (173):** correct and standard: for a unit `a`, `|sum_j a_j rho_j| >= |a_j| dist(rho_j, span of the others)` and `max_j |a_j| >= K^{-1/2}`.
- **Conditioning with a general (non-product) covariance (174-175):** correct. Conditionally on the other rows, `rho_j` is Gaussian with a mean that is affine in them (and in the shift `y`) and covariance equal to a Schur complement of `Sigma`; its inverse is a principal block of `Sigma^{-1}`, so the Schur complement is `>= lambda_min(Sigma) I`. A unit normal to the span of the other rows exists (`K-1 < d`), can be chosen measurably, and is fixed by the conditioning. The union bound gives `P(sigma_K <= t) <= 2 K^{3/2} t / sqrt(2 pi lambda_min)`, uniform in `y` and over compact sets of covariances.
- **Numerical check** with a random dense 9x9 covariance on 3x3 matrices (`lambda_min = 0.054`): `P(sigma_3 <= t)/t` stabilizes at 1.78, 1.43, 0.65 for three shifts (bound 17.8). For 2x4 the probability is `o(t)`, as expected.
- **Oscillation (176-178):** for `sigma_r(x) > 2 eps`, Li-Sun gives `2 eps/(sigma_r - eps) <= 4 eps/sigma_r`; otherwise the bound 2 is trivial. The layer-cake integral gives `C eps log(e/eps)`. So "Assumption O holds with any `a < 1`" is **correct** for the spectral geometry (and `a = 1` for sign; for Euclidean the same `eps log` bound). Any `a > 0` suffices in `lem:edge` because the radius is `2 e^{-cb}`. Cosmetic: a factor 2 is missing in the sign case (`2 N eps sup phi`).
- **`lem:edge`:** the sign of the `P_1` density and the integration by parts are right. Uniformity in `h` is correct because Assumption O has a supremum over shifts, which also covers the variant of the oscillation functional with a supremum over translations. The only external dependency is the form of Bhattacharya-Rao Theorem 20.1; it matches my recollection of the statement, but like the authors I could not check the book.

**Verdict item 4: CORRECT.**

---

## 5. Theorem `thm:state` (427-450)

- **What is proved:** every `W` in `U` with `|h_b(W)| <= delta b^{-1/2}` (so every zero) lies within `eps/b` of `W_* + H^{-1} g_*/b`. Step 1 (lower bound on `Phi`, forcing `|W - W_*| <= C_3/b`) and Step 2 (linearization, `J H H^{-1} g_* = -T/6`) are correct.
- **Gradient structure:** none is used. Unlike `thm:cf`, the proof does not use convexity or a potential; it uses only the uniform expansion, the lower bound of `lem:gauss`, and the invertibility of `J_{Sigma_*} H`. `H` need not be symmetric for the argument.
- **Existence:** claimed only "if moreover `h_b` is continuous on `U`" (434), by a degree argument that is correct. Without continuity no zero is claimed, and none need exist (the manuscript's two-point example). The abstract ("all zeros ... are at") is consistent.
- **Uniqueness:** not claimed, and not proved. Correct as stated.
- Minor points:
  1. Assumption O is listed as a hypothesis but is not used (it enters only through U(iii)).
  2. Step 1 uses the constant `c_0` of `lem:gauss` uniformly over a compact set of covariances; the lemma states uniformity only for the derivative bounds. It is true (`c_0` depends on `lambda_min(J_Sigma)` and a third-derivative bound, both uniform) and should be stated.
  3. A free strengthening that needs no continuity: at `w = H^{-1} g_*` the expansion gives `|h_b| = o(b^{-1/2})`, so the set of `delta`-approximate zeros is non-empty for large `b`.
  4. The theorem locates zeros; it does not say the iteration converges to one. (The linearization `J_{Sigma_*} H` has positive spectrum when `H` is symmetric positive definite, so the zero is linearly stable, but this is not claimed.)
  5. Continuity of `h_b` holds when `g(.;omega)` is continuous and `hat g_b(W)` has full rank a.s. for every `W`; Cramér alone does not guarantee this.

**Verdict item 5: CORRECT.**

---

## Recommended edits, in order of importance

1. Remark `rem:mom`(i): add the small-ball hypothesis on `xi_b` (bounded conditional row densities of `zeta`).
2. `thm:mombias`: replace the proof by the regrouping argument; drop `E|zeta|^4 < inf`; replace `w_0 != 0` by `w != 0`; note that `b_eff` is meaningful only for `rho_3 > 0`.
3. `prop:lin`: add Assumption O; put the first block of `chi` in `Sym(d)`; state the moment and density conditions on `(x, eps)`.
4. `cor:dim`, `cor:rank1`, `cor:compare`: state "under the hypotheses of Theorem `thm:two`" (Assumption N, `Sigma = sigma^2 I`, `K,d >= 2`); in `cor:rank1` add the sufficient condition for Cramér; in (iv) add `kappa_3(epsilon) != 0`, `M != 0`.
5. Abstract line 67: mark the lower bound `12/5` of `theta` as numerical.
6. `thm:momconv` proof: the three one-line additions listed in item 1.
7. `thm:state`: drop Assumption O; state the uniformity of `c_0` in `lem:gauss`.
