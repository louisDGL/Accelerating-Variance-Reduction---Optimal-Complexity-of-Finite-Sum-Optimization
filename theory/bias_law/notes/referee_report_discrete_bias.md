# Referee report 2 — `theory/bias_law/notes/discrete_bias.tex`

Date: 25 November 2026. Line numbers refer to `discrete_bias.tex`. Test scripts are in `experiments/bias_law/referee_note2/` (`r1`–`r9`). No existing file was modified. Rerunning `v01_wick_moments.py 8` rewrote `results/wick_moments.pkl` with byte-identical content (checked with `cmp`).

## Summary of verdicts

| # | Item | Verdict |
|---|------|---------|
| 1 | Section 3 (Lemmas cube, near, sobolev, ball) | CORRECT |
| 2 | Section 4 (Assumption R, examples, Prop. field, Thm. codim, corollaries) | CORRECT WITH MINOR FIX |
| 3 | Section 5 (Thm. sign) | CORRECT WITH MINOR FIX (tie case; abstract contradicts the theorem) |
| 4 | Section 6 (2x2) | CORRECT WITH MINOR FIX (shell cut-off at R must be 2R) |
| 5 | Section 8 (Thm. a_1 < 0) | CORRECT (certificates reproduced); Remark (iii) has an unproved input |
| 6 | Prop. lattice2 and Table 1 | Constant CORRECT, label honest; **2x2 numbers in Table 1 are WRONG** (bug in `e01_exact_lattice.py`); "three digits" and "monotonically" are overclaims |
| 7 | Overclaims | Several, listed in section 7 |

I found no error in any theorem, proposition or lemma. The one hard error is numerical: the 2x2 exact computations.

---

## 1. Section 3 — CORRECT

**Lemma cube (l.152–173).** Every step checks.
- `E cos<u, zeta - zeta'> <= 1 - <u,Sigma u> + (2/3)M^4|u|^4`: uses `cos x <= 1 - x^2/2 + x^4/24`, `E<u,zeta-zeta'>^2 = 2<u,Sigma u>`, `|zeta-zeta'| <= 2M`. The threshold `u_0 = (3 lambda/4)^{1/2} M^{-2}` is exactly where `(2/3)M^4|u|^2 <= lambda/2`.
- Splitting into A, B: `eps_w <= 1/2` gives `max w_k^2 <= rho_2/4`. The greedy fill stops at a finite index even for infinite sequences, with sum in `[rho_2/4, rho_2/2]`, so B has at least `rho_2/2`.
- AM–GM gives `prod|phi| <= (g_A + g_B)/2`.
- Fubini is legitimate (`Delta` is bounded with compact support, `Y_A` is bounded since `sum w_k = 1`). The transform of the triangle is `alpha (sin(alpha y/2)/(alpha y/2))^2 >= 0`. The left side is real, so the bound by `E check-Delta(Y_A)` holds.
- On `supp Delta`, `|s| <= sqrt(N) alpha = u_0/max w_k`, so `g_A(s) <= exp(-lambda |s|^2 sum_A w_k^2 / 2) <= exp(-lambda rho_2 |s|^2/8)`.
- Change of variables: half-side `sqrt(rho_2) alpha/2 = u_0/(2 sqrt(N) eps_w)`, and the Jacobian `rho_2^{N/2}` cancels. `C_1 = 2^N (8 pi/lambda)^{N/2}`.

Numerical test (`r1_cube.py`), `zeta in {-1,2}` with probabilities 2/3, 1/3, sup over `t_0` of the integral of `|phi(t/sqrt b)|^b` over a window of half-length `c sqrt(b)`:

| c | b=4 | 64 | 1024 | 65536 |
|---|-----|----|------|-------|
| 0.25 | 0.922 | 1.760 | 1.772 | 1.7724 |
| 1 | 1.704 | 1.767 | 1.772 | 1.7724 |
| 3 | 5.111 | 5.302 | 5.316 | 5.3173 |

Bounded in b; the limit for small c is `sqrt(pi) = sqrt(2 pi/sigma^2)`, one resonance. With momentum weights (`beta = 0.99`, `b = 1`) the sup is 1.75–1.77.

**Lemma near (l.179–191).** Correct. `rho_5 rho_2^{-5/2} <= eps_w^3`. The cross terms of `A^2/2 - eps_3^2 p_1^2/2` are `O(eps_w^3 |t|^7)` or smaller on `eps_w|t| <= sqrt(N) c_2`. `|A|^3 <= C eps_w^3 |t|^9`. `e^{|A|} e^{-<t,Sigma t>/2} <= e^{-lambda|t|^2/4}`.

**Lemma sobolev (l.193–210).** Correct.
- The `Q_j` tile E.
- On `Q_j`, `|t|_inf >= (2|j|_inf - 1)a >= a|j|_inf`, so the answer to "is `|t| >= a|j|_inf`?" is yes.
- `Q_j` has half-side `c_2/eps_w <= c_1/eps_w`, so Lemma cube applies.
- `#{j : |j|_inf = n} ~ n^{N-1}`, so the sum converges exactly when `m > N`.

**Lemma ball (l.222–237).** Correct.
- (i): `k >= (2/pi)^{2N}` on `[-pi,pi]^N`; the transform of `k(./r)` is supported in `|t|_inf <= 1/r`; the cover needs `(1 + eps_w/(c_1 r))^N` cubes.
- (ii): the balls `B(z_i, rho/2)` lie in `{B(0,2R), dist <= 2 rho}` because `rho <= R`; homogeneity gives `C R^{N-c} rho^c`.
- (iii): Hoeffding gives `2 exp(-u^2/(2M^2))` per coordinate, since `sum (w_k/sqrt(rho_2))^2 = 1`. The union over N coordinates at level `u/sqrt(N)` gives the stated bound.

---

## 2. Section 4 — CORRECT WITH MINOR FIX

**Lemma examples (c), l.253–261.**
- The integral representation is right: `A^{-1/2} = (2/pi) int_0^inf (A + tau^2)^{-1} d tau`. Numerically the error is 3e-15 on a random 2x3 matrix.
- Word count, redone: with `n_1` letters of type `HM^T` or `MH^T`, `n_2` letters `HH^T`, and Y in {M, H}, there are `2(n_1+n_2+1)` halves and `n_1 + [Y=M]` letters M. Each M claims a distinct adjacent half. Unpaired halves: `n_1 + 2n_2 + 2 - [Y=M] = k+1`. Correct.
- **Minor fix:** the tau-integral of `(sigma_K^2 + tau^2)^{-(k+1)/2}` converges only for `k >= 1`. Say "for `k >= 1`; `k = 0` is `|polar| <= 1`".
- Finite differences (`r2_polar.py`, 300 matrices x 10 directions, including directions in the small singular subspace): `max |D^k polar| sigma_K^k` is stable as `sigma_K` goes from 1e-1 to 1e-3.

| size | k=1 | k=2 | k=3 |
|------|-----|-----|-----|
| 2x3 | 0.99 | 1.1 | 2.9 |
| 3x5 | 0.99 | 1.0 | 2.9 |

- Volume: the row argument is right (`|u_i| >= 1/sqrt(K)`). Monte Carlo exponents of `vol{sigma_K <= r}`: 1.99 (2x3), 3.00 (2x4), 3.00 (3x5), 1.00 (2x2).

**Prop. field (l.265–284).** Correct.
- `R`: `P(|X_w| > R-1) <= 2N exp(-3 L_w) <= 2N eps_w^3`.
- `F_1`: derivatives of `Theta` live where `eps <= dist <= 2 eps`, so `eps^{-j} <= 2^j dist^{-j}`. The shell sum converges because `m = N+1 > c`. The result is `eps^3 R^N + eps^c R^{N-c} + eps^{N+1} R^N`, and `N >= 2` gives `eps^{N+1} <= eps^3`.
- Presentation only: "the same for the Gaussian weights" (l.272) deserves one line. The tail of `|y|^6 phi_Sigma` beyond `M sqrt(6 N L_w)` is `<= C L^{(N+4)/2} eps^{3N}` because `lambda_max(Sigma) <= M^2`; this is fine for `N >= 2`.

**Thm. codim (l.286–320).** Correct.
- Parity: `D^4 Phi(x)` and `D^6 Phi(x)` are `O(|x|)`, so the `eps_4` and `eps_3^2` terms are `O(eps_w^3)`. `D^3 Phi(x) = D^3 Phi(0) + O(|x|^2)` because `D^4 Phi(0) = 0`.
- The localisation by the subgradient inequality is correct. `E o(y + X_w)` is a subgradient for any Borel selection, so atoms on V are harmless.
- Scaling: centre `sqrt(rho_2) eps_3 g_* = (rho_3/rho_2) g_*`. Radius `sqrt(rho_2) eps_w^{min(c,3)} L_w^{N/2}`: right. Batch means: `b^{-1/2} b^{-1} = b^{-3/2}` for `c = 2`, `b^{-2}` for `c >= 3`: right.

**Cor. finite sums (l.322–333).** Correct as stated: with replacement, fixed W, `n >= Kd+1`, affinely spanning gradients.

**Cor. batch and momentum (l.335–347).** Algebra verified: `rho_2 = (1-beta)/(b(1+beta))`, `rho_3/rho_2 = (1-beta^2)/(b(1+beta+beta^2))`, `eps_w^2 = (1-beta^2)/b`, `sqrt(rho_2) = eps_w/(1+beta)`, `rho_3/rho_2 >= eps_w^2/3`. The relative exponent `(min(c,3)-1)/2` is right.
- **Minor fix:** the display writes a multiplicative `(1 + O(.))` on the vector `g_*`. That is meaningless if `g_* = 0` (symmetric noise) and loose for a vector. Write `Med = -g_*/b_eff + O(b_eff^{-1} theta^{(min(c,3)-1)/2} log^{N/2}(e/theta))`.

---

## 3. Section 5 — CORRECT WITH MINOR FIX

Esseen's lattice expansion (l.391–394) is quoted with the right sign and argument: `S_1(y) = floor(y) - y + 1/2`, which is `+1/2` at lattice points and tends to `-1/2` just before the next. The jump `h phi(x)/(sigma sqrt b)` agrees with the local limit theorem. The values `1/2 + phi(0)(m + g_* +- h/2)/(sigma sqrt b)`, the median characterisation, and (i)–(iii) are correct. Unbounded m is handled by monotonicity of the distribution function, which the text leaves implicit.

Exact checks (`r3_sign.py`):
- `{-1,2}`, probabilities 2/3, 1/3: `b G_b = 0, +1, -1` for `b = 0, 1, 2 mod 3` for **every** b from 1 to 3002. No violation.
- `{-1,0,2}` with `P(-1) = 2p`, `P(2) = p` (`a = 0`, `h = 1`, `g_* = 1/6`): the median set is `{0}` for all b tested with `p = 0.05` and `0.2`, and for `b >= 14` with `p = 0.323`. Statement (iii) is confirmed.
- Irrational `a/h`: `zeta in {-s, 1-s}`, `P(1-s) = s`, `s = 1/sqrt 2`. Mean zero forces `a/h = -s`, so the case is not empty. For b from 200 to 20000 the median equals the nearest point of `bs - Z` to `g_* = (1-2s)/6` in all but one case, which is a numerical tie at distance 1/2. The offsets are equidistributed on `[-1/2, 1/2]` (deciles 1980 +- 1). Statement (i) is confirmed.
- `q = 2`: `{-3,-1,1,5}` with probabilities (.1,.5,.3,.1), `g_* = 0.381`: `b G_b = 0` for even b and 1 for odd b, as predicted.

**Minor gap (l.378–379, "converges iff q=1, to h round(g_*/h)").** If `q = 1` and `g_*` lies in `h/2 + hZ`, the proof gives nothing: both neighbours are at distance exactly `h/2`, and the parenthesis on l.377–378 excludes every b. The case is not empty: `zeta in {-1,0,4}`, `P(-1) = 4p`, `P(4) = p` has `h = 1`, `a = 0`, `g_* = 1/2` exactly. Numerically `b G_b = {1}` for all b tested up to 3001 (`p = 0.05`, `0.15`), so the claim looks true but is decided by the next term. Fix: add "if `g_*` is not in `h/2 + hZ`".

**The abstract contradicts the theorem.** l.66–67 says "it oscillates in a window of width h around `g_*` and never converges to it". By (ii) and the last sentence of the theorem (l.383), when `a` is in `hZ` it converges, to `h round(g_*/h)`, with no oscillation; and it converges to `g_*` itself when `g_*` is in `hZ` (for example every symmetric lattice law with `a` in `hZ`). The same applies to the table, l.100: "fails iff the coordinate is lattice". Fix: "does not converge to `g_*` unless `a` and `g_*` are in `hZ`".

Remark, l.514, "for ever": the theorem gives `b >= b_0`. For this law I verified every b up to 3002, so the statement is fine with "for all b computed".

---

## 4. Section 6 — CORRECT WITH MINOR FIX

**Formulas (l.427–436).** `det = (|y+|^2 - |y-|^2)/2`, `||M||_* = sqrt(||M||_F^2 + 2|det|) = sqrt2 max(|y+|,|y-|)` and the polar formula were verified against the SVD on 2000 random matrices (errors <= 3e-15).

**Lemma decay (l.438–464).** Correct.
- `g_r`: on its support `|2 det| < 2r <= 1/128`, so `|y_pm|^2 >= (1/64 - 1/128)/2 = 1/256`. It is invariant under each one-sided rotation because det and chi depend on `(|y+|, |y-|)` only. The `J_0` average and the bound `min(1, C u^{-1/2})` are fine. Coarea with `|grad(2 det)| = 2|y|`.
- Divergence formula: redone with outward normals `-n` on `Omega_+` and `+n` on `Omega_-`. The signs `-i/|t|` and `-1/|t|^2` are right.
- Phase `const + |t_pm||y_pm| cos(theta - theta_0)` with `|y_pm| = |y|/sqrt2` on V. Van der Corput: second-derivative test where `|cos| >= 2^{-1/2}`; on the rest the phase derivative is monotone only on each half, hence four arcs. Correct as written.

**Prop. field2 (l.466–496).** All exponents re-derived.
- `sum over j in Z^4 of |j|^{-3/2} (1 + delta|j|/eps)^{-n} ~ (eps/delta)^{5/2}`, giving `mu^{5/2} eps^{3/2} (eps/delta)^{5/2}`.
- `sum of |j|^{-1/2}(...) ~ (eps/delta)^{7/2}`; with `r = 16 delta/mu` this gives `mu^4 r mu^{-1/2} eps^{1/2} (eps/delta)^{7/2} = 16 mu^{5/2} eps^{3/2} (eps/delta)^{5/2}`.
- With `delta = eps^{8/7}`: `eps^{3/2 - 5/14} = eps^{8/7}`, balanced against `mu^3 delta`.
- The majorant: `|2 det z| <= 9 mu delta` when `B(z, delta)` meets V; variation of `2 det/mu^2` on `B(z, delta')` is at most `0.41 r < 7r/16`; `9/16 + 7/16 = 1`; the cut-off equals 1 on `{1/5 < |y| < 5}`. Fine (eta must be nonnegative).
- Small shells: `P(B(0, 2 eps^{1/2})) <= C eps^2`. Final factor `R^4 ~ L_w^2`. Rate `b^{-1/2} b^{-4/7} = b^{-15/14}`: right.

**Minor fix (l.493).** "Shells with `2^j > R` contribute `C eps^3`" is false as written. Such a shell is supported in `|z| > 2^{j-1} > R/2`, and `P(|x + X_w| > R/2)` is only about `eps^{3/4}` with `R = 2 + M sqrt(6 N L_w)`. Put the cut at `2^j > 2R`. The shells with `R < 2^j <= 2R` join the middle sum, which stays `<= C R^4 eps^{8/7}`. The result is unchanged.

**Thm. square (l.498–510).** Follows.

---

## 5. Section 8 — CORRECT; Remark (iii) rests on an unproved limit

**Equivalences (l.583–588).** Verified with sympy (`r4_a1.py`): `3 r(2) - r(4) = 2 c*`; `m - v/(2m) = c*`; `v = Kd + K + d + 3`. By hand: `E_nu x^{+-1/2} = D_{2+-1}/(Kd)`; for a Gamma law both sides equal `theta(alpha - 1/2)`.

**K = 2 (l.590–597).** Verified by hand: the density, the change of variables, the independence of `rho` and `gamma`, `q_{2,2} = (3 sqrt2 - 2)/7 = 0.320377`, the integration by parts and the final bound. Numerically `q_{2,d}` exceeds `(d-1)/(2(d+3))` for d = 2, 3, 4, 10, 50, and `D_3/D_1 = 2 + d - q` holds to 1e-30.

**K >= 3.** `g^{(n)}` has sign `(-1)^n` for `n >= 1`. The Taylor remainder involves `g^{(6)} >= 0` and the Hermite remainder `g^{(8)} >= 0` times a square, so both minorants are valid on `(0, inf)`. The count of exceptional pairs is 19+13+9+6+4+1 = 52.

Reruns:
- `v01_wick_moments.py 8` then `v02_a1_proof.py`: all eight moment polynomials agree, NUM matches eq. (num) coefficient by coefficient, all seven positivity statements are True.
- `a1/finite.py`: 52 cases certified, smallest bound 0.032317 at (3,21).

Independent checks of mine:
- A separate Wick enumeration for `k <= 6` agrees with the stored polynomials.
- The K = 2 density reproduces the polynomials for `k <= 8`, `d = 2..6`, to 1e-15.
- Monte Carlo (2e6 samples) of `(c* D_1 - D_3)/(Kd)`: 0.191 at (3,3), 0.0344 +- 0.0006 at (3,21) (certificate 0.0323), 0.235 at (4,4).

**Remark (iii), l.629–631.** The logic is valid: a concave h with `h -> 0` at infinity is nondecreasing, hence `<= 0`, while `r(4) - 4 - (K+d-1) = (K-1)(d-1)/(K+d+1) > 0` (verified). **Gap:** the limit `r(p) - p -> K+d-1` is asserted without proof, and it is needed to `o(1)` precision, which requires the tail of the largest eigenvalue to second order. For K = 2 my numerics agree (`r(p) - p` = 3.2, 3.03, 3.012 at p = 4, 64, 256 for d = 2; 4.33, 4.06, 4.02, 4.005 for d = 3). Fixable; until then write "we expect" or supply the Laplace asymptotics.

---

## 6. Prop. lattice2 and Table 1

**Constant: CORRECT.** `J_Sigma = (sqrt(pi)/4) I` for `Sigma = 2I` in R^2, and `J^{-1}/(2 pi sqrt(det Sigma)) = pi^{-3/2}`. `e02_lattice_term.py` rerun gives `E_Lambda = +5.6346, -1.8520, +0.9885`, hence -1.0119, +0.3326, -0.1775. The script uses the weight `e^{-|l|^2/2 tau^2}` rather than `e^{-|l|^2/4 tau^2}`; this is harmless because the limit does not depend on the scale of tau.

**Label.** The statement is honestly labelled "formal; confirmed numerically", and l.554–555 says a proof is not written. Two sentences are not at that standard:
- l.538: "The remainder of Theorem codim for c=2 **is attained** by lattice laws, with an explicit coefficient." This is asserted as fact. Write "is, numerically, attained".
- Abstract l.71–72: "confirm it … **to three digits**." At the batch sizes in the note the agreement is 1–3%: -1.045 against -1.012, 0.3367 against 0.3326, -0.1791 against -0.1775.
- l.558: "each drifting **monotonically**" is false for class 0. The log reads -0.530 (b=18), -1.0155 (66), -1.0795 (258), -1.0449 (1026).

My independent exact computation (`r5_lattice.py`) supports the proposition at larger b; the gap halves when b is multiplied by 4.

| class | b | sqrt(b) err | prediction |
|-------|---|-------------|------------|
| 0 | 4098 | -1.0282 | -1.0119 |
| 0 | 16386 | -1.0200 | -1.0119 |
| 1 | 16384 | +0.3346 | +0.3326 |
| 2 | 32768 | -0.1779 | -0.1775 |

**Table 1 against the logs.** Rows sign, 1x2, 1x3, 2x3 and 2x2 reproduce `results/e01_*.log`. Three points.

(a) **WRONG: the 2x2 numbers.** `checks/e01_exact_lattice.py`, lines 48–52, sets `val = 0` whenever `den <= 1e-300`, that is whenever `M M^T` is singular. The atoms `L` with equal rows or equal columns make `L + m 11^T` rank one for **every** m. On them `m -> ||L + m 11^T||_*` is smooth, with derivative `2 y+_1/|y+|`, which is not zero. Their mass is of order 1/b: 3.7% at b = 17, 1.9% at b = 34, 0.5% at b = 130. They belong in the field. In addition, equal-column atoms are kept or dropped according to the sign of a rounding error in `a*c - bb*bb`.

Two independent implementations (closed form in `r6_2x2.py`, brute-force SVD in `r8_compare.py` and `r9_svd130.py`) agree with each other to all printed digits and disagree with e01:

| b | m | e01 field | correct field |
|---|---|-----------|---------------|
| 17 | 0.03 | -2.41e-3 | -5.07e-3 |
| 130 | 0.1341 | +7.6e-5 | -2.88e-4 |
| 130 | 0.1386 | +1.1e-4 | +1.0e-6 |

Corrected minimizers:

| b | log | corrected | sqrt(b) err, log -> corrected |
|---|-----|-----------|-------------------------------|
| 17 | 0.0370 | 0.0588 | -0.316 -> -0.227 |
| 130 | 0.1341 | 0.1386 | +0.232 -> +0.283 |
| 256 | 0.1261 | 0.1271 | +0.197 -> +0.214 |
| 514 | 0.12131 | 0.12156 | +0.171 -> +0.177 |
| 515 | 0.10559 | 0.10669 | -0.186 -> -0.161 |

The table entry "+0.17, -0.19" should read "+0.18, -0.16". The qualitative claims survive: with the correct subgradient 0 is still a minimizer at b = 129, 255, 384 and 513 (at 513, `f(0-) = -1.6e-2`, `f(0+) = +1.5e-4`). I did not recompute b = 768 and 1026. The 2x3 runs have the same defect with smaller mass (equal rows, order `b^{-3/2}`); I did not recompute them. Fix: on singular atoms use the rank-one part `u_1 v_1^T`, then rerun 2x2 and 2x3.

(b) **Locking is not confined to class 0.** For b = 34 and b = 64 (class 1) the exact minimizer is `1/7` in both; for b = 65 (class 2) it is `1/11`. These are kinks: the field jumps from -2.6e-3 to +4.4e-3 at `m = 1/7` for b = 34, with mass 2.2e-3 of singular matrices at that m. So the class 1 and 2 values are not a smooth trend. "sqrt(b) err = +0.17, -0.19" at one b per class says little, and the sentence on l.517–518 ("compatible with a relative error of order b^{-1/2}") is weakly supported. l.667–673 should mention it.

(c) Small points.
- l.656: "(b <= 400)" — class 0 was computed only up to b = 129.
- e01 hard-codes `q_{2,3} = 0.39347028878`; the exact value from the note's own formula is 0.3933980, so `g_* = 0.094670`, not 0.094674. Immaterial.
- l.665–667, reading (2), "2x3: the error is of order b^{-1}": this rests on `b <= 36`, and the explanation is heuristic. Say so.

---

## 7. Overclaims (abstract and introduction against what is proved)

1. **Abstract l.66–67, "never converges to it"**, and table l.100, "fails iff the coordinate is lattice": false by Theorem sign itself (section 3 above).
2. **Abstract l.71–72, "to three digits"**; l.538, "is attained"; l.558, "monotonically" (section 6 above).
3. **Finite sums, title and abstract l.57–58** ("this excludes the gradient noise of every finite sum. We determine what happens without it"). What is proved concerns the nuclear median of with-replacement minibatch noise at a fixed W, with nondegenerate covariance, hence `n >= Kd + 1` affinely spanning gradients. The abstract does not state the last restriction. The case `n <= Kd` (overparametrised layers, rank-one per-sample gradients) and sampling without replacement appear only in "Not established" (l.718–721). The note never mentions neural networks, which is honest, but "Muon-type steps … finite sums" invites that reading. Put "with replacement, `n >= Kd+1`, fixed parameter" in the abstract.
4. **Dynamics.** l.82–83 ("the iteration drives the gradient to arg min psi") is quoted from the manuscript. For discrete noise and for momentum the note proves statements about medians only; this is acknowledged in Remark pb (iv) and l.724–727. The phrases "batch and momentum at once, uniformly as beta -> 1" (abstract) and "Cor. mom settles it" (l.352) concern `Med(N_beta)`, not the momentum iteration. Say so in the abstract.
5. **K = d >= 3.** Correctly labelled a conjecture. But l.513–516 ("for polar steps on the same noise `b G_b -> g_*`") is unrestricted in (K,d) and must exclude `K = d >= 3`. l.424–425 ("A lattice cannot put a mass of order eps_w on a neighbourhood of width o(eps_w) of a curved hypersurface") is stated generally and proved for 2x2 only. l.671–672 ("polar steps on square matrices inherit …") generalises from one 2x2 law. "We conjecture the same" (abstract l.70) is stronger than the conjecture, which has no rate.
6. **Abstract l.74, "with equality exactly for Gamma laws (K=1)".** Only "equality for Gamma laws" is proved, not the converse.
7. **Abstract l.75, "An audit of PB found no error."** Section 10 itself lists missing proof steps in Theorem 15, a remark needing an extra assumption, unstated hypotheses in four statements, and an abstract-level bound "that rests on a monotonicity that is only numerical". "No counterexample found; several gaps listed" is the accurate summary.
8. **l.175–177, "the mass of |phi|^b near any frequency is at most its mass near the origin".** The lemma proves this up to the factor `2^N` and for the majorants `g_A`, `g_B`. Harmless.

## Required changes, in order of importance

1. Fix `e01_exact_lattice.py` for rank-deficient atoms, rerun 2x2 and 2x3, update Table 1 and l.667–673, and report the locking at nonzero kinks.
2. Correct the abstract: "never converges", "three digits", scope of the finite-sum claim.
3. l.493: cut at `2R`. l.378: exclude the tie case. l.339: additive error. l.259: `k >= 1`.
4. Prove or soften the limit used in Remark (iii), l.629.
5. l.538, l.558: soften.
