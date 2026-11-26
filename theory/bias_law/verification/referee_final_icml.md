# Referee report: "Where Does Muon Converge? The Two-Moment Law of the Bias of Orthogonalized Stochastic Steps"

Files reviewed: `paper/bias_law/icml/twomoment.tex` (+ `exp_section.tex`, `exp_appendix.tex`), `theory/bias_law/shared/statements.tex`, `macros.tex`, `theory/bias_law/proofs/p_{gauss,conv,cf,state,two,a1}.tex`, `figures/bias_law/NUMBERS_twomoment.md`, `table_constants.json`, `data/{fig_digits.json, digits_frozen.jsonl, digits_state.jsonl, fig_digits_largeb.json}`, and (for protocol only) `experiments/bias_law/paper_figures/digits_state.py`, `digits_frozen.py`.

Overall: the mathematics is sound. I found no wrong theorem. The problems are (a) a handful of compact statements and abstract sentences that claim slightly more than what is proved, (b) Section 6 wording on the digits experiment that is more favourable than the data, (c) two novelty claims and several citations I cannot vouch for.

What I recomputed myself (sympy / numpy, a few seconds each):
- the 2x2 linear system of the proof of Theorem 5 gives exactly `a_1`, `a_2` of (a12); determinant `(N^2/18)(K-1)(d-1)(K+2)(d+2)`; `a_1+a_2 = 3(3+2q)/((K+2)(d+2))`; `r a_1+a_2 = 3(dbar-1-2q)/((dbar-1)(dbar+2))`;
- at K=d=2 with q=(3 sqrt2-2)/7: a_1=-0.413207, a_2=1.095849, sum 0.682642 (matches Table 1);
- Lemma `count` by brute-force contraction of the nine pairing tensors for (2,3) and (3,4): all four values agree;
- `E tr W^k`, k<=6, by my own Wick enumeration; the identity `sqrt(m) E T_5 = NUM/(256 m^6)` with the printed NUM: exact match; the seven shifted polynomials have all coefficients positive (minima 112, 1120, 2016, 3136, 4480, 6048, 7840); NUM is negative at (3,21), (4,16), (5,13), (6,11), (7,10), (8,8), so Step 2 is really needed and the split into 52 cases is right (19+13+9+6+4+1);
- Monte Carlo (3e5 matrices): q_{2,2}=0.3201, q_{3,3}=0.6869, q_{3,21}=0.969 (threshold 0.8), and the Stein identity `m_3=(K+d-q)m_1` within MC error;
- every row of Table 1 from the q values in `table_constants.json`; b_eff factors 14.26, 29.26.

NOT rechecked by me: `E tr W^7`, `E tr W^8` and the 52 Hermite certificates (the value 0.0323 at (3,21)); the Pfaffian evaluation of q for large sizes; any simulation.

---

## 1. Mathematical correctness

### Lemma `gauss` (p_gauss.tex)
| item | verdict | comment |
|---|---|---|
| (1) smoothness, oddness, bounded derivatives | CORRECT | |
| (2) Hermite formula, J and T as Gaussian expectations | CORRECT | signs check: `D_h^k phi(y-h) = He_k phi` |
| (3) even derivatives vanish | CORRECT | |
| (4) J positive definite | CORRECT | s and Z' independent, Var s = 1/<v,Sigma^{-1}v>; `<Jv,v> = int phi_tau d(hbar')`, total mass 2 nu(v). Continuity of `Sigma -> J_Sigma` is used but not justified; one line (formula `int o(y) (x) Sigma^{-1}y phi_Sigma(y) dy`) suffices. |
| (5) scaling | CORRECT | `Phi_{c Sigma}(h)=Phi_Sigma(h/sqrt c)` |
| (6) lower bound on the field | CORRECT | |

### Theorem 1, proof (p_conv.tex)
| step | verdict | comment |
|---|---|---|
| lag recursion (Step 1) | CORRECT | `e_t = beta e_{t-1} + beta (G_{t-1}-G_t)` verified; (esum) uses monotone steps and sum eta^2 < inf |
| pathwise inequality (Step 2) | CORRECT | matches the main-text display |
| Poisson equation, Holder bound (Step 3) | CORRECT | `min(x,y) <= x^a y^{1-a}` termwise, sum beta^{j(1-a)} finite |
| martingale + summation by parts (Step 4) | CORRECT | the shift of index is right; alpha in (1/theta-1, 1) exists iff theta>1/2; theta>1/2 is also used for sum eta^2<inf (martingale, boundary term). Joint measurability of U_g(n) (continuity) is what makes `E[U_{G_t}(N_t)|F_{t-1}]=V_{G_t}(N_{t-1})` legitimate; say so. |
| Lyapunov (Step 5) | CORRECT | theta<=1 used for sum eta = inf |
| single limit (Step 6) | CORRECT | |

Remarks: (i) the main-text claim "the limit can be random (sign descent, xi=+-1, beta=1/3)" is not proved; the set [-1/3,1/3] is right, the randomness of the limit is asserted. (ii) Theorem 1 is for i.i.d. noise independent of W. This restriction is not listed in "Scope" or "Limitations", and Section 6 applies the theorem's conclusion to a problem with W-dependent noise.

### Lemma `osc`: CORRECT WITH (cosmetic) FIX
- l1: the probability is at most `2 N eps sup(marginal density)`, not `N eps sup phi` (factor 2, width 2 eps). Constant only.
- Euclidean: the final expectation is not written; add "and `P(|y+Z|<=r) <= C r`".
- Nuclear: the row-distance bound, the conditional covariance bound (Schur complement >= lambda_min I) and the union bound are right. The Li-Sun bound `||polar(A)-polar(B)||_F <= 2||A-B||_F/(sigma_K(A)+sigma_K(B))` for equal-rank rectangular matrices is, to my memory, the main theorem of Li and Sun (SIMAX 2002); I cannot check the constant against the paper here. Please confirm the theorem number and that it is stated for the Frobenius norm and real rectangular full-row-rank matrices.

### Lemma `edge`: CORRECT, conditional on the quotation
The quoted form of Bhattacharya-Ranga Rao, Theorem 20.1, agrees with my memory of it: i.i.d. vectors, finite s-th moment (integer s>=3), Cramer's condition; for every Borel f with `M_{s'}(f)<inf`, 0<=s'<=s,
`|int f d(Q_n - sum_{r<=s-2} n^{-r/2} P_r(-Phi_{0,V}:{chi_nu}))| <= M_{s'}(f) delta_1(n) + c(s,k) omega-bar_f(2e^{-dn}:Phi_{0,V})`, with `delta_1(n)=o(n^{-(s-2)/2})` and d, delta_1 independent of f. To check against the book:
1. that the theorem is stated for general covariance V (not only V=I);
2. that `delta_1(n)` and `d` do not depend on f (this gives uniformity in the shift h);
3. the definition of `omega-bar_f(eps:Phi)` as a supremum over translates of the Gaussian average of the oscillation on balls of radius eps (this is what the lemma uses), and the radius `2e^{-dn}`;
4. that s'=0 is allowed, with `M_0(f)=sup|f|`;
5. that the density of `P_1(-Phi_{0,V}:{chi_nu})` is `(1/6)<K_3,He_3^Sigma> phi_Sigma` with the paper's sign convention.

### Theorem 2 (Cornish-Fisher law), proof: CORRECT
Expansion at `h=(g*+u)/sqrt b`, cancellation `J g* = -T/6`, and the convexity argument on the sphere of radius eps/b are right. (`D^3 Phi(h)=T+O(|h|^2)` uses the vanishing of the 4th derivative of Phi at 0; O(|h|) would suffice.)

### Theorem 3 (effective batch), proof by regrouping: CORRECT
- Cramer for Y: `|E e^{i<t,Y>}| = prod_k |phi(w_k t)| <= |phi(w_{k0} t)|`, and limsup of the right side as |t|->inf is <1. Justified.
- Three moments are enough: Theorem 2 needs exactly `E|Y|^3<inf`, covariance `rho_2 Sigma > 0` (w != 0), Cramer. L^3 convergence of the series follows from sum|w_k|<inf; third cumulants add in the L^3 limit.
- `rho_3/rho_2 = (1-beta^2)/(1+beta+beta^2)` verified.
- Note for the reader: for weights not summing to 1 the gradient offset is `rho_3/(rho_1 rho_2) g*/b`; the theorem as stated (about argmin psi_{N_w}) is correct.

### Theorem 4 (parameter-dependent noise) and Proposition 10: CORRECT (full statements)
Step 1, Step 2, the degree argument and the excess-loss paragraph are right (Peano form of Taylor is enough because F is C^1 and grad F is differentiable at W*). Proposition 10: correct; the push-forward of the Edgeworth term is verified. The continuity of `Sigma -> T_Sigma` is used silently.
The COMPACT statement is not implied by the full one: see Section 2.

### Theorem 5 (two-moment theorem), proof (p_two.tex)
| part | verdict | comment |
|---|---|---|
| invariant-theory reduction | CORRECT | (V(x)W)^{GxH}=V^G(x)W^H, three pairings per factor, S_3 orbits of sizes 3 and 6; the orbit averages are A and B (index pattern of B checked against `zeta zeta^T zeta`) |
| two test contractions | CORRECT | `-m_1` and `m_3-(K+d+1)m_1` verified by hand, including the three delta terms d+1+K |
| counting lemma | CORRECT | verified by brute force |
| linear system, a_1, a_2 | CORRECT | verified symbolically |
| divergence formula, Stein identity | CORRECT WITH FIX | formula and algebra right (`2r+|K-d|=K+d`). "f is C^1" is asserted without proof; Stein's identity only needs f locally Lipschitz with integrable derivative, say that instead. The Candes-Sing-Long-Trzasko formula holds at matrices with simple nonzero singular values (a.e.), which is enough. |
| bounds on q, a_2>0 | CORRECT | |
| (c) | CORRECT | verified symbolically |
| (d) | CORRECT | does not need Theorem 6 |
| (e) | CORRECT WITH FIX | limits verified (`kappa(1)=2/5`, mu_{1/2}(1)=8/(3 pi), mu_{3/2}(1)=64/(15 pi)). Delete the orphan sentence "The hypergeometric form of mu_s is the Narayana expansion for integer s; ..." (no hypergeometric form appears in this paper). |
| K=1 | CORRECT | gives 3/(d+2); for d=1 recovers kappa_3/(6 sigma^2) |
| rank Kd | CORRECT | needs E absolutely irreducible (it is); say "real type" |

### Corollaries 7, 8, 9: CORRECT WITH FIX (9 only)
- Cor 7, Cor 8: correct (bounds verified; `(Kd+2)/(dbar+2)<r` since r>=2... in fact since 2<2r).
- Cor 9 (ii): the construction needs `E eps^3 = -2 sqrt2 rho^3/(1-rho^2)^{3/2}`, which is negative for rho>0, and a centred Gamma variable has positive skewness. Write "minus a centred Gamma variable (for rho>0)". (iii) correct given a_1 != 0 (Theorem 6); the realizability sentence is a sketch, acceptable. (iv) needs kappa_3(eps) != 0, M != 0 for the strict inequality.

### Theorem 6 (a_1<0), proof (p_a1.tex)
| part | verdict | comment |
|---|---|---|
| equivalences (i), (ii), (iii), Gamma equality | CORRECT | `3r(2)-r(4)=2c*`, `m-v/(2m)=c*`, v=Kd+K+d+3 all verified |
| K=2 | CORRECT | change of variables, independence, density (d-1)gamma^{d-2}, the integrand `(n+9)gamma-(n+1)`, the integration by parts and the two lower bounds verified line by line; by symmetry covers d=2 |
| K>=3 logic | CORRECT | `g^{(n)}` has sign (-1)^n for n>=1 (both `c* x^{-1/2}` and `-x^{1/2}`, c*>0); T_5 is a lower bound because the Lagrange remainder is `g^{(6)}(xi)(x-m)^6/6! >= 0`; the 4-node Hermite interpolant is a lower bound because the error is `g^{(8)}(xi) prod (x-x_i)^2/8! >= 0` |
| Step 1 | CORRECT | identity (num) and positivity re-derived independently |
| Step 2 | GAP (exposition) | the 52 certificates cannot be checked from the paper: the nodes t_i are not given. Machine-checked and not human-checkable: `E tr W^k` for k<=8 (Wick), (num), coefficient positivity, 52 rational evaluations. |
| final equivalence with `a_1+a_2>3/(K+d+1)` | CORRECT | |

---

## 2. Compact statements versus full statements

| compact item | status |
|---|---|
| Thm 1 | implied. "non-anticipating measurable choice" = "measurable function of (xi_1..xi_t)": fine. |
| Thm 2 | implied. |
| 1-d formulas `J=sqrt(2/pi)/sigma`, `T=-sqrt(2/pi) kappa_3/sigma^3`, median `-kappa_3/(6 sigma^2 b)` | correct. |
| Thm 3 | implied. 19 vs 14.3 and 39 vs 29.3: correct (14.26, 29.26). "ratio tends to 3/4": correct only as beta->1; the bullet in the Introduction omits "as beta->1". |
| Thm 4 | NOT implied as written. (a) "the zeros W_b ... exist for large b" needs continuity of the mean field (full statement: "If moreover h_b is continuous"); under Cramer's condition alone the batch law may have atoms and the field may jump over zero. (b) `Sigma(W*)>0` is not assumed in the compact version. (c) `\hat g_b` is undefined in the main text. (d) "linear images L_W chi" needs L_W onto and continuous in W. H>0 instead of invertible is harmless (stronger). The excess-loss formula is proved (paragraph after p_state) but is absent from the full statement in Appendix A; add it there. |
| Sentence after Thm 4, Contributions bullet 4, Abstract | OVERCLAIM. Theorem 4 locates zeros of the mean field. No theorem in the paper shows that the iteration converges to that zero when the noise depends on W (Theorem 1 is for W-independent i.i.d. noise), and Theorem 4 has no momentum. "the location ... of the limit", "run to convergence" and the abstract's "excess loss at the limit is <g*,H^{-1}g*>/(2 b_eff^2)" are not statements proved in the paper. For additive noise the b_eff formula follows from Thms 1+3 and twice-differentiability at W* in two lines, but it is not stated anywhere. |
| "in contrast with the 1/b variance floor of constant-step SGD" | misleading comparison: with the same annealed steps SGD has no floor (as the Introduction says). |
| Thm 5 | implied. |
| Thm 6 | implied. "equality for Gamma laws" refers to form (iii); fine. "computer-assisted only in that integer polynomial identities are checked" understates: Wick enumeration to k=8 and 52 rational certificates are also by machine. |
| "Reading the formula" | `zeta zeta^T zeta = ||zeta||_F^2 zeta/min(K,d)` holds for a multiple of a partial isometry of full rank (not "orthogonal matrix" unless K=d). Positivity claims correct. "directions of skewness that g* reverses" is not shown in Cor 9 (only "ignores", item (iii)); either add the one-line example (take `K_3^{(11),(22),(22)}` beyond `-(a_1+a_2)/a_1`) or drop "reverses". |
| Cor 7 and "about 5/12 max(K,d)" | correct (limit 12/5 for square). |
| Abstract: "the bias of a K x d orthogonalized step is that of sign descent divided by a factor between ..." | holds for independent entries with common variance and third cumulant only; the abstract states it without the hypothesis. |
| Abstract (ii): "this median is g*/b_eff+o(1/b)" | sign: the median is `-g*/b_eff`; the gradient at the limit is `+g*/b_eff`. |
| Cor 8 "factor of order min(K,d), never more" | correct. |
| Discussion: "its direction is given by two moments of the per-sample gradient noise that can be estimated from samples" | true only for Sigma = sigma^2 I. For general Sigma the paper itself uses Monte Carlo on (4). |

---

## 3. Numbers and honesty of Section 6 / Appendix G

All quoted numbers were found in the data files. Table 1: every entry correct. Constants paragraph: correct (34 sizes, 2.357 at 96x96).

Sentences more favourable than the data:

1. Abstract: "Experiments, including a real-data least-squares problem, match the predictions without fitted constants." On digits the law is matched only at b=2048, 8192, 32768, all larger than the data set (n=1797, sampling with replacement); for 16<=b<=512 the measured gradient is 1.1 to 2.2 times the prediction and the excess loss up to 4.8 times. The fact b>n is never stated in the text.
2. "The ratio tends to 1 in all cases" (law across noise laws). The one-factor law is flat at 0.93-0.94 for b=4..1024 (pooled 0.939+-0.019, >3 se below 1) and shows no trend to 1. The appendix admits it; the main text does not. If common random numbers are shared across b (NUMBERS notes chi2=0.02 on 3 dof), the pooled standard errors assume an independence that does not hold; state how pooling was done.
3. Rank-one explanation: "a sum of b samples has rank at most b and the Gaussian approximation of its spectrum sets in late" is given as fact. On 2x3 matrices the rank constraint is void for b>=2; this is a hypothesis and probably not the right one. The wrong sign at b=4 (-0.021) is not reported.
4. Sign-blind: "0.078 to 0.082 for b=16..256" omits b=1024 (0.0757+-0.0029, 1.6 se low). Sign step: all three values for b>=64 are negative (-0.0030, -0.0026, -0.0056, about 2 se pooled); "compatible with 0" is acceptable but borderline.
5. Digits (ii): "for b<=512 the stationary gradient overshoots". At b=4 and 8 it undershoots (0.38, 0.72); overshoot is for 16<=b<=512 (1.13, 1.60, 2.00, 2.18, 1.60).
6. Digits (ii): "With the noise frozen at W* the overshoot disappears (0.73, 1.11, 1.006)". Frozen noise is itself far from first order: 0.29 at b=32 (omitted), 0.73 at b=128, and still 1.11+-0.006 (an 11% overshoot, 19 se) at b=512. Accurate wording: the overshoot is reduced from 2.2 to 0.73 at b=128 and from 1.60 to 1.11 at b=512.
7. Digits (ii): "so the transient is a second-order effect of the dependence of the noise on W, not a failure of the law" and "its scale is consistent with g*^T Sigma^{-1} g* ~ 26". Both are hypotheses presented as conclusions. The frozen/state comparison shows that W-dependence accounts for most of the overshoot; it does not identify a second-order term, and the transient extends to b ~ 500-2000, i.e. 20 to 80 times 26.
8. Digits (i): "excess loss within 15%": true (1.148, 1.113, 1.148) but all three are above 1 and the estimator (loss of the run-averaged iterate, `digits_state.py`) is not noise-debiased; the tail quarters give 1.08-1.40. Say "11-15% above".
9. Digits (i): "the cosine reaches 0.997": at b=2048 only; 0.975 and 0.92 at b=8192, 32768. The appendix attributes the decrease to Monte Carlo error; plausible, not shown.
10. Digits (iii): "ratio 0.97 to 1.37" is for the projected gradient; "the same quantities" includes the excess loss, whose ratio is 0.93 to 1.86.
11. "range of validity ... b_eff >~ 10^3": at b_eff ~ 500-900 the ratio is still 1.28-1.6 and at 1826 (beta=0.9) 1.18; 1.06 at 2048. "b_eff >~ 2 x 10^3" is what the data support.
12. Appendix: "dividing eta_0 by 4 changes the result by less than 3%": beta=0.9, b=16: 3.0% in the norm and 6.6% in the excess loss (5.05e-5 vs 5.41e-5). "two halves agree within 1% (beta=0)": 0.8% in norm, up to 1.6% in loss. "64 runs": b=512 used 32.
13. Appendix: the large-batch runs use a different protocol that is not described: `eta_t = eta_0 (1+t/200)^{-0.6}` with `eta_0 = 0.5 sc/(sqrt b lambda_max)`, start at W*, T=2e4 (from `digits_state.py`); frozen runs use the same schedule without lambda_max, T=2e4-4e4. (The b=128 run under this protocol gives 2.16, consistent with 2.18 under the other: worth saying, it supports the comparison.)
14. "The coupled iteration on a quadratic agrees with the zero of the frozen field ... within 0.5% to 3%": no data for this claim in the files I was given (NUMBERS only cites `results/c04_frozen.json` in passing). Cannot verify.
15. Caption of Fig. 3: "limit of the iteration (1), whose noise depends on W (Thm 4)": Theorem 4 does not cover the iteration or momentum (see Section 2).

Fair points: the text does say the approach is slow and non-monotone, gives the 2.2 and 4.7 factors, says the two momentum curves do not collapse, and the appendix reports the unexplained 3-se deviations.

---

## 4. Citations and novelty

Cannot vouch (dated after my knowledge, or not known to me): `xie2026muon` (arXiv:2609.32861), `chen2026when` (2610.31728), `do2026muon` (2608.04607), `huang2026muon` (2611.18124), `shulgin2026quadratic` (2610.30280), `paquette2026phases` (2605.09552), `schwank` (2508.12926). The paper relies on the first three for "Upper bounds on the resulting error are known", "exhibit non-convergence and bound the error", and for `J_I = c Id`; the authors must recheck each attribution against the cited text.

Believed accurate: Jordan et al. 2024; Bernstein-Newhouse 2024; Carlson-Cevher-Carin 2015; Candes-Sing-Long-Trzasko 2013 (divergence of spectral functions); Li-Sun 2002 (constant to be confirmed, see Lemma osc); Duchi et al. 2012; Cardot-Cenac-Zitt; Minsker 2015; Minsker-Strawn 2023; Cornish-Fisher 1938; Bhattacharya-Ranga Rao (the bbl prints "Rao, R. R."; the author is R. Ranga Rao); Lau-Long-Su 2025; Liu-Yuan-Gu 2025.

Loose attribution: "That sign descent tracks medians rather than means is well known [5 refs]". Chen et al. (2020) says exactly this. Karimireddy et al. (2019) give non-convergence counterexamples and error feedback; Balles et al. (2020) is about the l_inf geometry; neither is about medians. Split the citation.

Novelty claims:
- "The nuclear median appears to be new as an object": cannot be verified; the stationarity condition `E polar(G+xi)=0` is implicit in any analysis of Muon's fixed points (possibly in the three 2026 papers cited). Write "we are not aware of a previous study of".
- "The effect of momentum on higher cumulants seems not to have been isolated before": too strong. `kappa_p(sum w_k X_k) = sum w_k^p kappa_p` is textbook. The new part is its consequence for the bias (b_eff). Rephrase.
- "This paper gives the answer with equality": only to first order in 1/b, under Cramer, for strongly convex F and additive noise. Add "to first order".
- Companion paper: the Scope paragraph leans on unverifiable claims of an anonymous concurrent submission; keep them clearly marked as external and make sure no result here depends on it (none does).

---

## 5. Presentation

- Notation clashes: `o` is both the direction map and Landau's symbol (`E o(G+xi)` next to `o(1/b)`); beta = momentum and = coefficient of B in the proof of Thm 5 and = shift variable in NUM(9+alpha, 9+alpha+beta); alpha = Holder exponent, coefficient of A, shift variable, Gamma shape; theta = step exponent and Gamma scale; gamma = aspect ratio and = cos(phi) in the K=2 proof; r = min(K,d) and r(p); m = K+d+1, m_p, and m_1 = E|N_beta| in the proof of Thm 1; hbar in Lemma gauss and in the proof of Thm 1; H = Hessian and Hermite interpolant; W = iterate and Wishart matrix; varrho = ratio in Cor 8, rows in Lemma osc, correlation in Cor 9; K_3 / K / K_nu; kappa(gamma) / kappa_3; N = dimension, N_t = noise; T_Sigma, T_5.
- Undefined in the main text: `\hat g_b` (Thm 4); `cA`, `cB` are defined only inside Cor 9; `ones`.
- Thm 1: `N_beta=(1-beta) sum_{j>=0} beta^j xi_j` uses xi_0 while the sequence starts at t=1.
- Dangling sentence in p_two (e) (hypergeometric form).
- Anonymity: fine (PDF metadata "Anonymous Authors"; no names or repository links in the sources). The supplementary paths `a1/`, `checks/v01, v02` exist as `experiments/bias_law/a1`, `experiments/bias_law/lattice/v01_wick_moments.py`, `v02_a1_proof.py`; make sure the supplementary archive uses the printed paths and contains no user names.
- Format: 15 pages; main body ends on page 6 (about 5.5 pages of text), references pages 6-7, appendix from page 8. Two overfull boxes of 1.4pt. About two pages are free.

---

## REQUIRED fixes

R1. `paper/bias_law/icml/twomoment.tex`, abstract: "(ii) For a minibatch of size $b$ this median is $g_\ast/b_{\mathrm{eff}}+o(1/b)$" -> "(ii) For a minibatch of size $b$ the limiting gradient is $g_\ast/b_{\mathrm{eff}}+o(1/b)$".

R2. Abstract: "the bias of a $K\times d$ orthogonalized step is that of sign descent divided by a factor between ..." -> insert "for noise with independent, identically skewed entries,".

R3. Abstract: "the excess loss at the limit is $\langle g_\ast,H^{-1}g_\ast\rangle/(2b_{\mathrm{eff}}^2)$" -> add "to leading order" and either (a) add a corollary after Thm 3: "for additive noise and F twice differentiable at W*, `F(W_inf)-F(W*) = <g*,H^{-1}g*>/(2 b_eff^2)+o(b^{-2})`" (proof: Thms 1, 3 and `W_inf=(grad F)^{-1}(G_inf)`), or (b) write `b^2` and "at the zero of the mean field".

R4. Abstract, last sentence -> "Experiments match the predictions without fitted constants on synthetic noise laws; on a real-data least-squares problem the law is reached only at large effective batch size ($b_{\mathrm{eff}}\gtrsim 2\cdot10^3$), after a non-monotone transient."

R5. Theorem 4 (compact), `twomoment.tex`: define `\hat g_b(W)` (mean of b per-sample gradients); add "$\Sigma(W_\ast)\succ0$"; replace "(true for linear images $L_W\chi$ ...)" by "(true for $L_W\chi$ with $L_W$ onto and continuous in $W$ and $\chi$ satisfying Assumption A)"; replace "Then the zeros $W_b$ ... exist for large $b$ and satisfy" by "Then every zero $W_b$ of the mean field near $W_\ast$ satisfies ...; a zero exists for large $b$ if the mean field is continuous." Add the excess-loss line to `\StmtState` in `theory/bias_law/shared/statements.tex`.

R6. `twomoment.tex`: (a) Contributions bullet 4: "the location $W_\ast+H^{-1}g_\ast/b$ of the limit" -> "of the zero of the mean field"; (b) sentence after Thm 4: "Thus the excess loss of an orthogonalized ... method run to convergence scales as $b^{-2}$ ..., in contrast with the $1/b$ variance floor of constant-step SGD" -> "Thus the excess loss at the zero of the mean field scales as $b^{-2}$ with a computable constant. We do not prove that the iteration converges to this zero when the noise depends on $W$; \cref{thm:conv} covers parameter-independent noise." (c) Limitations: add "(6) \cref{thm:conv} assumes i.i.d. noise that does not depend on $W$, and \cref{thm:state} has no momentum; the finite-sum experiment of \cref{sec:exp} is outside both." (d) caption of Fig. 3 in `exp_section.tex`: "(\cref{thm:state})" -> "(compare \cref{thm:state}, which locates the zero of the mean field for $\beta=0$)".

R7. Introduction bullet 2: "the ratio of the two effective batch sizes tends to $\frac34$" -> "... tends to $\frac34$ as $\beta\to1$". Introduction: "gives the answer \emph{with equality}" -> "gives the answer with equality to first order in $1/b$".

R8. `exp_section.tex`, digits paragraph:
 - state that $b=2048, 8192, 32768$ exceed $n=1797$ (sampling with replacement);
 - "for $b\le512$ the stationary gradient overshoots" -> "for $16\le b\le512$ the stationary gradient overshoots (it undershoots at $b=4,8$: $0.38$, $0.72$)";
 - "With the noise frozen at $W_\ast$ the overshoot disappears ($0.73$, $1.11$, $1.006\pm0.012$ ...), so the transient is a second-order effect of the dependence of the noise on $W$, not a failure of the law; its scale is consistent with ..." -> "With the noise frozen at $W_\ast$ the ratio is $0.29$, $0.73$, $1.11$, $1.006\pm0.012$ at $b=32,128,512,2048$: most of the overshoot is therefore due to the dependence of the noise on $W$, while a slower, also non-monotone, approach remains for frozen noise. We conjecture that the transient is governed by a second-order term in the $W$-dependence; we have no theory for it. For reference, $g_\ast^\top\Sigma^{-1}g_\ast\approx26$ is the batch size at which the offset equals one standard deviation of the noise; the transient lasts 20 to 80 times longer.";
 - "(ratio $0.97$ to $1.37$)" -> "(ratio $0.97$ to $1.37$ for the gradient, $0.93$ to $1.86$ for the excess loss)";
 - "is $\beff\gtrsim10^3$" -> "is $\beff\gtrsim2\cdot10^3$";
 - "within $15\%$ of" -> "$11$ to $15\%$ above".

R9. `exp_section.tex`, law paragraph: "tends to $1$ in all cases" -> "approaches $1$ for four of the five laws; for the one-factor law it stays at $0.94\pm0.02$ for all $b\ge4$, a deviation of three standard errors that we cannot explain"; replace the rank-one explanation by "(we do not have an explanation for the slow convergence; at $b=4$ the isotropic rank-one law even has the opposite sign)". Sign-blind paragraph: give the $b=1024$ value ($0.076\pm0.003$).

R10. `exp_appendix.tex`: "dividing $\eta_0$ by $4$ changes the result by less than $3\%$" -> "dividing $\eta_0$ by $4$ changes nothing for $\beta=0$ ($b=128$) and lowers the gradient norm by $3\%$ and the excess loss by $7\%$ for $\beta=0.9$ ($b=16$)"; "agree within $1\%$" -> "within $1\%$ in norm and $2\%$ in loss"; "$64$ runs" -> "$64$ runs ($32$ at $b=512$)"; describe the large-batch and frozen protocol ($\eta_t=\eta_0(1+t/200)^{-0.6}$, $\eta_0=0.5\,\bar\sigma/(\sqrt b\,\lambda_{\max})$, start at $W_\ast$) and note that at $b=128$ it reproduces the other protocol (2.16 vs 2.18); state whether common random numbers are shared across $b$ in Fig. 2(a) and how the pooled errors were formed; either give the data behind the "coupled dynamics" claim or remove the numbers.

R11. `theory/bias_law/proofs/p_a1.tex`: list the Hermite nodes $t_i$ for the 52 cases (table or rule) so that Step 2 can be checked from the paper; in `twomoment.tex` replace "computer-assisted only in that integer polynomial identities are checked by machine" by "computer-assisted: the Wishart moment polynomials (Wick enumeration up to $\E\tr W^8$), one polynomial identity, the positivity of its shifted coefficients, and 52 rational evaluations are checked by machine".

R12. `theory/bias_law/proofs/p_two.tex`: Cor 9(ii) "an independent centred Gamma variable whose skewness cancels" -> "an independent variable, minus a centred Gamma variable when $\varrho>0$, whose third moment cancels"; delete the sentence "The hypergeometric form of $\mu_s$ ... is used;"; replace "which is $C^1$" by "which is locally Lipschitz with derivative of polynomial growth".

R13. `twomoment.tex`, Related work: "The nuclear median appears to be new as an object" -> "We are not aware of a previous study of the nuclear median"; "The effect of momentum on higher cumulants seems not to have been isolated before" -> "Cumulants of linear filters are classical; their consequence for the bias, the skewness-effective batch size, is to our knowledge new". Split the sign-descent/median citation (keep Chen et al. and Safaryan-Richtarik for medians; cite Karimireddy et al., Balles et al. for non-convergence and geometry). Recheck every statement attributed to the 2026 preprints.

R14. `twomoment.tex`, Discussion: "its direction is given by two moments of the per-sample gradient noise that can be estimated from samples" -> add "when the covariance is isotropic; in general by \eqref{eq:gs}". "Reading the formula": "a multiple of an orthogonal matrix" -> "a multiple of a matrix with orthonormal rows or columns"; drop "reverses or" or add the example. Sentence after Thm 1: "the limit can be random" -> "the limit may depend on the realization".

R15. Verify against the sources: Theorem 20.1 of Bhattacharya-Ranga Rao (the five points listed under Lemma edge) and the constant and scope of the Li-Sun bound.

## Optional suggestions

O1. Use the two spare pages, in this order of value:
 1. A proof sketch of Theorem 5 in the main text (10-15 lines): equivariance, two invariants, the two contractions `-m_1` and `m_3-(K+d+1)m_1`, the 2x2 system, Stein. It is the paper's most original argument and is currently invisible.
 2. A practical recipe for g* from samples with general Sigma, without forming K_3: `T_Sigma[K_3] = E_{Z,zeta}[ polar(Z) ( (zeta^T Sigma^{-1} Z)^3 - 3 (zeta^T Sigma^{-1} Z)(zeta^T Sigma^{-1} zeta) ) ]`, `J_Sigma = E[polar(Z) (x) Sigma^{-1}Z]`, Z ~ N(0,Sigma) independent of zeta; cost and variance; this is what Section 6 already does.
 3. An explicit limitations paragraph on the digits transient (law reached only for b>n; conjectured second-order term), with the frozen/state numbers in a small table.
 4. A corollary for Nesterov-style Muon buffers from the general-filter part of Theorem 3 (weights `(1+beta), beta^2, beta^3, ...` normalised), since the title names Muon and the implemented method is not the plain EMA.
 5. The additive-noise excess-loss corollary with b_eff (R3).
O2. Rename the direction map (e.g. `\mathsf o` or `D_\nu`) to avoid the clash with little-o; rename alpha, beta in the proof of Thm 5 and in NUM.
O3. Lemma gauss (4) and Theorem 4: add one line on continuity of `Sigma -> J_Sigma, T_Sigma`.
O4. Lemma osc: constant `2N eps`; write the Euclidean expectation.
O5. Report the excess loss in Fig. 3(b) with a noise-debiased estimator at large b.
O6. Theorem 3: remark that for weights not summing to one the gradient offset carries an extra `1/rho_1`.
O7. State in Table 1's caption how q was computed (Pfaffian formula) and that Monte Carlo agrees.
