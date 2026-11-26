# Literature check for T1–T3 (norm medians of sums, no Cramér condition)

Date of search: 2026-11-25. Only generic queries were sent to external services.

## Reading guide: how each statement was checked

Every statement below carries one of three tags.

- **[P]** read in the primary text (arXiv PDF converted to text and read directly).
- **[S]** read in a secondary source that quotes or restates the result; the source is named.
- **[M]** my recollection of the result, NOT verified in this search. Treat as a pointer, not a citation.

### What I could not access

- Bhattacharya–Rao, *Normal Approximation and Asymptotic Expansions* (Wiley 1976 / Krieger 1986 / SIAM 2010): no readable copy. The Internet Archive scan (`normalapproximat0000bhat`) is access-restricted; Google Books gave no snippets. Theorems 20.6 and 20.8 were recovered from papers that quote them; 20.1, 22.1, 23.1 are [M] only.
- Götze–Hipp (1978), Barbour (1986), Esseen (1968), Breuillard (2005): Springer redirects to a login. HAL copy of Breuillard returned HTML only.
- Esseen (1945), Yarnold (1972), Hipp (1977), Sweeting (1977), Bhattacharya (1977 survey), Kolassa–McCullagh (1990): Project Euclid returned empty pages to automated fetches.
- Littman (1963): AMS returned 403. Only the first-page text surfaced through the search engine.
- Stein, *Harmonic Analysis*; Katznelson (1993); Kaas–Buhrman (1980); Hamza (1995); Göb (1994); Matthes (1975): abstracts or citations only.
- Not searched or not found: Dolgopyat–Hebbar; a second "Song" paper beyond arXiv:1806.01431.

---

## Verdicts at a glance

| Claim | Verdict | Closest prior work |
|---|---|---|
| T1 Fourier lemma (smooth F, L1 derivative norms, no Cramér) | Partially overlapping | Breuillard 2005 Thm 3.2; Fernando–Liverani 2021; Götze–Hipp 1978; Barbour 1986 |
| T1 full statement (norm median, singular set of codimension >= 2, all laws incl. lattice) | Not found | — |
| T2 principle (curvature of the boundary rescues lattice laws) | Partially overlapping (known for ellipsoids) | Esseen 1945; Bentkus–Götze; Götze–Zaitsev |
| T2 for {det = 0} / polar factor of square matrices | Not found | Katznelson 1993 is consistent with it |
| T3 failure for lattice coordinates, sawtooth | Essentially known in substance; explicit median formula not found | Esseen 1945 Thm 3 p. 56; Buhler–Gamst–Graham–Hales 2017 |
| T3 validity for non-lattice coordinates | Already known | Esseen 1945 |
| Leading coefficient g* (skewness-type bias of a geometric median) | Partially overlapping, different asymptotic regime | Schwank–Drton 2025 |
| Muon bias from skewed bounded noise | Partially overlapping (scalar case only) | Do–Dereich–Jentzen 2026 |

---

## 1. Bhattacharya & Rao

**Theorem 20.8 (strongly nonlattice, third moment)** — [S], quoted as "Proposition 1 ([14, Theorem 20.8])" in J. Honda, *Exact Asymptotics for the Random Coding Error Probability*, arXiv:1506.03355 (stated there in R^2):

> Let V_1, V_2, ... be i.i.d. strongly nonlattice random variables with mean zero and covariance matrix Σ. Then there exists a three-degree polynomial h(z) such that for any function f(z)
> | ∫ f(z) (1 − h(z)/√n) φ_Σ(z) dz − E[f(V̄)] | <= ω_f(R^2) δ_n + ω̄_f(δ_n; Φ_Σ),
> where δ_n satisfies lim √n δ_n = 0 and does not depend on f.

So your recollection is right: one-term expansion, error o(n^{-1/2}) uniformly over classes with ω̄_f(ε : Φ) = O(ε). The quoted form omits the moment hypothesis; in the book it is finite third moment [M]. "Strongly nonlattice" means |φ(t)| < 1 for all t != 0 [S: a 1996 PTRF paper surfaced by search; M].

**Theorem 20.6 (independent, non-identically distributed, Cramér)** — [S], quoted verbatim as Theorem 4.1 in Angst–Poly (arXiv:1511.08750), "as it is stated in Theorem 20.6 of [BR86]":

> independent zero-mean vectors in R^k; smallest eigenvalue of V_n bounded away from zero; an integer s > 2 with ρ_s(n) bounded and a Lindeberg-type condition on s-th moments; uniform Cramér condition. Then for every real Borel f with M_{s'}(f) < ∞ for some 0 <= s' <= s,
> | ∫ f dQ_n − ∫ f dQ̃_n | <= M_{s'}(f) δ_1(n) + c(s,k) ω̄_f(2 e^{-dn} : Φ), with δ_1(n) = o(n^{-(s-2)/2}).

**Theorem 20.1 (i.i.d., Cramér)** — [M] plus [S]: Song (arXiv:1806.01431) states his proof "follows the arguments in the proof of Theorem 20.1 of BR" and that 20.1 is the i.i.d. version. From memory the statement is the same as 20.6 with i.i.d. summands, finite s-th moment, Cramér's condition limsup |φ| < 1, and the same right-hand side M_{s'}(f) δ_1(n) + c ω̄_f(2e^{-dn} : Φ). Not verified verbatim. Corollary 20.5 (convex sets) is confirmed as existing by Song [S].

**Theorem 22.1 (local expansion for lattice vectors)** — [S, weak]: a 2026 arXiv paper (arXiv:2611.09592) cites it as the local Edgeworth expansion for lattice sums with minimal lattice Z^k, error o(n^{-(k+s-2)/2}) on point probabilities. Not read in the book.

**Theorem 23.1 (lattice distribution functions)** — not verified. [M]: expansion of the distribution function of a lattice sum with the periodic S_j correction terms, error o(n^{-(s-2)/2}).

Verdict: these are the tools of the prior note. None covers T1 or T2: 20.1/20.6 need Cramér, 20.8 needs strongly nonlattice, and neither exploits the codimension of the singular set of the test function.

## 2. Smooth test functions without Cramér

**Götze & Hipp (1978)**, Z. Wahrsch. Verw. Gebiete 42, 67–87, DOI 10.1007/BF00534208 — full text not accessed.
- [S] arXiv:2511.01603: "Götze and Hipp (1978) derived an asymptotic expansion of the expectation of a smooth function of the sample mean under the moment condition", and (Remark 6) their function class is smooth f, excluding indicators.
- [S] Barbour (1986), via search snippet: for expectations of smooth functions "there is no need to impose smoothness conditions on the distributions of the summands"; Götze–Hipp obtained such expansions "by Fourier methods, under conditions which, although similar to those used here, are not equivalent".
- [M] The theorem is for f with s−2 derivatives of polynomial growth, s finite moments, error o(n^{-(s-2)/2}), no arithmetic condition (lattice laws allowed). I believe the derivative control is pointwise (sup/growth type), not integrated. I could not confirm the number of derivatives or the norm. I found no version for functions singular on lower-dimensional sets.

**Hipp (1977)**, Ann. Probab. 5, 1004–1011, "Edgeworth expansions for integrals of smooth functions"; **Sweeting (1977)**, Ann. Probab. 5, 28–41, and **Sweeting (1980)**, Ann. Probab. 8, 281–297 (abstract: "asymptotic expansions for smooth functions are derived"); **Barbour (1986)**, PTRF 72, 289–303 (Stein's method, Lyapunov-type error) — citations and abstracts only.

**Breuillard (2005)**, PTRF 132, 39–73 — [S], Theorem 3.2 as restated in M. Weber's survey arXiv:2208.02700, Theorem 1.19 (stated on R):

> Let r be a non-negative integer and l >= 0. Assume μ is centered with finite moment of order r+2. If μ is l-Diophantine, then for any f of class C^k with k > l(r+1)/2 + 1 and C^k(f) := max_{0<=j<=k} ||f^{(j)}||_1 < ∞,
> E f(S_n) = Σ_{p=0}^{r} n^{-p/2} ∫ f(x√n) P_p(x) g(x) dx + C^k(f) · o(n^{-(r+1)/2}).

l-Diophantine is equivalent to |μ̂(x)| <= 1 − C/|x|^l for large |x|. Note the L1 norms of derivatives. The Springer abstract says the paper characterises "in diophantine terms" the measures for which the full local expansion holds for smooth enough f, and its introduction says counterexamples exist for every order. The title is on R^d; I verified only the one-dimensional statement.

**Fernando & Liverani (2021)**, AIHP 57, 469–505, arXiv:1803.07667 — [P]. One-dimensional setting. Norms C^m(f) = max_{0<=j<=m} ||f^{(j)}||_{L1}, C_k(f) = max_{j<=k} ||x^j f||_{L1}, space F^m_k. "Weak global expansion of order r": E f(S_N − NA) = Σ_{p<=r} N^{-p/2} ∫ P_{p,g}(z) n(z) f(z√N) dz + ||f|| o(N^{-(r+1)/2}). Theorem 2.5: under (A1)–(A4) with s = r+2 and q > (r+1)/(2 r_1), holds for f in F^{q+2}_0. In the i.i.d. case (Sec. 6.1) with an l-Diophantine law this needs q + 2 > 2 + (r+1)l/2 derivatives, "similar to" Breuillard but one derivative worse. l-Diophantine implies non-lattice.

Verdict for T1's Fourier lemma: **partially overlapping**. Bounding the Edgeworth error for smooth F by L1 norms of derivatives through Fourier inversion, with no Cramér condition, is an established technique (Breuillard, Fernando–Liverani; Götze–Hipp for the no-arithmetic-condition case). What I did not find:
- a statement in R^N valid for every law, lattice included, in the explicit form C ε^3 ||F||_1 + C ε^m ||D^m F||_1 with m > N;
- any use of such a bound with a cutoff near a singular set of codimension >= 2.

## 3. Weak Cramér / Diophantine conditions for discrete laws

**Angst & Poly (2017)**, EJP 22, paper 59, arXiv:1511.08750 — [P].
- Definition 2.1: X in R^d satisfies the weak Cramér condition with exponent b > 0 if there are C, R > 0 with |φ_X(t)| <= 1 − C/||t||^b for ||t|| > R.
- Proposition 2.1 (genericity): for p >= 3 atoms at U in R^p with arbitrary positive weights, for all b > 1/(p−2) the set of U whose laws fail C(1, 2b) has Lebesgue measure zero. Proposition 2.2: algebraic, rationally independent atoms also work.
- Theorem 4.2: independent vectors in R^k, same moment hypotheses as BR 20.6, mean weak Cramér with 3/2 + 1/b − s/2 > 0. For every Borel f with M_{s'}(f) < ∞: error <= M_{s'}(f) δ_1(n) + c(s,k) ω̄_f(2 n^{-(s-2)/2} : Φ), δ_1(n) = o(n^{-(s-2)/2}). Hence O(n^{-(s-2)/2}) when ω̄_f(2ε : Φ) <= Cε. Remark 4.2: only big-O, not little-o.

**Song (2018)**, arXiv:1806.01431 — [P via text]: uniform-in-P version; any Borel f; bound C n^{-(s-2)/2}(1 + M_s(f)) + C ω̄_f(n^{-(s-2)/2}; Φ) under mean weak Cramér with 0 < b < 2/max{s−3, 1}.

**Dolgopyat & Fernando (2023)**, IMRN 2023(21), arXiv:2303.10235 — [P]. Real-valued X with d+1 atoms, d >= 2.
- "It is known that S_n admits the first order Edgeworth expansion if and only if X is non-lattice (see [12])" where [12] is Esseen 1945.
- S_n never admits the order-d expansion (jumps of order n^{-d/2}).
- β-Diophantine: d(s) = max_j dist(b_j s, 2πZ) >= K/|s|^β. Theorem 1: if 2(R − 1/2)β < 1 then n^R [P(S_n/(σ√n) <= z) − E_{d−1}(z)] → 0 uniformly in z. Theorems 2–4 describe the oscillating order-n^{-d/2} error for typical atoms.
- Their results concern distribution functions, not smooth test functions. On smooth test functions they only point elsewhere: "[2, 5] discusses a weak Edgeworth expansion where the LHS of (1.1) is convolved with smooth compactly supported functions" ([2] Barbour 1986, [5] Breuillard 2005) and Remark 2.2: "[5] shows, among other things, that a Diophantine condition with any exponent is sufficient for obtaining a weak Edgeworth expansion for sufficiently smooth functions."

Answer to the specific question: I found no Dolgopyat–Fernando theorem of the type "discrete law, r derivatives => order-r expansion in R^d". The result of that type is Breuillard's Theorem 3.2 (and Fernando–Liverani), with L1 norms of derivatives, verified by me in dimension one only, and requiring a Diophantine condition.

Verdict: **partially overlapping** with T1. All of these impose an arithmetic condition on the law (weak Cramér / Diophantine) and so exclude lattice laws; T1 instead imposes a geometric condition on the test function and allows every law. I found no result of the T1 type.

## 4. Lattice laws and non-smooth sets

**Esseen (1945)**, Acta Math. 77, 1–125.
- [S] Bobkov–Ulyanov (TPA, copy at par.nsf.gov/servlets/purl/10339631): non-lattice X_1 gives P{S_n <= x} = Φ_3((x−μ)/σ) + o(1/√n) uniformly in x. For integer-valued X_1 with span 1, "Theorem 3 in [12, p. 56] can be equivalently stated as" P{S_n <= x} = Φ_3((x−μ)/σ) + (1/σ) ψ(x) φ((x−μ)/σ) + o(1/√n), a discontinuous periodic correction of order 1/√n. (The extraction garbled the formula for ψ; it is the centred sawtooth, [x] − x + 1/2 in the usual statement [M].)
- [S] Götze–Zaitsev (Bielefeld SFB 701 preprint 08119, "Uniform rates of convergence in the CLT for quadratic forms"): the bound "O(N^{-d/(d+1)}) by Esseen (1945) for the CLT for ellipsoids with axes parallel to coordinate axes", the probabilistic counterpart of Landau's lattice-point estimate O(r^{-2d/(d+1)}).

**Götze–Zaitsev** (same preprint) — [P abstract]: i.i.d. vectors in R^d not concentrated on a proper subspace, fourth moment, d >= 5: sup_x |P(Q[S_N] <= x) − P(Q[G] <= x)| = O(N^{-1}) for non-degenerate quadratic forms. No arithmetic condition, so lattice laws are included; they note the O(N^{-1}) is optimal because of lattice jumps. Earlier: Bentkus–Götze (1997) for d >= 9.

**Yarnold (1972)**, Ann. Math. Statist. 43, 1566–1580 — abstract fragments only [S]: starting from Ranga Rao's lattice expansion, a simpler expansion for convex B, with a term counting lattice points in the set and remainder O(n^{-1}); "the Edgeworth approximation assuming a continuous distribution should never be used when Z_i has a lattice distribution".

**Kolassa–McCullagh (1990)**, Ann. Statist. 18, 981–985 — abstract [S]: Sheppard-corrected Edgeworth series for lattice sums is accurate to the same order as in the continuous case.

Verdict for T2: **partially overlapping at the level of principle**. That curvature of the boundary restores accuracy better than n^{-1/2} for every law, lattice included, is classical for ellipsoids (Esseen 1945; Götze and coauthors). I found no general statement for functions singular on a curved hypersurface, none for sets of codimension >= 2, and nothing for {det = 0} or the polar factor.

## 5. Medians of lattice sums; geometric median of a sample mean

**Buhler, Gamst, Graham, Hales (2017)**, "Explicit error bounds for lattice Edgeworth expansions", arXiv:1710.08845 — [P]. X bounded, lattice with span b and shift a; tilt T_n = P(X[n] > mean) − P(X[n] < mean). Theorem 1:

> T_n = L(na)/√(2πn) + E, L(c) = ((−c) mod b − c mod b)/σ − μ_3/(3σ^3), E = o(1/√n).

"the sign of T_n for large n depends on the third moment μ_3 ... but also, perhaps more surprisingly, on the congruence class of n modulo the so-called 'span' of X." This is T3's sawtooth for the sign of (median − mean), with explicit error bounds.

**Binomial / Poisson** — abstracts only [S]: Kaas–Buhrman (1980), Statist. Neerlandica 34, 13–18 (mean, median, mode coincide when np is integer; |mean − median| bounds); Hamza (1995), Statist. Probab. Lett. 23, 21–25 (|mean − median| < ln 2, sharp, binomial and Poisson); Choi (1994), Proc. AMS 121 (μ − ln 2 <= median(Poisson μ) < μ + 1/3; gamma median n + 2/3 + ...). The Poisson window (−ln 2, 1/3) is the range swept by "nearest lattice point to mean − κ_3/(6σ^2) = μ − 1/6". Göb (1994) not accessed.

I found no source stating in general "median of a lattice i.i.d. sum = shifted-lattice point nearest to the Cornish–Fisher value". It is an immediate inversion of Esseen's Theorem 3.

**Minsker & Strawn (2024)**, SIMODS 6, arXiv:2307.03111 — [fetched HTML, summarised]: Theorem 8 bounds ||mean − geometric median|| for laws absolutely continuous on a subspace by min(√tr Σ, √||Σ|| × ratio of negative moments). For block means the bias bound is of order √(||Σ|| k/N), i.e. O(n^{-1/2}) in the block size. No second-order expansion, no discrete laws.

**Schwank & Drton (2025/26)**, "On the distance between mean and geometric median in high dimensions", arXiv:2508.12926 — [P]. Dimension p → ∞, weakly dependent coordinates, q > 3 moments. Theorem 3.5: m_{p,i} = μ_i − α_i/p + o(1/p), α_i = (1/(2σ̄^2)) Σ_{|j−i|<=M} E[(X_i − μ_i)(X_j − μ_j)^2]. They remark that for a mean of K i.i.d. copies "α_i = α_i^{(1)}/K ... the first order bias of the population geometric median of means scales exactly as 1/K".

Verdict: T3 is **essentially known in substance** (Esseen 1945; BGGH 2017). The explicit closed form for the median, and its use as a counterexample to a norm-median law, was **not found**. The Euclidean case of the leading term g*/b is **partially overlapping** with Schwank–Drton, but their limit is p → ∞, not fixed N with b → ∞, with no rate, no lattice discussion and only the Euclidean norm.

## 6. Concentration function

- [S] Eliseeva, arXiv:1303.4005, citing Rudelson–Vershynin Lemma 3.4: for Y in R^d with law F, Q(F, √d) <<_d ∫_{B(√d)} |F̂(t)| dt (the multidimensional Esseen inequality). She cites Esseen 1966/1968 for the one-dimensional inequalities and unnamed references for multidimensional analogues.
- Esseen (1968), Z. Wahrsch. 9, 290–308: [S] only that the last section treats random vectors. [M]: for i.i.d. nondegenerate vectors the consequence Q(S_n; r) <= C n^{-N/2} for fixed r is standard.
- I did not find the inequality sup_{s0} ∫_{|s−s0|<=c} |φ(s)|^n ds <= C n^{-N/2} stated explicitly. It is the dual form of the Esseen bound and is very likely folklore; the positive-definiteness/Fejér proof is the standard mechanism behind Esseen's inequality.

Verdict: **not found as an explicit statement**; expect a referee to regard it as classical.

## 7. Fourier decay of surface measures; the determinantal hypersurface

**Littman (1963)**, Bull. AMS 69, 766–770 — [S: first-page text via search snippet, plus restatements in Cuenin–Schippa and Cowling–Mauceri]: S an n-surface (possibly with boundary) in R^{n+1}, μ a smooth mass density on S vanishing near the boundary; if at each point k of the n principal curvatures are nonzero, then ∫_S e^{iX·Y} dμ = O(|Y|^{-k/2}). The constant depends on lower bounds for the nonvanishing curvatures and on the density.

**Stein, Harmonic Analysis, Ch. VIII §3** — [M], not accessed: Theorem 1 gives |dμ̂(ξ)| <= A |ξ|^{-(n-1)/2} for nonvanishing Gaussian curvature; the k-curvature version O(|ξ|^{-k/2}) is attributed to Littman.

**Katznelson (1993)**, Duke Math. J. 69, 121–136 — [S] arXiv:2503.13768: Duke–Rudnick–Sarnak (d != 0) and Katznelson (d = 0) obtained asymptotics for #{A in M_n(Z; H): det A = d} with main terms of order H^{n^2−n} and H^{n^2−n} log H respectively. Exact constant not seen.

Remark (my own, not from a source): combined with a local limit theorem, Katznelson's count suggests that an integer-matrix-valued walk puts mass of order b^{-K/2} log b on {det = 0} in the CLT window, which is o(b^{-1/2}) exactly when K >= 2. This is consistent with T2 and with its failure at K = 1 (T3).

Search hit, abstract only: Zhu, "Fourier dimension of constant rank hypersurfaces", arXiv:2410.09711 (Fourier dimension equals the number of nonvanishing principal curvatures for constant-rank hypersurfaces). I found nothing on the Fourier transform or curvature of {det = 0} specifically.

Verdict: tools are classical; their application to {det = 0} in a CLT expansion was **not found**.

## 8. Muon / spectral / sign methods (2024–2026)

**Do, Dereich, Jentzen (Aug 2026)**, arXiv:2608.04607 — [P]. The closest hit.
- Theorem 1.2: Muon with generalised Newton–Schulz steps and momentum, bounded i.i.d. data, strongly convex Lipschitz-gradient loss: E||Θ_n − ϑ||^2 <= C (M^{-1} + γ_n), M the mini-batch size.
- Theorem 1.3: scalar problem L(θ, x) = |θ − x|^2, bounded i.i.d. X with E[(X − EX)^3] != 0: "for almost all M in N", limsup_n E[min{1, |Θ^M_n − E X|}] > 0.

So a batch-size-dependent bias driven by the third moment is established, with an O(1/M) scale in mean square. It is scalar, uses a polynomial approximation of sign, and contains no Cornish–Fisher constant, no matrix case, no lattice analysis. I did not determine what "almost all M" excludes; worth reading against T3.

Other hits, abstracts only [S]:
- Beneventano–Abdelmoneum–Poggio, arXiv:2606.08388: spectral-flattening bias of the polar update; not a noise-median analysis.
- Sato–Naganuma, arXiv:2507.01598 (critical batch size); Pethick et al., arXiv:2502.07529 (norm-constrained LMO, guarantees need growing batch); arXiv:2509.15816 (Muon O(T^{-1/4}), MVR variant); arXiv:2506.04192 (stochastic Frank–Wolfe view); arXiv:2605.06884 (heavy tails). All work with unbiased-oracle variance bounds.
- Sign methods: arXiv:1906.01736 (signSGD tracks the median of the gradient noise; skewness creates bias); arXiv:2402.12828 (median of gradients); arXiv:2411.12135 (exact risk of signSGD); arXiv:2510.00236 (per-example gradients).

Verdict: **partially overlapping** only through Do–Dereich–Jentzen in the scalar case. Nothing found on medians for the nuclear norm, on Edgeworth/Cornish–Fisher bias laws, or on lattice effects for finite-sum noise.

## 9. Wishart moments at non-integer exponents

**Cunden, Mezzadri, O'Connell, Simm (2019)**, CMP, arXiv:1805.08760 — [P].
- LUE (Haagerup–Thorbjørnsen), eq. (4.15), Q^C_k(m,n) = E Tr X_n^k, α = m − n:
  (k+2) Q^C_{k+1} = (2k+1)(2n+α) Q^C_k + (k−1)(k^2 − α^2) Q^C_{k−1}.
  Theorem 4.4 gives the hypergeometric representation for complex k with Re k > −α − 1 (via Carlson's theorem).
- LOE: no three-term recurrence for the moments themselves. Theorem 8.3: a combination S^R_k(m,n) of real moments (expressed through LUE moments at (m−1, n−1), continuous dual Hahn / Hahn polynomials in x with k = ix − 1/2) satisfies the three-term recurrence (8.23):
  (k+3) S^R_{k+1} = (2k+1)(2n+α−1) S^R_k + (k−2)(k^2 − α^2) S^R_{k−1},
  invariant under k → −1−k with zeros on Re k = −1/2.
- Corollary 8.4, eq. (8.25), five-term recursion for Q^R_k(m,n) = E Tr X_n^k, "To our knowledge these recursion formulae are new":
  (k+1) Q^R_k = A Q^R_{k−1} + B Q^R_{k−2} + C Q^R_{k−3} + D Q^R_{k−4},
  A = (4k−1)(m+n−1),
  B = −(4k−6)(m+n−1)^2 + (k−4)((k−2)^2 − α^2) + (k+1)((2k−3)^2 − α^2),
  C = −(m+n−1)[(2k−3)((2k−5)^2 − α^2) − (2k−8)((k−2)^2 − α^2)],
  D = −(k−4)((k−2)^2 − α^2)((2k−7)^2 − α^2).
  (Bracketing of C reconstructed from a text extraction; check against the PDF before use.)
- Complex exponents for LOE: the moments are defined "for all k in C for which the expectations exist", but I did not find an explicit statement that (8.25) holds for complex k. Normalisation of the Wishart matrix must also be checked against the paper.
- Duality (8.2): Q^H_k(m,n) = (−1)^{k+1} 2^{-1} Q^R_k(−2m, −2n).

Log-concavity/convexity of s -> E tr W^{s+1} / E tr W^s: **not found** in this paper or by search. (Log-convexity of s -> E tr W^s itself is immediate from Hölder; the ratio is therefore nondecreasing in s. That is my remark, not a citation.)

---

## Bottom line

1. **T1** — no prior statement found. Its Fourier lemma belongs to a known family (Götze–Hipp; Breuillard; Fernando–Liverani), and should be presented as a variant of those, not as new in kind. The novelty is the trade: no arithmetic condition on the law, paid for by codimension >= 2 of the singular set.
2. **T2** — no prior statement found for {det = 0}. The mechanism is the classical one behind Esseen's ellipsoid theorem and the Götze school's quadratic-form results, which should be cited.
3. **T3** — the lattice failure and sawtooth are Esseen (1945, Thm 3 p. 56) in substance, and Buhler–Gamst–Graham–Hales (2017) prove the n-mod-span dependence of sign(median − mean). Cite both; claim only the explicit norm-median formulation.
4. Cite **Schwank–Drton** for the skewness-type first-order bias of the geometric median and **Do–Dereich–Jentzen** for skewness-driven non-convergence of Muon at fixed batch size.
5. Before submission, check in a library: BR Theorems 20.1, 22.1, 23.1; the exact Götze–Hipp theorem (number and type of derivative norms); Breuillard's R^d statement; Esseen 1968 last section.
