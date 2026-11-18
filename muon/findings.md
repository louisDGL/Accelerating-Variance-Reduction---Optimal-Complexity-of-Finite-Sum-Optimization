# Variance reduction for Muon on finite sums: how far it goes, where it breaks, new configurations

Side study of the project "Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization"
(Statistical Learning Theory, UC Berkeley, November 2026). Code and conventions: [`README.md`](README.md). Proofs of
the structural facts F1-F11: [`notes_theory.md`](notes_theory.md). All numbers: [`results/digest.md`](results/digest.md)
(generated from the JSON summaries; the tables below are copied from it).

## 1. Question

Muon replaces the (momentum of the) gradient of a matrix parameter by its polar factor, the steepest-descent direction
for the spectral norm. Almost all the variance-reduced versions published so far are momentum-based (STORM / MVR:
MARS-M, arXiv:2510.21800; Chang et al., arXiv:2509.15816; Qian et al., arXiv:2512.16598; LiMuon, arXiv:2509.14562;
Kovalev and Borodich, arXiv:2511.11466):
they improve a nonconvex rate from T^(-1/4) to T^(-1/3) and, at constant parameters, keep a noise floor on a finite
sum. The few exceptions are listed in Section 3. The main project studies the other family of variance reduction, the
finite-sum estimators (SAGA, L-SVRG, L-Katyusha), whose error vanishes at the minimizer and which converge linearly
at a constant step.

**Question.** To what extent can finite-sum variance reduction be applied to Muon-type steps? Which combinations
converge linearly, which break and why, and are there configurations, not explored in the literature, that do better
than the naive combination "polar factor of the SAGA estimator"?

The study is on strongly convex matrix problems (l2-regularized softmax regression and multi-output least squares)
where the minimizer, the constants and the conditional expectations are computed exactly, as in the main project.

<!-- SUMMARY -->

## 2. Setting, tools and protocol

**Problems.** F(W) = (1/n) sum_i l(W x_i; y_i) + (lam/2) ||W||_F^2, W in R^{K x d}; features as in the main project
(covariance spectrum j^(-alpha), max_i ||x_i|| = 1); softmax labels drawn from a planted model, least-squares targets
from a planted matrix plus noise. kappa = R^2 / lam is set by the generator, but what decides the difficulty is the
true conditioning at the optimum: we report R^2 / mu_true (mu_true the smallest eigenvalue of the Hessian at W_*,
in the gauge subspace for softmax). Six instances separate three situations:

| key | instance | R^2 / mu_true | regime |
|---|---|---|---|
| soft-mem | softmax, n = 1000, d = 50, K = 10, alpha = 1, kappa = 1e3 | 0.9 n | memory-limited |
| ls-mem | least squares, same sizes | 0.5 n | memory-limited |
| soft-cond | softmax, n = 200, d = 40, K = 8, alpha = 2, kappa = 1e4 | 48 n | conditioning-limited |
| ls-cond | least squares, same sizes | 36 n | conditioning-limited |
| ls-orth | least squares, n = 500, K = d = 20, alpha = 2, kappa = 1e5, planted matrix with flat singular values, noise 0.1 | 8.7 n | conditioning-limited, favourable to Muon |
| soft-sq | softmax, n = 500, K = d = 20, alpha = 2, kappa = 1e4 | 18 n | conditioning-limited, square |

In `ls-orth` the polar factor of the full gradient at W_0 = 0 is almost exactly the direction of W_0 - W_*
(cosine 0.999): one full-gradient polar step with exact line search removes 99.7% of the squared distance, against
11% for a gradient step with exact line search. It is the situation where Muon-type steps do not see the conditioning
of the features (`planted="orthogonal"` in `lab/problems.py`). Each instance is one draw of the data (seed 0);
Section 5.9 repeats the main comparison on other draws.

**Laboratory.** `lab/vrmuon.py` writes every variance-reduced estimator as omega = m + theta c, a predictable part m
(memory mean plus the exact regularizer gradient) and an innovation c (the sampled correction, rank one at batch size
one), and factors a method into estimator x placement of the nonlinear map x step length x orthogonalization:

| placement | step S (W <- W - S) | conditional expectation |
|---|---|---|
| joint | T(m + theta c) | no closed form (computed by enumeration) |
| split | T(m) + gamma_c theta c | T(m) + gamma_c theta (grad F - m) |
| tangent | T(m) + DT(m)[theta c] | T(grad F) up to second order in grad F - m (theta = 1) |
| radial | gamma_c omega + gamma <omega, O_m> O_m, O_m = polar(m) | gamma_c grad F + gamma <grad F, O_m> O_m |
| metric | gamma_P omega C, C a frozen "polar metric" (F10) | gamma_P grad F C |
| euclid | gamma (m + theta c) | gamma grad F (theta = 1) |

with T(M) = eta(M) orth(M); lengths eta: nuclear-scaled gamma ||M||_* ("sharp"), fixed, clipped
min(rho, gamma ||M||_*); orth: exact polar factor, Newton-Schulz, or the bounded-gain map s -> s / sqrt(s^2 + eps^2).
The metric placement takes C from the polar decomposition of a gradient, either refreshed (every pass for SAGA, at
each reference point for L-SVRG) or computed once at W_0 and never refreshed ("one-shot"); as a control it can take
C from the feature covariance instead, C proportional to (c_l Sigma_hat + lam I)^(-1) (`cov-metric`: the
statistically preconditioned SAGA of the note B1 of the main project).
`lab/exact.py` enumerates the n possible next steps of a single-index method at a given state: every one-step
expectation is exact (no Monte-Carlo error), including for steps that are nonlinear in the estimator.

**Protocol** (`experiments/common.py`). Budgets and results are in *passes* (per-sample gradient evaluations / n; a
full gradient is one pass). The metric is the number of passes needed to bring ||W - W_*||_F^2 / ||W_0 - W_*||_F^2
below a tolerance (a first hitting time; the tables also give the number of seeds that go on to a level 1e-4 times
lower, to separate convergence from a touch followed by a stall). Every family has a geometric grid of step constants
in problem units. Tuning: (0) screening of every grid point at 20% of the budget on one seed, the best third goes
on; (1) the kept points are run on two tuning seeds, and the grid neighbours of the best point that the screening
had dropped are run too; (2) the three best points are run on three more tuning seeds and the selected point
minimizes the geometric mean over the five. Report: the selected point is run on ten fresh confirmation seeds, which
give the median and the range. Deterministic methods use one run per grid point, on fine grids (their optimum sits
just below their stability limit). Selections on the edge of a grid are flagged in the digest.

Three conventions shape the comparison and are stated once. (a) SVRG-type methods (L-SVRG, PAGE, L-Katyusha and their
orthogonalized versions) are charged 2 gradients per iteration plus n per refresh, the convention of the main project;
storing the residuals at the reference point, which costs the memory of SAGA, would make it 1 per iteration, i.e.
about 2/3 of the passes reported for these rows. (b) Orthogonalized methods cost one factorization per iteration in
addition; this cost is not in the budget and is discussed in Section 5.10. (c) With 5 tuning seeds, 10 confirmation
seeds and grids of factor 2 (finer for the damping), the bootstrap standard error of a median over the confirmation
seeds is 1 to 4% of the median for the typical family and 6 to 9% for one family in ten (digest, "M1 resolution"):
differences below about 10% between two families are not resolved, and the text only interprets larger ones.

The protocol, the code and the theory notes were revised after an adversarial review by independent agents (code,
protocol, theory, citations); the points they raised that changed a conclusion are marked "(review)" below.

## 3. Where the literature stands

Two literature checks were run on 2026-11-12 with automated search agents. The first, before the experiments,
searched each candidate configuration with 15 to 30 queries. The second, after the adversarial review of the study,
used 179 web searches: five agents searched the configurations by theme, four others tried to refute their verdicts
from the vocabularies of neighbouring fields (Frank-Wolfe, preconditioning, sign and compression methods, the
2025-2026 literature on Muon), and a last one opened again the documents they cited (162 checks: none points to a
wrong paper, one author list was wrong and is not used here). The full record, with what was read of each paper and
the queries, is [`literature_check.md`](literature_check.md). Theorem and equation numbers quoted in this report
come from that reading and should be re-read before being quoted elsewhere. "Not found" always means "not found by
these checks", not "provably unpublished"; several of the closest works are a few weeks old.

What the checks establish:

* **Finite-sum variance reduction inside a spectral or LMO step is not absent from the literature.** Li, Tang and Zou
  (arXiv:2602.11557) run a minibatch SVRG estimator inside a fixed-length polar step with a decaying step size on
  separable data. Katyusha (Allen-Zhu, JMLR 2018, Section 6) feeds an SVRG estimator, with importance sampling of the
  index, to the steepest-descent step of an arbitrary norm inside its accelerated scheme, with a linear rate; the
  template is instantiated and tested for the l1 norm (ASGCD, Song et al., arXiv:1702.07842), not found for the
  spectral norm. Control variates inside a spectral map exist in federated learning: FedMuon of Takezawa et al.
  (arXiv:2509.26337), FedCALM (arXiv:2610.23434) and NBCLT (arXiv:2610.17913: a damped control-variate estimator
  with a relaxed table, clipped on a spectral ball). Qian et al. (arXiv:2604.10689) feed a PAGE-type estimator, then
  momentum, to the Gluon / Muon LMO.
* **In the Frank-Wolfe literature, SVRG, SAGA, SAG and SARAH estimators inside a linear minimization oracle are
  classical**: SVRF and STORC (Hazan and Luo, arXiv:1602.02101), SVFW and SagaFw (Reddi et al., arXiv:1607.08254),
  SAG-type memories (Negiar et al., arXiv:2002.11860; Lu and Freund, arXiv:1807.07680), SPIDER-FW (Yurtsever et al.,
  ICML 2019), Sarah Frank-Wolfe (Beznosikov et al., arXiv:2304.11737), the L-SVRG, SAGA and DIANA variants of
  Nazykov et al. (arXiv:2406.06788), AdaSVRF (arXiv:2009.14114). Since Muon with weight decay is a stochastic
  Frank-Wolfe method on a spectral-norm ball (Sfyraki and Wang, arXiv:2506.04192), this literature is directly
  relevant. Its rates are sublinear, restart-based or, with a Euclidean proximal oracle, logarithmic (Garber and
  Kaplan, arXiv:1809.10477).
* **Sign, normalized and compressed neighbours.** SignSVRG (arXiv:2305.13187), SSVR (arXiv:2406.00489) and SignRVR
  (arXiv:2310.15976) put an SVRG-type estimator inside the sign map (the l_infinity LMO). SPIDER (Fang et al.,
  arXiv:1807.01695) pairs a recursive finite-sum estimator with a normalized step of fixed length or with a clipped
  length. In the compression literature a contractive operator is applied, with error feedback, to a loopless-SVRG
  estimator with a linear rate (EC-LSVRG, Gorbunov et al., arXiv:2010.12292).
* **Variance reduction with a preconditioner computed once is published**: SAG and SVRG on data whitened by the
  feature covariance (Yang, Jin, Zhu and Lin, arXiv:1408.3115), sketched preconditioned SVRG (Gonen et al.,
  arXiv:1602.02350), SVRG and Katyusha X in a fixed metric with rates in the constants of that metric (Liu, Feng and
  Yin, arXiv:1905.09734), a preconditioner frozen at its initial value inside SAGA and SVRG (Kereta et al.,
  arXiv:2106.03172), and PROMISE (Frangella et al., arXiv:2309.02014), which updates its preconditioner once in
  total for least squares.
* **The identity behind our Muon-favourable instance is published**: on a matrix least-squares problem the polar
  factor of the gradient is the Newton direction, whatever the conditioning, exactly when the error has orthonormal
  columns up to scale (Shulgin, Laing, Orvieto and Richtarik, arXiv:2610.30280, Theorem 1(iii); the same mechanism
  in a Kronecker model in Du and Su, arXiv:2604.01472).
* **Not found**: a SAGA or loopless-SVRG estimator inside the nuclear-scaled polar step, without error buffer, at a
  constant step size on a strongly convex finite sum; a comparison of placements of a spectral map relative to a
  finite-sum variance-reduction correction, and any method with the map on the predictable part only (placement is
  discussed for compression operators relative to SVRG / SAGA corrections in VR-DIANA, arXiv:1904.05115, for
  gradient tracking in decentralized Muon in SUDA-Muon, arXiv:2604.23980, and for momentum in arXiv:2606.03899 and
  PolarGrad); the polar decomposition of one gradient used as the frozen metric of a variance-reduced method; an
  exact (enumerated) computation of the expected orthogonalized step (closest: the batch-to-population coherence of
  Chen et al., arXiv:2609.34915, estimated with 16 or 32 probe batches).

The configurations tested here and their closest published neighbours. Verdicts of the second check: *close* = a
published work shares the main idea and differs in a way that matters; *not found* = nothing closer than the works
listed was found.

| configuration of this study | verdict | closest published work | what differs here |
|---|---|---|---|
| SAGA-Muon / L-SVRG-Muon (joint placement, nuclear-scaled step); in the lab before this study | not found | Li-Tang-Zou (SVRG, fixed length, decaying step, separable data); Katyusha Section 6 and its l1 instance ASGCD; the Frank-Wolfe family above; EC-LSVRG (contractive operator with error feedback on an L-SVRG estimator, linear rate); control variates inside a spectral map (FedMuon of Takezawa et al., FedCALM, NBCLT); SignSVRG, SSVR | unconstrained nuclear-scaled step with a finite-sum memory and no error buffer, constant step, strongly convex finite sum |
| damped SAGA-Muon: polar step of m + theta c, theta < 1 (theta = 1/n: the SAG direction) | close | NBCLT (damped control-variate estimator m + theta c through a spectral-ball projection); scaled STORM correction inside the polar factor (MARS, arXiv:2411.10438; MARS-M); alpha-SVRG (arXiv:2311.05589); Euclidean B-SAGA (Driggs et al., arXiv:1906.01133) and SVAG (Morin-Giselsson, arXiv:1903.09009); SAG-type memory inside a Frank-Wolfe LMO (Negiar et al.; Lu-Freund, whose Lemma 2.1 is the published analogue of our Section 5.2) | polar factor with nuclear scaling; single-index finite sum with exact per-sample gradients, constant step; theta = 1/n; the MISO reading of Section 5.2 (not found) |
| split placement: gamma sharp(m) + gamma_c c | not found | PMRC (arXiv:2610.19844) and MuSGD (arXiv:2606.03748): polar factor of one component plus another component added linearly. The opposite assignment is published (map on the sampled part or on the innovation, predictable part linear): FedMuon of Liu et al. (arXiv:2510.27403), VR-DIANA (arXiv:1904.05115), ECLK (arXiv:2010.00091), Clip21 (arXiv:2305.18929), DIANA (arXiv:1901.09269), EF21-Muon (arXiv:2510.00643); error feedback around the nuclear-scaled polar step (Parshakova et al., arXiv:2605.08980) | map on the predictable part of a variance-reduced estimator, innovation kept linear; explicit conditional expectation |
| tangent placement: innovation propagated through the Frechet derivative of the map | close | OptiSelect (arXiv:2610.03432: the Frechet derivative of the polar map at the momentum, applied to per-sample gradients to score training examples); linearization of the polar factor as an analysis device (Beneventano et al., arXiv:2606.08388; Xie, arXiv:2609.32861; MuonEq, arXiv:2603.28254) | used as the step of the optimizer, on the innovation of a finite-sum estimator |
| radial placement: predictable direction polar(m), unbiased scalar length | close | GLyDER (Yehudai et al., arXiv:2311.13877: step <grad f, d> / (L ||d||^2) along an arbitrary direction, pairing estimated from fresh samples; its optimal length is Eq. (7) of Schaul et al., arXiv:1206.1106); heavy-ball Frank-Wolfe (arXiv:2110.04243: direction = LMO of a gradient memory, length = pairing with the current gradient); LBGM (arXiv:2202.00280); Demidovich et al. (arXiv:2605.18999: length (||m||_* - ||g - m||_*)_+ / L along the Muon direction of the momentum) | predictable polar direction of a finite-sum memory mean, unbiased signed length from a variance-reduced estimator, added to a Euclidean variance-reduced step (F9) |
| polar-metric SAGA / L-SVRG, lazily refreshed or one-shot | close | fixed or lazily refreshed preconditioners inside SAGA / SVRG (Yang et al.; Gonen et al.; Liu-Feng-Yin; Kereta et al.; PROMISE; Mime, arXiv:2008.03606; OSGM-SVRG, arXiv:2609.11751); polar(D) = D (D^T D)^(-1/2) as a gradient-built preconditioner (arXiv:2610.30280, Eq. (10); Ma et al., arXiv:2601.13474; ASGO, arXiv:2503.20762); cached polar-type transforms applied linearly (CacheMuon, arXiv:2606.16371) | the metric is read in closed form off the polar decomposition of one full gradient and used as the preconditioner of a finite-sum estimator; on `ls-orth` it is the Newton metric by the published identity above |
| anchor-PolarGrad L-SVRG: one PolarGrad step at each refresh of the reference point | not found | the full-gradient step of SARAH / PAGE at a refresh (arXiv:1703.00102); PolarGrad (Theorem 3.2 of arXiv:2505.21799); one-step decrease of the nuclear-scaled step on quadratics (arXiv:2610.30280, Theorem 3) | spectral step on exact gradients only, inside a Euclidean variance-reduced method |
| mean-scaled length (pairing of the direction with m); clipped length with the SAGA estimator | close | clipped: SPIDER, Option II; (L0, L1)-Spider (arXiv:2303.00883); EPISODE (arXiv:2302.07155); AdaSVRF; NBCLT; generalized gradient norm clipping with momentum (arXiv:2506.01913). Mean-scaled: Grams (arXiv:2412.17107), AdaGO (arXiv:2509.02981), NAMO (arXiv:2602.17080), Demidovich et al. (arXiv:2605.18999) | nuclear-norm length with the polar direction and a SAGA table, constant step, strongly convex finite sum; length taken from the predictable part |
| bounded-gain map with an offset proportional to the innovation | close | PRISM (Yang, arXiv:2602.03096: polar map damped by the innovation G_t - M_t of the current step); Variance-Adaptive Muon (arXiv:2601.14603); the scalar map s / sqrt(s^2 + eps^2) with other offsets (arXiv:2610.30280, Section 4.2; Kim et al., arXiv:2603.26554; Mustafi et al., arXiv:2605.23871; soft spectral clipping, arXiv:2603.14315); bounded-gain spectral maps on a STORM estimator (Oikonomidis et al., arXiv:2605.11850); regularized Newton-Schulz (arXiv:2609.30546); Jawanpuria et al. (arXiv:2610.27747) | isotropic offset proportional to the norm of the sampled correction of a finite-sum estimator (it vanishes at the minimizer); nuclear-scaled length |
| relaxed-memory SAGA-Muon | close | NBCLT (relaxed control-variate table feeding a spectral-ball projection); FedCALM; DIANA Frank-Wolfe (arXiv:2406.06788); dual-free SDCA (Shalev-Shwartz, arXiv:1502.06177); the relaxed predictions of Lu-Freund | nuclear-scaled polar step, single-index finite sum; relaxation by the dual-free-SDCA fraction and the link with SDCA / MISO |
| importance sampling inside the polar step (IS SAGA-Muon) | close | Katyusha Section 6 (index drawn with p_i proportional to L_i, reweighted correction, dual-norm step of an arbitrary norm); Euclidean importance sampling for SAGA / SVRG (Xiao-Zhang, arXiv:1403.4699; Schmidt et al., arXiv:1504.04406; Gower et al., arXiv:1805.02632; Qian et al., arXiv:1906.01481) | SAGA table and stand-alone nuclear-scaled polar step in the spectral norm; exponent 1/2; bounded-gain map |
| lazy split L-SVRG-Muon: one factorization per reference point | not found | CacheMuon, MuonBP (arXiv:2510.16981), PROMISE; cached curvature factors with a fresh polar factor (Newton-Muon, arXiv:2604.01472) | polar factor computed once per reference point, stochastic correction added linearly |
| PAGE-Muon with the nuclear-scaled step | close | Sarah Frank-Wolfe (PAGE-type estimator with exact refresh inside an LMO); SPIDER-FW; SPIDER (normalized step); Qian et al. (arXiv:2604.10689) | unconstrained nuclear-scaled step, constant step, strongly convex finite sum |
| Katyusha-Muon: loopless Katyusha with the spectral y step | close | Katyusha Section 6 (general norms); ASGCD (the l1 instance, tested on Lasso); Katyusha X in a fixed metric (Liu-Feng-Yin) | spectral instance, loopless, non-proximal |
| minibatch SAGA-Muon with b of the order of the rank | close | Phases of Muon (Paquette et al., arXiv:2605.09552: the minibatch gradient has rank min(N, B); transition of Muon at an effective batch of the order of the dimension; momentum supplies the missing rank); Naganuma et al. (arXiv:2602.03001); Carlson et al. 2015 (Section 3.4); Li-Tang-Zou; Sato et al. (arXiv:2507.01598) | minibatch finite-sum memory inside the nuclear-scaled step; tuned batch compared with the rank against Euclidean minibatch SAGA |

Controls, not claimed: Euclidean B-SAGA, SAGA with importance sampling, their combination, and SAGA preconditioned
by the feature covariance (`cov-metric SAGA`: the method of Yang et al. and the object of the note B1 of the main
project, here with the loss-part memory of the lab); the nuclear-scaled step on a STORM buffer (a variant of
OptMuon-I, arXiv:2606.08783, which scales by the Frobenius norm and an AdaGrad-norm coefficient). Several structural
facts of Section 4 are known and are attributed there.

The second check changed the status of several configurations that the first one had left without a close
neighbour: the Newton-metric identity (published); the one-shot preconditioned variance-reduced structure
(published; only the source of the metric is not found); the bounded-gain map with an innovation-dependent offset
(PRISM); the radial length (GLyDER and others); importance sampling in a general-norm step (Katyusha); the damped and
relaxed tables inside a spectral map (NBCLT, October 2026). No published work was found to contain one of the
algorithms of this study as such.

## 4. Structural facts

Proved in [`notes_theory.md`](notes_theory.md), with attributions; the main identities and inequalities are checked
numerically by unit tests (`tests/test_facts.py` and the tests of the code), except F8, which is a reading of a
published proof. The notes were refereed twice by independent agents who re-derived the statements and tested them
with their own scripts; F11 was written after the second report.

* **F1 (rank one).** On a rank-one matrix the nuclear-scaled polar map is the identity and the polar factor is the
  normalized matrix. At batch size one, without memory and with the regularizer outside the orthogonalized matrix,
  the nuclear-scaled polar step is the SGD step and the fixed-length one is normalized SGD (Lemma H.1 of
  Li-Tang-Zou): the spectral structure of a single-index method comes only from what it aggregates. The nuclear-norm
  second moment of the estimator error is at most 4 times its Euclidean variance, with no rank factor.
* **F2 (fixed length).** A step of fixed length eta in spectral norm keeps the iterates at spectral distance at least
  eta / 2 from the minimizer every other iteration, for any estimator; in Frobenius norm the bound is
  sqrt(rank) eta / 2 for an exact polar factor (for exact gradients: Lemma 4 of Jawanpuria et al., arXiv:2610.27747;
  the 2-cycles of Gonon et al., arXiv:2602.11948; "null-gradient consistency" in PolarGrad). Variance reduction
  cannot remove this floor; the length rule must.
* **F3 (memory ceiling).** The SAGA table error satisfies E[M_{t+1} | past] >= (1 - b/n) M_t for every step rule: per
  pass the *mean* memory error contracts by at most (1 - b/n)^(n/b), about e^(-1) when b << n, in any geometry. It
  is a statement in expectation, carried by the entries not yet refreshed: single runs sit around this factor and
  can beat it (0.30 to 0.46 per pass for Euclidean SAGA on the memory-limited instances). It concerns the SAGA table
  (not an SVRG reference point); that the iterate cannot stay much ahead of its memory is an observation of the
  runs, not a theorem.
* **F4 (predictable part).** m_{t+1} = m_t - lam S_t + (b/n) c_t for any step S_t. m is the gradient of an
  *isotropic* quadratic surrogate (Hessian lam I) minimized at W_B = -zbar/lam; the Euclidean member with
  theta = b/n, gamma = 1/lam is MISO with lower-bound surrogates (the Finito update with alpha = 1). For a frozen
  table m depends on W isotropically: the curvature of the losses enters m only through past innovations.
* **F5 (pathwise descent).** F(W - gamma ||omega||_* polar(omega)) <= F(W) - gamma ||omega||_* [(1 - gamma L_op/2)
  ||omega||_* - ||omega - grad F||_*]: a decrease is guaranteed only if the nuclear-norm error of the estimator is
  below (1 - gamma L_op / 2) times its nuclear norm. The descent steps of the published analyses of stochastic LMO
  methods are of this type and do not use the unbiasedness of the matrix fed to the LMO.
* **F6 (expected step).** Explicit for the split and tangent placements; for theta = 1 the tangent one is
  second-order accurate when the memory error is small compared with the smallest singular value of m.
* **F7 (gain of the polar derivative).** ||D polar(M)|| = 1 / s_min(M) (rectangular case; classical): the worst
  perturbation rotates the smallest singular pair, and a generic rank-one innovation changes the polar factor, to
  first order, by a fraction of ||c|| / s_min. The polar factor of an estimator is guaranteed to be accurate only if
  the innovation is small compared with the *smallest* singular value of the matrix at which the map is evaluated
  (innovations along a singular pair are harmless).
* **F8 (acceleration).** Katyusha's general-norm template (Katyusha2) applies to the spectral norm with the Frobenius
  distance; its certified complexity has the Euclidean order. The loopless, non-proximal variant implemented here is
  covered by no theorem.
* **F9 (radial step).** For a direction O known before the index is drawn and the length gamma <omega_j, O>,
  E F(W_+) <= F(W) - gamma a^2 + (L_op/2) gamma^2 (a^2 + v) with a = <grad F, O>, v the variance of the length: a
  descent inequality in expectation with no condition on the accuracy of the estimator (a decrease for
  gamma < 2 a^2 / (L_op (a^2 + v))), proved for the pure radial step. The progress is governed by
  a^2 <= ||grad F||_*^2, which is guaranteed to be large only when the memory is fresh (with a stale memory a can
  even be negative).
* **F10 (frozen polar metric).** The linear map omega -> omega C, with C built from the SVD of an anchor matrix D
  (C proportional to (D^T D)^(-1/2) on the row space of D), equals sharp(D) at the anchor and, while C is frozen,
  turns the method into a Euclidean variance-reduced method on the reparametrized problem
  Fhat(What) = F(What C^(1/2)), with per-sample smoothness R2P = max_i c_l x_i^T C x_i + lam ||C||_op. For least
  squares with K = d and an anchor where W - W_* is a multiple of an orthogonal matrix, C is the Newton metric and
  R2P / mu_P = max_i x_i^T H^(-1) x_i + lam / lambda_min(H); this does not survive an active floor, K < d, or later
  anchors.
* **F11 (one-shot polar metric: certified rate).** If C is computed once and never refreshed, the iteration is, for
  the whole run, SAGA on Fhat in the form used by the lab (table of loss gradients, exact gradient of a regularizer
  that is no longer isotropic), and the proof of Proposition 5.9 of the book goes through in the metric after two
  changes: with step 1 / (4 R2P), a Lyapunov function contracts by 1 - min{1/(3n), 3 mu_P / (16 R2P)} per
  iteration, mu_P being the *global* strong-convexity constant of Fhat. For least squares this constant is exact and
  can be far better than the Euclidean one (42 against 4364 on `ls-orth`). For softmax it is lam lambda_min(C), and
  R2P / mu_P >= R^2 / lam: the theorem never certifies a gain there, as for every fixed metric (note B1 of the main
  project); the Hessian at the optimum gives a local constant only. Nothing is proved for a refreshed metric.

## 5. Experiments

Scripts `experiments/exp_M*.py`; all tables in [`results/digest.md`](results/digest.md).

### 5.1 Two floors, two ingredients (M3)

Runs with constant parameters, three seeds. "Stationary level": the run is started at relative squared distance 1e-14
from W_* (memory initialized there); a method without floor stays below, a method with a floor climbs to it.

| configuration (constant parameters) | stationary level, soft-mem | stationary level, ls-mem |
|---|---|---|
| SAGA estimator, fixed length 1e-1 / 1e-2 / 1e-3 | 4.6e-5 / 4.6e-7 / 4.6e-9 | 5.3e-5 / 5.2e-7 / 5.3e-9 |
| SAGA estimator, nuclear-scaled length | 4.4e-26 | 1.4e-22 |
| SAGA estimator, clipped length (rho = 1e-1 or 1e-2) | 4.4e-26 | 1.4e-22 |
| SAGA estimator, bounded-gain map with a fixed length | 5.7e-18 | 2.3e-25 |
| minibatch gradient (b = 32), nuclear-scaled length | 1.2e-1 | 4.8e-2 |
| Muon (b = 32, Nesterov, Newton-Schulz), lr 1e-1 / 1e-2 | 5.7e-2 / 5.7e-3 | 1.6e-2 / 1.7e-3 |
| MARS-M (b = 32), lr 1e-2 | 4.8e-3 | 1.7e-3 |
| STORM-Muon (b = 32), lr 1e-2 | 4.7e-3 | 1.6e-3 |
| STORM buffer with the nuclear-scaled length, b = 32 / b = 1 | 4.1e-2 / 3.5e-1 | 9.1e-3 / 1.4e-1 |
| Euclidean SAGA | 6.1e-21 | 2.3e-25 |

* **Length.** With the SAGA estimator and a fixed length lr, the iterates settle at ||W - W_*||_F^2 = 3.8 lr^2
  (softmax) and 6.0 lr^2 (least squares) for the three values of lr: the floor of F2, exactly quadratic in lr and of
  the order of its Frobenius form (rank x lr^2 / 4 = 2.25 lr^2 and 2.5 lr^2). The deterministic versions of these
  statements are known (Section 4, F2); what the experiment adds is the finite-sum estimator: variance reduction
  changes the order of the floor (with minibatch noise the floor of Muon is linear in lr: 5.7e-2, 5.7e-3) but cannot
  remove it. A nuclear-scaled or clipped length removes it: the same runs go down to rounding (1e-22 to 1e-26), and
  from W_0 = 0 they reach 2e-27 and 3e-20 in 60 passes. The clipped rule keeps a fixed length rho far from the optimum
  and coincides with the nuclear-scaled rule near it. A bounded-gain map with a fixed length also converges: near
  the optimum it is SAGA with step lr / eps (the deterministic analogue is Proposition 2 of Jawanpuria et al.).
* **Estimator.** A momentum-based buffer (STORM, MARS-M) run with constant parameters keeps a noise floor on a finite
  sum whatever the length rule: with the nuclear-scaled length it settles at 4e-2 and 9e-3 (b = 32) and at 0.35 and
  0.14 (b = 1). With the decaying schedules of the published analyses the floor vanishes, but only at a sublinear
  rate (O(T^(-1/3)) for the objective gap under a Polyak-Lojasiewicz condition, Chang et al.); a linear rate at
  constant parameters needs a finite-sum estimator.

Figure: `figures/figM3_floors.png`.

### 5.2 The polar factor of the SAGA estimator is MISO in disguise (M6)

F4 says that the part of the SAGA estimator that orthogonalization can act on, m = zbar + lam W, is the gradient of
an isotropic quadratic minimized at W_B = -zbar / lam. At batch size one the estimator is m plus a rank-one innovation
that is much larger than m (Section 5.4): its polar factor is the direction of the innovation plus, essentially, the
polar factor of the rest of m, and the nuclear-scaled step gives all these directions the length set by the innovation.
This drives m to zero: the iterate is slaved to W_B. A method that keeps W = W_B exactly is MISO with lower-bound
surrogates (Mairal 2015; the Finito update with alpha = 1), the Euclidean member theta = 1/n, gamma = 1/lam of the
family. M6 compares them at fixed, untuned parameters, with the same sampled indices (n = 1000, d = 50, K = 10,
alpha = 1; three seeds; passes to 1e-6):

| instance | SAGA, c = 1/4 | MISO | joint SAGA-Muon, c = 1/4 | joint SAGA-Muon, c = 1 | dual-free SDCA | joint SAGA-Muon, relaxed table, c = 1/4 |
|---|---|---|---|---|---|---|
| softmax, kappa/n = 0.1 | 9.2 | 8.7 | 8.8 | 8.8 | 8.7 | 8.8 |
| softmax, kappa/n = 1 | 20.2 | 11.9 | 11.8 | 12.3 | 14.4 | 14.4 |
| softmax, kappa/n = 10 | not reached (2e-3) | stalls (1.0) | stalls (0.19) | stalls (0.83) | not reached (1.1e-4) | not reached (1.0e-4) |
| least squares, kappa/n = 0.1 | 10.6 | 10.4 | 10.5 | 11.3 | 10.4 | 10.5 |
| least squares, kappa/n = 1 | 12.9 | 18.9 | 18.1 | 20.0 | 17.4 | 17.0 |
| least squares, kappa/n = 10 | 19.1 | diverges | diverges | diverges | not reached (1.3e-3) | not reached (5.9e-4) |

(In parentheses: relative squared distance after 40 passes.) The joint method at the transferred textbook step needs
the same number of passes as MISO wherever MISO converges, fails where MISO fails (kappa >> n; MISO is guaranteed for
n >= 2 kappa only), and is insensitive to its step constant (c = 1/4 or 1). When the table is relaxed by the
dual-free-SDCA fraction (`mem_relax='sdca'`), it follows dual-free SDCA instead, including on the two instances with
kappa/n = 10, where both are stable and slow (in expectation, and as long as the memory error dominates, a relaxed
table contracts by at most about exp(-delta (2 - delta)) per pass, F4). The agreement is pathwise: with the same
indices the traces are superposed (`figures/figM6_miso.png`).
Along the joint runs the nuclear norm of m is 1% to 16% of that of the gradient (26% to 116% for Euclidean SAGA).

Consequences. (i) At batch size one the "polar factor of the SAGA estimator" is not a spectral method: the matrix it
orthogonalizes is dominated by a rank-one innovation, the rest (m) is kept near zero, and the algorithm it implements
is a known Euclidean one. Its speed-up over SAGA at textbook steps on some instances (softmax, kappa/n = 1: 11.8
against 20.2 passes) is MISO's, and tuned SAGA recovers it. (ii) Its failures when kappa >> n and the narrow range of
usable step constants reported below are those of MISO outside its "big data" regime. (iii) The Euclidean family
gamma (m + theta c) interpolates between SAGA (theta = 1) and MISO (theta = 1/n, gamma = 1/lam); the tuned
orthogonalized methods of Section 5.3 sit at intermediate points of an equivalent interpolation. The published
analogue on the Frank-Wolfe side is Lemma 2.1 of Lu and Freund (a SAG-type memory inside a generalized LMO is a
randomized dual coordinate method).

<!-- M1M2 -->

### 5.7 Rounding sensitivity of the exact polar factor (M4)

Two runs with the same sampled indices, started 1e-12 apart (relative to ||W_*||_F). Relative separation
||W_a - W_b||_F / ||W_a - W_*||_F, median over 3 seeds:

| configuration | soft-mem, after 1 / 10 passes | ls-mem, after 1 / 10 passes |
|---|---|---|
| Euclidean SAGA, c = 1/4 | 1e-12 / 2e-12 | 1e-12 / 1e-12 |
| joint, exact polar, c = 1/64 | 1e-12 / 1e-12 | 1e-12 / 3e-12 |
| joint, exact polar, c = 1/16 | 1e-12 / 3e-12 | 2e-12 / 2e-11 |
| joint, exact polar, c = 1/4 (transferred textbook step) | 4e-4 / 0.10 | 0.31 / 0.30 |
| joint, Newton-Schulz (ns5), c = 1/4 | 8e-13 / 2e-12 | 1e-12 / 1e-12 |
| joint, bounded gain, c = 1/4 | 1e-12 / 2e-12 | 1e-12 / 1e-12 |
| joint, damped (theta = 0.1), c = 1/4 | 8e-13 / 2e-12 | 2e-12 / 4e-12 |
| split, c = 1/4 | 1e-12 / 2e-12 | 2e-12 / 1e-12 |

With the exact polar factor, the joint placement and the transferred textbook step, a perturbation of relative size
1e-12 is amplified to order one within about one pass (F7: the gain of the polar map is 1 / s_min, and the estimator
is close to rank one). Such runs are deterministic on a given machine with single-threaded BLAS (a random sample of
the cached runs was recomputed and compared, see the cache audit at the end of the digest) but they are not robust
to a change of rounding (BLAS threading, platform): only their statistics over seeds are meaningful, which is how
they are reported here. A smaller step, the Newton-Schulz polynomial (bounded gain 3.4445^5), the bounded-gain map,
damping, or the split placement all remove the effect. Figure: `figures/figM4_sensitivity.png`.

<!-- M7M8 -->

### 5.10 Cost of the factorization

Budgets count gradient evaluations only. In this numpy implementation at K = 10, d = 50 an iteration of the
orthogonalized single-index methods (one 10 x 50 SVD or five Newton-Schulz steps) takes 3 to 4 times as long as a
SAGA iteration (unit test `test_timing` of `tests/test_lab.py`: about 11,000 against 34,000 to 46,000 iterations per
second on an idle machine). The placements differ in the number of factorizations:

| placement | factorizations |
|---|---|
| joint, split, tangent, radial | one per iteration (n per pass at batch size one, n / b with minibatches) |
| polar metric, refreshed | one per pass (SAGA) or per reference point (L-SVRG) |
| lazy split, anchor PolarGrad | one per reference point |
| one-shot polar metric | one for the whole run |

The linear placements (metric) then cost one K x d by d x d product per iteration on top of SAGA.

<!-- CONCLUSION -->
