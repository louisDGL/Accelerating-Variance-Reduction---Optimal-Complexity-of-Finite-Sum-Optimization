# Variance reduction for Muon on finite sums: how far it goes, where it breaks, new configurations

Side study of the project "Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization"
(Statistical Learning Theory, UC Berkeley, November 2026). Code and conventions: [`README.md`](README.md). Proofs of
the structural facts F1-F16: [`notes_theory.md`](notes_theory.md). All numbers: [`results/digest.md`](results/digest.md)
(generated from the JSON summaries; the tables below are copied from it). Sections 5.11 and 5.12, the facts
F12-F16 and the sentences marked "2026-11-19" in Sections 2 to 4 were added on 2026-11-19; the last section lists
the statements of the other write-ups that they affect. The paper of the project is `../paper/paper_saga_to_muon.pdf`.
The unit tests and the numerical checks named in this file were archived on 2026-11-20 in the local, unversioned
`_archive/` folder of the project.

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
(Added 2026-11-19.) A grid of factor 2 is not always fine enough: with minibatches the fast steps of the
nuclear-scaled polar step end at a cliff and a factor-2 grid can step over them, which is what happened in M5
(Section 5.12 redoes it on grids of ratio 2^(1/4)). The minibatch rows of M1 (`b = r`) were tuned on factor-2
grids too and have not been redone: for the orthogonalized families they may overstate the passes needed.

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

(Added 2026-11-19.) A third check, on 2026-11-18, concerned only the additions of Sections 5.11 and 5.12 and the
facts F12-F16. One agent, asked to refute their novelty, ran 27 web searches, 21 abstract-level queries of the
arXiv API and 16 title searches on OpenReview, and read the relevant sections and proofs of ten papers in full
text (the others by keyword scan or abstract; Goffin 1977 could not be opened). Its verdict is *close*: the
bullet and the table row marked "2026-11-19" below give what it found, and the addendum of
[`literature_check.md`](literature_check.md) the record.

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
* **(Added 2026-11-19.) The two halves of the schedule law of Section 5.11 are published separately.** *Exact
  gradients:* steps of fixed length on a geometric schedule converge linearly, with the error within a constant of
  the step and a critical ratio below which the guarantee is lost: Goffin (Math. Programming 13, 1977; not read,
  known through the next paper), Davis, Drusvyatskiy, MacPhee and Paquette (arXiv:1803.02461, Theorem 5.1), Li,
  Zhu, So and Lee for incremental methods on sharp finite sums without variance reduction (arXiv:1907.11687,
  Theorem 2); for Muon and spectral descent, Ma et al. (arXiv:2601.13474: ratio at least 1/2), Rhee et al.
  (arXiv:2606.30509), Yang, He and Li (arXiv:2605.26977). *Finite sums:* a descent inequality "first-order decrease
  minus lr^2 times a staleness constant" for an SVRG estimator inside a sign or spectral LMO step under
  reshuffling, with arbitrary step sizes and without linear rate: Qin, Liu and Xu (SignRVR, arXiv:2310.15976,
  Theorem 4.1) and Li, Tang and Zou (arXiv:2602.11557, Lemma F.1: Muon's recipe fed with SVRG, polynomial
  schedules, separable data); for Muon with momentum under shuffling, without table, Huang, Pu and Johansson
  (arXiv:2611.18124). Their Theorem 3.2 is a published form of "without a table no schedule removes the floor", for
  a fixed momentum coefficient: on a family of convex sums the gradient stays above 1/N in every order for every
  step-size sequence that is constant within a pass and at most 1/(2N), decreasing ones included. Their Theorem 3.1
  removes the floor without a table when the momentum coefficient tends to one with the number of components, at a
  sublinear rate (abstract and both statements re-read on 2026-11-19).
  Chzhen and Schechtman (arXiv:2305.13187, Section 4.2) argue that a linear rate is not to be expected from sign +
  SVRG, at a fixed step size only. Neighbours of the other statements: the floor of order lr^2 and "cooldown is
  necessary" for sign and normalized steps under noise that vanishes at the optimum (Singh, Mishra and Raut,
  arXiv:2607.12360); SAGA as delayed gradient descent, with a path-length staleness bound and a Lyapunov function
  "gap + stale window", in the Euclidean case (Zhu, Heath and Mitra, arXiv:2602.05304); SAG-type tables inside an
  LMO with a (1 - 1/n) memory recursion (Negiar et al., arXiv:2002.11860); SAGA under reshuffling at a constant
  step (Ying, Yuan and Sayed, arXiv:1708.01383).
* **Not found**: a SAGA or loopless-SVRG estimator inside the nuclear-scaled polar step, without error buffer, at a
  constant step size on a strongly convex finite sum; a comparison of placements of a spectral map relative to a
  finite-sum variance-reduction correction, and any method with the map on the predictable part only (placement is
  discussed for compression operators relative to SVRG / SAGA corrections in VR-DIANA, arXiv:1904.05115, for
  gradient tracking in decentralized Muon in SUDA-Muon, arXiv:2604.23980, and for momentum in arXiv:2606.03899 and
  PolarGrad); the polar decomposition of one gradient used as the frozen metric of a variance-reduced method; an
  exact (enumerated) computation of the expected orthogonalized step (closest: the batch-to-population coherence of
  Chen et al., arXiv:2609.34915, estimated with 16 or 32 probe batches). (Added 2026-11-19, check of
  2026-11-18:) a SAGA table inside the fixed-length polar step with a geometric schedule and a linear rate; a
  path-length bound on the nuclear-norm error of the table estimator without rank or dimension factor (F12); the
  exact scale covariance of the SAGA polar iteration on least squares (F14); a Lyapunov function "gap +
  operator-norm staleness of the table" for the nuclear-scaled polar step (F15); a link between the admissible
  speed of a cooldown, the memory rate (1 - b/n)^(n/b) of the table and reshuffling (Section 5.11).

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
| (added 2026-11-19) Muon's own recipe with a SAGA table: fixed-length polar or Newton-Schulz step of the SAGA estimator, geometric cooldown lr_0 q^passes (Section 5.11, F12-F16) | close | Li-Tang-Zou (the same recipe with an SVRG anchor under reshuffling: polynomial schedules, separable data, no linear rate); Qin-Liu-Xu (sign + SVRG, arbitrary steps, nonconvex); the geometric-schedule law for exact gradients (Goffin; Davis et al.; Li-Zhu-So-Lee; for Muon, Ma et al., Rhee et al., Yang-He-Li); Huang-Pu-Johansson (Muon with momentum under shuffling, no table) | SAGA table instead of an epoch anchor; strongly convex finite sum, linear rate observed down to rounding; critical cooldown measured against the memory of the table and the sampling order; rank-free staleness lemma |

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
with their own scripts; F11 was written after the second report. (Added 2026-11-19.) F12-F16 concern polar steps
fed by a SAGA table; their proofs were written by a referee agent during the verification of 2026-11-18 and
re-derived for the notes, where each of them ends with what it does not say.

* **F1 (rank one).** On a rank-one matrix the nuclear-scaled polar map is the identity and the polar factor is the
  normalized matrix. At batch size one, without memory and with the regularizer outside the orthogonalized matrix,
  the nuclear-scaled polar step is the SGD step and the fixed-length one is normalized SGD (Lemma H.1 of
  Li-Tang-Zou): the spectral structure of a single-index method comes only from what it aggregates. The nuclear-norm
  second moment of the estimator error is at most 4 times its Euclidean variance, with no rank factor.
* **F2 (fixed length).** A step of fixed length eta in spectral norm keeps the iterates at spectral distance at least
  eta / 2 from the minimizer every other iteration, for any estimator; in Frobenius norm the bound is
  sqrt(rank) eta / 2 for an exact polar factor (for exact gradients: Lemma 4 of Jawanpuria et al., arXiv:2610.27747;
  the 2-cycles of Gonon et al., arXiv:2602.11948; "null-gradient consistency" in PolarGrad). Variance reduction
  cannot remove this floor; the length must go to zero. (Updated 2026-11-19: the text read "the length rule
  must", and Section 5.1 only tried constant lengths. The length can vanish with the estimator, as in the
  nuclear-scaled and clipped rules, or on a schedule: with a SAGA table a geometric cooldown of the fixed length
  converges linearly, F13 and Section 5.11.)
* **F3 (memory ceiling).** The SAGA table error satisfies E[M_{t+1} | past] >= (1 - b/n) M_t for every step rule: per
  pass the *mean* memory error contracts by at most (1 - b/n)^(n/b), about e^(-1) when b << n, in any geometry. It
  is a statement in expectation, carried by the entries not yet refreshed: single runs sit around this factor and
  can beat it (0.30 to 0.46 per pass for Euclidean SAGA on the memory-limited instances). It concerns the SAGA table
  (not an SVRG reference point); that the iterate cannot stay much ahead of its memory is an observation of the
  runs, not a theorem. (Added 2026-11-19.) For fixed-length steps on a geometric cooldown this factor is where the
  critical cooldown sits on the memory-limited instances, and a sampling order that refreshes every entry once
  per pass moves it (Section 5.11): measured, not proved.
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
* **F12 (staleness lemma; F12-F16 added 2026-11-19).** For the table estimator
  omega = zbar + lam W + (theta/b) sum_{j in B} (g_j'(W) - z_j),
  ||omega - grad F(W)||_* <= (theta/b) sum_{j in B} L_j l_j + (1/n) sum_i L_i l_i, where l_i is the length, in
  spectral norm, of the path travelled since entry i was refreshed. Per-sample gradients are rank one, so the
  bound carries no rank factor (path-length bounds with dimension or rank factors, for SVRG anchors or momentum:
  Qin-Liu-Xu; Li-Tang-Zou; Huang-Pu-Johansson). It is a worst-case bound that grows with the staleness; it does
  not say that the error is small along the runs.
* **F13 (fixed-length polar steps fed by a table).** If every entry is at most A iterations old,
  D_{t+1} <= D_t - lr_t sqrt(2 mu_op D_t) + (L_eff/2) lr_t^2 for D = F - F_*, with
  L_eff = L_op + 4 (theta L_max + L_mean) (sum of the last A lengths) / lr_t. Hence a floor O(lr^2) at a constant
  length, and D_t <= C lr_t^2 on geometric schedules whose per-iteration factor of lr^2 is at least
  1 - mu_op / L_eff: surely for a cyclic order (A = n/b) and for a fresh permutation per pass (A = 2n/b - 1);
  under independent uniform batches, almost-sure convergence when sum lr = infinity and sum lr^2 < infinity, by a
  separate argument. The induction is the classical one for steps of fixed length (Goffin; Davis et al.). It
  certifies the mechanism, not the speed: on `soft-cond` and `ls-cond` the certified cooldown is 3.3e4 to 8.9e4
  times slower than the q = 0.8 that works in Section 5.11, and along such a run the nuclear-norm error of the
  estimator is 2.3 to 3.5 times ||grad F||_*, so the one-step inequality does not certify a decrease there
  (numbers printed by `test_F13_limits_on_the_benchmark`). Momentum is not covered.
* **F14 (scale covariance).** On least squares, with a map such that orth(s M) = orth(M), the state
  u = (W - W_*, table errors) of the fixed-length iteration satisfies Phi_{s lr}(s u) = s Phi_lr(u): along a
  geometric schedule u_t / lr_t follows a recursion that does not depend on t (a time-homogeneous Markov chain
  under independent batches), and the iteration has no intrinsic scale. False for softmax and for the lab's
  Newton-Schulz map with its normalization constant eps = 1e-7, which is what sets its floor in Section 5.11.
* **F15 (a Lyapunov function for the nuclear-scaled step).** At batch one, for the joint nuclear-scaled step with
  gamma <= 1 / (14 n L_max) and gamma L_op <= 1, Psi = F - F_* + (1 / (8 n gamma)) mean_i ||W - phi_i||_op^2 (phi_i
  the iterate at which entry i was refreshed) contracts in expectation by
  1 - min{(1 - gamma L_op) gamma mu_op / 2, 1/(4n)} per iteration: a proof of linear convergence for SAGA-Muon,
  the Euclidean analogue being the incremental-aggregated-gradient analysis of SAGA (Zhu-Heath-Mitra). The
  admissible step is 3.574e-4 / R2_op on `soft-cond` and `ls-cond` (printed by `test_F15_lyapunov`), against the
  tuned 0.5 / R2_op and 0.125 / R2_op of M1: the theorem does not reach the steps that are used.
* **F16 (frozen polar factor).** On least squares with K >= d, the SAG direction (theta <= b/n) keeps the polar
  factor of the full gradient at W_0 as long as the total length travelled is below a threshold tau_*; this is
  false for the SAGA direction (theta = 1).

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

### 5.11 Muon's own recipe with a SAGA table (M9)

(Added 2026-11-19. Script `experiments/exp_M9_schedule.py`. Every number of this section is in
`results/M9_summary.json` under the key given in brackets, and in the section "M9" of the digest, from which the
tables are copied.)

Section 5.1 tried fixed lengths at constant parameters only, and concluded that the length rule had to change. Muon
as used does something else: it keeps steps of fixed length and cools the learning rate down. M9 runs that recipe on
a finite sum, with the SAGA estimator in the place of the minibatch gradient:

    W <- W - lr_t orth(omega_t),        lr_t = lr_0 q^(t b / n)        (the learning rate loses the factor q per pass).

Two numbers are tuned, q and lr_0, with the protocol of Section 2 [`definitions/recipe`, `protocol`]. The maps are
the exact polar factor, Muon's five Newton-Schulz steps (`ns5`), and two controls that are *not* orthogonalizations,
the Frobenius normalization M / ||M||_F and the coordinatewise sign, each with and without Nesterov momentum 0.9.
"Muon's recipe" below is `ns5` with Nesterov momentum. The parts A to D of the experiment are those of the digest.

**The schedule law: with a table the squared distance follows lr_t^2 down to rounding; without a table the run
freezes** (part C) [`C/<instance>/deep/<configuration>/{on_schedule, level, min_rel}`,
`C/<instance>/controls/<configuration>/final_rel`].

| batch one, 300 passes, seeds 10-14 | soft-cond | ls-cond |
|---|---|---|
| SAGA table, exact polar factor, no momentum: seeds on schedule; level of dist^2 / lr^2; lowest relative squared distance | q = 0.8, lr_0 = 0.1: 5/5; 6.3e+04; 2.8e-29 [2.3e-29, 4.1e-29] | q = 0.8, lr_0 = 0.3: 5/5; 9.8e+04; 2.3e-29 [1.4e-29, 3.8e-29] |
| SAGA table, Muon's recipe at its tuned point: seeds on schedule; level of dist^2 / lr^2; lowest relative squared distance | q = 0.7, lr_0 = 0.3: 1/5; 1.1e+05; 1.3e-07 [5.2e-13, 3.0e-05] | q = 0.75, lr_0 = 0.3: 4/5; 2.2e+04; 4.6e-18 [3.6e-18, 2.6e-06] |
| SAGA table, Muon's recipe at q = 0.8, lr_0 = 0.1: seeds on schedule; level of dist^2 / lr^2; lowest relative squared distance | 5/5; 9.0e+03; 9.8e-19 [7.7e-19, 2.8e-18] | 5/5; 1.1e+04; 1.1e-18 [1.0e-18, 1.4e-18] |
| the same with eps = 1e-30 in `ns5`: seeds on schedule; level of dist^2 / lr^2; lowest relative squared distance | 5/5; 8.8e+03; 1.5e-29 [1.1e-29, 1.6e-29] | 5/5; 1.1e+04; 4.1e-30 [3.2e-30, 4.6e-30] |
| no table, Muon's recipe on the minibatch gradient, b = 1 / 32: final relative squared distance | 0.256 / 0.823 | 0.083 / 0.629 |
| no table, MARS-M, b = 1 / 32: final relative squared distance | 0.340 / 0.931 | 0.129 / 0.816 |
| no table, STORM-Muon, b = 1 / 32: final relative squared distance | 0.341 / 0.931 | 0.143 / 0.813 |

"On schedule": the run goes below a relative squared distance 1e-12. The controls without a table use the schedule
tuned for Muon's recipe (`soft-cond`: q = 0.7, lr_0 = 0.3; `ls-cond`: q = 0.75, lr_0 = 0.3)
[`C/<instance>/controls_schedule`]. With the table, dist^2 / lr_t^2 stays within a band of width 3 to 7 (max / min
over the second half of a run [`deep/<configuration>/band/width_median`]) until the distance meets a floor: the
rounding floor for the exact polar factor (the ratio leaves the range it had on schedule at a relative squared
distance of 5.9e-29 and 4.8e-29, medians of [`depth_rel_in_range`]), and about 1e-18 for `ns5`, whose normalization
constant eps = 1e-7 breaks the scale invariance of the map (F14); with eps = 1e-30 the floor of `ns5` is the rounding
floor too. Without a table the same schedules freeze at 8% to 93% of the initial squared distance, for the minibatch
gradient as for the two momentum-based variance-reduced estimators of the literature: the cooldown outruns an
estimator that does not vanish at W_*. This is the geometric-schedule law of steps of fixed length, known for exact
gradients (Section 3), made usable on a finite sum by the table; F13 proves a version of it.

**Batch one: passes to the target on the six instances** (part B: both recipes tuned on a finer grid of q, then run
without tuning at one fixed point; median [min, max] over the confirmation seeds; target 1e-6, and 1e-9 on the two
memory-limited instances) [`B/<instance>/rows`, `B/<instance>/fixed_point/rows`, `B/<instance>/M1/rows`].

| passes to the target, batch one | soft-cond | ls-cond | ls-orth | soft-sq | soft-mem | ls-mem |
|---|---|---|---|---|---|---|
| fixed length, exact polar factor, no momentum (q, lr_0) | 28.0 [26.7, 30.4] (0.8, 0.1) | 36.1 [35.4, 37.0] (0.8, 0.3) | 22.2 [22.0, 22.5] (0.75, 0.03) | 17.1 [16.7, 17.4] (0.75, 0.03) | 18.6 [17.3, 21.7] (0.65, 0.03) | 23.1 [22.2, inf] (9/10) (0.7, 0.01) |
| fixed length, Muon's recipe (q, lr_0) | 21.8 [19.8, inf] (7/10) (0.7, 0.3) | 25.5 [24.7, inf] (9/10) (0.75, 0.3) | 16.6 [16.1, inf] (9/10) (0.75, 0.03) | 13.4 [12.8, 13.9] (0.65, 0.1) | 16.9 [16.4, 19.9] (0.6, 0.03) | 19.0 [17.6, inf] (8/10) (0.6, 0.03) |
| exact polar factor at the fixed point q = 0.8, lr_0 = 0.1 | 28.0 [26.7, 30.4] | 31.3 [30.7, 33.1] | 32.7 [32.2, 32.9] | 25.0 [23.9, 25.7] | 25.2 [24.7, 25.4] | 26.9 [26.5, 27.4] |
| Muon's recipe at the fixed point | 23.4 [22.3, 24.8] | 26.4 [25.4, 26.7] | 25.3 [24.7, 25.7] | 21.3 [20.8, 21.8] | 38.6 [38.4, 38.8] | inf [39.9, inf] (4/10) |
| M1 (constant step, factor-2 grids): SAGA | 69.5 | 98.7 | 27.1 | 26.1 | 17.1 | 18.9 |
| M1 (constant step, factor-2 grids): B-SAGA | 33.8 | 31.9 | 18.4 | 15.9 | 17.0 | 18.6 |
| M1 (constant step, factor-2 grids): SAGA-Muon (joint) | 29.0 | 43.5 (9/10) | 28.1 | 20.3 | 18.0 | 23.1 (9/10) |
| M1 (constant step, factor-2 grids): soft SAGA-Muon | 23.9 | 29.8 | 20.5 | 14.9 | 16.6 | 18.5 |
| M1 (constant step, factor-2 grids): one-shot polar-metric B-SAGA | 22.1 | 24.4 | 13.9 | 12.9 | 16.4 | 18.4 |
| M1 (constant step, factor-2 grids): cov-metric B-SAGA | 15.1 | 16.1 | 13.9 | 13.5 | 15.8 | 17.5 |

On the four conditioning-limited instances Muon's recipe with a SAGA table needs 13.4 to 25.5 passes: fewer than
B-SAGA and than the nuclear-scaled joint SAGA-Muon of M1, comparable to the one-shot polar-metric B-SAGA (12.9 to
24.4), and more than the covariance-metric B-SAGA (13.5 to 16.1) except for a tie on `soft-sq`. The comparison with M1
is on unequal terms in one respect, M1's factor-2 step grids: on 2^(1/4) grids (Section 5.12) B-SAGA needs 27.1, 31.9
and 17.7 passes at batch one on `soft-cond`, `ls-cond` and `ls-orth`, still more than the tuned recipe. But the tuned
recipe loses confirmation seeds on three of the four instances (7/10, 9/10, 9/10). At the fixed point q = 0.8, lr_0 =
0.1, chosen once for all instances, no seed is lost on these four and Muon's recipe needs 21.3 to 26.4 passes: ahead
of B-SAGA on `soft-cond` and `ls-cond`, behind it on `ls-orth` and `soft-sq`. On the memory-limited instances nothing
is gained (16.9 and 19.0 with 8/10 seeds, against 17.1 and 18.9 for SAGA), and the fixed point is too slow or fails.

**What belongs to the spectral map: nothing at batch one, a factor 0.6 to 0.9 at batches of 16 and 32** (part A:
eight variants tuned per batch size; same-batch references with constant steps on 2^(1/4) grids; `*` = selected q on
the edge 0.95 of its grid) [`A/<instance>/cells/<b>/rows/<family>/summary`, `A/<instance>/cells/<b>/compare/ratios`].

| batch one | soft-cond | ls-cond | ls-orth | soft-sq |
|---|---|---|---|---|
| exact polar factor | 28.0 | 36.1 | 22.6 | 19.5 |
| exact polar factor + Nesterov | 21.4 (7/10) | 26.8 | 17.4 | 14.9 |
| `ns5` | 27.6 | 36.0 | 22.1 | 19.7 |
| `ns5` + Nesterov (Muon's recipe) | 21.8 (7/10) | 27.9 (8/10) | 17.2 | 14.6 |
| control: Frobenius normalization | 22.9 | 29.0 | 16.3 | 14.8 |
| control: Frobenius normalization + Nesterov | 32.0 | 50.5 | 19.8 | 18.8 |
| control: sign | 31.1 | 31.4 | 19.6 | 18.1 |
| control: sign + Nesterov | 35.7 | 59.3 | 21.6 | 18.9 |
| *fastest spectral / fastest control (grid of part A)* | 0.94 | 0.92 | 1.06 | 0.99 |
| *exact polar factor without momentum (part B) / fastest control* | 1.22 | 1.24 | 1.36 | 1.16 |

| passes to 1e-6 | soft-cond, b = 16 | soft-cond, b = 32 | ls-cond, b = 16 | ls-cond, b = 32 | ls-orth, b = 16 | ls-orth, b = 32 | soft-sq, b = 16 | soft-sq, b = 32 |
|---|---|---|---|---|---|---|---|---|
| exact polar factor | 22.4 | 25.7 | 27.7 (9/10) | 30.8 | 16.9 | 17.4 | 13.6 | 13.2 |
| exact polar factor + Nesterov | 31.0 | 29.3 | 28.8 | 32.7 | 18.5 | 20.4 | 16.4 (9/10) | 16.5 |
| `ns5` | 22.9 | 30.4 | 29.7 | 31.3 | 16.6 | 17.9 | 13.6 | 15.3 |
| `ns5` + Nesterov (Muon's recipe) | 30.4 | 28.8 | 33.3 | 32.2 | 18.2 | 20.2 | 16.1 (9/10) | 19.0 |
| control: Frobenius normalization | 48.3 | 75.0 | 144.3* | not reached* | 40.6 | 68.6 | 22.8 | 32.3 |
| control: Frobenius normalization + Nesterov | 27.8 | 30.5 | 38.9 | 53.2 | 17.9 | 19.4 | 17.5 | 16.0 |
| control: sign | 63.2 | 134.9* | not reached* | not reached* | 55.5 | not reached* | 23.6 (9/10) | 39.5 |
| control: sign + Nesterov | inf (5/10) | 33.2 | 53.2 | 71.7 | 22.8 | 22.2 | 19.0 | 17.7 (8/10) |
| reference: SAGA, constant step | 82.2 | 100.4 | 232.6 | not reached | 51.4 | 85.2 | 36.3 | 50.6 |
| reference: B-SAGA, constant step | 31.0 | 36.4 | 51.2 | 75.4 | 18.8 | 20.6 | 16.8 | 17.7 |
| reference: nuclear-scaled joint SAGA-Muon, constant step | 23.2 | 29.6 | 28.0 | 37.9 | 16.5 | 16.7 | 13.4 | 13.4 |
| *fastest spectral / fastest control* | 0.80 | 0.84 | 0.71 | 0.58 | 0.93 | 0.89 | 0.78 | 0.83 |
| *fastest spectral / fastest control without momentum* | 0.46 | 0.34 | 0.19 | 0.00 | 0.41 | 0.25 | 0.59 | 0.41 |
| *fastest spectral / best of SAGA and B-SAGA* | 0.72 | 0.71 | 0.54 | 0.41 | 0.89 | 0.84 | 0.81 | 0.75 |
| *fastest spectral / nuclear-scaled joint SAGA-Muon* | 0.96 | 0.87 | 0.99 | 0.81 | 1.01 | 1.04 | 1.01 | 0.99 |

At batch one the effect is not specific to Muon: the fastest spectral variant over the fastest control gives 0.94,
0.92, 1.06 and 0.99 (with 7/10 seeds for the spectral variant on `soft-cond`), and without momentum the Frobenius
normalization of the SAGA estimator, i.e. normalized SAGA on the same cooldown, needs *fewer* passes than the exact
polar factor (ratios 1.16 to 1.36 [`cells/1/part_B/ratios`]). This is consistent with F1 and Section 5.2: at batch
one the estimator is close to rank one, where the polar factor is the normalized matrix. The map earns its keep with
minibatches. At b = 16 and 32 the polar and `ns5` maps need no momentum (13.2 to 31.3 passes; momentum costs them
passes in 15 of the 16 cells), while the controls need Nesterov momentum to stay in range: without it they need 22.8
to 144.3 passes or miss the target (on `ls-cond` and `ls-orth`), with it the faster of the two needs 16.0 to 53.2
(the sign control with momentum loses seeds in two cells). The fastest spectral variant
needs 0.58 to 0.93 of the passes of the fastest control (`ls-cond` 0.71 and 0.58, `ls-orth` 0.93 and 0.89), 0.41 to
0.89 of those of the faster of SAGA and B-SAGA at the same batch, and 0.81 to 1.04 of those of the nuclear-scaled
joint SAGA-Muon: the fixed-length recipe does as well as the nuclear-scaled step of Section 5.12, and both are
ahead of the Euclidean references at these batch sizes.

**Sampling order and the memory of the table** (part D: exact polar factor, no momentum, 5 seeds; rho is the factor
of lr^2 per pass, q = rho^(1/2); rho_star is the smallest rho of the grid at which at least 80% of the seeds stay on
schedule) [`D/<instance>/cells/<b>/schemes/<scheme>/{rho_star, memory_factor, fastest}`].

| critical factor rho_star of lr^2 per pass | exp(-(1 + m/b)) | soft-mem, b = 5 | soft-mem, b = 20 | soft-mem, b = 50 | ls-mem, b = 5 | ls-mem, b = 20 | ls-mem, b = 50 | soft-cond, b = 20 | soft-cond, b = 50 |
|---|---|---|---|---|---|---|---|---|---|
| independent batches | 0.368 | 0.368 | 0.32 | 0.27 | 0.43 | 0.368 | 0.32 | 0.61 | 0.61 |
| independent batches + m = b extra refreshes | 0.135 | 0.135 | 0.135 | 0.1 | 0.18 | 0.18 | 0.135 | 0.43 | 0.61 |
| independent batches + m = 3b extra refreshes | 0.018 | 0.012 | 0.012 | 0.008 | 0.025 | 0.018 | 0.018 | 0.27 | 0.61 |
| fresh permutation per pass | - (linear SAGA on the model of hypothesis B4: exp(-2) = 0.135; see below and Section 5.13) | 0.22 | 0.1 | 0.135 | 0.43 | 0.18 | 0.18 | 0.5 | 0.7 |
| one permutation for the whole run | - | 0.18 | 0.18 | 0.18 | 0.368 | 0.32 | 0.368 | 0.7 | 0.8 |
| passes to 1e-9 at the fastest rho, independent batches | | 17.7 | 17.9 | 16.7 | 20.7 | 20.0 | 19.1 | 38.7 | 53.0 |
| the same with a fresh permutation per pass | | 10.4 | 9.1 | 10.4 | 16.7 | 11.3 | 12.2 | 29.5 | 52.5 |
| the same with m = b extra refreshes (charged) | | 17.4 | 18.6 | 18.0 | 21.0 | 21.4 | 20.6 | 46.6 | 78.5 |
| memory factor (1 - (b + m)/n)^(n/b): m = 0, b, 3b | | 0.367, 0.134, 0.018 | 0.364, 0.130, 0.015 | 0.358, 0.122, 0.012 | 0.367, 0.134, 0.018 | 0.364, 0.130, 0.015 | 0.358, 0.122, 0.012 | 0.349, 0.107, 0.006 | 0.316, 0.062, 0.000 |

Where memory is the limit (`soft-mem`, `ls-mem`) the thresholds sit next to the memory factor of the table. With
independent batches the critical factor is 0.27 to 0.43 for a memory factor of 0.36-0.37; refreshing m = b or m = 3b
extra entries per step moves it to 0.10-0.18 and 0.008-0.025, next to exp(-(1 + m/b)) = 0.135 and 0.018, but the
extra gradients are charged and the passes do not improve. A fresh permutation per pass refreshes every entry once
per pass for free: at b = 20 and 50 the threshold moves to 0.10-0.18 and the passes to 1e-9 go from 16.7-20.0 to
9.1-12.2.

*What F3 does and does not say here* (corrected on 2026-11-19, after experiment M9b, Section 5.13). F3 bounds the
*mean* memory error: in expectation the table forgets by at most its memory factor per pass. It does not bound a
single run, and seven of the 18 thresholds with independent batches at b >= 5 are indeed below their memory factor
(m = 0: 0.32 and 0.27 on `soft-mem` at b = 20 and 50, 0.32 on `ls-mem` at b = 50; m = b: 0.10 on `soft-mem` at
b = 50; m = 3b: 0.012, 0.012 and 0.008 on `soft-mem`). The level
dist^2 / lr^2 degrades gradually there (on `soft-mem` at b = 50: 6.1, 10.3, 22.2 and 455 at rho = 0.61, 0.368, 0.32
and 0.27 [`schemes/iid/by_rho/<k>/C`, medians]), and the reading "on schedule" allows a factor 100. An earlier
version of this paragraph read the two thresholds 0.27-0.43 and 0.10-0.18 as the two memory rates of linear SAGA in
the main study (hypothesis B4: per pass, exp(-1) in expectation under independent sampling and exp(-2) under
reshuffling, on sums with identical Hessians). M9b runs the fixed-length recipe on that very model, and the reading
does not survive as stated: there a single run with independent batches follows cooldowns *faster* than exp(-1)
(thresholds 0.22-0.32 with the design of this part, 1.2-1.3 nats of lr^2 per pass on long runs), reshuffling moves
the threshold to 0.18-0.22 (1.7-1.9 nats on long runs, below the 2 nats of linear SAGA), and with extra refreshes
at b >= 20 every rho of the grid is followed. What stands is the measurement of this section and one comparison:
with reshuffling at b = 20 and 50 the two instances follow cooldowns at least as fast as the model does (0.10-0.18
against 0.22) and reach 1e-9 in about its number of passes (9.1-12.2 against 9.1-9.4), while with independent batches
they need about twice the passes of the model (16.7-20.0 against 7.0-8.7). No statement of the B4 note covers the fixed-length iteration, which is not
linear.

On `soft-cond`, where conditioning is the limit, the threshold is 0.61 with independent batches at both
batch sizes, far above the memory factor: at b = 20 reshuffling and extra refreshes move it to 0.5, 0.43 and 0.27,
still far from the memory factors 0.107 and 0.006; at b = 50 nothing improves it (0.61 with extra refreshes, 0.7
with reshuffling); one permutation for the whole run is worse at both batch sizes (0.7, 0.8).
This part is not reliable below b = 20: at b = 5 reshuffling gains on `soft-mem` only, and at batch one (digest) the
threshold is not monotone in rho under reshuffling on `soft-mem` (5, 0, 4, 5, 4 seeds of 5 on schedule at rho =
0.61, 0.5, 0.43, 0.368, 0.32) and the deterministic orders track at a degraded level (reference C of 3.52e+05 and
7.16e+05 with one permutation, against 4.24 and 8.94 with independent batches [`schemes/<scheme>/reference_C`]).

**Theory.** F12 to F16 (Section 4) are about this iteration. F13 proves the law when the ages of the table are
bounded, which holds surely for cyclic and reshuffled orders, with a certified cooldown 3.3e4 to 8.9e4 times slower
than the q = 0.8 used here; F14 explains why the law has no intrinsic scale on least squares and why `ns5` has a
floor; nothing explains the observed speed, the value of the critical q on the conditioning-limited instances, or
the effect of momentum. On the memory-limited instances the thresholds are compared with a solvable model in
Section 5.13: at b = 20 and 50 they are within three steps of the grid of those of the model, above them for
independent batches and below them for reshuffling; the thresholds with extra refreshes are not reproduced.

**Limits.**

* *Not specific to Muon at batch one.* Normalized SAGA on the same cooldown does as well or better; the spectral
  map matters at b = 16 and 32, and that comparison covers four conditioning-limited instances, one draw of the
  data each (seed 0), and no memory-limited instance.
* *Two tuned numbers and a cliff in q.* A run that leaves the schedule freezes for good (it does not diverge), and
  the fastest q is next to the cliff. Tuned points lose confirmation seeds in part B on `soft-cond` (7/10),
  `ls-cond` (9/10), `ls-orth` (9/10) and `ls-mem` (9/10 and 8/10), and in part A at b = 16 on `ls-cond` (polar,
  9/10) and `soft-sq` (both maps with Nesterov, 9/10) and at b = 1 with Nesterov momentum (7/10 on `soft-cond`, 8/10
  for `ns5` on `ls-cond`). More runs reach the target and leave the schedule later: on
  `soft-cond` at b = 16, 8/10 runs of the polar map and 5/10 of `ns5` go on to the stop level
  [`rows/<family>/stop_level`], and in part C Muon's recipe at its tuned point keeps 1 of 5 seeds on schedule on
  `soft-cond`. The fixed point keeps all seeds on the four conditioning-limited instances, at the price of more
  passes on `ls-orth` and `soft-sq`.
* *Not the best method of the study.* The covariance-metric B-SAGA of M1 needs 13.5 to 17.5 passes on the six
  instances at batch one.
* *The theory certifies the mechanism only* (F13: a cooldown 3.3e4 to 8.9e4 times slower than the observed one).
* *Grids.* The b = 1 references on 2^(1/4) grids were not run (M1's medians on factor-2 grids are quoted; Section
  5.12 has them for three instances). The controls without momentum select q = 0.95, the largest of the grid, in
  six cells: their numbers are upper bounds. At batch one the two recipes are tuned twice, on the grid of part A
  and on the finer grid of part B, and the selections differ on `ls-orth` and `soft-sq`; the text quotes part B.
  The recipe families were screened at 20% of the budget like the constant-step families (the screening run of a
  geometric cooldown is a prefix of the full run) [`protocol/screening`].
* *Cost.* Passes count gradients. The polar and `ns5` maps need one factorization per step (n / b per pass); the
  two controls need none. Parts C and D use 5 seeds.

**Against the values verified on the prototypes.** The numbers of the verification of 2026-11-18 are reproduced where
the designs coincide (the values quoted for b = 16 and 32 on `soft-cond` and `ls-cond`, the fixed point, the
thresholds with independent batches, and with reshuffling at b = 20 and 50). Differences: (i) with the finer grid of
part B the exact polar factor at batch one needs 17.1 passes on `soft-sq` (prototype and part A: 19.5), and Muon's
recipe on `ls-orth` 16.6 with 9/10 seeds (prototype and part A: 17.2 with all seeds); (ii) the Frobenius control at
batch one on `soft-cond` gives 22.9 on 10 seeds (prototype: 22.1 on 4 seeds); (iii) the law holds deeper than was
reported, down to about 5e-29 for the exact polar factor (reported: about 1e-27) and with a floor near 1e-18 for `ns5`
(reported: about 1e-16); (iv) one threshold with m = 3b is 0.025 (`ls-mem`, b = 5; reported range 0.008-0.018); (v) on
`soft-cond` the threshold was reported as 0.61 whatever the device: this holds for independent batches, and at b = 50
for the extra refreshes; reshuffling gives 0.5 at b = 20 and 0.7 at b = 50, and the extra refreshes at b = 20 give
0.43 and 0.27.

Figure: `figures/figM9_schedule.png` (top: part A; middle: part C; bottom: part D).

### 5.12 M5 at resolved steps (M5b)

(Added 2026-11-19. Script `experiments/exp_M5b_batch_fine.py`. Numbers from `results/M5b_summary.json`, cell =
`instances/<instance>/<data seed>/batches/<b>`, key in brackets, and from the section "M5b" of the digest. The ratios
of this section supersede those of the section "M5" of the digest wherever they differ.)

M5 compared SAGA, B-SAGA and the joint nuclear-scaled SAGA-Muon at batch sizes 1, 4, 16 and 64, each tuned on step
grids of factor 2, and found the orthogonalized step 1.7 to 3.3 times slower than the best Euclidean method at
b >= 16 on `soft-cond` and at b >= 4 on `ls-cond`. That was an artifact of the grid. M5b repeats the comparison (same
instances, budgets, target, families and protocol) on grids of ratio 2^(1/4) centred on the steps selected by M5,
with three controls: the same protocol on the powers of two only ("factor-2 control"), a grid shifted by half a step,
and runs at the selected step times 2^(+-1/8) and 2^(+-1/4) ("window"). Passes to 1e-6, medians over the
confirmation seeds, data seed 0; ratio = SAGA-Muon over the faster of SAGA and B-SAGA [`fine/rows/<family>/summary`,
`fine/ratio`, `fine/boot`, `shifted/ratio`, `factor2/ratio`, `m5/ratio`, `conservative`]:

| instance | b | SAGA | B-SAGA | SAGA-Muon | ratio, 2^(1/4) grid [5%, 95%] | shifted grid | factor-2 control | M5 | conservative: fine, shifted |
|---|---|---|---|---|---|---|---|---|---|
| soft-cond | 1 | 49.8 | 27.1 | 24.6 | 0.91 [0.88, 0.96] | 0.92 | 0.86 | 0.86 | 0.91, 0.97 |
| soft-cond | 4 | 58.7 | 28.6 | 21.2 | 0.74 [0.73, 0.75] | 0.81 | 0.69 | 0.69 | 0.75, 0.81 |
| soft-cond | 16 | 82.2 | 31.0 | 23.2 | 0.75 [0.68, 0.79] | 0.73 | 2.69 | 2.71 | 0.77, 0.73 |
| soft-cond | 64 | 163.2 | 57.6 | 37.8 | 0.66 [0.61, 0.68] | 1.04 | 3.33 | 3.32 | 0.70, 1.04 |
| ls-cond | 1 | 83.5 | 31.9 | 33.3 | 1.04 [1.01, 1.08] | 0.96 | - | - | 1.04, 0.98 |
| ls-cond | 4 | 117.0 | 34.2 | 25.4 | 0.74 [0.72, 0.76] | 0.72 | 2.04 | 2.04 | 0.75, 0.72 |
| ls-cond | 16 | 232.6 | 51.2 | 28.0 | 0.55 [0.52, 0.61] | 0.48 | 1.78 | 1.74 | 0.56, 0.55 |
| ls-cond | 64 | not reached | 181.8 | 50.6 | 0.28 [0.23, 0.34] | 0.42 | - | - | 0.28, 0.42 |
| ls-orth | 1 | 27.1 | 17.7 | 22.9 | 1.30 [1.25, 1.32] | 1.48 | 1.53 | 1.53 | 1.30, 1.49 |
| ls-orth | 4 | 31.7 | 18.2 | 20.2 | 1.11 [1.04, 1.15] | 1.12 | 1.37 | 1.37 | 1.11, 1.12 |
| ls-orth | 16 | 51.4 | 18.8 | 16.5 | 0.87 [0.85, 0.89] | 0.89 | 0.94 | 0.94 | 0.88, 0.89 |
| ls-orth | 64 | not reached | 28.9 | 20.6 | 0.71 [0.68, 0.75] | 0.64 | 0.52 | 0.52 | 0.73, 0.64 |

* **The control reproduces M5.** On the powers of two the protocol selects M5's point for every family in every
  cell (36 of 36, with confirmation medians within 2.2% of M5's [`factor2_vs_m5/<family>/{same_point, rel_diff}`])
  and returns M5's ratios: the difference between the columns "2^(1/4) grid" and "M5" is the grid, not the code or
  the machine.
* **At resolved steps the polar step is ahead at equal batch for b >= 4 on `soft-cond` and `ls-cond`** (0.74 / 0.75 /
  0.66 and 0.74 / 0.55 / 0.28 at b = 4 / 16 / 64), on a par at batch one (0.91; 1.04), and on `ls-orth` behind at
  b = 1 and 4 (1.30, 1.11) and ahead at b = 16 and 64 (0.87, 0.71). No selected value is on the edge of a grid.
* **Why the factor-2 grid failed: the fast steps end at a cliff** [`window/SAGA-Muon/<j>`]:

| passes of SAGA-Muon at the selected step times | 2^(-1/4) | 2^(-1/8) | 1 | 2^(1/8) | 2^(1/4) |
|---|---|---|---|---|---|
| soft-cond, b = 1 | not reached | 185.7 | 24.6 | 26.4 | 29.0 |
| soft-cond, b = 4 | 121.8 | 22.0 | 21.2 | 22.8 | 24.1 |
| soft-cond, b = 16 | 94.0 | 67.4 | 23.2 | 22.0 | 24.2 |
| soft-cond, b = 64 | 87.7 | 56.5 | 37.8 | 220.8 (6/10) | not reached |
| ls-cond, b = 1 | 43.5 (9/10) | 31.2 | 33.3 | 36.2 | 42.0 |
| ls-cond, b = 4 | 68.8 | 24.1 | 25.4 | 27.2 | 30.3 |
| ls-cond, b = 16 | 175.3 | 104.6 | 28.0 | 27.5 | 32.7 |
| ls-cond, b = 64 | 254.8 | 155.6 | 50.6 | 76.5 | not reached (10 diverged) |
| ls-orth, b = 1 | 25.2 (7/10) | 22.2 (9/10) | 22.9 | 23.9 | 24.9 |
| ls-orth, b = 4 | 36.0 (6/10) | 20.4 | 20.2 | 20.8 | 21.2 |
| ls-orth, b = 16 | 90.3 (7/10) | 20.2 | 16.5 | 16.6 | 16.7 |
| ls-orth, b = 64 | 16.7 | 18.0 | 20.6 | 26.0 | 40.3 |

  On `soft-cond` and `ls-cond` a step 2^(-1/4) below the selected one is 2.3 to 6.3 times slower for b >= 4
  [`window/SAGA-Muon/-2/over_selected`] and loses seeds at b = 1; above the selected step the cost is at most 26% up
  to b = 16, and at b = 64 the window closes on both sides (seeds lost or divergence at 2^(1/8) or 2^(1/4)). B-SAGA
  at the same offsets of its own step stays within 0.97 to 1.37 of its selected median for b <= 16 (at b = 64: 1.70
  at 2^(1/4) on `soft-cond`, divergence above the selected step on `ls-cond`) [`window/B-SAGA/<j>/over_selected`]. A
  grid of factor 2 that misses the fast steps reports the slow regime below them.
* **Robustness.** The shifted grid and two other draws of the data (b = 16 and 64 on `soft-cond` and `ls-cond`;
  fine grid and control only) [`<data seed>/batches/<b>/{fine, factor2, conservative}`]:

| other draws of the data: ratio on the 2^(1/4) grid (factor-2 control) [conservative] | data seed 1 | data seed 2 |
|---|---|---|
| soft-cond, b = 16 | 0.73 (0.73) [0.73] | 0.69 (0.69) [0.69] |
| soft-cond, b = 64 | 0.94 (1.26) [0.94] | 0.73 (1.04) [0.73] |
| ls-cond, b = 16 | 0.62 (1.23) [0.62] | 0.50 (4.94) [0.56] |
| ls-cond, b = 64 | 0.47 (-) [0.47] | 0.47 (-) [0.47] |

  At b = 16 the ratio is 0.69 to 0.75 on `soft-cond` and 0.48 to 0.62 on `ls-cond` over the two grids and the three
  draws; the two lowest values (0.48 on the shifted grid, 0.50 on data seed 2) come from a slower B-SAGA selection,
  and against the best confirmed Euclidean point of the cell they are 0.55 and 0.56 ("conservative").

**Limits.**

* *b = 64 is fragile*: 0.66 to 1.04 on `soft-cond` and 0.28 to 0.47 on `ls-cond` over grids and draws; on `ls-orth`
  the step 2^(-1/4) below the selected one is faster in median (16.7 against 20.6 passes), so the tuning on five
  seeds did not pick the best point and M5's 0.52 becomes 0.71 (0.64 on the shifted grid). The b = 64 ratios should
  not be quoted as results.
* *The polar step still loses on `ls-orth` at b = 1 and 4*, and its selected step sits next to the cliff: slow
  seeds occur at the selected step itself (largest passes over the ten seeds: 112.7 on `soft-cond` at b = 1, 89.3 on
  `ls-cond` at b = 16 [`fine/rows/SAGA-Muon/summary/1e-06/max`]).
* *Passes are not flat in b.* SAGA-Muon needs 23.2 and 37.8 passes at b = 16 and 64 on `soft-cond`, 28.0 and 50.6
  on `ls-cond`: the ratio improves with b because the Euclidean methods degrade faster, and the smallest pass counts
  are at b = 4 to 16.
* *Scope.* Exact polar factor, joint placement, no momentum; passes only (one factorization per step for
  SAGA-Muon: 200, 50, 12.5 and 3.1 per pass on `soft-cond` and `ls-cond` [`factorizations_per_pass`]). "Best
  Euclidean" is the faster of SAGA and B-SAGA: the covariance-metric B-SAGA of M1 (15.1, 16.1 and 13.9 passes at
  batch one on `soft-cond`, `ls-cond` and `ls-orth`, Section 5.11) is not in this comparison and needs fewer passes
  than every cell of the table.
  Other data draws: two, for two instances and two batch sizes, without shifted grid or window. The minibatch rows
  of M1 (`b = r`, factor-2 grids) were not redone.

**Against the values verified on the prototypes.** The seven ratios of the verification of 2026-11-18 and M5's
values are confirmed to the digit, the reproduction of M5 on the powers of two within 2.2%, and so is the fragility
at b = 64. Two
statements are corrected: the fast window was described as "about +-10% wide", which holds at b = 64 only (for
b <= 16 the cliff is on the low side only); and the stability "up to b = 16" was reported with a range 0.55-0.74 on
`ls-cond`, where this run gives 0.48 to 0.74 because the lab protocol selected a slower B-SAGA point in two cells
(conservative values 0.55 and 0.56).

Figure: `figures/figM5b_batch_fine.png` (top: the ratio against the batch size; bottom: the step window).

### 5.13 The sampling-order part of M9 on a sum with identical Hessians (M9b)

(Added 2026-11-19. Script `experiments/exp_M9b_order_model.py`. Numbers from `results/M9b_summary.json`, key in
brackets, and from the section "M9b" of the digest. Tests: `tests/test_order_model.py`.)

Part D of M9 leaves a question that a reader of F3 will ask: F3 says that the mean error of the table decreases by
at most its memory factor per pass, and several thresholds are below that factor. M9b answers on the model in which
nothing but the table limits the iteration, f_i(W) = ||W - y_i||^2 / 2 on K x 1 matrices: the Hessians are identical, the SAGA estimator is
E + mean_i Psi_i - mean_{j in B} Psi_j (errors of the iterate and of the stored points), and its polar factor is the
estimator divided by its norm. This is the model on which the main study solves linear SAGA (hypothesis B4): per
pass, the squared error contracts by exp(-1) in expectation under independent sampling at large steps and by exp(-2)
under reshuffling, and single runs under independent sampling are faster than the expectation.

**Part D of M9 on the model** (n = 1000, K = 10; the same tasks, grids, seeds, horizon and criteria, run by the lab)
[`D/cells/<b>/schemes/<scheme>/rho_star`, `D_vs_M9/<b>/<scheme>`]:

| critical factor rho_star of lr^2 per pass | memory factor | model, b = 1 / 5 / 20 / 50 | soft-mem, b = 1 / 5 / 20 / 50 | ls-mem, b = 1 / 5 / 20 / 50 |
|---|---|---|---|---|
| independent batches | 0.36-0.37 | 0.32 / 0.32 / 0.22 / 0.22 | 0.43 / 0.368 / 0.32 / 0.27 | 0.61 / 0.43 / 0.368 / 0.32 |
| independent batches + m = b extra refreshes | 0.12-0.135 | 0.135 / 0.07 / <= 0.05 / <= 0.05 | 0.22 / 0.135 / 0.135 / 0.1 | 0.27 / 0.18 / 0.18 / 0.135 |
| independent batches + m = 3b extra refreshes | 0.012-0.018 | 0.008 / <= 0.005 / <= 0.005 / <= 0.005 | 0.035 / 0.012 / 0.012 / 0.008 | 0.07 / 0.025 / 0.018 / 0.018 |
| fresh permutation per pass | - | 0.18 / 0.18 / 0.22 / 0.22 | 0.32 / 0.22 / 0.1 / 0.135 | 0.32 / 0.43 / 0.18 / 0.18 |
| one permutation for the whole run | - | 0.135 / 0.27 / 0.18 / 0.18 | 0.18 / 0.18 / 0.18 / 0.18 | 0.27 / 0.368 / 0.32 / 0.368 |

| median passes to 1e-9 at the fastest rho | model, b = 1 / 5 / 20 / 50 | soft-mem, b = 1 / 5 / 20 / 50 | ls-mem, b = 1 / 5 / 20 / 50 |
|---|---|---|---|
| independent batches | 10.5 / 9.4 / 8.7 / 7.0 | 18.2 / 17.7 / 17.9 / 16.7 | 23.3 / 20.7 / 20.0 / 19.1 |
| fresh permutation per pass | 7.1 / 9.1 / 9.4 / 9.1 | 11.2 / 10.4 / 9.1 / 10.4 | 19.8 / 16.7 / 11.3 / 12.2 |

("<=": every seed is on schedule at the smallest rho of the grid. Passes: [`D_vs_M9/<b>/<scheme>/passes_at_fastest_rho`].)

**Long runs at batch one** (80 passes, where the design of part D has 23 / log(1/rho); 20 runs per point; x =
log(1/rho), the nats of lr^2 lost per pass; x_star = largest x of the grid with at least 80% of the runs on schedule)
[`L/cells/n=<n>/orders/<order>/level_end/{x_star, on_schedule, level_over_reference}`, `L/cells/n=<n>/linear_saga`]:

| | n = 100 | n = 1000 | n = 10000 |
|---|---|---|---|
| independent indices: x_star after 80 passes (after 40) | 1.2 (1.2) | 1.3 (1.2) | 1.3 (1.2) |
| fresh permutation per pass: x_star after 80 passes (after 40) | 1.9 (2) | 1.8 (1.7) | 1.7 (1.6) |
| linear SAGA on the model, independent sampling: expectation; best single-run exponent (main study, B4b) | 0.995; 1.546 | 0.999; 1.634 | 1.000; 1.675 |
| linear SAGA on the model, reshuffling: best exact mean-square exponent (same steps) | 1.933 | 1.990 | 1.959 |

What this shows.

* **A single run is not bound by the memory factor.** On the model every threshold with independent batches is
  below the memory factor (0.22-0.32 against 0.36-0.37), and on long runs the recipe follows 1.2 to 1.3 nats per pass
  where the mean memory error loses at most 1: the runs of part D that go below the memory factor do not contradict
  F3. The recipe does not reach the best single-run exponent of linear SAGA on the same model (1.55 to 1.67).
* **With independent batches the instances are less tolerant than the model.** Their thresholds are one to three
  steps of the grid above those of the model for b >= 5 (0.27-0.43 against 0.22-0.32; four steps on `ls-mem` at batch
  one), and they need about twice its passes (16.7-23.3 against 7.0-10.5 over the four batch sizes). The model is the
  favourable case, as it should be: nothing but the table limits it.
* **Reshuffling: the instances catch up with the model.** On the model a fresh permutation per pass moves the
  threshold from 0.32 to 0.18 at b = 1 and 5 and changes nothing at b = 20 and 50 (0.22 both ways); on long runs at
  batch one it moves it from 1.2-1.3 to 1.7-1.9 nats per pass, short of the 2 nats of linear SAGA, and the runs still
  follow the schedule at a degraded level up to 2.1 nats (level 43 to 7e5 times the reference at x = 2.0) before
  they freeze. On the instances at b = 20 and 50 the reshuffled thresholds (0.10-0.18) are one to three steps of the
  grid below that of the model (0.22) and the passes are those of the model (9.1-12.2 against 9.1-9.4); at b = 1 and
  5, where part D is not reliable, they are above it (0.22-0.43 against 0.18). The large gain of reshuffling measured in
  part D is therefore not a property of the table alone: the instances lose more than the model to independent
  sampling, and reshuffling removes that loss. Why they lose more (heterogeneous per-sample Hessians are the
  obvious candidate) is not established.
* **The extra-refresh thresholds are not reproduced.** The model follows every rho of the grid at b >= 20 with m = b
  or 3b extra refreshes, where the instances stop next to exp(-(1 + m/b)). That regularity is measured on two
  instances and is not explained.

**Limits.** One model instance (n = 1000, K = 10, one draw) and 5 seeds for the design of part D; batch one and K =
10 for the long runs. The thresholds depend on the reading (factor 100 on the level; with a factor 10 the long-run
values are 1.2 and 1.4-1.7 [`level_end/x_star_clean`]), on the start (the runs start 4 times closer to W_* than
the total length of their steps, as in part D) and, for independent indices, on the horizon: more runs leave the
schedule between 40 and 80 passes at the largest x followed. The model has no conditioning and its estimator has rank
one: it says nothing on the spectral map, on momentum or on `soft-cond`. The simulator of the long runs is checked
step by step against the lab on the model instance (largest relative difference 2.4e-12 over two passes, printed by
`test_model_run_matches_the_lab`), and the threshold rule reproduces the 50 stored cells of part D of M9
(`test_thresholds_rule_reproduces_part_D_of_M9`).

Figure: `figures/figM9b_order_model.png` (left: share of the long runs on schedule against the speed of the cooldown;
right: largest cooldown followed, log(1/rho_star), on the model and on the two instances, by batch size; both in
nats of lr^2 per pass, the dashed lines at 1 and 2 being the mean memory rate of F3 and the reshuffled rate of
linear SAGA).

<!-- CONCLUSION -->

## Statements of the write-ups affected

(Added 2026-11-19; none of the files below was edited.) Sentences that Sections 5.11 and 5.12 and the facts F12-F16
make wrong or incomplete, with a proposed wording. Line numbers are those of the files on 2026-11-19.
*Update 2026-11-20:* the paper was moved to `paper/` (root of the project), revised and condensed, and the wordings proposed below for
`paper_saga_to_muon.tex` are applied there (its new Section 3.3 reports Sections 5.11-5.13); the other three write-ups
were moved unchanged to `muon/archive/` (since moved to `archive/muon/`), so the rows that concern them stay open; the main
report was not touched (it was archived unchanged in `archive/report/` the same day). The files
searched are `muon/paper_saga_to_muon.tex`, `muon/paper_spectral_muon.tex`, `muon/report_spectral_muon.tex`,
`muon/report_spectral_muon_new.tex` and the main report (`report/main.tex`, `report/synthesis.tex`,
`report/sections/*.tex`).

**M5's ratios, or "orthogonalization loses at larger batches".** No sentence of these files quotes a ratio of M5 or
makes that statement: the four write-ups of the side study discuss batch one only (and minibatches for the noise
floors and the offset of the limit), and the main report does not mention Muon, polar steps or orthogonalization at
all. The ratios of M5 appear in the section "M5" of the digest only, where the section "M5b" that follows says that it
supersedes them. One sentence is incomplete rather than wrong:

| file: line | sentence | proposed wording |
|---|---|---|
| `report_spectral_muon_new.tex`: 606-607 | "For Muon-type steps, finite-sum variance reduction matters less for speed, where Euclidean devices do as well, than for precision" | keep, and add: "(at batch one; with minibatches of 4 to 16 samples on the two instances limited by conditioning, the nuclear-scaled polar step of SAGA needs 0.55 to 0.75 of the passes of the better of SAGA and damped SAGA at the same batch, on step grids of ratio 2^(1/4); the covariance-preconditioned SAGA at batch one stays ahead)" [Section 5.12] |
| `report_spectral_muon_new.tex`: 196-198 | "where conditioning limits, orthogonalization at every step divides the passes of SAGA by 1.3-5 at batch one, as these Euclidean devices do (ratio 0.78-1.11)" | unchanged (batch one, M1); the same addition can be made here |

**"A fixed length cannot converge", "variance reduction does not remove the floor", "rounding precision needs a step
that vanishes with the estimator".** True at a constant length, which is what was tested (Section 5.1); wrong or
incomplete once the length is cooled geometrically (Section 5.11).

| file: line | sentence | proposed wording |
|---|---|---|
| `paper_saga_to_muon.tex`: 103-104 (abstract) | "a finite-sum estimator makes the minimizer the only fixed point and, with a step that vanishes with it, reaches rounding precision in experiments" | "... and reaches rounding precision in experiments when the step length goes to zero, with the estimator or, for steps of fixed length, on a geometric schedule" |
| `paper_saga_to_muon.tex`: 177 (Table 1); `report_spectral_muon_new.tex`: 190 (Table 1) | "rounding precision needs a step that vanishes with the estimator" | "rounding precision needs a step length that goes to zero: with the estimator (nuclear-scaled, clipped) or, for fixed lengths, on a geometric cooldown with a SAGA table" |
| `paper_saga_to_muon.tex`: 612-617; `report_spectral_muon_new.tex`: 372-375 | "Polar steps of fixed length stall at a floor [...] Variance reduction does not remove it: with the SAGA estimator and a fixed length eta the iterates settle at an absolute squared distance 3.8 eta^2 (softmax) or 6.0 eta^2 (least squares)" (new report: "variance reduction makes the floor quadratic in the length [...] but does not remove it") | "Polar steps of *constant* length stall at a floor [...] At a constant length variance reduction does not remove it: [...]" |
| `paper_saga_to_muon.tex`: 620-621; `report_spectral_muon_new.tex`: 376-378 | "Only a finite-sum estimator with a step that vanishes with it (nuclear-scaled or clipped length; on least squares also the soft map with a fixed epsilon) reaches rounding precision" | "Only a finite-sum estimator reaches rounding precision, with a length that vanishes with it (nuclear-scaled or clipped length; on least squares also the soft map) or with a fixed length cooled geometrically: with the SAGA table and the exact polar factor, lr_t = lr_0 q^passes at batch one reaches a relative squared distance of 2e-29 to 4e-29 on the two instances limited by conditioning, where the same schedule on a minibatch gradient or a momentum-based buffer freezes at 0.08 to 0.93" [Section 5.11, part C] |
| `paper_saga_to_muon.tex`: 657 (Table 2) | "Finding 2: with a step that vanishes with the estimator, rounding precision" | "Finding 2: rounding precision with a step that vanishes with the estimator, or with fixed lengths on a geometric cooldown" |
| `paper_spectral_muon.tex`: 540-541; `report_spectral_muon.tex`: 378-379 | "a length floor for constant lengths (quadratic in the length, removed by a nuclear-scaled or decreasing length)"; "besides the floor of constant step lengths" | consistent with Section 5.11; no change needed |

**Prior work and proofs** (outside the three statements asked for, affected by the check of 2026-11-18 and by
F12-F16).

| file: line | sentence | proposed wording |
|---|---|---|
| `paper_saga_to_muon.tex`: 676-679; `report_spectral_muon_new.tex`: 584-588 | "finite-sum estimators in fixed-length polar steps [Li, Tang, Zou] [...] are published; a SAGA table in the nuclear-scaled polar step at a constant step size [...] were not found by the check of 12 November 2026" | still true; add: "a check of 18 November 2026 found the geometric-schedule law of fixed-length steps for exact gradients (Goffin; Davis et al.; for Muon, Ma et al.) and the descent inequality for an SVRG anchor in a sign or spectral step under reshuffling (Qin, Liu and Xu; Li, Tang and Zou), not the SAGA-table version with a linear rate" |
| `paper_spectral_muon.tex`: 554 and 559-560; `report_spectral_muon.tex`: 386-387 and 392-393 | "(the convergence to it is observed in Figure [...], not proved)"; "No convergence of a variance-reduced polar method is proved: Proposition [...](d) only identifies its fixed point." | "Convergence of a variance-reduced polar method is proved in two restricted settings only: fixed lengths fed by a table whose entries have bounded ages, on a geometric cooldown far slower than the one that works in the experiments, and the nuclear-scaled step at batch one for gamma <= 1/(14 n L_max) (notes_theory, F13 and F15)" |
| `paper_spectral_muon.tex`: 571-572 (open questions) | "a Lyapunov function for a SAGA table driven by nuclear-scaled polar steps" | "a Lyapunov function for a SAGA table driven by nuclear-scaled polar steps at the steps that are used (F15 of notes_theory gives one for gamma <= 1/(14 n L_max), that is 3.574e-4 / R2_op on the two instances limited by conditioning, where the tuned steps are 0.125 / R2_op and 0.5 / R2_op)" |
| `report_spectral_muon_new.tex`: 609-610 | "Open questions are convergence proofs when the polar factor is recomputed at every step, momentum, iterate-dependent minibatch noise, and real data." | "Open questions are convergence proofs at the steps and cooldowns that are used (F13 and F15 of notes_theory hold for much smaller ones), momentum, [...]" |
