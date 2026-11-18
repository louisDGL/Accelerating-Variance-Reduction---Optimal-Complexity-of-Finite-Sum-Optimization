# Structural facts used in the Muon study

Short, self-contained statements with proofs. They are referred to as F1-F11 in [`findings.md`](findings.md).
Several of them are known in some form; the attribution is given with each statement. The main identities and
inequalities are checked numerically by the unit tests named with them (`muon/tests/`); F8 is a reading of a
published proof and has no test.

## Notation

W in R^{K x d}, r = min(K, d), <A, B> = tr(A^T B). Norms: ||.||_F (Frobenius), ||.||_op (largest singular value),
||.||_* (nuclear norm, the dual of ||.||_op: |<A, B>| <= ||A||_* ||B||_op). For M with thin singular value
decomposition M = U diag(s) V^T,

    polar(M) = U_+ V_+^T   (restricted to the nonzero singular values),      sharp(M) = ||M||_* polar(M).

polar(M) maximizes <M, O> over ||O||_op <= 1 and <M, polar(M)> = ||M||_*. The Muon step is W - eta polar(M) with a
scheduled length eta; the dualized steepest-descent step is W - gamma sharp(M) (the "sharp operator" of Nesterov and of
Kelner et al.; stochastic spectral descent of Carlson, Cevher and Carin 2015; PolarGrad of Lau, Long and Su,
arXiv:2505.21799).

Problem: F(W) = (1/n) sum_i g_i(W) + (lam/2) ||W||_F^2 with g_i(W) = l(W x_i; y_i) convex and twice differentiable,
so that g_i'(W) = rho_i(W) x_i^T with rho_i(W) = grad l(W x_i; y_i) in R^K: **every per-sample loss gradient has rank
at most one**. W_* is the minimizer. F is L_op-smooth with respect to the spectral norm:
F(W - D) <= F(W) - <grad F(W), D> + (L_op / 2) ||D||_op^2.

Variance-reduced estimators (single index j uniform in {1..n}; minibatches of size b without replacement are similar)
are written omega_j = m + c_j with a predictable part m and an innovation c_j:

    SAGA    m = zbar + lam W,          c_j = g_j'(W) - z_j,           z_i the stored gradients, zbar their mean
    L-SVRG  m = grad G(w) + lam W,     c_j = g_j'(W) - g_j'(w),       G = (1/n) sum_i g_i, w the reference point

and E_j[c_j] = grad F(W) - m. For batch size one c_j has rank one. The regularizer gradient lam W is exact and belongs
to m.

## F1. Rank-one identity

*For a in R^K, b in R^d nonzero: polar(a b^T) = a b^T / (||a|| ||b||), sharp(a b^T) = a b^T, and
||a b^T||_* = ||a b^T||_F = ||a b^T||_op = ||a|| ||b||.*

Proof. The thin SVD of a b^T is (a/||a||) (||a|| ||b||) (b/||b||)^T. []

Consequences.

(i) With batch size one and no memory, the nuclear-scaled polar step applied to a per-sample loss gradient is the SGD
step, and the fixed-length polar step is normalized SGD: orthogonalization is void. This observation is Lemma H.1 of
Li, Tang and Zou (arXiv:2602.11557: "Spectral-SGD and Normalized-SGD are equivalent when batch size = 1"). It holds
when the regularizer gradient stays outside the orthogonalized matrix (decoupled weight decay). In this lab lam W
belongs to m, so the memoryless estimator (`estimator='sgd'`) at batch one, lam W + rho_j x_j^T, has full rank at a
generic W (it is rank one for every index only in degenerate cases such as W = 0). The spectral structure of a
single-index method can only come from what it aggregates: a momentum buffer, a minibatch, the regularizer, or the
memory of a variance-reduced estimator (the predictable part m, a full-rank matrix). For minibatch gradients without
memory the corresponding statement is that the gradient has rank min(K, d, b); it is the starting point of the
batch-size analysis of Muon by Paquette et al. (arXiv:2605.09552), where momentum supplies the missing rank.

(ii) The dual-norm second moment of a single-index innovation equals its Euclidean one,
E_j ||c_j||_*^2 = E_j ||c_j||_F^2, and the error of the estimator satisfies

    E_j ||omega_j - grad F(W)||_*^2  <=  4 E_j ||omega_j - grad F(W)||_F^2  <=  4 E_j ||c_j||_F^2.

Proof. omega_j - grad F = c_j - E_i c_i = E_i[c_j - c_i], and c_j - c_i has rank at most 2, so
||c_j - c_i||_* <= sqrt(2) ||c_j - c_i||_F. By Jensen ||omega_j - grad F||_*^2 <= 2 E_i ||c_j - c_i||_F^2, and
E_{i,j} ||c_j - c_i||_F^2 = 2 Var_j(c_j) = 2 E_j ||omega_j - grad F||_F^2. The last inequality is "variance <= second
moment". []

The factor sqrt(r) or r by which analyses of LMO methods convert a Euclidean variance into a nuclear-norm variance is
therefore not incurred at batch size one; only the constant 4 remains: 2 from the rank of c_j - c_i and 2 from
E ||c_j - c_i||_F^2 = 2 Var_j(c_j). It is sharp as r grows (c_j = e_j e_j^T, n = r: the ratio to the variance is
4 (1 - 1/n)). The triangle-inequality route, E ||xi - E xi||_*^2 <= 2 E ||xi||_*^2 + 2 ||E xi||_*^2, gives the same
constant against the second moment only; it is what Section 6.2 of the Katyusha paper pays (constants 8 L instead of
2 L).
(Tests: `test_maps_values`, `test_F1_dual_norm_error_at_batch_one`.)

## F2. A step of fixed spectral length cannot converge

*If W_{t+1} = W_t - eta_t O_t, then max(||W_t - W_*||_op, ||W_{t+1} - W_*||_op) >= eta_t ||O_t||_op / 2, and
max(||W_t - W_*||_F, ||W_{t+1} - W_*||_F) >= eta_t ||O_t||_F / 2 (= eta_t sqrt(rank O_t) / 2 for an exact polar
factor).*

Proof. eta_t ||O_t|| = ||W_{t+1} - W_t|| <= ||W_{t+1} - W_*|| + ||W_t - W_*|| in either norm. []

For exact gradients this is known: it is Lemma 4 of Jawanpuria, Garai and Mishra (arXiv:2610.27747), including the
version for Newton-Schulz maps; Gonon, Musat and Boumal (arXiv:2602.11948) show that exact polar steps of constant
length stay on a lattice and end in a 2-cycle on strongly convex quadratics; PolarGrad (arXiv:2505.21799) isolates the
missing property as "null-gradient consistency" (its Definition 3.2) and proves the constant-step floor of matrix sign
descent (Theorem 3.4) and the linear rate of the nuclear-scaled step (Theorem 3.2). For variance-reduced estimators
the nearest published statements are in other geometries: Chzhen and Schechtman (arXiv:2305.13187, Section 4.2)
argue with sign descent on x^2 / 2 that a linear rate cannot be expected from sign + SVRG at a fixed step, and say
that this is not a proof; SPIDER (Fang et al., arXiv:1807.01695, Algorithm 1) pairs a recursive finite-sum estimator
with a normalized step whose fixed length is tied to the target accuracy (Option I) or with a clipped length
(Option II). What matters here is that the inequality involves no property of O_t: it holds for any estimator. With eta_t = eta constant the iterates stay at
distance at least eta / 2 from W_* every other iteration: a variance-reduced estimator cannot remove this floor; only
the length rule can (a length that vanishes with the estimator, a decreasing schedule, or a map with bounded gain near
zero). For exact gradients a geometric decay of the fixed length is the published remedy (Ma et al.,
arXiv:2601.13474; Rhee et al., arXiv:2606.30509); it is not tested in this study. For an approximate orthogonalization only the largest singular value of O_t matters (for the five-step
Newton-Schulz map it stays in about [0.70, 1.20], while small singular values can remain far below one). In Frobenius
norm the bound reads rank(O_t) eta^2 / 4 for the squared distance, i.e. 2.25 eta^2 (softmax, rank 9) and 2.5 eta^2
(least squares, rank 10) on the two instances of experiment M3: the order of the measured stationary levels.
(Test: `test_F2_fixed_length_floor`.)

## F3. The memory error contracts by at most 1 - b/n per iteration, whatever the step rule

*Let M_t = (1/n) sum_i ||z_i^t - g_i'(W_*)||_F^2 be the error of the SAGA memory and
S_t = (1/n) sum_i ||g_i'(W_t) - g_i'(W_*)||_F^2. If the batch B_t is a uniformly random subset of size b and the
entries of B_t are replaced by the gradients at W_t, then, for every rule producing W_{t+1},*

    E[M_{t+1} | past] = (1 - b/n) M_t + (b/n) S_t  >=  (1 - b/n) M_t.

Proof. Entry i is replaced with probability b/n, in which case its term becomes ||g_i'(W_t) - g_i'(W_*)||_F^2. []

Hence E[M_t] >= (1 - b/n)^t M_0: per pass (n/b iterations, n gradient evaluations) the *mean* memory error decreases
at most by the factor (1 - b/n)^(n/b), which is at most e^{-1} and close to it when b << n (0.30 for b/n = 0.32). No
choice of geometry, map or step length changes this. The bound is a statement on the mean, and the mean is carried by
the entries that have not been refreshed yet (probability (1 - b/n)^t each): once every entry has been refreshed
(about ln n passes), a single run can fall below M_0 (1 - b/n)^t. With Euclidean SAGA on the two memory-limited
instances of the study (n = 1000, steps 0.25 / R^2 and 0.5 / R^2, three seeds each), the per-pass factor of M_t
between passes 6 and 16 lies between 0.30 and 0.46 depending on the seed, around e^{-1} = 0.37, and the squared
distance follows the same factors. The statement concerns the SAGA table only: an L-SVRG reference point obeys no
such recursion (after a refresh its error is S at the reference point), and Hannah, Liu, O'Connor and Yin (2018) show
that SVRG-type methods can indeed beat n log(1/eps) when n >> kappa while methods of the SAGA / SDCA type cannot
(pointer: abstract only, not checked here).

What F3 implies for the iterate. One step, batch size one: E_j ||W_{t+1} - W_*||_F^2 >= Var_j(S_j), and when the
innovation enters the step linearly, S_j = P + gamma_c theta c_j (split placement; Euclidean placement with
gamma_c = gamma),

    Var_j(S_j) >= gamma_c^2 theta^2 ( sqrt(V_t) - sqrt(S_t) )_+^2,      V_t = M_t - ||zbar_t - grad G(W_*)||_F^2

(c_j = d_j - e_j with d_j = g_j'(W_t) - g_j'(W_*) and e_j = z_j - g_j'(W_*); Var(c) >= (sqrt(Var e) - sqrt(Var d))^2,
Var e = V_t, Var d <= S_t). This is not a rate statement (gamma_c theta may be small, and a common offset of the table
changes M_t but not omega_j). That the iterate error and the memory error decay at the same rate is an *observation*
on the runs of the study, not a theorem. We call *memory-limited* the regime where the memory, not the conditioning,
sets the rate (per-pass factor of the order of e^{-1}; R^2 / mu_true of the order of n or smaller), and
*conditioning-limited* the opposite regime.
(Tests: `test_F3_memory_recursion`, `test_F3_one_step_variance`; trace `mem` of `vrmuon.vr_muon`; experiment M1.)

## F4. Recursion of the predictable part; the surrogate behind it

*For the SAGA memory, any step S_t (W_{t+1} = W_t - S_t) and any batch B_t with innovation c_t,*

    m_{t+1} = m_t - lam S_t + (b/n) c_t.

Proof. zbar_{t+1} = zbar_t + (1/n) sum_{j in B_t} (g_j'(W_t) - z_j^t) = zbar_t + (b/n) c_t and W_{t+1} = W_t - S_t. []

m_t is the gradient at W_t of the surrogate h_t(W) = (1/n) sum_i [g_i(phi_i) + <z_i, W - phi_i>] + (lam/2) ||W||_F^2
(phi_i the point where entry i was last refreshed), a lower bound of F by convexity, which is an *isotropic*
quadratic: its Hessian is lam times the identity, its minimizer is W_B = - zbar / lam, and ||W - W_B||_F = ||m||_F / lam.

Two consequences.

(i) For a frozen table, m depends on W isotropically (dm/dW = lam Id): the curvature of the losses enters m only
through the table, that is through past innovations (m = grad F(W) when the table is fresh). A step that only uses m
with a frozen table "sees" a perfectly conditioned quadratic, for which orthogonalization brings nothing (the plain
step m / lam reaches W_B).

(ii) The Euclidean member of the family with damping theta = b/n (the SAG direction) and step gamma = 1 / lam gives
m_{t+1} = 0 for all t: its iterate is the minimizer of the current surrogate. This is MISO with lower-bound
surrogates (Mairal, SIAM J. Optim. 2015, eq. (3.8) and Proposition 3.4, guaranteed for n >= 2 R^2 / lam), i.e. the
Finito update with alpha = 1 (the analysis of Finito, Defazio, Domke and Caetano 2014, takes alpha = 2, a different
iterate), and formally dual-free SDCA at the excluded endpoint eta lam n = 1 (see the relaxed table below). The
relations between SAG, SAGA, Finito / MISO and SDCA are described in Section 2 of the SAGA paper (Defazio, Bach and
Lacoste-Julien 2014). Lu and Freund (arXiv:1807.07680) give the published analogue on the LMO side: with the
regularizer R = (lam/2) ||.||^2 their generalized LMO, argmin_s <d, s> + R(s), is W_B = -zbar / lam for a SAG-type
memory, and their Lemma 2.1 identifies the method with a randomized dual coordinate method.

The orthogonalized steps of the joint placement act on m through polar(m + c_j): outside the direction of the
rank-one innovation they move W along the polar factor of (the rest of) m with a length set by ||omega_j||_*, which
drives m towards zero; experiment M2 measures how close to W_B each method stays (`base_ratio`, `surrogate_gap` in
exact.py), experiment M6 compares the trajectories with MISO.
(Tests: `test_sag_limit_and_memory`, `test_F4_surrogate`.)

### F4 with a relaxed table

If the sampled entries are moved by a fraction delta towards the fresh gradients instead of being replaced
(`mem_relax`), F4 reads m_{t+1} = m_t - lam S_t + delta (b/n) c_t, and the Euclidean member theta = delta b/n,
gamma = 1/lam keeps m = 0: it is dual-free SDCA with uniform sampling and step eta = delta / (lam n)
(Shalev-Shwartz, "SDCA without Duality", arXiv:1502.06177, Theorem 1: eta <= 1 / (L + lam n) with L the smoothness of
the losses, i.e. delta <= lam n / (L + lam n) < 1; `mem_relax='sdca'` uses R^2 = L + lam in place of L, slightly
conservative). The ICML 2016 version of that paper (arXiv:1602.01582) samples non-uniformly and requires
eta <= min(1/(4 L), 1/(4 lam n)), a smaller range. The same update is MISO-Prox (Lin, Mairal and Harchaoui 2015,
Section 4.2). With a relaxed table F3 becomes

    E[M_{t+1} | past] >= (1 - (b/n) delta (2 - delta)) M_t - 2 (b/n) delta (1 - delta) sqrt(M_t S_t),

so that, when S_t << M_t, the mean memory error contracts by at most about exp(-delta (2 - delta)) per pass (0.84 for
delta = 1/11): a relaxed table is stable for any conditioning but slow. The cross term of the bound can be negative
in practice, and it is again a statement on the mean: for large delta single runs go below this value (0.38 against
0.47 per pass at delta = 0.5 on a small instance, n = 200); at delta = 1/11 the two agree (0.837 against 0.841).
(Tests: `test_relaxed_memory_and_miso`, `test_F3_memory_recursion`.)

## F5. What a nuclear-scaled step guarantees pathwise

*Let omega be any nonzero matrix, O = polar(omega), gamma > 0 and W_+ = W - gamma ||omega||_* O. Then*

    F(W_+) <= F(W) - gamma ||omega||_* [ (1 - gamma L_op / 2) ||omega||_* - ||omega - grad F(W)||_* ].

Proof. Smoothness in the spectral norm with ||O||_op = 1 gives
F(W_+) <= F(W) - gamma ||omega||_* <grad F(W), O> + (L_op/2) gamma^2 ||omega||_*^2, and
<grad F(W), O> = <omega, O> - <omega - grad F(W), O> >= ||omega||_* - ||omega - grad F(W)||_*. []

With omega = grad F(W), F5 is the deterministic guarantee F(W_+) <= F(W) - gamma (1 - gamma L_op/2) ||grad F||_*^2.
With an estimator, the decrease is guaranteed only when the nuclear-norm error of the estimator is smaller than
(1 - gamma L_op / 2) ||omega||_*. The key step, <grad F, polar(omega)> >= ||omega||_* - ||omega - grad F||_*
(hence >= ||grad F||_* - 2 ||omega - grad F||_*), is standard; it is for instance Lemma 1 of SUDA-Muon (Zhang et
al., arXiv:2604.23980). The descent steps of the published analyses of stochastic LMO methods are of this
type: the error of the matrix fed to the LMO enters through its dual norm, and the unbiasedness of that matrix is not
used (checked here in the proof of Katyusha2, Lemma E.1 of the Katyusha paper; Pethick et al., arXiv:2502.07529, put
it as "the lmo of an unbiased gradient is not unbiased"). Unbiased stochastic gradients are still used in these
proofs, to bound the error of the momentum buffer. They therefore need momentum, large batches or MVR corrections to
make the *relative* dual-norm error small. Experiment M2 measures E_j ||omega_j - grad F||_* / ||grad F||_* for the
single-index SAGA estimator along trajectories: it is of order one or larger. Inequalities of this type cannot
certify the single-index methods of this study, which nevertheless converge linearly in the experiments; a proof
would have to use the unbiasedness of the innovation, which is exactly what the nonlinearity of sharp prevents
(E_j sharp(omega_j) != sharp(E_j omega_j)). A published device that restores unbiasedness for the sign map is the
random dither of SignSVRG (Chzhen and Schechtman, arXiv:2305.13187); no analogue is known to us for the polar factor.
(Tests: `test_F5_pathwise_descent` for stochastic estimators, `test_anchor_step` for exact gradients.)

## F6. Conditional expectation of the split and tangent steps

*Let T be the step map (T(M) = gamma sharp(M) for the nuclear-scaled rule) and omega_j = m + c_j an estimator with
E_j[c_j] = grad F(W) - m (SAGA, L-SVRG). For the split placement S_j = T(m) + gamma_c theta c_j and for the tangent
placement S_j = T(m) + DT(m)[theta c_j],*

    E_j[S_j] = T(m) + gamma_c theta (grad F(W) - m)                (split),
    E_j[S_j] = T(m) + DT(m)[theta (grad F(W) - m)]                (tangent).

*For theta = 1 the tangent expectation equals T(grad F(W)) up to a remainder of second order in grad F(W) - m,
provided the whole segment [m, grad F(W)] consists of full-rank matrices (T is real-analytic there).*

Proof. T(m) is known given the past and both expressions are linear in c_j. The last claim is Taylor's formula. []

The constant of the remainder is of order ||m||_* / s_min^2 (F7): "second order" is informative only when
||grad F(W) - m|| << s_min(m), and the clipped length rule is not differentiable at its kink. The joint placement
has no such formula: E_j[T(m + c_j)] is a genuinely nonlinear expectation, computed exactly in this study by
enumerating the n indices (exact.py). A PAGE estimator is biased given the past and is not covered.
(Tests: `test_split_expectation`, `test_tangent_expectation_and_order`.)

## F7. Gain of the derivative of the polar factor

*Let M in R^{K x d}, K <= d, have full rank with singular values s_1 >= ... >= s_K > 0, M = U diag(s) V^T, and for a
direction E let A = U^T E V. Then*

    D polar(M)[E] = U ((A - A^T) / (s_i + s_j))_{ij} V^T + U diag(1/s) U^T E (I - V V^T),

*a symmetric operator (for the Frobenius inner product) with eigenvalues 0 (on the symmetric part of A),
2/(s_i + s_j) for i < j (on its antisymmetric part) and, when K < d, 1/s_i (on the component of E outside the row
space of M). Its operator norm (Frobenius to Frobenius) is 1/s_K if K < d and 2/(s_K + s_{K-1}) if K = d >= 2.
Moreover D sharp(M)[E] = <polar(M), E> polar(M) + ||M||_* D polar(M)[E], whose largest eigenvalue is
||M||_* / s_K >= K when K < d.*

Proof. polar(M) = (M M^T)^{-1/2} M. With S = M M^T = U diag(s^2) U^T, dS = E M^T + M E^T = U (A diag(s) + diag(s) A^T) U^T,
and by the Daleckii-Krein formula U^T d(S^{-1/2}) U has entries -(A_ij s_j + s_i A_ji) / (s_i s_j (s_i + s_j)).
Then U^T (d(S^{-1/2}) M + S^{-1/2} E) V has entries
[-(A_ij s_j + s_i A_ji) + A_ij (s_i + s_j)] / (s_i (s_i + s_j)) = (A_ij - A_ji) / (s_i + s_j),
and U^T (d(S^{-1/2}) M + S^{-1/2} E) V_perp = diag(1/s) U^T E V_perp since M V_perp = 0. The spectrum is read off the
formula: E = U (e_i e_j^T - e_j e_i^T) V^T is mapped to 2/(s_i + s_j) times itself, E = U (e_i e_j^T + e_j e_i^T) V^T
to zero, and E = U e_i v^T with v orthogonal to the rows of M to 1/s_i times itself. The formula for sharp follows from
the product rule and d||M||_* = <polar(M), E>; in the same basis its eigenvalues are ||M||_* times those of D polar,
plus K on polar(M) itself. []

The formula is classical in the perturbation theory of the polar decomposition (see Higham, *Functions of Matrices*,
2008, Chapter 8, and the references of arXiv:2606.08388, where it is called the Mathias formula); the proof is
included to keep the note self-contained.

Consequence. The worst direction is a *rotation* of the smallest singular pair: E = u_K v^T with v orthogonal to the
rows of M changes the polar factor, to first order, by ||E||_F / s_K (by 2 ||E||_F / (s_K + s_{K-1}) for a rotation
within the last two pairs when K = d). A perturbation *along* a singular pair, delta u_k v_k^T, changes nothing (it
belongs to the kernel of the derivative, and the factor is unchanged as long as delta > -s_k) and flips the factor
once delta < -s_k. For a generic rank-one innovation c the first-order change is a fraction of ||c||_F / s_K. The
exact polar factor of a variance-reduced estimator is therefore *guaranteed* to be accurate only if the innovation is
small compared with the *smallest* singular value of the matrix at which the map is evaluated (ratio `smin_ratio` of
exact.py), a much stronger requirement than a small relative error in norm; this is also what makes the trajectories
of the exact-polar methods sensitive to rounding (experiment M4) and the tangent placement noisy.
(Tests: `test_frechet_finite_differences`, `test_frechet_structure`, `test_frechet_accuracy_corner_cases`,
`test_F7_worst_direction`.)

## F8. An accelerated template exists in general norms

Allen-Zhu's Katyusha (JMLR 2018, Section 6 and Appendix E, Algorithm "Katyusha2") is stated for smoothness in an
arbitrary norm, for a composite objective f + psi. Its y step is the proximal steepest-descent step in that norm,
argmin_y (9 Lbar / 2) ||y - x||^2 + <g, y> + psi(y), which is the sharp step of length 1/(9 Lbar) when psi = 0; its
z step is a proximal mirror step for a distance-generating function that is strongly convex in the same norm; psi must
be sigma-strongly convex relative to the Bregman divergence; indices are drawn with probabilities proportional to
L_i. In the proof, the unbiasedness of the estimator is used once, in the term E <g, z_k - u> of the mirror step
(Lemma E.4), which is linear in the estimator; the nonlinear y step is handled pathwise with smoothness and Young's
inequality (Lemma E.1, an inequality of the type of F5) and a dual-norm variance bound (Lemma E.2, the constant 4 of
F1). Theorem 6.1 gives the complexity O((n + sqrt(n Lbar / sigma)) log(1/eps)), the Euclidean order with constants
9 Lbar in place of 3 L.

For the spectral norm the steepest-descent step is the nuclear-scaled polar step of this study and ||.||_F^2 / 2 is
an admissible distance-generating function (1-strongly convex with respect to ||.||_op since ||.||_F >= ||.||_op);
with psi = (lam/2) ||.||_F^2, sigma = lam. Because the per-sample smoothness constants are the same in both
geometries up to the regularizer (rank-one Hessian directions), the certified complexity is of the same order as in
the Euclidean case: no gain is expected from the theory. `lab/katyusha.py` implements a loopless variant with the
sharp y step ("Katyusha-Muon"), the regularizer in the estimator instead of a proximal term, and uniform sampling:
it is covered by no theorem (with psi = 0 the paper only has the sublinear Theorem 6.2). The experiments measure what
happens in practice. The template has been instantiated and tested for the l1 norm (ASGCD, Song et al.,
arXiv:1702.07842: l1 steepest-descent y step on a minibatch SVRG estimator, p-norm mirror step); no instance for the
spectral norm was found.

## F9. A predictable direction with an unbiased length (radial placement)

*Let O be a matrix known given the past with ||O||_op <= 1, omega_j an unbiased estimator of grad F(W),
q_j = <omega_j, O>, a = <grad F(W), O> and v = Var_j(q_j). For the step W_+ = W - gamma q_j O,*

    E_j[q_j] = a,        E_j F(W_+) <= F(W) - gamma a^2 + (L_op / 2) gamma^2 (a^2 + v).

Proof. Smoothness of F in the spectral norm along O gives F(W - eta O) <= F(W) - eta a + (L_op/2) eta^2 for every
real eta; take eta = gamma q_j and the expectation over j, with E q_j = a and E q_j^2 = a^2 + v. []

The Euclidean form of this inequality and of its optimal length is known: the best step along an arbitrary direction
d is <grad f, d> / (L ||d||^2), and the best step in expectation along a stochastic gradient is
(1/L) ||grad f||^2 / (||grad f||^2 + sigma^2) (GLyDER, Yehudai et al., arXiv:2311.13877, Corollary 2.2 and
Theorem 2.4; per coordinate, Eq. (7) of Schaul, Zhang and LeCun, arXiv:1206.1106). F9 is the same computation for a
spectral direction and a variance-reduced length.

The inequality holds with no condition on the error of the estimator: the nonlinear map is applied to a predictable
matrix (O = polar(m) in `placement='radial'`) and the sampled index enters only through a scalar, so its
unbiasedness can be used. It is a decrease when gamma < 2 a^2 / (L_op (a^2 + v)); the best gamma,
a^2 / (L_op (a^2 + v)), gives E_j F(W_+) <= F(W) - a^4 / (2 L_op (a^2 + v)): the noise of the scalar enters at order
gamma^2, as for SGD. The price is that the progress is governed by a^2, the squared pairing of the true gradient with
a direction built from the memory: |a| <= ||grad F||_*, and for O = polar(m), a >= ||m||_* - ||grad F - m||_*, so a
large a^2 is *guaranteed* only when the memory is fresh. With a stale memory a can be small or negative (the signed
length then reverses the step and the inequality is unchanged); at SAGA states of the test problems a^2 / (a^2 + v)
ranges from 0.98 down to 3e-4. The statement covers the pure radial step; `placement='radial'` adds it to the
Euclidean SAGA step, E_j[S] = gamma_c grad F + gamma <grad F, O> O. Every placement that is linear in the estimator
has an inequality of the same kind (the metric placement, F10).
(Tests: `test_radial_expectation` for E_j[S], `test_F9_radial_descent` for the inequality, `test_F9_pairing_bounds`
for the bounds on a.)

## F10. A frozen polar metric makes the step linear (metric placement)

*Let D = U diag(s) V^T (thin SVD on the nonzero singular values) have rank r, nu = ||D||_* and
C = V diag(nu / s_k) V^T + r (I_d - V V^T), a symmetric positive definite d x d matrix. Then D C = sharp(D), and for
any unbiased estimator omega_j and any C known given the past the step S_j = gamma omega_j C satisfies
E_j[S_j] = gamma grad F(W) C. In the variables W = What C^{1/2} it is the Euclidean step What - gamma omegahat_j,
omegahat_j = omega_j C^{1/2}, on Fhat(What) = F(What C^{1/2}), whose components fhat_i are R2P-smooth with
R2P = max_i c_l x_i^T C x_i + lam ||C||_op.*

Proof. D C = U diag(s) V^T V diag(nu/s) V^T = nu U V^T. Linearity gives the expectation. grad Fhat(What) =
grad F(W) C^{1/2}, hence What_+ = (W - gamma omega C) C^{-1/2} = What - gamma omega C^{1/2}. For a direction Dh,
the loss part of fhat_i has second derivative hess l(z_i)[Dh C^{1/2} x_i, Dh C^{1/2} x_i] <= c_l ||Dh C^{1/2} x_i||^2
<= c_l (x_i^T C x_i) ||Dh||_F^2, and the regularizer lam ||Dh C^{1/2}||_F^2 <= lam ||C||_op ||Dh||_F^2. []

In the lab D is the anchor reduced to the gauge subspace (r = K - 1 for softmax, min(K, d) otherwise), and the gains
nu / s_k are floored at nu / (f s_1) (`metric_floor` = f). While C is frozen the iteration is, in the variables What,
a Euclidean variance-reduced method on Fhat. For L-SVRG it is the standard estimator (the regularizer terms cancel
in the correction). For SAGA it is SAGA in the form used by the lab: a table of loss gradients, and the exact
gradient lam What C of the regularizer (lam/2) ||What C^{1/2}||_F^2, which is no longer isotropic. F11 gives the rate
that can be certified for it when C is never refreshed. No rate is claimed across the refreshes of the metric (note
B1 of the main project: a non-constant metric breaks the transfer argument). The identity
D ((D^T D)^+)^{1/2} = polar(D) (pseudo-inverse when K < d) is the usual way of writing the polar step as a
gradient-built preconditioner: Eq. (10) of Shulgin et al. (arXiv:2610.30280, left-sided form), Eq. (17) of Ma et al.
(arXiv:2601.13474, right-sided form); it is one-sided Shampoo without accumulation (ASGO, An et al.,
arXiv:2503.20762, states the equivalence with Muon). Here it is used as a lazily refreshed metric inside a
variance-reduced method, as the curvature sketches of PROMISE (Frangella et al., arXiv:2309.02014) or the cached
transform of CacheMuon (arXiv:2606.16371, applied to a momentum, without finite-sum estimator).

*When the polar metric is the Newton metric.* Least squares, K = d, anchor at a point where W - W_* = s Q with Q
orthogonal: grad F = s Q H with H = Sigma_hat + lam I, so that C = tr(H) H^{-1}, Fhat is perfectly conditioned
(Hessian tr(H) Id) and

    kappa_P = R2P / mu_P = d_inf(lam) + lam / lambda_min(H),      d_inf(lam) = max_i x_i^T H^{-1} x_i,

d_inf being n times the largest ridge leverage score, the quantity that governs statistically preconditioned SAGA in
the note B1 (kappa_P is between d_inf and d_inf + 1). Here mu_P is the strong-convexity constant of Fhat, which is
global because the Hessian of a least-squares objective is constant.

Attribution. The identity C proportional to H^{-1} is published: Theorem 1(iii) of Shulgin, Laing, Orvieto and
Richtarik (arXiv:2610.30280) states, for matrix quadratics, that the polar step with exact line search reaches the
minimizer in one step, whatever the conditioning of the Hessian, if and only if the error has orthonormal columns up
to scale in our orientation (K >= d; an orthogonal matrix up to scale when K = d), and its proof is the computation
above; Du and Su (arXiv:2604.01472, Theorem 1 and Corollary 1) have the same mechanism in a Kronecker model. The
instance `ls-orth` of the study is this situation. The bound of max_i x_i^T H^{-1} x_i by an incoherence times an
effective dimension, and the resulting complexity of SAG and SVRG on data whitened by H^{-1/2}, are in Yang, Jin, Zhu
and Lin (arXiv:1408.3115, Lemma 2 and Theorem 5); the additive term lam / lambda_min(H) comes from the convention of
this lab (regularizer outside the table).

*When it is not.* This requires an anchor of that special form and no active floor. (a) For K < d the metric whitens
only the row space of the anchor. (b) A floor f caps the gains at nu / (f s_1); with f = 1 all gains are equal: C is a
multiple of the identity when the reduced anchor has at least d rows (the method is then exactly Euclidean SAGA /
L-SVRG with step c_metric / R^2) and a two-level scaling otherwise. (c) At later anchors W - W_* is no longer a
multiple of an orthogonal matrix: the gradient is small in the directions that have already converged, the metric
gives them the largest gains, and kappa_P can exceed the Euclidean condition number (for exact polar steps this is
Proposition 2 and Theorem 3 of arXiv:2610.30280: the implicit preconditioner of the polar step has the conditioning
of the error). On the instance `ls-orth`
(experiment M7) kappa_P is 42 at W_0 without floor, against 4364 in the Euclidean geometry, but grows with the floor
and with the age of the anchor.
(Tests: `test_polar_metric`, `test_F10_newton_metric`.)

## F11. One-shot polar-metric SAGA: what is certified

Setting: C is a symmetric positive definite d x d matrix fixed before the first index is drawn (in the study, the
polar metric of grad F(W_0) with floor f, computed once: `placement='metric'` with `metric_every` larger than the
run). Batch size one, uniform sampling, theta = 1, full replacement of the table:

    W_t = W_{t-1} - gamma omega_t C,     omega_t = g_j'(W_{t-1}) - z_j + zbar + lam W_{t-1},     z_j <- g_j'(W_{t-1}).

Notation: ||A||_{C^-1}^2 = tr(A C^{-1} A^T) for parameters and ||G||_C^2 = tr(G C G^T) for gradients (the Frobenius
norms of A C^{-1/2} and G C^{1/2}); q_i = x_i^T C x_i; R2P = c_l max_i q_i + lam ||C||_op (F10); mu_P the *global*
strong-convexity constant of F in the metric, <grad F(W) - grad F(W'), W - W'> >= mu_P ||W - W'||_{C^-1}^2 for all
W, W'.

*(a) Equivalence. For the whole run the iteration is SAGA, in the form used by the lab, on
Fhat(What) = (1/n) sum_i l(What xhat_i; y_i) + (lam/2) tr(What C What^T) with xhat_i = C^{1/2} x_i and
What = W C^{-1/2}: the table holds the loss gradients (it is the same n x K array of residuals), and the gradient
lam What C of the regularizer is exact at each step.*

*(b) Rate. Let gamma = c / R2P with c < 1/2, let k > 2 with c (1 + k/2) < 1, and
V_t = ||W_t - W_*||_{C^-1}^2 + k gamma^2 sum_i ||z_i^t - g_i'(W_*)||_C^2. Then*

    E[V_t | past] <= (1 - min{ 2 gamma mu_P (1 - c (1 + k/2)),  (1 - 2/k) / n }) V_{t-1}.

*For c = 1/4 and k = 3 the factor is 1 - min{1/(3n), 3 mu_P / (16 R2P)}, the rate of Proposition 5.9 of the book
with (R^2, mu) replaced by (R2P, mu_P), and with the table initialised at W_0,*

    E ||W_t - W_*||_{C^-1}^2 <= (1 - min{1/(3n), 3 mu_P / (16 R2P)})^t (1 + 3n/16) ||W_0 - W_*||_{C^-1}^2.

Proof. (a) Multiply the update by C^{-1/2} on the right: What_t = What_{t-1} - gamma omega_t C^{1/2}, and
omega_t C^{1/2} = (rho_j(W_{t-1}) - rtab_j) xhat_j^T + (1/n) sum_i rtab_i xhat_i^T + lam What_{t-1} C, where
rho_j(W_{t-1}) = grad l(What_{t-1} xhat_j; y_j) because W x_j = What xhat_j.

(b) Write Delta = W_{t-1} - W_*, G = <grad F(W_{t-1}), Delta> and M = (1/n) sum_i ||z_i - g_i'(W_*)||_C^2 (table
before the step).
(1) ||W_t - W_*||_{C^-1}^2 = ||Delta||_{C^-1}^2 - 2 gamma <omega_t, Delta> + gamma^2 ||omega_t||_C^2, and
E_j omega_t = grad F(W_{t-1}).
(2) omega_t = A_j + B_j with A_j = f_j'(W_{t-1}) - f_j'(W_*) for f_j = g_j + (lam/2) ||.||_F^2, and
B_j = g_j'(W_*) - z_j + zbar + lam W_*. Since (1/n) sum_i g_i'(W_*) = - lam W_*, B_j is Z_j - E_j Z_j for
Z_j = g_j'(W_*) - z_j, so E_j ||B_j||_C^2 <= E_j ||Z_j||_C^2 = M and E_j ||omega_t||_C^2 <= 2 E_j ||A_j||_C^2 + 2 M.
(3) ||A_j||_C is the Frobenius norm of a difference of gradients of fhat_j(What) = f_j(What C^{1/2}), which is
convex and R2P-smooth (F10); by co-coercivity, E_j ||A_j||_C^2 <= R2P <grad Fhat(What_{t-1}), What_{t-1} - What_*>
= R2P G.
(4) Each entry is replaced with probability 1/n, so the conditional expectation of the new memory error is
(1 - 1/n) M + S_g / n with S_g = (1/n) sum_i ||g_i'(W_{t-1}) - g_i'(W_*)||_C^2. The loss part
ghat_i(What) = l(What xhat_i; y_i) is convex and c_l q_i-smooth, so by co-coercivity
S_g <= c_l max_i q_i <(1/n) sum_i (g_i'(W_{t-1}) - g_i'(W_*)), Delta> = c_l max_i q_i (G - lam ||Delta||_F^2)
<= R2P G.
(5) Adding (1)-(4), E[V_t | past] <= ||Delta||_{C^-1}^2 - 2 gamma (1 - gamma R2P (1 + k/2)) G
+ k gamma^2 n M (1 - (1 - 2/k) / n), and G >= mu_P ||Delta||_{C^-1}^2.
(6) With the table at W_0, ||g_i'(W_0) - g_i'(W_*)||_C <= c_l q_i ||W_0 - W_*||_{C^-1}, hence
V_0 <= (1 + k c^2 n) ||W_0 - W_*||_{C^-1}^2. []

Relation to the published statements. Variance reduction in a metric fixed once is published: SAG and SVRG on data
whitened by the covariance (Yang et al., arXiv:1408.3115), SVRG and Katyusha X with a fixed preconditioner and
linear rates in the constants of its norm (Liu, Feng and Yin, arXiv:1905.09734, Theorems 1 and 2), a preconditioner
frozen at its initial value inside SAGA and SVRG (Kereta et al., arXiv:2106.03172), and PROMISE, which updates its
preconditioner once in total for least squares. F11 is the corresponding statement for the SAGA of this lab; what
was not found in the literature is the polar decomposition of one gradient as the source of the metric. The argument
is the proof of Proposition 5.9 of the book run in a fixed
metric, as in the note B1 of the main project (P-SAGA is Euclidean SAGA on a linearly transformed problem). Neither
statement covers the lab's iteration as it stands, because both store the full gradients f_i' in the table. Two
lines change: step (2) centers the loss-part memory, and step (4) uses the co-coercivity of the loss parts. The
shortcut that works in the Euclidean geometry (the loss-part memory error is at most the full-gradient one because
the loss gradients are monotone) fails in a metric: the cross term <(g_i'(W) - g_i'(W_*)) C, W - W_*> has no sign.
Two iterations that the equivalence should not be confused with: SAGA of Proposition 5.9 on the fhat_i (table of
full gradients; its trajectory differs), and the lab's Euclidean SAGA run on the transformed features, whose
regularizer is (lam/2) ||What||_F^2 (another objective, another minimizer). For softmax, C acts on the feature side
and commutes with the class-side projection: the gauge subspace is invariant under the step and (b) applies on it.

What (b) certifies, and what it does not.
(1) C must be fixed before the run. With a refreshed metric nothing is proved.
(2) c < 1/2. Tuned steps above this value are not covered.
(3) The contracting quantity is the C^{-1}-weighted distance; in Frobenius norm the bound is multiplied by cond(C).
(4) mu_P must be global. For least squares the Hessian is constant, mu_P = lambda_min(C^{1/2} (Sigma_hat + lam I)
C^{1/2}), and the certificate can be much better than the Euclidean one: on `ls-orth` at W_0 without floor,
kappa_P = R2P / mu_P = 42 against 4364, i.e. max{3n, 16 kappa_P / 3} = 3n iterations (3 passes) per factor e of the
bound, against 16 kappa / 3 = 23,275 iterations (about 47 passes) for Euclidean SAGA. For softmax the curvature of the losses vanishes at infinity, so
mu_P = lam lambda_min(C) and R2P / mu_P >= R^2 / lam: the theorem never certifies a gain over Euclidean SAGA (the
"no free lunch for global constants" of the note B1). The Hessian of Fhat at W_* gives a local constant, which
describes the asymptotic regime only; this is the constant reported by experiment M7 for softmax, next to the global
one.
(5) kappa_P depends on the anchor and on the floor and can exceed the Euclidean value (F10, experiment M7).
(6) The certified rate is at best 1 - 1/(3n) per iteration, a factor e^{-1/3} per pass: (b) certifies that the
method is limited by its memory when kappa_P is of the order of n or smaller; it does not explain the observed speed
(about e^{-1} per pass, the ceiling of F3).
(Tests: `test_F11_one_shot_equivalence`, `test_F11_lyapunov`.)
