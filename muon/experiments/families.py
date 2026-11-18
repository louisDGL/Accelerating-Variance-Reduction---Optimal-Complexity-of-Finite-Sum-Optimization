"""Catalogue of the method families compared in the Muon experiments.

A family is {'name', 'method', 'base': fixed keyword arguments, 'grid': {parameter: [values]}, 'group', 'status',
'what'}: `method` and the arguments are those of common.run_one; steps are in problem units (common.over / times).
  group   'euclid' (no orthogonalization), 'full' (full-batch), 'stream' (minibatch methods without finite-sum memory),
          'spectral' (orthogonalized finite-sum variance reduction)
  status  'sota'  published method, or a minor variant of one (reference and difference in `what`)
          'lab'   the two orthogonalized variance-reduced methods already in muon/lab before this study
                  (saga_muon and lsvrg_muon with the nuclear-scaled step)
          'new'   configuration introduced in this study (see muon/findings.md for the closest published work)
The grids are geometric in the step constants; `tune_and_confirm` reports when the selected value sits on the edge of
its grid.

History of the grids (the cached runs of the earlier passes are reused, see common.run_tasks). Passes 1 to 3 widened
the grids in the directions flagged by edge selections. Pass 4 (this version) follows an adversarial review of the
protocol: (a) Euclidean controls with importance sampling (IS-SAGA, IS-B-SAGA) and their orthogonalized counterparts;
(b) fine grids for the deterministic full-batch methods, whose optimum sits just below their stability limit, and two
stronger full-batch references (accelerated gradient, L-BFGS); (c) a finer damping grid, shared by B-SAGA and damped
SAGA-Muon, with theta = 1/n (the SAG direction) in both; (d) the dual-free-SDCA fraction in the MISO grid;
(e) Katyusha-Muon on the same (L, mu) grid as L-Katyusha; (f) the polar-metric families restricted to an active metric
(metric_floor < 1: with metric_floor = 1 the metric is isotropic on the row space of the anchor and the family
contains SAGA / L-SVRG; the earlier passes, which included this value, selected it on most instances, see
results/M1_pass3_summary.json); (g) the previous-batch variant of MARS-M and a damped minibatch control.
Pass 5 adds the one-shot polar-metric families (metric of the full gradient at W_0, never refreshed), motivated by
the conditioning computed in experiment M7, and their controls preconditioned by the feature covariance.
"""
from __future__ import annotations

from common import over, times

THETAS = [0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03]      # damping grid shared by B-SAGA and damped SAGA-Muon
ONE_SHOT = 10 ** 9                                         # metric_every of a metric that is never refreshed


def _g(unit, lo, hi, step=1, den=1):
    """Geometric grid 2^(k/den), k = lo..hi (every `step`), of step constants in the unit `unit`."""
    return [over(unit, 2.0 ** (k / den)) for k in range(lo, hi + 1, step)]


def catalogue(n, quick=False):
    """List of families for a problem with n samples (theta = 1/n is the SAG direction)."""
    st = 2 if quick else 1
    thetas = [0.3, 1.0 / n] if quick else THETAS + [1.0 / n]
    fams = [
        # ---------------- Euclidean references ----------------------------------------------------------------
        dict(name="SAGA", method="vr", group="euclid", status="sota",
             what="Euclidean SAGA (Defazio, Bach, Lacoste-Julien 2014; Bach Prop. 5.9), loss-part memory",
             base=dict(estimator="saga", placement="euclid"), grid=dict(gamma=_g("R2", -4, 5, st))),
        dict(name="B-SAGA", method="vr", group="euclid", status="sota",
             what="Euclidean SAGA with damped innovation theta < 1 (SAG-SAGA interpolation: Driggs et al. B-SAGA, "
                  "Morin-Giselsson SVAG; theta = 1/n is SAG)",
             base=dict(estimator="saga", placement="euclid"),
             grid=dict(theta=thetas, gamma=_g("R2", -2, 8, st))),
        dict(name="IS-SAGA", method="vr", group="euclid", status="sota",
             what="Euclidean SAGA with importance sampling p_i ~ L_i^a, a in {1/2, 1} (non-uniform sampling: Xiao-Zhang "
                  "2014, Schmidt et al. 2015)",
             base=dict(estimator="saga", placement="euclid"),
             grid=dict(sampling=[0.5, 1.0], gamma=_g("R2", -2, 8, st))),
        dict(name="IS-B-SAGA", method="vr", group="euclid", status="sota",
             what="damped Euclidean SAGA with importance sampling p_i ~ L_i^(1/2) (combination of two published "
                  "ingredients)",
             base=dict(estimator="saga", placement="euclid", sampling=0.5),
             grid=dict(theta=[0.6, 0.3, 0.2, 0.1, 0.05] if not quick else [0.3], gamma=_g("R2", -1, 9, st))),
        dict(name="L-SVRG", method="vr", group="euclid", status="sota",
             what="Euclidean loopless SVRG (Kovalev et al. 2020)",
             base=dict(estimator="lsvrg", placement="euclid"), grid=dict(gamma=_g("R2", -4, 6, st))),
        dict(name="PAGE", method="vr", group="euclid", status="sota",
             what="Euclidean PAGE / loopless SARAH (Li et al. 2021), batch 1, p = 1/(n + 1)",
             base=dict(estimator="page", placement="euclid"), grid=dict(gamma=_g("R2", -4, 6, st))),
        dict(name="L-Katyusha", method="katyusha", group="euclid", status="sota",
             what="loopless Katyusha (Kovalev et al. 2020); L = R2 / c_L, mu = c_mu lam",
             base=dict(ystep="coupling"),
             grid=dict(L=[times("R2", c) for c in ((1.0, 0.25) if quick else
                                             (2.0, 1.0, 0.5, 0.25, 0.125, 1.0 / 16, 1.0 / 32, 1.0 / 64))],
                       mu=[times("lam", c) for c in ((1.0, 10.0) if quick else
                                               (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0))])),
        dict(name="MISO", method="vr", group="euclid", status="sota",
             what="MISO with lower-bound surrogates (Mairal 2015; the Finito update with alpha = 1): iterate at the "
                  "minimizer of the surrogate, i.e. the SAG direction with step 1 / lam; here with the relaxation "
                  "delta of the table (delta = 1: MISO; delta < 1: dual-free SDCA, Shalev-Shwartz 2015; 'sdca' = "
                  "lam n / (R2 + lam n), its largest admissible value)",
             base=dict(estimator="saga", placement="euclid"),
             grid=dict(miso_delta=[1.0, 0.5, 0.25, 0.1, 0.03, 0.01, "2sdca", "sdca", "sdca/2"])),
        dict(name="cov-metric SAGA", method="vr", group="euclid", status="sota",
             what="SAGA preconditioned by the feature covariance, C proportional to (c_l Sigma_hat + lam I)^-1 with "
                  "floored gains: the statistically preconditioned SAGA of the note B1 of the main project (the "
                  "Newton metric for least squares), with the loss-part memory; control for the polar metric (the "
                  "covariance, d^2 n operations, is not charged)",
             base=dict(estimator="saga", placement="metric", metric_source="cov"),
             grid=dict(metric_floor=[0.0, 0.001, 0.01, 0.1, 0.3] if not quick else [0.0, 0.01],
                       c_metric=[2.0 ** k for k in range(-4, 4, st)])),
        dict(name="cov-metric B-SAGA", method="vr", group="euclid", status="sota",
             what="damped SAGA (theta < 1) preconditioned by the feature covariance (combination of two published "
                  "ingredients); control for the one-shot polar-metric B-SAGA",
             base=dict(estimator="saga", placement="metric", metric_source="cov"),
             grid=dict(theta=[0.6, 0.3, 0.1] if not quick else [0.3],
                       metric_floor=[0.0, 0.001, 0.01, 0.1] if not quick else [0.01],
                       c_metric=[2.0 ** k for k in range(-4, 5, st)])),
        # ---------------- full-batch references ---------------------------------------------------------------
        dict(name="GD", method="gd", group="full", status="sota", what="gradient descent, step c / L_F",
             base={}, grid=dict(step=_g("L_F", 0, 24 if not quick else 8, 1, 4))),
        dict(name="AGD", method="agd", group="full", status="sota",
             what="Nesterov's accelerated gradient for strongly convex objectives, constant momentum; L = L_F / c, "
                  "mu = c_mu lam",
             base={},
             grid=dict(L=[times("L_F", 2.0 ** -k) for k in range(0, 7 if not quick else 3)],
                       mu=[times("lam", c) for c in ((1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
                                               if not quick else (1.0, 30.0))])),
        dict(name="L-BFGS", method="lbfgs", group="full", status="sota",
             what="limited-memory BFGS (scipy L-BFGS-B, strong-Wolfe line search), one pass per evaluation of "
                  "(F, grad F); memory m",
             base={}, grid=dict(memory=[5, 10, 20] if not quick else [10])),
        dict(name="PolarGrad", method="muon_full", group="full", status="sota",
             what="full-batch nuclear-scaled polar step (stochastic spectral descent, Carlson et al. 2015; dualized "
                  "steepest descent, Bernstein-Newhouse; PolarGrad, Lau-Long-Su), step <G, O> c / L_op",
             base=dict(rule="sharp"),
             grid=dict(L=[times("L_op", 2.0 ** (-k / 8.0)) for k in range(-8, 49 if not quick else 9)])),
        # ---------------- orthogonalized finite-sum variance reduction ----------------------------------------
        dict(name="SAGA-Muon (joint)", method="vr", group="spectral", status="lab",
             what="nuclear-scaled polar step of the SAGA estimator: W - gamma ||omega||_* polar(omega)",
             base=dict(estimator="saga"), grid=dict(gamma=_g("R2_op", -9, 4, st))),
        dict(name="L-SVRG-Muon (joint)", method="vr", group="spectral", status="lab",
             what="nuclear-scaled polar step of the loopless-SVRG estimator",
             base=dict(estimator="lsvrg"), grid=dict(gamma=_g("R2_op", -10, 0, st))),
        dict(name="damped SAGA-Muon", method="vr", group="spectral", status="new",
             what="joint placement with the damped estimator m + theta c (theta = 1/n: polar factor of the SAG "
                  "direction)",
             base=dict(estimator="saga"),
             grid=dict(theta=thetas, gamma=_g("R2_op", -6, 5, st))),
        dict(name="mean-scaled SAGA-Muon", method="vr", group="spectral", status="new",
             what="direction polar(omega), length gamma <m, polar(omega)>_+ (pairing with the predictable part)",
             base=dict(estimator="saga", pair_with="mean"), grid=dict(gamma=_g("R2_op", -7, 6, st))),
        dict(name="soft SAGA-Muon", method="vr", group="spectral", status="new",
             what="joint placement with the bounded-gain map s / sqrt(s^2 + eps^2), eps = eps_innov ||theta c||_F",
             base=dict(estimator="saga", orth="soft"),
             grid=dict(eps_innov=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], gamma=_g("R2_op", -4, 6, st))),
        dict(name="IS SAGA-Muon (joint)", method="vr", group="spectral", status="new",
             what="joint placement with importance sampling p_i ~ L_i^a of the index (weighted innovation)",
             base=dict(estimator="saga"),
             grid=dict(sampling=[0.5, 1.0], gamma=_g("R2_op", -6, 5, st))),
        dict(name="IS soft SAGA-Muon", method="vr", group="spectral", status="new",
             what="bounded-gain joint placement with importance sampling p_i ~ L_i^(1/2)",
             base=dict(estimator="saga", orth="soft", sampling=0.5),
             grid=dict(eps_innov=[0.03, 0.1, 0.3, 1.0] if not quick else [0.3], gamma=_g("R2_op", -2, 7, st))),
        dict(name="split SAGA-Muon", method="vr", group="spectral", status="new",
             what="gamma sharp(m) + gamma_c c: nuclear-scaled polar step of the predictable part, innovation added "
                  "linearly",
             base=dict(estimator="saga", placement="split"),
             grid=dict(gamma_c=_g("R2", -4, 2, 1), gamma=_g("R2_op", -8, 5, st))),
        dict(name="tangent SAGA-Muon", method="vr", group="spectral", status="new",
             what="gamma (sharp(m) + D sharp(m)[c]): innovation propagated through the Frechet derivative",
             base=dict(estimator="saga", placement="tangent"), grid=dict(gamma=_g("R2_op", -12, 0, st))),
        dict(name="lazy split L-SVRG-Muon", method="vr", group="spectral", status="new",
             what="gamma sharp(grad F(w)) + gamma_c (innovation + lam (W - w)): one factorization per reference point",
             base=dict(estimator="lsvrg", placement="split", lazy=True),
             grid=dict(gamma_c=_g("R2", -2, 3, 1), gamma=_g("R2_op", -9, 2, st))),
        dict(name="PAGE-Muon (joint)", method="vr", group="spectral", status="new",
             what="nuclear-scaled polar step of the PAGE / loopless-SARAH recursive estimator",
             base=dict(estimator="page"), grid=dict(gamma=_g("R2_op", -9, 4, st))),
        dict(name="radial SAGA-Muon", method="vr", group="spectral", status="new",
             what="gamma_c omega + gamma <omega, O_m> O_m with O_m = polar(m): Euclidean SAGA step plus a step along the "
                  "predictable polar direction whose length is an unbiased estimate of <grad F, O_m>",
             base=dict(estimator="saga", placement="radial"),
             grid=dict(gamma_c=_g("R2", -5, 2, 1), gamma=_g("R2_op", -5, 5, 2))),
        dict(name="polar-metric SAGA", method="vr", group="spectral", status="new",
             what="SAGA preconditioned by the frozen polar metric of its predictable part (refreshed every n "
                  "iterations): linear in the estimator, equal to the nuclear-scaled polar step at the anchor; step "
                  "c / R2P (certified per-sample smoothness in the metric); gains floored at metric_floor < 1",
             base=dict(estimator="saga", placement="metric"),
             grid=dict(metric_floor=[0.03, 0.1, 0.3], c_metric=[2.0 ** k for k in range(-4, 6)])),
        dict(name="polar-metric L-SVRG", method="vr", group="spectral", status="new",
             what="loopless SVRG preconditioned by the frozen polar metric of the exact gradient at the reference point",
             base=dict(estimator="lsvrg", placement="metric"),
             grid=dict(metric_floor=[0.03, 0.1, 0.3], c_metric=[2.0 ** k for k in range(-4, 6)])),
        dict(name="one-shot polar-metric SAGA", method="vr", group="spectral", status="new",
             what="SAGA preconditioned by the polar metric of the full gradient at W_0, computed once and never "
                  "refreshed: SAGA on a fixed linear reparametrization of the problem (facts F10, F11), one "
                  "factorization for the whole run",
             base=dict(estimator="saga", placement="metric", metric_every=ONE_SHOT),
             grid=dict(metric_floor=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6] if not quick else [0.01, 0.1],
                       c_metric=[2.0 ** k for k in range(-4, 4, st)])),
        dict(name="one-shot polar-metric B-SAGA", method="vr", group="spectral", status="new",
             what="damped SAGA (theta < 1) preconditioned by the polar metric of the full gradient at W_0",
             base=dict(estimator="saga", placement="metric", metric_every=ONE_SHOT),
             grid=dict(theta=[0.6, 0.3, 0.1] if not quick else [0.3],
                       metric_floor=[0.003, 0.01, 0.03, 0.1, 0.3] if not quick else [0.03],
                       c_metric=[2.0 ** k for k in range(-4, 5, st)])),
        dict(name="anchor-PolarGrad L-SVRG", method="vr", group="spectral", status="new",
             what="Euclidean loopless SVRG plus one nuclear-scaled polar step (c_a / L_op) sharp(grad F(w)) at each "
                  "refresh of the reference point (the nonlinear map sees exact gradients only)",
             base=dict(estimator="lsvrg", placement="euclid"),
             grid=dict(anchor_step=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], gamma=_g("R2", -3, 6, st))),
        dict(name="relaxed-memory SAGA-Muon", method="vr", group="spectral", status="new",
             what="joint placement with the table moved by the dual-free-SDCA fraction delta = lam n / (R2 + lam n) "
                  "towards the fresh gradient instead of replaced",
             base=dict(estimator="saga", mem_relax="sdca"), grid=dict(gamma=_g("R2_op", -6, 4, st))),
        dict(name="Katyusha-Muon", method="katyusha", group="spectral", status="new",
             what="loopless Katyusha whose y step is the nuclear-scaled polar step of the estimator (the general-norm "
                  "template of Allen-Zhu's Katyusha2 in the spectral norm); gamma_y = c / R2_op with c = 1/9 (the "
                  "constant of Katyusha2) or 1; same (L, mu) grid as L-Katyusha",
             base=dict(ystep="sharp"),
             grid=dict(L=[times("R2", c) for c in ((1.0, 0.25) if quick else
                                             (2.0, 1.0, 0.5, 0.25, 0.125, 1.0 / 16, 1.0 / 32, 1.0 / 64))],
                       mu=[times("lam", c) for c in ((1.0, 10.0) if quick else
                                               (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0))],
                       gamma_y=[over("R2_op", 1.0 / 9.0), over("R2_op", 1.0)])),
    ]
    return fams


def minibatch_families(b, quick=False):
    """SAGA with batch size b: Euclidean (plain and damped), joint and split orthogonalized versions (steps grow
    with b)."""
    st = 2 if quick else 1
    return [
        dict(name=f"SAGA b={b}", method="vr", group="euclid", status="sota", what="minibatch SAGA",
             base=dict(estimator="saga", placement="euclid", batch=b), grid=dict(gamma=_g("R2", -3, 9, st))),
        dict(name=f"B-SAGA b={b}", method="vr", group="euclid", status="sota", what="damped minibatch SAGA",
             base=dict(estimator="saga", placement="euclid", batch=b),
             grid=dict(theta=[0.6, 0.3, 0.1] if not quick else [0.3], gamma=_g("R2", -1, 10, st))),
        dict(name=f"SAGA-Muon (joint) b={b}", method="vr", group="spectral", status="new",
             what="minibatch SAGA estimator, nuclear-scaled polar step",
             base=dict(estimator="saga", batch=b), grid=dict(gamma=_g("R2_op", -8, 7, st))),
        dict(name=f"split SAGA-Muon b={b}", method="vr", group="spectral", status="new",
             what="minibatch split placement",
             base=dict(estimator="saga", placement="split", batch=b),
             grid=dict(gamma_c=_g("R2", -4, 6, 2), gamma=_g("R2_op", -6, 6, 2))),
    ]


def streaming_families(b=32, quick=False):
    """Minibatch methods without finite-sum memory: Muon as used, and the momentum-based variance reduction of the
    literature (at constant parameters they keep a floor on a finite sum; tuned for the lowest final error within the
    budget, with a cosine schedule spanning the budget)."""
    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0] if not quick else [1e-2, 1e-1]
    return [
        dict(name=f"SGD b={b}", method="sgd", group="stream", status="sota", what="minibatch SGD, cosine schedule",
             base=dict(batch=b, schedule="cosine"),
             grid=dict(lr=[over("R2", c) for c in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)])),
        dict(name=f"Muon b={b}", method="muon_sgd", group="stream", status="sota",
             what="Keller Jordan's Muon: momentum 0.95, Nesterov, 5 Newton-Schulz steps, fixed-length step, cosine "
                  "schedule",
             base=dict(batch=b, schedule="cosine", beta=0.95, nesterov=True, orth="ns5"), grid=dict(lr=lrs)),
        dict(name=f"MARS-M b={b}", method="mvr", group="stream", status="sota",
             what="MARS-M (Liu-Yuan-Gu), exact variant: same-batch correction g = 0.025 (two gradients per step), "
                  "clip 1, momentum 0.95, cosine schedule",
             base=dict(batch=b, schedule="cosine", beta=0.95, g=0.025, clip=1.0, same_batch=True, orth="ns5"),
             grid=dict(lr=lrs)),
        dict(name=f"MARS-M approx b={b}", method="mvr", group="stream", status="sota",
             what="MARS-M (Liu-Yuan-Gu), approximate variant: correction with the gradient of the previous batch (one "
                  "gradient per step), g = 0.025, clip 1, momentum 0.95, cosine schedule",
             base=dict(batch=b, schedule="cosine", beta=0.95, g=0.025, clip=1.0, same_batch=False, orth="ns5"),
             grid=dict(lr=lrs)),
        dict(name=f"STORM-Muon b={b}", method="mvr", group="stream", status="sota",
             what="STORM / MVR2 buffer (g = 1) then Newton-Schulz, cosine schedule (Chang et al.; Qian et al. "
                  "arXiv:2512.16598)",
             base=dict(batch=b, schedule="cosine", beta=0.95, g=1.0, same_batch=True, orth="ns5"),
             grid=dict(lr=lrs)),
        dict(name=f"sharp STORM-Muon b={b}", method="mvr", group="stream", status="sota",
             what="STORM buffer with a norm-scaled step (variant of OptMuon-I, Yuan arXiv:2606.08783, which scales by "
                  "the Frobenius norm times an AdaGrad-norm coefficient; here nuclear norm and constant gamma): "
                  "removes the fixed-length floor, keeps the noise floor of the buffer",
             base=dict(batch=b, beta=0.95, g=1.0, same_batch=True, orth="svd", length="sharp"),
             grid=dict(gamma=[over("L_op", c) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)])),
    ]
