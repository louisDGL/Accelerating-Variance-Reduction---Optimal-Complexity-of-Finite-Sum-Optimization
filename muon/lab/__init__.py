"""lab: numpy laboratory for Muon-type (orthogonalized) methods on matrix-parameter finite sums.

Modules
-------
problems         softmax cross-entropy and multi-output least squares with W in R^{K x d}, features generated as in
                 vrlab.problems.make_logistic, exact Euclidean and spectral-norm constants, exact derivatives
                 (gradients, per-sample and minibatch gradients, Hessian-vector products, dense Hessian) and an accurate
                 F(W1) - F(W0) (Bregman identity).
optimum          high-precision minimizer W_* (closed form for LS, damped Newton + polish for softmax).
polar            exact polar factor (SVD), Keller Jordan's quintic Newton-Schulz (ns5), cubic Newton-Schulz,
                 bf16 emulation, dispatcher `orth`.
maps             spectral maps and their Frechet derivatives: sharp (nuclear-norm-scaled polar factor), bounded-gain
                 soft polar factor, derivative of the polar factor / of sharp / of a generalized matrix function,
                 softmax gauge basis.
algorithms       GD, Muon with full gradients (const / sharp / line-search / BB steps), SGD, Muon (minibatch),
                 STORM-Muon, SAGA, L-SVRG, SAGA-Muon and L-SVRG-Muon (const / decay / sharp / BB steps), with the
                 gradient accounting of vrlab and traces of F - F_*, ||W - W_*||_F^2, ||W - W_*||_op^2.
vrmuon           generalized variance-reduced Muon: estimator (SAGA, L-SVRG, PAGE, minibatch) x placement of the
                 nonlinear map (joint, split, tangent, Euclidean) x step-length rule (nuclear-scaled, fixed, clipped)
                 x orthogonalization (exact, Newton-Schulz, bounded gain); damping, minibatches, momentum, lazy
                 factorization.
exact            exact conditional expectations of one iteration of a vrmuon configuration (enumeration of the sampled
                 index), reference full-gradient steps.
mvr              momentum-based variance reduction for Muon (Nesterov / MARS-M / STORM family).
katyusha         loopless Katyusha on matrix problems, with the spectral step on the y sequence.
instrumentation  effective constants mu_eff, L_eff in the Euclidean and spectral geometries, curvatures along the
                 update, exact one-step decrease.
The package imports vrlab (read-only) for `covariance_spectrum`; it adds the project root to sys.path if needed.
"""
from . import polar, problems, optimum, instrumentation, algorithms  # noqa: F401
from . import maps, vrmuon, exact, mvr, katyusha  # noqa: F401
from .problems import make_softmax, make_matrix_ls, SoftmaxProblem, MatrixLSProblem  # noqa: F401
from .optimum import solve_optimum  # noqa: F401
from .polar import polar_svd, ns5, ns_cubic, orth, nuclear_norm, spectral_norm  # noqa: F401
from .instrumentation import instrument  # noqa: F401
from .algorithms import (gd, muon_full, sgd, muon_sgd, storm_muon, saga, lsvrg, saga_muon,  # noqa: F401
                         lsvrg_muon, saga_estimator, lsvrg_estimator, lr_factors, Trace)
from .vrmuon import vr_muon, VRMuon  # noqa: F401
from .mvr import mvr_muon  # noqa: F401
from .katyusha import lkatyusha  # noqa: F401
