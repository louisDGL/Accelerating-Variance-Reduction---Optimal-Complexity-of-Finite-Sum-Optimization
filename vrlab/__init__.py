"""vrlab: numerical laboratory for the project
"Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization".

Modules
-------
problems        synthetic finite-sum problems (l2-regularized logistic regression, block-separable
                quadratics / Nesterov-chain hard instance), with exact constants (R^2, mu, kappa).
solvers         high-precision computation of the minimizer eta_* (Newton with backtracking).
algorithms      GD, AGD, SGD, SAGA (optionally preconditioned, fully instrumented), L-SVRG,
                L-Katyusha, naive Nesterov + SVRG estimator.
instrumentation exact conditional expectations of every quantity appearing in the proof of
                Proposition 5.9 (Bach, Learning Theory from First Principles) and tightness ratios.
bounds          theoretical rates (Prop. 5.9, general (gamma, Delta) version, L-Katyusha, GD, AGD).
analysis        attribution of the theory/practice gap to each inequality of the proof.
"""
from . import problems, solvers, instrumentation, bounds, algorithms, analysis  # noqa: F401
