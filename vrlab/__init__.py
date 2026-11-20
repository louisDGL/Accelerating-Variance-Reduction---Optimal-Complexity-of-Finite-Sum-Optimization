"""vrlab: numerical laboratory for the project
"Accelerating Variance Reduction: Optimal Complexity of Finite-Sum Optimization".

Modules
-------
problems        synthetic finite-sum problems (l2-regularized logistic regression, block-separable
                quadratics / Nesterov-chain hard instance), with exact constants (R^2, mu, kappa).
solvers         high-precision computation of the minimizer eta_* (Newton with backtracking).
algorithms      GD, AGD, SGD, SAGA (optionally preconditioned, fully instrumented; index order i.i.d.,
                reshuffled, shuffled once or cyclic), L-SVRG, L-Katyusha, naive Nesterov + SVRG estimator.
instrumentation exact conditional expectations of every quantity appearing in the proof of
                Proposition 5.9 (Bach, Learning Theory from First Principles) and tightness ratios.
bounds          theoretical rates (Prop. 5.9, general (gamma, Delta) version, L-Katyusha, GD, AGD)
                and the lower bounds of the A1 note on the shared-regularizer hard instance.
analysis        attribution of the theory/practice gap to each inequality of the proof.
order_model     solvable model of the sampling order in SAGA (identical Hessians): exact moment
                recursion for i.i.d. indices, recurrence and critical step of a fixed order, pass map
                and fluid limit of random reshuffling, simulator of the scalar model (hypothesis B4).
"""
from . import problems, solvers, instrumentation, bounds, algorithms, analysis, order_model  # noqa: F401
