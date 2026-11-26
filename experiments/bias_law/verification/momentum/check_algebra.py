"""Numeric checks of (i) the weight algebra in Corollary cor:joint, (ii) the min-sum -> Hoelder step, (iii) TV claim."""
import numpy as np
rng = np.random.default_rng(0)
worst = 0
for _ in range(2000):
    b = int(rng.integers(1, 50)); beta = rng.uniform(0, 0.97)
    J = int(np.log(1e-18) / np.log(max(beta, 1e-9))) + 2
    w = np.repeat((1 - beta) * beta ** np.arange(J) / b, b)
    r2, r3 = (w ** 2).sum(), (w ** 3).sum(); eps2 = (w.max() ** 2) / r2
    B = b * (1 + beta + beta ** 2) / (1 - beta ** 2)
    errs = [abs(w.sum() - 1), abs(r3 / r2 * B - 1), abs(eps2 / ((1 - beta ** 2) / b) - 1),
            abs(np.sqrt(r2) / (np.sqrt(eps2) / (1 + beta)) - 1)]
    assert 1 - 1e-9 <= B * eps2 <= 3 + 1e-9
    worst = max(worst, max(errs))
print("corollary weight identities: worst relative error", worst)
N = 9; q = N / (N - 0.5); print("(1+q)/2 - (1+1/(4N-2)) =", (1 + q) / 2 - (1 + 1 / (4 * N - 2)))
# (ii) sum_j min(A, B beta^j) <= A^alpha B^(1-alpha)/(1-beta^(1-alpha))
worst = -np.inf
for _ in range(20000):
    A_, B_ = np.exp(rng.uniform(-10, 10, 2)); beta = rng.uniform(0.01, 0.99); alpha = rng.uniform(0.01, 0.99)
    j = np.arange(5000); lhs = np.minimum(A_, B_ * beta ** j).sum()
    worst = max(worst, lhs / (A_ ** alpha * B_ ** (1 - alpha) / (1 - beta ** (1 - alpha))))
print("min-sum / Hoelder bound, worst ratio (must be <=1):", worst)
# (iii) beta=1/2, signs: j-step law from a point has 2^j atoms; N_beta is uniform on [-1,1] => TV distance 1.
n = 10 ** 6; U = np.zeros(n)
for _ in range(60): U = 0.5 * U + 0.5 * (2.0 * (rng.random(n) < 0.5) - 1)
from scipy import stats
print("KS test of N_beta vs U[-1,1]:", stats.kstest(U, 'uniform', args=(-1, 2)))
