"""cite-stat: check of the scalar offset attributed to cornish1938moments (paper line 456-457):
"the scalar offset kappa_3/(6 sigma^2 b) of the median of a mean".

Cornish-Fisher (1938) expansion of the p-quantile: x_p = m + s (z_p + (z_p^2 - 1) g1/6 + ...), g1 the skewness.
For the mean of b i.i.d. centred variables: s = sigma/sqrt(b), g1 = kappa_3/(sigma^3 sqrt(b)); at p = 1/2 (z_p = 0):
median = -kappa_3/(6 sigma^2 b) + o(1/b).  The paper's G_b^dagger = -Med = +kappa_3/(6 sigma^2 b) (K = d = 1,
a_{1,1}/c_{1,1} = 1).  Checked with exact quantiles of gamma-type means (closed form) and of a centred
lognormal by Monte Carlo.

Run from the project root:  OPENBLAS_NUM_THREADS=1 python3 paper/audit/checks/cite-stat_cornish_median.py
"""
import numpy as np
from scipy import stats

# 1) xi = Gamma(a,1) - a: sigma^2 = a, kappa_3 = 2a; mean of b copies = Gamma(ab,1)/b - a
for a in [1.0, 0.3]:
    for b in [2, 8, 32, 128, 512]:
        med = stats.gamma.ppf(0.5, a * b) / b - a
        pred = -(2 * a) / (6 * a * b)
        print(f"Gamma(a={a}) b={b:4d}: median of mean = {med:+.6f}, -k3/(6 s^2 b) = {pred:+.6f}, "
              f"b*median = {b*med:+.4f} (limit {b*pred:+.4f})")

# 2) centred lognormal(0, 0.5): Monte Carlo
rng = np.random.default_rng(1)
s = 0.5
m1 = np.exp(s**2 / 2)
var = (np.exp(s**2) - 1) * np.exp(s**2)
k3 = (np.exp(s**2) + 2) * np.sqrt(np.exp(s**2) - 1) * var**1.5
for b in [8, 32]:
    n = 400000
    x = rng.lognormal(0.0, s, size=(n, b)).mean(axis=1) - m1
    med = np.median(x)
    se = 1.2533 * x.std() / np.sqrt(n)
    print(f"lognormal(0,{s}) b={b}: median of mean = {med:+.5f} (se {se:.5f}), -k3/(6 s^2 b) = {-k3/(6*var*b):+.5f}")
