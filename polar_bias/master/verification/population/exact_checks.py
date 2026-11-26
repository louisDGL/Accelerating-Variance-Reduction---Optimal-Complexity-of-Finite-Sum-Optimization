"""Exact / symbolic checks for Section 3 (Theorem "Finite-population bias").

Items 1-5 of the referee brief.  Everything here is exact enumeration, exact
rational arithmetic, sympy, or deterministic quadrature (no Monte Carlo).
"""
import itertools, math
from fractions import Fraction
import numpy as np
import sympy as sp
from scipy.optimize import minimize

rng = np.random.default_rng(20261125)
np.set_printoptions(precision=6, suppress=True, linewidth=140)


def header(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# ----------------------------------------------------------------------------
header("ITEM 1. conditioning identity and lattice inversion formula for Psi_0")
# ----------------------------------------------------------------------------
def psi0_enum(z, b, t):
    """exact conditional c.f. of T = b xi/h by enumeration of subsets"""
    n = len(z); p = b / n; h = math.sqrt(n * p * (1 - p))
    vals = [np.exp(1j * (z[list(S)].sum(0) @ t) / h) for S in itertools.combinations(range(n), b)]
    return np.mean(vals)


def H0(z, p, h, t, u):
    """u may be an array; z (n,N), t (N,)"""
    a = z @ t
    s = (a[:, None] + np.atleast_1d(u)[None, :]) / h
    return np.prod((1 - p) * np.exp(-1j * p * s) + p * np.exp(1j * (1 - p) * s), axis=0)


def psi0_formula(z, b, t, m=None):
    n = len(z); p = b / n; h = math.sqrt(n * p * (1 - p))
    m = m or 4 * (n + 2)
    # integrand is a trigonometric polynomial in u/h of degree <= n (times e^{-i b u/h}, b integer):
    # the periodic trapezoid rule with m > n nodes is exact.
    u = -math.pi * h + 2 * math.pi * h * np.arange(m) / m
    num = H0(z, p, h, t, u).mean() * 2 * math.pi * h
    den = H0(z, p, h, 0 * t, u).mean() * 2 * math.pi * h
    return num / den, den


worst = 0
for (n, b, N) in [(6, 2, 1), (6, 2, 2), (7, 3, 2), (8, 5, 3), (9, 4, 1), (10, 7, 2)]:
    z = rng.standard_normal((n, N)) ** 3; z -= z.mean(0)
    for _ in range(5):
        t = rng.standard_normal(N) * 3
        e = psi0_enum(z, b, t); f, den = psi0_formula(z, b, t)
        worst = max(worst, abs(e - f))
    p = b / n; h = math.sqrt(n * p * (1 - p))
    exact_den = 2 * math.pi * h * math.comb(n, b) * p ** b * (1 - p) ** (n - b)
    print(f"n={n} b={b} N={N}: max|Psi0(enum)-Psi0(formula)| so far {worst:.2e};  denominator {den.real:.12f}"
          f" vs 2 pi h P(Bin=b) {exact_den:.12f}")
print("-> identity verified to rounding; denominator = 2 pi h P(Bin(n,p)=b).")

# what happens if np is not an integer: the event has probability zero and the 'formula' is not the cond. law
print("\nDenominator / sqrt(2 pi) versus 1 - (1-r)/(12 v)   [r=p(1-p), v=n r]:")
for (n, b) in [(40, 10), (400, 100), (4000, 1000), (4000, 40), (40000, 40), (4000, 3960)]:
    p = b / n; r = p * (1 - p); v = n * r
    logP = math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(n - b + 1) + b * math.log(p) + (n - b) * math.log(1 - p)
    ratio = math.sqrt(2 * math.pi * v) * math.exp(logP)
    print(f"  n={n:6d} b={b:5d} v={v:9.2f}: ratio={ratio:.8f}   1-(1-r)/(12v)={1 - (1 - r) / (12 * v):.8f}")

# ----------------------------------------------------------------------------
header("ITEM 2. large frequencies")
# ----------------------------------------------------------------------------
s, p_ = sp.symbols("s p", real=True)
mod2 = sp.simplify(sp.expand(((1 - p_) + p_ * sp.cos(s)) ** 2 + (p_ * sp.sin(s)) ** 2 - (1 - 2 * p_ * (1 - p_) * (1 - sp.cos(s)))))
print("|E e^{is(I-p)}|^2 - (1-2r(1-cos s)) =", mod2)
x = np.linspace(0, 1, 100001)
print("max over x in [0,1] of sqrt(1-x) - exp(-x/2) =", np.max(np.sqrt(1 - x) - np.exp(-x / 2)), "(<=0: |factor| <= exp(-r(1-cos s)))")
print("   note 2r(1-cos s) <= 4r <= 1 so x stays in [0,1].")
bad = 0; worst_ratio = 0; idmax = 0
for trial in range(2000):
    n = int(rng.integers(3, 40)); N = int(rng.integers(1, 4)); b = int(rng.integers(1, n))
    z = rng.standard_normal((n, N)); z -= z.mean(0)
    p = b / n; r = p * (1 - p); h = math.sqrt(n * r)
    t = rng.standard_normal(N) * rng.exponential(5); u = rng.uniform(-math.pi * h, math.pi * h)
    si = (z @ t + u) / h
    D = math.exp(-r * np.sum(1 - np.cos(si)))
    Hv = abs(H0(z, p, h, t, u)[0])
    worst_ratio = max(worst_ratio, Hv / D)
    bad += Hv > D * (1 + 1e-12)
    Sig = z.T @ z / n
    idmax = max(idmax, abs(r * np.sum(si ** 2) - (t @ Sig @ t + u * u)) / (1 + t @ Sig @ t + u * u))
print(f"2000 random (population,t,u): violations of |H0|<=D: {bad}; max |H0|/D = {worst_ratio:.6f}")
print(f"max relative error in  r sum s_i^2 = t'Sigma_n t + u^2 : {idmax:.2e}")

# D as characteristic function of symmetric compound Poisson: check positive-definiteness numerically
n, N = 7, 2; z = rng.standard_normal((n, N)); z -= z.mean(0); b = 3; p = b / n; r = p * (1 - p); h = math.sqrt(n * r)
pts = rng.standard_normal((60, N + 1)) * 4
def Dfun(tau):
    si = (z @ tau[:N] + tau[N]) / h
    return math.exp(-r * np.sum(1 - np.cos(si)))
G = np.array([[Dfun(a - c) for c in pts] for a in pts])
print("min eigenvalue of Gram matrix [D(tau_a-tau_b)] on 60 random points of R^{N+1}:", np.linalg.eigvalsh(G).min(),
      "(>= 0 up to rounding: D is positive definite, as it must be for a c.f.)")
# exact: exp(r(cos<tau,w>-1)) = c.f. of  w*(Pois(r/2)-Pois'(r/2)); product over i -> sum of independent such vectors
print("exp(r(cos x -1)) = E exp(i x (P1-P2)), P1,P2 iid Poisson(r/2):",
      sp.simplify(sp.exp(sp.Rational(1, 2) * sp.Symbol('r') * (sp.exp(sp.I * s) - 1)) * sp.exp(sp.Rational(1, 2) * sp.Symbol('r') * (sp.exp(-sp.I * s) - 1))
                  - sp.exp(sp.Symbol('r') * (sp.cos(s) - 1))).rewrite(sp.cos).simplify())

# tent inequality in N+1 dims, direct numerical test in the scalar case (N=1 -> 2 dims)
n = 30; z1 = rng.choice([-1.0, 2.0], size=n, p=[2 / 3, 1 / 3]); z1 = np.array([-1.0] * 20 + [2.0] * 10); b = 10
p = b / n; r = p * (1 - p); h = math.sqrt(n * r)
def D2(T, U):
    si = (z1[:, None, None] * T[None] + U[None]) / h
    return np.exp(-r * np.sum(1 - np.cos(si), axis=0))
a = 0.25 * h
g = np.linspace(-2 * a, 2 * a, 401); dg = g[1] - g[0]
T, U = np.meshgrid(g, g, indexing="ij")
tent = (1 - np.abs(T) / (2 * a)) * (1 - np.abs(U) / (2 * a))
I0 = (D2(T, U) * tent).sum() * dg * dg
mx = 0
for _ in range(300):
    t0 = rng.uniform(-40 * h, 40 * h); u0 = rng.uniform(-math.pi * h, math.pi * h)
    mx = max(mx, (D2(T + t0, U + u0) * tent).sum() * dg * dg)
# lattice population {-1,2}: s_i = (z_i t + u)/h in 2 pi Z for all i  <=>  (t,u) = (2 pi h/3)(k, k + 3l):
# genuine resonances INSIDE the strip |u| <= pi h, where D = |H_0| = 1.
t0 = 2 * math.pi * h / 3
Ires = (D2(T + t0, U + t0) * tent).sum() * dg * dg
Hres = abs(np.prod((1 - p) * np.exp(-1j * p * (z1 * t0 + t0) / h) + p * np.exp(1j * (1 - p) * (z1 * t0 + t0) / h)))
print(f"tent test (lattice population -1/2, n=30,b=10): centered integral {I0:.5f}; max over 300 random shifts {mx:.5f};")
print(f"   at the resonance (t,u)=(2 pi h/3)(1,1): D = {D2(np.array([[t0]]), np.array([[t0]]))[0, 0]:.6f}, |H_0| = {Hres:.6f}, shifted tent integral {Ires:.5f}")
print("   -> the bound int_{cube} D <= C is attained at resonances: |Psi_0| is NOT small in L^1(Q_j); only boundedness is available, as the proof uses.")

# ----------------------------------------------------------------------------
header("ITEM 3. small frequencies: cumulants, uniform remainder, u-integration")
# ----------------------------------------------------------------------------
zc = sp.Symbol("z")
pp = sp.Symbol("p")
f = sp.log(1 + pp * (sp.exp(zc) - 1)) - pp * zc
ser = sp.series(f, zc, 0, 6).removeO()
r_ = pp * (1 - pp)
for k, name, target in [(2, "r", r_), (3, "r d", r_ * (1 - 2 * pp)), (4, "r d4", r_ * (1 - 6 * r_))]:
    ck = sp.factor(ser.coeff(zc, k) * sp.factorial(k))
    print(f"cumulant {k}: {ck}   minus {name}: {sp.simplify(ck - target)}")
c5 = sp.factor(ser.coeff(zc, 5) * sp.factorial(5)); print("cumulant 5:", c5, " -> divisible by p(1-p):", sp.factor(c5 / r_))

# uniform remainder: sup over p in (0,1), complex |z|<=rho of |f - poly4| / (r |z|^5)
def rem_ratio(pv, zv):
    fv = np.log1p(pv * np.expm1(zv)) - pv * zv
    rv = pv * (1 - pv)
    poly = rv * zv ** 2 / 2 + rv * (1 - 2 * pv) * zv ** 3 / 6 + rv * (1 - 6 * rv) * zv ** 4 / 24
    return np.abs(fv - poly) / (rv * np.abs(zv) ** 5)
pg = np.concatenate([10.0 ** -np.arange(1, 13), 1 - 10.0 ** -np.arange(1, 13), np.linspace(0.01, 0.99, 99)])
for rho in [0.25, 0.5, 1.0]:
    sv = np.linspace(-rho, rho, 41); sv = sv[np.abs(sv) > rho / 4]  # avoid cancellation noise near 0
    val = max(rem_ratio(pv, 1j * sv).max() for pv in pg[(pg > 1e-7) & (pg < 1 - 1e-7)])
    print(f"sup_p sup_{{rho/4<|s|<={rho}}} |remainder| / (r |s|^5) = {val:.5f}   (bounded uniformly in p; limit value |kappa_5|/(120 r) <= 1/120={1/120:.5f})")
# the analytic-family argument: g(p,z) = f(p,z)/(p(1-p)) extends continuously to p in {0,1}
g0 = sp.simplify(sp.diff(f, pp).subs(pp, 0)); g1 = sp.simplify(-(sp.exp(zc) - 1) / sp.exp(zc) + zc)  # = -d_p f at p=1, using log(e^z)=z for small |z|
print("limit of f/(p(1-p)) at p=0:", g0, "   at p=1:", g1, "  (both entire in z: no singularity is created)")

# u-integration
u, eps, d, d4 = sp.symbols("u epsilon d d_4", real=True)
m2, m3, m4, m1 = sp.symbols("m_2 m_3 m_4 m_1", real=True)   # m_k = n^{-1} sum a_l(t)^k
al = sp.Symbol("a")
def avg(expr):
    """n^{-1} sum_l of a polynomial in a_l  -> moments m_k (m_1 kept symbolic to display the role of sum a_i=0)"""
    expr = sp.expand(expr)
    return sum(expr.coeff(al, k) * {0: 1, 1: m1, 2: m2, 3: m3, 4: m4}[k] for k in range(5))
A3 = avg((sp.I * (al + u)) ** 3)
A4 = avg((sp.I * (al + u)) ** 4)
gauss = sp.exp(-u ** 2 / 2) / sp.sqrt(2 * sp.pi)
Eu = lambda e: sp.simplify(sp.integrate(sp.expand(e) * gauss, (u, -sp.oo, sp.oo)))
EA3 = Eu(A3); print("E_u A_3 =", EA3, "  -> with m_1=0:", EA3.subs(m1, 0), " = n^{-1} sum (i a_l)^3")
EA4 = Eu(A4.subs(m1, 0)); print("E_u A_4 (m_1=0) =", EA4)
EA32 = Eu(sp.expand(A3.subs(m1, 0) ** 2)); print("E_u A_3^2 (m_1=0) =", EA32)
R = d4 * EA4 / 24 + d ** 2 * EA32 / 72
R0 = R.subs({m2: 0, m3: 0, m4: 0})
print("order eps^2 coefficient after u-integration  R(t) =", sp.expand(R))
print("R(0) =", R0, "  with d^2=1-4r, d4=1-6r:", sp.simplify(R0.subs({d4: 1 - 6 * sp.Symbol('r'), d: sp.sqrt(1 - 4 * sp.Symbol('r'))})),
      " (= -(1-r)/12, matches the local-limit expansion above)")
Qn = sp.expand(R - R0)
print("Q_n(t) = R(t)-R(0) =", Qn)
print("Q_n with d4-d^2 = -2r: quadratic part =", sp.simplify((Qn.coeff(m2, 1)).subs({d4: 1 - 6 * sp.Symbol('r'), d: sp.sqrt(1 - 4 * sp.Symbol('r'))})), "* m_2",
      "  => covariance (1 + eps^2 r) Sigma_n = (1+1/n) Sigma_n, consistent with exact n/(n-1) Sigma_n to O(n^-2)")

# numerical test of the whole small-frequency statement: h^3 * sup_t |Psi_0 - Gamma_n| bounded, scalar populations
def gamma_n(z, b, t):
    n = len(z); p = b / n; r = p * (1 - p); h = math.sqrt(n * r); e = 1 / h
    dd = 1 - 2 * p; dd4 = 1 - 6 * r
    a = np.outer(t, z)
    M2, M3, M4 = (a ** 2).mean(1), (a ** 3).mean(1), (a ** 4).mean(1)
    Q = dd4 / 24 * (M4 + 6 * M2) - dd ** 2 / 72 * (M3 ** 2 + 9 * M2 ** 2 + 18 * M2)
    return np.exp(-M2 / 2) * (1 + e * dd * (1j) ** 3 * M3 / 6 + e * e * Q)

def psi0_scalar(z, b, tgrid):
    n = len(z); p = b / n; h = math.sqrt(n * p * (1 - p))
    m = n + 8   # > n nodes: periodic trapezoid rule exact
    uu = -math.pi * h + 2 * math.pi * h * np.arange(m) / m
    out = np.empty(len(tgrid), complex)
    # work in logs to avoid under/overflow
    den = None
    for k, t in enumerate(np.concatenate([[0.0], tgrid])):
        s_ = (z[:, None] * t + uu[None, :]) / h
        val = np.exp(np.sum(np.log((1 - p) * np.exp(-1j * p * s_) + p * np.exp(1j * (1 - p) * s_)), axis=0)).mean()
        if k == 0: den = val
        else: out[k - 1] = val / den
    return out

print("\nScalar test of the Q_0 expansion: E3 = h^3 * sup_{|t|<=min(6,h/2)} |Psi_0(t) - Gamma_n(t)|, E2 = h^2 * sup |Psi_0 - Gamma_n^{(1)}| (first-order only)")
tg = np.linspace(-6, 6, 81)
pops = {"lattice {-1,2} (2/3,1/3)": lambda n: np.array([-1.0] * (2 * n // 3) + [2.0] * (n // 3)),
        "bounded skewed non-lattice (cubed uniform quantiles)": lambda n: (lambda q: (q - q.mean()) / q.std())(((np.arange(n) + 0.5) / n) ** 3)}
for name, mk in pops.items():
    print(" population:", name)
    for frac in [(1, 4), (1, 2), (3, 4), (1, 12)]:
        row = []
        for n in [48, 192, 768, 3072]:
            z = mk(n); z = z - z.mean(); b = n * frac[0] // frac[1]
            h = math.sqrt(b * (n - b) / n)
            tg = np.linspace(-1, 1, 81) * min(6.0, 0.5 * h)      # stay inside Q_0 = {|t| <= c h}
            ps = psi0_scalar(z, b, tg); ga = gamma_n(z, b, tg)
            p = b / n
            ga1 = np.exp(-(tg ** 2) * (z ** 2).mean() / 2) * (1 + (1 - 2 * p) / h * (1j * tg) ** 3 * (z ** 3).mean() / 6)
            row.append(f"n={n}: E3={h ** 3 * np.abs(ps - ga).max():7.3f} E2={h * h * np.abs(ps - ga1).max():6.3f}")
        print(f"   p={frac[0]}/{frac[1]}: " + " | ".join(row))
print(" (E3 bounded as n grows => Gamma_n, including my explicit Q_n, is the correct second-order transform on Q_0;")
print("  E2 bounded but not -> 0 => the eps^2 term is genuinely present.)")

# distance of s_i from 2 pi Z on the region c2 h <= |u| <= pi h, |t| <= c1 h with M c1 < c2/2
c2 = 0.3; M = 2.0; c1 = 0.9 * c2 / (2 * M)
worst = 10
for _ in range(20000):
    hh = rng.uniform(3, 50); uu_ = rng.uniform(c2 * hh, math.pi * hh) * rng.choice([-1, 1]); a_ = rng.uniform(-M * c1 * hh, M * c1 * hh)
    si = (a_ + uu_) / hh
    worst = min(worst, abs(si - 2 * math.pi * round(si / (2 * math.pi))))
print(f"min distance of s_i to 2 pi Z on the stated region (c2={c2}, M c1={M * c1:.3f}): {worst:.4f}  >=  c2 - M c1 = {c2 - M * c1:.4f} > c2/2")

# ----------------------------------------------------------------------------
header("ITEM 4. exact conditional cumulants (rational arithmetic), Maclaurin, scalings")
# ----------------------------------------------------------------------------
def exact_moments(zint, b):
    """zint: list of integer tuples summing to zero. Returns exact Cov and third moment of the *sum* over a b-subset."""
    n = len(zint); N = len(zint[0])
    cnt = 0
    S2 = [[Fraction(0)] * N for _ in range(N)]
    S3 = {}
    for S in itertools.combinations(range(n), b):
        sm = [sum(zint[i][k] for i in S) for k in range(N)]
        cnt += 1
        for i in range(N):
            for j in range(N):
                S2[i][j] += sm[i] * sm[j]
        for idx in itertools.product(range(N), repeat=3):
            S3[idx] = S3.get(idx, 0) + sm[idx[0]] * sm[idx[1]] * sm[idx[2]]
    return [[v / cnt for v in row] for row in S2], {k: Fraction(v, cnt) for k, v in S3.items()}

cases = [("n=6,b=2 scalar", [(-3,), (-2,), (-1,), (0,), (1,), (5,)], 2),
         ("n=6,b=2 2-dim", [(-3, 1), (-2, -4), (-1, 0), (0, 2), (1, 3), (5, -2)], 2),
         ("n=6,b=4 2-dim", [(-3, 1), (-2, -4), (-1, 0), (0, 2), (1, 3), (5, -2)], 4),
         ("n=7,b=3 2-dim", [(-3, 1), (-2, -4), (-1, 0), (0, 2), (1, 3), (5, -2), (0, 0)], 3),
         ("n=9,b=5 3-dim", [(-3, 1, 2), (-2, -4, 0), (-1, 0, -5), (0, 2, 1), (1, 3, 1), (5, -2, 1), (0, 0, 0), (2, 1, -1), (-2, -1, 1)], 5)]
for name, zint, b in cases:
    n = len(zint); N = len(zint[0])
    assert all(sum(zz[k] for zz in zint) == 0 for k in range(N))
    cov, third = exact_moments(zint, b)
    Sig = [[Fraction(sum(zz[i] * zz[j] for zz in zint), n) for j in range(N)] for i in range(N)]
    K3 = {idx: Fraction(sum(zz[idx[0]] * zz[idx[1]] * zz[idx[2]] for zz in zint), n) for idx in itertools.product(range(N), repeat=3)}
    # claims, in units of the unscaled sum: Cov(sum) = h^2 n/(n-1) Sigma, E sum^3 = h^3 (1-2p)/h n^2/((n-1)(n-2)) K3, h^2 = b(n-b)/n
    h2 = Fraction(b * (n - b), n); p = Fraction(b, n)
    ok2 = all(cov[i][j] == h2 * Fraction(n, n - 1) * Sig[i][j] for i in range(N) for j in range(N))
    ok3 = all(third[idx] == h2 * (1 - 2 * p) * Fraction(n * n, (n - 1) * (n - 2)) * K3[idx] for idx in K3)
    print(f"{name}: Cov(T|C0=0) = n/(n-1) Sigma_n exactly: {ok2};   E[T^3|C0=0] = (1-2p)/h n^2/((n-1)(n-2)) K_3n exactly: {ok3}")
nn, bb, hh = sp.symbols("n b h", positive=True)
ratio = sp.simplify(((1 - 2 * bb / nn) / hh * nn ** 2 / ((nn - 1) * (nn - 2))) / (nn / (nn - 1)))
print("ratio of the scalar multipliers:", sp.factor(ratio), "  [claimed (n-2b)/(h(n-2))]")
print("(n-2b)/(n-2) - (1-2p) =", sp.factor((nn - 2 * bb) / (nn - 2) - (1 - 2 * bb / nn)), " = O(1/n)")

# Maclaurin
worst = 0; viol = 0
for _ in range(400):
    n = int(rng.integers(4, 13)); m0 = int(rng.integers(1, n)); y = rng.standard_normal(n) ** 3; y -= y.mean(); Mb = np.abs(y).max()
    sarg = rng.standard_normal() * 2 / Mb
    xs = np.exp(sarg * y)
    mgf = np.mean([np.prod(xs[list(S)]) for S in itertools.combinations(range(n), m0)])
    mgf_direct = np.mean([math.exp(sarg * y[list(S)].sum()) for S in itertools.combinations(range(n), m0)])
    assert abs(mgf - mgf_direct) < 1e-9 * mgf
    b1 = xs.mean() ** m0; b2 = math.exp(Mb ** 2 * m0 * sarg ** 2 / 2)
    viol += (mgf > b1 * (1 + 1e-12)) + (b1 > b2 * (1 + 1e-12))
    worst = max(worst, mgf / b1)
print(f"Maclaurin: 400 random cases, violations of  e_m/C(n,m) <= (mean x_i)^m <= exp(M^2 m s^2/2): {viol}; max ratio mgf/(mean)^m = {worst:.6f}")
print("m0=min(b,n-b) <= 2 h^2 :", all(min(b, n - b) <= 2 * b * (n - b) / n + 1e-12 for n in range(2, 200) for b in range(1, n)))
kap, NN = sp.symbols("kappa N", positive=True)
q = NN / (NN - kap)
print("exponent: h*d_eps = eps^(q-1) -> v^{-(q-1)/2};  (q-1)/2 =", sp.simplify((q - 1) / 2), ";  kappa=1/2,N=k^2 ->", sp.simplify(((q - 1) / 2).subs({kap: sp.Rational(1, 2), NN: sp.Symbol('k') ** 2})))

# ----------------------------------------------------------------------------
header("ITEM 5. complement duality and half sampling (exact enumeration, Euclidean geometric median in R^2)")
# ----------------------------------------------------------------------------
def geomed(points, w=None):
    w = np.full(len(points), 1 / len(points)) if w is None else w
    fobj = lambda m: np.sum(w * np.linalg.norm(points - m, axis=1))
    best = None
    for x0 in [points.T @ w, points.T @ w + 0.01]:
        res = minimize(fobj, x0, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000})
        if best is None or res.fun < best.fun: best = res
    return best.x, best.fun

n = 10
z = rng.standard_normal((n, 2)) ** 3 + rng.standard_normal((n, 2)); z -= z.mean(0)
for b in [3, 4]:
    xi_b = np.array([z[list(S)].mean(0) for S in itertools.combinations(range(n), b)])
    xi_c = np.array([z[list(S)].mean(0) for S in itertools.combinations(range(n), n - b)])
    mb, _ = geomed(xi_b); mc, _ = geomed(xi_c)
    print(f"n={n}: Med(xi_{{n,{n - b}}}) = {mc},  -(b/(n-b)) Med(xi_{{n,{b}}}) = {-(b / (n - b)) * mb}")
xi_h = np.array([z[list(S)].mean(0) for S in itertools.combinations(range(n), n // 2)])
mh, fh = geomed(xi_h)
f0 = np.mean(np.linalg.norm(xi_h, axis=1))
gsub = np.mean(xi_h / np.linalg.norm(xi_h, axis=1)[:, None], axis=0)
print(f"half sampling b=n/2 on a generic asymmetric population: objective at 0 = {f0:.12f}, min found = {fh:.12f}; |gradient at 0| = {np.linalg.norm(gsub):.2e}")
print("-> 0 is a median at half sampling for an arbitrary centered population (law of xi is symmetric by complementation).")
