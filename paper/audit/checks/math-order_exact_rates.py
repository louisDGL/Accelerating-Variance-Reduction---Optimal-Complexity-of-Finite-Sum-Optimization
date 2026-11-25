#!/usr/bin/env python3
"""Audit math-order, check 1: exact per-pass rates of the second moment of linear SAGA (identical Hessians).

Independent re-implementation (does not use the covariance recursion of the note B4 nor vrlab for the matrices):
  * scalar model  e+ = e - g (e - psi_i + psibar),  psi_i <- e  (error before the step), uniform memory psi_j = e_0;
  * i.i.d. indices: 4x4 one-step matrix T_n(g) on (A,B,C,P) = (E e^2, E mean psi^2, E psibar^2, E e psibar),
    re-derived by hand, validated by enumeration of ALL index sequences (small n);
  * fresh permutation per pass: 4x4 pass matrix R_n(g), built from the (n+1)x(n+1) linear map of one pass in visiting
    order and the covariance of a sample without replacement, validated by enumeration of ALL permutations (small n)
    and by a Monte Carlo at n = 100.
Then: exponents per pass (nats, squared error) at n = 100, 1000 against the laws quoted by the paper,
  i.i.d.:       min{2c, 1}   (fixed c = gamma n lambda),   2(1-g)/(2-g)  (fixed g = gamma lambda)
  permutation:  -2 log nu(c) (fixed c),                    min{2, log((2-g)/g)}  (fixed g),
and a comparison with vrlab.order_model (the code of the repository).

Run from the project root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/math-order_exact_rates.py
"""
import itertools
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True          # read-only audit: no bytecode cache written into the repository
sys.path.insert(0, os.getcwd())


# ------------------------------------------------------------------------------------------------ model
def run_indices(n, g, idx, e, psi):
    """Plain simulation of the scalar model along an index sequence; returns (e, psi, list of errors e_0..e_T)."""
    psi = np.array(psi, dtype=float)
    hist = [e]
    for i in idx:
        pbar = psi.mean()
        e_new = e - g * (e - psi[i] + pbar)
        psi[i] = e
        e = e_new
        hist.append(e)
    return e, psi, hist


def moments(e, psi):
    m = psi.mean()
    return np.array([e * e, np.mean(psi ** 2), m * m, e * m])


# ------------------------------------------------------------------------------------------------ i.i.d.
def T_iid(n, g):
    T = np.zeros((4, 4))
    T[0] = [(1 - g) ** 2, g * g, -g * g, 0.0]
    T[1] = [1 / n, 1 - 1 / n, 0, 0]
    T[2] = [1 / n ** 2, 1 / n ** 2, 1 - 2 / n, 2 / n - 2 / n ** 2]
    T[3] = [(1 - g) / n, -g / n, g / n, (1 - g) * (1 - 1 / n)]
    return T


def brute_iid(n, g, t, e0, psi0):
    acc = np.zeros(4)
    for idx in itertools.product(range(n), repeat=t):
        e, psi, _ = run_indices(n, g, idx, e0, psi0)
        acc += moments(e, psi)
    return acc / n ** t


def r_iid(n, c):
    return -n * math.log(np.abs(np.linalg.eigvals(T_iid(n, c / n))).max())


# ------------------------------------------------------------------------------------------------ permutation
def pass_map(n, g):
    """Linear map of ONE pass in visiting order: returns (a, Q) with e_k = a[k] e + sum_l Q[k, l] psi_{pi(l+1)},
    k = 0..n (pi(l+1) = slot visited at step l+1). Obtained by running the model on the n+1 basis states."""
    # rows = basis states (e; psi in visiting order); the identity order visits slot k at step k+1
    E = np.zeros(n + 1)
    E[0] = 1.0
    Psi = np.zeros((n + 1, n))
    Psi[1:, :] = np.eye(n)
    hist = np.zeros((n + 1, n + 1))            # hist[k] = e_k for each basis state
    hist[0] = E
    pbar = Psi.mean(axis=1)
    for k in range(n):
        d = E - Psi[:, k]
        E_new = E - g * (d + pbar)
        pbar = pbar + d / n
        Psi[:, k] = E
        E = E_new
        hist[k + 1] = E
    a = hist[:, 0].copy()
    Q = hist[:, 1:].copy()
    return a, Q


def R_perm(n, g):
    """4x4 pass matrix on (A, B, C, P) for a uniform permutation independent of the state."""
    a, Q = pass_map(n, g)
    beta = Q.sum(axis=1)                       # coefficient of m (psi_{pi(l)} = m + y_l)

    def cov(w, u):                             # E[(w.y)(u.y)] / v for a sample without replacement
        return n / (n - 1) * (w @ u) - w.sum() * u.sum() / (n - 1)

    # e' = e_n ; m' = mean_{k<n} e_k ; B' = mean_{k<n} e_k^2
    an, bn, qn = a[n], beta[n], Q[n]
    am, bm, qm = a[:n].mean(), beta[:n].mean(), Q[:n].mean(axis=0)
    cov_kk = n / (n - 1) * (Q[:n] ** 2).sum(axis=1) - beta[:n] ** 2 / (n - 1)
    # quadratic forms in (e, m): x^2 = xa^2 e^2 + 2 xa xb e m + xb^2 m^2 ; then e^2->A, e m->P, m^2->C, v->B-C
    def row(ca, cb, da, db, cv):               # E[(ca e + cb m + ..)(da e + db m + ..)] with fluctuation cv * v
        return [ca * da, cv, cb * db - cv, ca * db + cb * da]

    R = np.zeros((4, 4))
    R[0] = row(an, bn, an, bn, cov(qn, qn))
    R[1] = [np.mean(a[:n] ** 2), cov_kk.mean(), np.mean(beta[:n] ** 2) - cov_kk.mean(), 2 * np.mean(a[:n] * beta[:n])]
    R[2] = row(am, bm, am, bm, cov(qm, qm))
    R[3] = row(an, bn, am, bm, cov(qn, qm))
    M = np.array([[an, bn], [am, bm]])         # first-moment pass map (e, m) -> (E e', E m')
    return R, M


def brute_perm(n, g, e0, psi0):
    acc = np.zeros(4)
    cnt = 0
    for perm in itertools.permutations(range(n)):
        e, psi, _ = run_indices(n, g, perm, e0, psi0)
        acc += moments(e, psi)
        cnt += 1
    return acc / cnt


def r_perm(n, c):
    R, M = R_perm(n, c / n)
    return (-math.log(np.abs(np.linalg.eigvals(R)).max()), -2 * math.log(np.abs(np.linalg.eigvals(M)).max()))


def fluid_matrix(c):
    """Fluid limit of the first-moment pass map (closed form of the note, re-derived: I'' + c I' + c I = c m u)."""
    d2 = c * c / 4 - c
    if d2 > 0:
        d = math.sqrt(d2)
        sh, ch = math.sinh(d) / d, math.cosh(d)
    elif d2 < 0:
        w = math.sqrt(-d2)
        sh, ch = math.sin(w) / w, math.cos(w)
    else:
        sh, ch = 1.0, 1.0
    ex = math.exp(-c / 2)
    h1, dh1 = ex * sh, ex * (ch - c / 2 * sh)
    h0, dh0 = ex * (ch + c / 2 * sh), -c * ex * sh
    return np.array([[dh1, 1 + dh0 - dh1], [h1, h0 - h1]])


def r_rr_fluid(c):
    return -2 * math.log(np.abs(np.linalg.eigvals(fluid_matrix(c))).max())


def fluid_ode_check(c, steps=200000):
    """Independent check of the fluid matrix: explicit integration (RK4) of I' = e, e' = -c e - c I + c m u."""
    out = []
    for e0, m in ((1.0, 0.0), (0.0, 1.0)):
        y = np.array([0.0, e0])
        h = 1.0 / steps
        f = lambda u, y: np.array([y[1], -c * y[1] - c * y[0] + c * m * u])
        u = 0.0
        for _ in range(steps):
            k1 = f(u, y); k2 = f(u + h / 2, y + h / 2 * k1); k3 = f(u + h / 2, y + h / 2 * k2); k4 = f(u + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            u += h
        out.append((y[1], y[0]))
    return np.array(out).T


def empirical_exponent(Mat, per_pass_power, p0, p1):
    """Exponent of the A-coordinate of Mat^(per_pass_power * p) (1,1,1,1) between passes p0 and p1."""
    P = np.linalg.matrix_power(Mat, per_pass_power)
    x = np.ones(4)
    logs = {}
    scale = 0.0
    for p in range(1, p1 + 1):
        x = P @ x
        s = np.abs(x).max()
        x /= s
        scale += math.log(s)
        if p in (p0, p1):
            logs[p] = scale + math.log(abs(x[0]))
    return -(logs[p1] - logs[p0]) / (p1 - p0)


# ------------------------------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(12345)
    print("=" * 100)
    print("1. Validation of the matrices by exhaustive enumeration")
    worst = 0.0
    for n, g, t in ((3, 0.3, 7), (4, 0.9, 6), (5, 0.17, 5), (2, 1.3, 9)):
        e0, psi0 = rng.normal(), rng.normal(size=n)
        ex = brute_iid(n, g, t, e0, psi0)
        th = np.linalg.matrix_power(T_iid(n, g), t) @ moments(e0, psi0)
        err = np.abs(ex - th).max() / max(1.0, np.abs(ex).max())
        worst = max(worst, err)
        print(f"   iid   n={n} g={g} t={t}: all n^t={n**t} sequences, max rel. dev. of (A,B,C,P) from T^t: {err:.2e}")
    for n, g in ((3, 0.4), (4, 0.25), (5, 0.9), (6, 0.32), (7, 1.2)):
        e0, psi0 = rng.normal(), rng.normal(size=n)
        ex = brute_perm(n, g, e0, psi0)
        R, M = R_perm(n, g)
        th = R @ moments(e0, psi0)
        err = np.abs(ex - th).max() / max(1.0, np.abs(ex).max())
        worst = max(worst, err)
        print(f"   perm  n={n} g={g}: all n!={math.factorial(n)} permutations, max rel. dev. of (A,B,C,P)' from R: {err:.2e}")
    print(f"   worst deviation: {worst:.2e}  ->  {'OK' if worst < 1e-10 else 'PROBLEM'}")

    print("=" * 100)
    print("2. Monte Carlo at n = 100 (permutation), uniform memory, E e^2 after p passes against R^p (1,1,1,1)")
    n = 100
    for c in (4.0, 32.0):
        g = c / n
        runs, passes = 40000, 4
        e = np.ones(runs)
        psi = np.ones((runs, n))
        ar = np.arange(runs)
        for p in range(passes):
            perm = np.argsort(rng.random((runs, n)), axis=1)
            pbar = psi.mean(axis=1)
            for k in range(n):
                i = perm[:, k]
                d = e - psi[ar, i]
                en = e - g * (d + pbar)
                pbar += d / n
                psi[ar, i] = e
                e = en
        R, M = R_perm(n, g)
        th = np.linalg.matrix_power(R, passes) @ np.ones(4)
        thm = (np.linalg.matrix_power(M, passes) @ np.ones(2))[0]
        se = np.std(e ** 2) / math.sqrt(runs)
        sem = np.std(e) / math.sqrt(runs)
        print(f"   c={c:g}: MC E e^2 = {np.mean(e**2):.4e} +- {se:.1e}   exact {th[0]:.4e} (z = {(np.mean(e**2)-th[0])/se:+.1f})"
              f"   | MC E e = {e.mean():.4e} +- {sem:.1e}  exact {thm:.4e} (z = {(e.mean()-thm)/sem:+.1f})")

    print("=" * 100)
    print("3. i.i.d. indices, fixed c: exact exponent -n log rho(T_n(c/n)) against min{2c,1}")
    print("   (A-coordinate exponent from the uniform memory between passes 150 and 300 in brackets for n=1000)")
    cs = [0.1, 0.25, 0.4, 0.5, 1, 2, 4, 8, 16, 32]
    print("   c        " + "".join(f"{c:>9g}" for c in cs))
    for n in (100, 1000, 10000, 100000):
        print(f"   n={n:<7d}" + "".join(f"{r_iid(n, c):9.4f}" for c in cs))
    print("   law      " + "".join(f"{min(2*c,1):9.4f}" for c in cs))
    print("   uniform memory, n=1000, passes 150->300: " +
          "".join(f"{empirical_exponent(T_iid(1000, c/1000), 1000, 150, 300):9.4f}" for c in cs))
    print("   first-order formula 1 - (c^2/(2c-1) - 1/2)/n at n=1000: " +
          "".join(f"{(1 - (c*c/(2*c-1) - 0.5)/1000):9.4f}" if c > 0.5 else "      n/a" for c in cs))

    print("=" * 100)
    print("4. i.i.d. indices, fixed g: exact exponent against 2(1-g)/(2-g)")
    gs = [0.05, 0.1, 0.2, 0.25, 0.32, 0.5, 0.9, 1.1]
    print("   g        " + "".join(f"{g:>9g}" for g in gs))
    for n in (100, 1000, 10000, 100000):
        print(f"   n={n:<7d}" + "".join(f"{r_iid(n, n*g):9.4f}" for g in gs))
    print("   law      " + "".join(f"{2*(1-g)/(2-g):9.4f}" for g in gs))

    print("=" * 100)
    print("5. fresh permutation per pass, fixed c: exact mean-square exponent -log rho(R_n(c/n)) [squared-mean exponent]")
    cs = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    print("   c        " + "".join(f"{c:>17g}" for c in cs))
    cache = {}
    for n in (100, 1000):
        vals = [r_perm(n, c) for c in cs]
        cache[n] = vals
        print(f"   n={n:<7d}" + "".join(f"{a:9.4f}[{b:6.4f}]" for a, b in vals))
    print("   fluid law" + "".join(f"{r_rr_fluid(c):17.4f}" for c in cs))
    print("   min{2c,2}" + "".join(f"{min(2*c,2):17.4f}" for c in cs))
    for c in (1.0, 4.0, 32.0):
        d = np.abs(fluid_matrix(c) - fluid_ode_check(c, 20000)).max()
        print(f"   fluid matrix against RK4 integration of the ODE, c={c:g}: max abs dev {d:.1e}")
    for c in (1e2, 5e2, 1e3):
        print(f"   c={c:g}: c (2 - r_rr(c)) = {c*(2-r_rr_fluid(c)):.4f}   (claim: -> 2(e-2) = {2*(math.e-2):.4f})")

    print("=" * 100)
    print("6. fresh permutation per pass, fixed g: exact exponent against min{2, log((2-g)/g)}")
    gs = [0.05, 0.1, 0.2, 0.2384, 0.25, 0.32, 0.5, 0.9, 1.1, 1.25]
    print("   g        " + "".join(f"{g:>9g}" for g in gs))
    for n in (100, 1000):
        print(f"   n={n:<7d}" + "".join(f"{r_perm(n, n*g)[0]:9.4f}" for g in gs))
    try:
        from vrlab import order_model as om
        for n in (10000, 100000):
            print(f"   n={n:<7d}" + "".join(f"{om.rr_ms_exponent(n, n*g):9.4f}" for g in gs) + "   (vrlab.order_model)")
    except Exception as exc:                                   # pragma: no cover
        om = None
        print("   vrlab not importable:", exc)
    print("   law      " + "".join(f"{min(2, math.log((2-g)/g)):9.4f}" for g in gs))
    print("   uniform memory, n=1000, A-coordinate exponent between passes 40 and 80: " +
          "".join(f"{empirical_exponent(R_perm(1000, g)[0], 1, 40, 80):9.4f}" for g in (0.05, 0.25, 0.5)) +
          "   (g = 0.05, 0.25, 0.5)")

    print("=" * 100)
    print("7. Largest exact permutation exponent over the step (is 'up to two' / 'nearly e^-2' right at finite n?)")
    for n in (100, 1000):
        grid = np.exp(np.linspace(math.log(0.5), math.log(0.6 * n), 60))
        vals = [r_perm(n, c)[0] for c in grid]
        k = int(np.argmax(vals))
        print(f"   n={n}: max over 60 log-spaced c in [0.5, {0.6*n:g}] = {vals[k]:.4f} at c = {grid[k]:.1f} (g = {grid[k]/n:.3f});"
              f" i.i.d. expectation at the same c: {r_iid(n, grid[k]):.4f}; at c=2: perm {r_perm(n, 2.0)[0]:.4f}, iid {r_iid(n, 2.0):.4f}")
    print("=" * 100)
    print("9. 'At most one nat per pass under independent sampling': sup over the step of the exact i.i.d. exponent")
    for n in (100, 1000):
        grid = np.exp(np.linspace(math.log(0.3), math.log(0.9 * n), 400))
        vals = [r_iid(n, c) for c in grid]
        k = int(np.argmax(vals))
        print(f"   n={n}: max over 400 log-spaced c in [0.3, {0.9*n:g}] = {vals[k]:.5f} at c = {grid[k]:.2f};"
              f" bound -n log(1-1/n) = {-n*math.log(1-1/n):.5f}; at g = 1/4: {r_iid(n, n/4):.4f} (6/7 = {6/7:.4f})")
    print("   memory factor (1-b/n)^(n/b), n=1000: " + ", ".join(f"b={b}: {(1-b/1000)**(1000/b):.4f}" for b in (1, 5, 20, 50))
          + f"; e^-1 = {math.exp(-1):.4f}")
    print(f"   kappa below which ((kappa-1)/(kappa+1))^2 < e^-1: {(math.sqrt(math.e)+1)/(math.sqrt(math.e)-1):.4f}")
    if om is not None:
        print("=" * 100)
        print("8. Agreement with vrlab.order_model (repository code)")
        dev = 0.0
        for n in (100, 1000):
            for c in (0.25, 1, 4, 32, 0.25 * n):
                dev = max(dev, abs(om.iid_exponent(n, c) - r_iid(n, c)), abs(om.rr_ms_exponent(n, c) - r_perm(n, c)[0]),
                          abs(om.rr_exponent(c, n=n) - r_perm(n, c)[1]), abs(om.rr_exponent(c) - r_rr_fluid(c)))
        print(f"   max |vrlab - this script| over iid_exponent, rr_ms_exponent, rr_exponent(n), rr_exponent(limit): {dev:.2e}")


if __name__ == "__main__":
    main()
