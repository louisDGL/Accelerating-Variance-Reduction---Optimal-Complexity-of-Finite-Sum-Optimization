"""Integer counterexample to the monotonicity of the spectral norm of the error along the polar flow (Proposition 4.2(e)).

One-sided quadratic F = (1/2) tr(E Sigma E^T), K = d = 3:
    E_0 = diag(300, 200, 1),   Sigma = [[2, 4, 12], [4, 9, 20], [12, 20, 89]]   (det Sigma = 2).
The top singular value of E_0 is simple, with singular vector e_1, and d/dt ||E||_op = - Q_11 at t = 0, where
Q = polar(E_0 Sigma). Here Q_11 = -0.01433 < 0: the spectral norm of the error increases, from 300 to 303.27 at t = 33.5,
while the Frobenius norm of the error and F decrease; the flow reaches the solution at T = 1.20 ||E_0||_op.
The example was found by the referee of Section 4 (review of 16 November 2026), who certified the sign of Q_11 by exact
rational arithmetic; this script recomputes Q in 40-digit arithmetic, evaluates the exact first phase (the segment
E_0 - t Q until tau* = 14.34, where ||E||_op = 300.86) and integrates the whole flow with small polar steps.
Run: .venv/Scripts/python muon/theory_checks/counterexample_opnorm_integer.py
"""
import numpy as np, mpmath as mp
mp.mp.dps = 40
E0 = mp.diag([300, 200, 1])
S = mp.matrix([[2, 4, 12], [4, 9, 20], [12, 20, 89]])
M = E0 * S
U, sv, Vh = mp.svd_r(M)
Q = U * Vh
print('det Sigma =', mp.det(S), '; eig Sigma =', [mp.nstr(x, 8) for x in mp.eigsy(S)[0]])
print('Q_11 =', mp.nstr(Q[0, 0], 15), '; det Q =', mp.nstr(mp.det(Q), 10), '; tr Q =', mp.nstr(Q[0,0]+Q[1,1]+Q[2,2], 10))
# first phase: the flow is the segment E_0 - t Q until tau* = smallest generalized eigenvalue of (|G_0|, Sigma)
absG = Vh.T * mp.diag([sv[i] for i in range(3)]) * Vh              # |G_0| = (Sigma E_0^2 Sigma)^(1/2)
C = mp.cholesky(S)                                                 # Sigma = C C^T
Ci = C ** -1
gen = sorted(mp.eigsy(Ci * absG * Ci.T)[0])
tau = gen[0]
print('generalized eigenvalues of (|G_0|, Sigma):', [mp.nstr(x, 10) for x in gen],
      '; product', mp.nstr(gen[0] * gen[1] * gen[2], 10), '(= 300 * 200 * 1)')
for frac in (0, mp.mpf(1) / 10, mp.mpf(1) / 2, 1):
    Et = E0 - frac * tau * Q
    print('  t =', mp.nstr(frac * tau, 8), ': ||E||_op =', mp.nstr(mp.svd_r(Et, compute_uv=False)[0], 10),
          '; ||E||_F =', mp.nstr(mp.mnorm(Et, 'f'), 8))
# flow by small polar steps in float64: track ||E||_op and ||E||_F
E = np.diag([300.0, 200.0, 1.0]); Sig = np.array([[2, 4, 12], [4, 9, 20], [12, 20, 89]], float)
eta = 0.01; t = 0.0; mx = (300.0, 0.0); nF0 = np.linalg.norm(E); Fmono = True; prevF = nF0; T = None
while t < 500:
    U_, s_, Vt_ = np.linalg.svd(E @ Sig)
    keep = s_ > 1e-9 * s_[0]
    E = E - eta * (U_[:, keep] @ Vt_[keep]); t += eta
    n2 = np.linalg.norm(E, 2); nF = np.linalg.norm(E)
    if nF > prevF + 1e-9: Fmono = False
    prevF = nF
    if n2 > mx[0]: mx = (n2, t)
    if n2 < 3 * eta:
        T = t; break
print(f'max ||E(t)||_op = {mx[0]:.4f} at t = {mx[1]:.2f} (start 300); ||E||_F nonincreasing: {Fmono}; extinction time {T:.2f} = {T/300:.4f} ||E_0||_op')
