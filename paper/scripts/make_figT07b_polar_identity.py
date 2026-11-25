"""Figure T07b of the lecture (paper/talk/figures/figT07b_polar_identity): the same anatomy for the polar step, on a
quadratic. Along a run of the nuclear-scaled polar step on least squares, the exact one-step contraction is the
certified one times a smoothness slack that is at most 2 times the sharpness that the run meets (mu_PL / mu_op, bounded
only by the condition number kappa_op). Explanatory figure, not in the paper; companion of figT07_two_identities
(SAGA, Eq. (1)). It explains Eq. (2) of paper/paper_saga_to_muon.tex and the paragraph "The same anatomy for polar
steps" in its corrected wording (paper/audit/AUDIT.md, findings G and L): the identity is stated for quadratics, and
the factor of about 4 is observed in the tail of a generic run.

The instance and the method (nothing is drawn by hand):
    F(W) = (1/2n) sum_i |W x_i - y_i|^2 + (lam/2) |W|_F^2,   W in R^{K x d}: the least-squares conditioning-limited
    instance of the paper, n = 200, K x d = 8 x 40 (specification read from muon/results/M9_summary.json, key
    A / ls-cond / spec; built with muon/lab/problems.make_matrix_ls and solved with muon/lab/optimum.solve_optimum, as
    muon/experiments/common.problem does);
    W+ = W - |G|_* polar(G) / L_op,   G = grad F(W) (exact gradients),   W_0 = 0,   STEPS steps.
F is quadratic with Hessian H = I_K (x) A, A = Sigma + lam I (Sigma = X^T X / n): L_op = sum of the min(K, d) largest
eigenvalues of A and mu_op = smallest eigenvalue of A (the exact constants of least squares; prob.mu_op = lam of the
project is only a lower bound and is not used). With O = polar(G), J = <O, H O> in [mu_op, L_op] and
mu_PL = |G|_*^2 / (2 (F(W) - F_*)) in [mu_op, J], Eq. (2) of the paper reads
    (F(W+) - F_*) / (F(W) - F_*) = 1 - (mu_PL / L_op) (2 - J / L_op),
and J = L_op, mu_PL = mu_op give the certified factor 1 - 1/kappa_op, kappa_op = L_op / mu_op. Hence, exactly,
    [1 - (F(W+) - F_*) / (F(W) - F_*)] / (1 / kappa_op) = (mu_PL / mu_op) x (2 - J / L_op).

What is drawn, against the step k (log scale), on a log scale: the ratio of the exact to the certified contraction
gap (ink); the sharpness ratio mu_PL / mu_op (colour of I4, the strong-convexity slack of SAGA, drawn as a wide line
under the ink curve, which it nearly carries alone) with its bound kappa_op; the smoothness slack 2 - J / L_op with
its bound 2 (colour of I2, the cocoercivity slack of SAGA); the level 1 is labelled "factor 1 = certificate" (the
smoothness slack lies on it); the instance (n, K x d, kappa_op) is written in the panel. Two numbers are annotated:
the level of the ratio in the tail ("about x 4 in the tail", observed; the script stops unless the median over steps
5000-10000 is in [3.5, 4.5]) and the factor between the number of steps after which the certificate guarantees
that F - F_* is divided by 10^6 and the number of steps the run takes (the whole run: a larger factor, as the audit
notes). The isolated one-step blips of the tail are steps at which J drops (the slack rises to about 1.17); they are
steps of this floating-point run, but their places and the third digit of the tail depend on rounding: rerunning
with a polar factor computed otherwise (eigendecomposition of G G^T) or with W_0 perturbed by 1e-12 moves the blips
and gives 3.85-4.09 over steps 5000-10000 and 3.95-3.97 at the last step (review check), hence "about 4", not "4.0".

The iteration is run in the error variable E = W - W_*, with G = E A and F - F_* = <E, E A> / 2 (exact for a quadratic
whose minimizer is W_*; no cancellation, so that the identity can be checked far below the rounding floor of F).
Checks (the script stops if one fails): the instance against muon/results/M9_summary.json (|W_*|_F^2, |W_*|_op) and
muon/results/digest.md (L_op, smallest eigenvalue of the Hessian); L_op against the top eigenvalues of A; the first
CHECK_STEPS steps against the functions of the project (prob.grad, prob.Fgap) at the same iterates; identity (2) at
every step to 1e-9 (relative error of the contraction gap); mu_op <= mu_PL <= J <= L_op at every step.

Status. Proved: Eq. (2) (Eq. (17) of Shulgin et al., Eq. (8) of Jawanpuria et al.) and the ranges of the two factors.
Measured, one deterministic run on one instance: the three curves and the two annotated numbers. Not measured: the
polar step on logistic or softmax losses (no curve is drawn for them); nothing here says that generic runs are
attracted to a ray.

Progressive build (states s1-s2): s1 the ratio exact / certified; s2 adds the two factors.

The script prints every number it plots. About 5 s.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT07b_polar_identity.py
Output: paper/talk/figures/figT07b_polar_identity.pdf and .png, and the states _s1, _s2
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

for _p in (es.ROOT, os.path.join(es.ROOT, "muon")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NAME = "figT07b_polar_identity"
SPEC = {"loss": "ls", "n": 200, "d": 40, "K": 8, "kappa": 10000.0, "alpha": 2.0, "seed": 0}   # A / ls-cond / spec
STEPS = 10000
CHECK_STEPS = 300            # steps compared with prob.grad and prob.Fgap of the project
TARGET = 1e-6                # "the whole run": F - F_* divided by 10^6
TAIL = (5000, STEPS)         # steps of "the tail"
K_LINE = 12                  # right end of the reference line at kappa_op (it would cross the legend)
STEPS_PRINT = (1, 2, 3, 6, 11, 21, 51, 101, 201, 501, 1001, 2001, 5001, 10000)
DIGEST_INSTANCE = "Instance `ls_n200_d40_K8_k10000_a2_s0`:"


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def instance():
    """The least-squares instance of the paper with its minimizer, and the checks against the repository."""
    from lab import optimum, problems
    with open(os.path.join(es.ROOT, "muon", "results", "M9_summary.json"), encoding="utf-8") as f:
        ref = json.load(f)["A"]["ls-cond"]
    if ref["spec"] != SPEC:
        raise SystemExit(f"M9_summary.json: A / ls-cond / spec is {ref['spec']}, expected {SPEC}")
    prob = problems.make_matrix_ls(**{k: v for k, v in SPEC.items() if k != "loss"})
    optimum.solve_optimum(prob)
    A = prob.Sigma_hat + prob.lam * np.eye(prob.d)
    eig = np.linalg.eigvalsh(A)
    mu_op = float(prob.lam + prob.cov_eigs.min())             # exact for least squares (prob.mu_op = lam: lower bound)
    L_op = float(prob.L_op)
    w_fro2 = float(np.vdot(prob.W_star, prob.W_star))
    w_op = float(np.linalg.svd(prob.W_star, compute_uv=False)[0])
    with open(os.path.join(es.ROOT, "muon", "results", "digest.md"), encoding="utf-8") as f:
        line = [s for s in f.read().splitlines() if s.startswith(DIGEST_INSTANCE)]
    bad = []
    if not np.isclose(w_fro2, ref["norms"]["D0"], rtol=1e-9) or not np.isclose(w_op, ref["norms"]["W_star_op"], rtol=1e-9):
        bad.append(f"|W_*|_F^2 = {w_fro2!r}, |W_*|_op = {w_op!r} against {ref['norms']} of M9_summary.json")
    if len(line) != 1 or f"L_op = {L_op:.4f};" not in line[0] or f"Hessian at the optimum in [{mu_op:.2e}," not in line[0]:
        bad.append(f"L_op = {L_op:.4f} or mu_op = {mu_op:.2e} is not in the instance line of muon/results/digest.md")
    if not np.isclose(L_op, eig[-prob.r:].sum(), rtol=1e-12) or not np.isclose(mu_op, eig[0], rtol=1e-10):
        bad.append("L_op or mu_op is not the sum of the top eigenvalues / the smallest eigenvalue of Sigma + lam I")
    if bad:
        raise SystemExit("the instance does not agree with the repository: " + "; ".join(bad))
    return prob, A, mu_op, L_op, {"w_fro2": w_fro2, "w_op": w_op, "ref": ref["norms"], "digest": line[0][:200]}


def data():
    prob, A, mu_op, L_op, checks = instance()
    kappa_op = L_op / mu_op
    E = -prob.W_star.copy()                                    # W_0 = 0
    W = np.zeros_like(E)                                       # the iterate itself, for the first CHECK_STEPS steps
    rec = {k: np.empty(STEPS) for k in ("gap", "contraction", "identity", "J", "mu_PL", "dist2")}
    code_err = 0.0
    for t in range(STEPS):
        G = E @ A                                              # grad F(W) = (W - W_*)(Sigma + lam I)
        gap = 0.5 * float(np.vdot(E, G))                       # F(W) - F_*
        if t < CHECK_STEPS:                                    # the same quantities through the code of the project
            G_code, gap_code = prob.grad(W), prob.Fgap(W)
            code_err = max(code_err, float(np.linalg.norm(G_code - G) / np.linalg.norm(G)), abs(gap_code / gap - 1.0))
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        O, nuc = U @ Vt, float(s.sum())                        # polar factor and nuclear norm of the gradient
        J = float(np.vdot(O, O @ A))                           # <O, H O>
        mu_PL = nuc ** 2 / (2.0 * gap)
        E_next = E - (nuc / L_op) * O
        gap_next = 0.5 * float(np.vdot(E_next, E_next @ A))
        rec["gap"][t], rec["dist2"][t] = gap, float(np.vdot(E, E))
        rec["contraction"][t] = 1.0 - gap_next / gap           # exact contraction gap of the step t -> t + 1
        rec["identity"][t] = (mu_PL / L_op) * (2.0 - J / L_op)  # the same, by Eq. (2)
        rec["J"][t], rec["mu_PL"][t] = J, mu_PL
        E = E_next
        if t < CHECK_STEPS:
            W = W - (nuc / L_op) * O
    identity_err = float(np.max(np.abs(rec["contraction"] - rec["identity"]) / rec["contraction"]))
    if code_err > 1e-9:
        raise SystemExit(f"the first {CHECK_STEPS} steps disagree with prob.grad / prob.Fgap: {code_err:.2e}")
    if identity_err > 1e-9:
        raise SystemExit(f"identity (2) fails: largest relative error of the contraction gap {identity_err:.2e}")
    tol = 1e-9
    if not (np.all(rec["mu_PL"] >= mu_op * (1 - tol)) and np.all(rec["mu_PL"] <= rec["J"] * (1 + tol))
            and np.all(rec["J"] <= L_op * (1 + tol)) and np.all(rec["J"] >= mu_op * (1 - tol))):
        raise SystemExit("mu_op <= mu_PL <= J <= L_op fails at a step")
    ratio = rec["contraction"] * kappa_op                      # exact / certified contraction gap
    sharp = rec["mu_PL"] / mu_op
    smooth = 2.0 - rec["J"] / L_op
    if float(np.max(np.abs(ratio / (sharp * smooth) - 1.0))) > 1e-9:
        raise SystemExit("exact / certified is not the product of the two factors")
    # the whole run: steps to divide F - F_* by 1 / TARGET, against the number the certificate allows
    rel = np.append(rec["gap"][1:], np.nan) / rec["gap"][0]    # (F(W_k) - F_*) / (F(W_0) - F_*) after k = t + 1 steps
    k_run = int(np.argmax(rel <= TARGET)) + 1
    k_cert = int(np.ceil(np.log(1.0 / TARGET) / -np.log1p(-1.0 / kappa_op)))
    dist_rel = rec["dist2"] / rec["dist2"][0]
    k_dist = int(np.argmax(dist_rel <= TARGET))                # steps to the target of the paper (squared distance)
    k_dist_cert = np.log(rec["gap"][0] / rec["gap"][k_dist]) / -np.log1p(-1.0 / kappa_op)
    tail = slice(TAIL[0] - 1, TAIL[1])
    tail_median = float(np.median(ratio[tail]))
    if not 3.5 <= tail_median <= 4.5:                         # the figure says "about 4 in the tail"
        raise SystemExit(f"the tail of exact / certified is not about 4: median {tail_median:.3f}")
    return {"tail_median": tail_median,"k": np.arange(1, STEPS + 1), "ratio": ratio, "sharp": sharp, "smooth": smooth, "gap": rec["gap"],
            "dist_rel": dist_rel, "mu_op": mu_op, "L_op": L_op, "kappa_op": kappa_op, "lam": prob.lam,
            "kappa_op_lab": prob.kappa_op, "identity_err": identity_err, "code_err": code_err, "checks": checks,
            "k_run": k_run, "k_cert": k_cert, "k_dist": k_dist, "k_dist_cert": float(k_dist_cert),
            "tail": (float(ratio[tail].min()), float(ratio[tail].max())), "last": float(ratio[-1]),
            "gap0": float(rec["gap"][0]), "gap0_code": float(prob.Fgap(np.zeros_like(E)))}


def print_all(D):
    c = D["checks"]
    print(f"instance A / ls-cond of M9_summary.json: n = {SPEC['n']}, K x d = {SPEC['K']} x {SPEC['d']}, lam = "
          f"{D['lam']:.6g}; |W_*|_F^2 = {c['w_fro2']:.6f}, |W_*|_op = {c['w_op']:.6f} (summary: {c['ref']['D0']:.6f}, "
          f"{c['ref']['W_star_op']:.6f})")
    print(f"L_op = {D['L_op']:.6g}, mu_op = lam + lambda_min(Sigma) = {D['mu_op']:.6g}, kappa_op = {D['kappa_op']:.2f} "
          f"(with the lower bound mu_op = lam of the project: {D['kappa_op_lab']:.2f}); certified contraction gap "
          f"1 / kappa_op = {1 / D['kappa_op']:.3e} per step; F(W_0) - F_* = {D['gap0']:.6f} (prob.Fgap: "
          f"{D['gap0_code']:.6f})")
    print(f"checks: first {CHECK_STEPS} steps against prob.grad / prob.Fgap, largest relative difference "
          f"{D['code_err']:.1e}; identity (2) at the {STEPS} steps, largest relative error of the contraction gap "
          f"{D['identity_err']:.1e}")
    print("  step k | (F(W_k-1) - F_*) / (F(W_0) - F_*) | exact / certified gap | mu_PL / mu_op | 2 - J / L_op")
    for k in STEPS_PRINT:
        j = k - 1
        print(f"  {k:6d} | {D['gap'][j] / D['gap'][0]:34.3e} | {D['ratio'][j]:21.3f} | {D['sharp'][j]:13.3f} | "
              f"{D['smooth'][j]:.4f}")
    print(f"over the run: exact / certified {D['ratio'].min():.2f}-{D['ratio'].max():.1f}; sharpness ratio "
          f"{D['sharp'].min():.2f}-{D['sharp'].max():.1f} (at most kappa_op = {D['kappa_op']:.0f}); smoothness slack "
          f"{D['smooth'].min():.4f}-{D['smooth'].max():.4f} (at most 2)")
    print(f"tail, steps {TAIL[0]}-{TAIL[1]}: exact / certified between {D['tail'][0]:.3f} and {D['tail'][1]:.3f}, "
          f"median {D['tail_median']:.3f}; {D['last']:.3f} at the last step (drawn: 'about x 4 in the tail')")
    print(f"whole run: F - F_* divided by {1 / TARGET:.0e} after {D['k_run']} steps; the certificate (1 - 1/kappa_op)^k "
          f"allows {D['k_cert']} steps: factor {D['k_cert'] / D['k_run']:.2f}")
    print(f"   (not drawn) target of the paper, |W - W_*|_F^2 <= {TARGET:.0e} |W_0 - W_*|_F^2: {D['k_dist']} steps; the "
          f"certificate allows {D['k_dist_cert']:.0f} steps for the same decrease of F - F_*: factor "
          f"{D['k_dist_cert'] / D['k_dist']:.2f}; mean of exact / certified over these steps: "
          f"{D['ratio'][:D['k_dist']].mean():.2f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    SM, SH = ps.INEQ["I2"], ps.INEQ["I4"]            # smoothness slack: colour of I2; sharpness met: colour of I4
    fig = es.figure("talk", height=2.6)
    ax = fig.add_subplot(111)
    b = es.Build()
    k = D["k"]
    dash = (0, (4.0, 2.2))
    note = dict(textcoords="offset points", fontsize=es.small(), color=ink["secondary"])
    ax.axhline(1.0, color=ink["axis"], lw=0.9, zorder=1)
    # "factor 1": the label names the level, not the smoothness slack that lies on it from s2 on
    ax.annotate("factor 1 = certificate", (k[-1], 1.0), xytext=(-3, -3), ha="right", va="top", **note)
    ax.annotate(rf"$n = {SPEC['n']}$, $W \in \mathbb{{R}}^{{{SPEC['K']} \times {SPEC['d']}}}$, "
                rf"$\kappa_{{\mathrm{{op}}}} = {D['kappa_op']:.0f}$", (1.0, 12.0), xytext=(4, 0), ha="left",
                va="center", **note)
    # s1: the ratio of the exact to the certified contraction gap
    ax.plot(k, D["ratio"], color=ink["primary"], lw=2.0, zorder=5, label="exact / certified contraction")
    # s2: the two factors (the product of the two is the ink curve, exactly)
    b.add(2, ax.plot(k, D["sharp"], color=SH, lw=5.5, alpha=0.75, zorder=3, solid_capstyle="butt",
                     label=r"sharpness met, $\mu_{\mathrm{PL}}/\mu_{\mathrm{op}}$"))
    b.add(2, ax.plot(k, D["smooth"], color=SM, lw=1.9, zorder=4,
                     label=r"smoothness slack, $2 - J/L_{\mathrm{op}}$"))
    b.add(2, ax.axhline(2.0, color=ink["secondary"], lw=0.9, ls=dash, zorder=2))
    b.add(2, ax.annotate("bound 2", (k[0], 2.0), xytext=(4, 2.5), ha="left", va="bottom", **note))
    b.add(2, ax.plot([k[0], K_LINE], [D["kappa_op"]] * 2, color=ink["secondary"], lw=0.9, ls=dash, zorder=2))
    b.add(2, ax.annotate(r"bound $\kappa_{\mathrm{op}}$", (K_LINE, D["kappa_op"]), xytext=(4, 0), ha="left",
                         va="center", **note))
    # the two annotated numbers: the tail and the whole run
    j = D["k_run"] - 1
    ax.plot([D["k_run"]], [D["ratio"][j]], ls="", marker="o", ms=6.5, mfc=ink["primary"], mec=ink["surface"], mew=1.2,
            zorder=7)
    ax.annotate(rf"$F - F_\ast$ divided by $10^6$ in {D['k_run']:,} steps;" "\n"
                rf"the certified rate takes {D['k_cert']:,} ($\times\,{D['k_cert'] / D['k_run']:.1f}$)",
                (D["k_run"], D["ratio"][j]), xytext=(-4, 26), ha="center", va="bottom", linespacing=1.2, **note)
    ax.annotate("", (D["k_run"], D["ratio"][j]), xytext=(-4, 24), textcoords="offset points",      # its leader
                arrowprops=dict(arrowstyle="-", color=ink["secondary"], lw=0.8, shrinkA=0.0, shrinkB=4.0))
    # "about 4": the third digit of the tail and the place of its blips depend on rounding (see the docstring)
    ax.annotate("about " r"$\times\,4$" "\nin the tail", (k[-1], D["last"]), xytext=(-3, 4), ha="right",
                va="bottom", linespacing=1.15, **note)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, STEPS)
    ax.set_ylim(0.45, 3000.0)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", r"$10^3$", r"$10^4$"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(r"step $k$ of $W^+ = W - \|G\|_*\,\mathrm{polar}(G)\,/\,L_{\mathrm{op}}$, $G = \nabla F(W)$")
    ax.set_ylabel("factor over the certificate")
    b.legend(ax, loc="upper right", handlelength=1.5, borderaxespad=0.15, labelspacing=0.28)
    es.save(fig, NAME, "talk", build=b,
            note=("least squares (quadratic), one instance, exact gradients; polar step on logistic / softmax: not "
                  "measured"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
