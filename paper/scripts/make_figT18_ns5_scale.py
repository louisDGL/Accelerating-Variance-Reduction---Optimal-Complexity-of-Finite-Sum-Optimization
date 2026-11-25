"""Figure T18 of the lecture (paper/talk/figures/figT18_ns5_scale): the exact polar factor does not see the scale of its
input; Newton-Schulz with its normalization constant eps = 1e-7 does, once the input is small compared with eps.
Under a cooldown fed by an estimator that vanishes at the minimizer, this constant sets the floor near 1e-18 of the
runs. Explanatory figure, not in the paper. It explains the floor of ns5 in finding (M5) of
paper/paper_saga_to_muon.tex and the role of the scale invariance o(sM) = o(M) in Proposition 5(c) (least squares
only), see fact F14 of muon/notes_theory.md and the docstring of muon/lab/polar.py.

Panels:
  (a) exact evaluation with muon/lab/polar.py. M: one 8 x 40 matrix with i.i.d. standard Gaussian entries (seed
      M_SEED), divided by its Frobenius norm. For scales s from 1e-14 to 1: the largest and the smallest singular
      value of o(sM) for the exact polar factor (polar_svd: identically 1), for ns5 with eps = 1e-7 (the default, the
      value of the public Muon code) and for ns5 with eps = 1e-30. The singular values of ns5 are computed twice, from
      the matrix polar.ns5(sM) and from the scalar map p^5(s_k s / (s + eps)) (polar.ns_scalar_map; the map itself is
      drawn in figT03); the script stops if they differ. Below the scale where p^5 is linear the output is
      proportional to the input: (3.4445^5 / eps) sM to first order.
      Thin dark line (state 3 of the build): one run of panel (b) (eps = 1e-7, seed RUN_SEED), rerun here with the
      project code (muon/lab/vrmuon.vr_muon, a recording wrapper around polar.ns5 as the map): at the first
      iteration of each pass and at the last iteration of the run, the Frobenius norm of the matrix fed to ns5 (the
      SAGA estimator through Nesterov momentum 0.9) against the largest singular value of what ns5 returns; dots at
      passes 30, 60, 90 and 300 (the same passes are marked in (b) on this run). The rerun is checked against the
      cached run (same relative squared distances at every record; the script stops otherwise), and the input of the
      map is checked to be frozen from pass FROZEN_FROM on. The input of the run is not M: its line need not lie on
      the curves of M.
  (b) the consequence, from the cache of experiment M9, part C (muon/results/M9_law.pkl, keys as in
      paper/scripts/make_figP4_recipe.py; instance ls-cond: least squares, n = 200, K x d = 8 x 40, kappa = 1e4;
      batch one; W <- W - lr_t ns5(.), lr_t = 0.1 * 0.8^(t/n), Nesterov momentum 0.9; 300 passes, seeds 10..14):
      relative squared distance against passes with eps = 1e-7 and with eps = 1e-30, each seed (thin) and the median
      over the seeds at each record. The floors printed on the panel are the medians over the five seeds of the
      lowest recorded value, M9_summary.json["C"]["ls-cond"]["deep"][<label>]["min_rel"]["median"]; they are checked
      against muon/results/digest.md and against the cached runs (the script stops if they disagree).

Status. Proved: the exact polar factor is scale invariant, ns5 is not (its singular values are
p^5(s_k s / (s + eps))). Measured (experiment M9, part C; two instances, five seeds each; only ls-cond is drawn):
with eps = 1e-7 the runs stop near 1e-18 (medians 1.1e-18 on ls-cond, 9.8e-19 on soft-cond), with eps = 1e-30 they go
on to the rounding level (4.1e-30 and 1.5e-29). Measured here on one run: where the input of the map stands when the
run leaves its schedule. NOT derived: the level 1e-18 itself; the figure shows that it is consistent with the scale
at which the plateau of the map ends. Five Newton-Schulz steps, eps = 1e-7 and Nesterov momentum are Muon's; the
geometric cooldown and the momentum coefficient 0.9 are choices of the paper.

Progressive build (states s1-s3): s1 panel (a), the three maps; s2 adds panel (b); s3 adds the run to (a) and its
marked passes to (b).

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT18_ns5_scale.py      (about 30 s)
Output: paper/talk/figures/figT18_ns5_scale.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "muon", "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "muon"))
import explain_style as es  # noqa: E402
from lab import optimum, polar, problems, vrmuon  # noqa: E402
from lab.exact import expected_cost  # noqa: E402

NAME = "figT18_ns5_scale"
M_SHAPE, M_SEED = (8, 40), 0
EPS_DEFAULT, EPS_SMALL = 1e-7, 1e-30
SCALES = np.logspace(-14.0, 0.0, 281)        # 20 scales per decade
INSTANCE = "ls-cond"
LABELS = {EPS_DEFAULT: "ns5 + Nesterov at q = 0.8, lr_0 = 0.1",
          EPS_SMALL: "ns5 + Nesterov, eps = 1e-30, at q = 0.8, lr_0 = 0.1"}
RUN_SEED = 10                                # the run of panel (b) whose map input is recorded for panel (a)
FROZEN_FROM = 100                            # the input of the map no longer moves from this pass on (checked)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def task_key(task):
    """Key of a task in the caches of muon/experiments/common.py (common.task_key)."""
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def thin(x, m=240):
    """The thinning of the cached traces (muon/experiments/common._thin)."""
    x = np.asarray(x)
    if len(x) <= m:
        return x
    return x[np.unique(np.round(np.linspace(0, len(x) - 1, m)).astype(int))]


def data_maps():
    """Panel (a): extreme singular values of o(sM) for the three maps, each ns5 value computed in two ways."""
    Z = np.random.default_rng(M_SEED).standard_normal(M_SHAPE)
    M = Z / np.linalg.norm(Z)
    sv = polar.singular_values(M)
    out = {"sv": sv, "polar": [], EPS_DEFAULT: [], EPS_SMALL: []}
    worst = 0.0
    for s in SCALES:
        p = polar.singular_values(polar.polar_svd(s * M))
        out["polar"].append((p[0], p[-1]))
        for eps in (EPS_DEFAULT, EPS_SMALL):
            a = polar.orth_singular_values(s * M, "ns5", eps=eps)
            b = np.sort(polar.ns_scalar_map(sv * s / (s + eps)))[::-1]
            worst = max(worst, float(np.max(np.abs(a / b - 1.0))))
            out[eps].append((a[0], a[-1]))
    if worst > 1e-6:
        raise SystemExit(f"ns5: the matrix and the scalar map disagree (relative difference {worst:.1e})")
    for k in ("polar", EPS_DEFAULT, EPS_SMALL):
        out[k] = np.array(out[k])
    if np.max(np.abs(out["polar"] - 1.0)) > 1e-12:
        raise SystemExit("the exact polar factor has a singular value different from 1")
    out["worst"] = worst
    top = out[EPS_DEFAULT][:, 0]
    plateau = polar.ns_scalar_map(np.linspace(0.01, 1.0, 9901))
    out["band"] = (float(plateau.min()), float(plateau.max()))
    below = np.nonzero(top < out["band"][0])[0]                  # scales where the largest singular value has left
    out["s_leave"] = float(SCALES[below.max()])                  # the band of the plateau (they form an interval)
    if below.max() != len(below) - 1:
        raise SystemExit("the largest singular value of ns5(sM) re-enters its band at a smaller scale")
    slope = float(polar.NS5_COEFFS[0]) ** 5
    out["slope"] = slope
    out["linear_ratio"] = float(top[0] / (slope * sv[0] * SCALES[0] / EPS_DEFAULT))    # at the smallest scale
    return out


def load_summary():
    with open(os.path.join(RESULTS, "M9_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def digest_floor(instance, label):
    """The cell 'lowest relative squared distance' (median) of the row `label` of the table of `instance` in part C
    of M9 in muon/results/digest.md, as a string such as '1.1e-18'."""
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        text = f.read()
    part = text[text.index("### M9 part C"):]
    block = part[part.index(f"**{instance}**"):]
    for line in block.splitlines():
        if line.startswith(f"| {label} |"):
            return re.match(r"\s*([0-9.e+-]+)", line.rstrip("|").split("|")[-1]).group(1)
    raise SystemExit(f"row {label!r} of {instance} not found in digest.md")


def data_runs(m9):
    """Panel (b): the cached runs of the two configurations (five seeds each) and their floors."""
    e = m9["C"][INSTANCE]
    path = os.path.join(RESULTS, "M9_law.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    out = {}
    for eps, label in LABELS.items():
        d = e["deep"][label]
        cfg = d["cfg"]
        if (cfg["lr_decay"], cfg["lr"], cfg["batch"], cfg["orth"], cfg.get("beta"), cfg.get("nesterov")) != \
                (0.8, 0.1, 1, "ns5", 0.9, True) or cfg.get("orth_kw", {}).get("eps", EPS_DEFAULT) != eps:
            raise SystemExit(f"{label}: the configuration is not the one stated in the docstring: {cfg}")
        runs = []
        for seed in e["seeds"]:
            task = {"spec": e["spec"], "method": d.get("method", "vr"), "cfg": cfg, "passes": e["passes"],
                    "seed": seed, "stop_rel": 0.0}
            r = cache.get(task_key(task))
            if r is None:
                raise SystemExit(f"run missing from M9_law.pkl: {label}, seed {seed}")
            runs.append(r)
        p = np.asarray(runs[0]["trace"]["passes"], dtype=float)
        for r in runs[1:]:
            if not np.allclose(np.asarray(r["trace"]["passes"], dtype=float), p):
                raise SystemExit(f"{label}: the seeds are not recorded at the same passes")
        rel = np.array([r["trace"]["rel"] for r in runs], dtype=float)
        lowest = np.array([float(r["min_rel"]) for r in runs])
        floor = float(d["min_rel"]["median"])
        dig = digest_floor(INSTANCE, label)
        if abs(float(np.median(lowest)) / floor - 1.0) > 1e-9 or f"{floor:.1e}" != dig:
            raise SystemExit(f"{label}: floor {floor:.3e} (summary), {np.median(lowest):.3e} (cache), {dig} (digest)")
        eta = np.asarray(runs[0]["trace"]["eta"], dtype=float)[1:]
        law = cfg["lr"] * cfg["lr_decay"] ** (p[1:] - 1.0)       # passes count the initial gradient of the table
        if np.max(np.abs(eta / law - 1.0)) > 5e-3:
            raise SystemExit(f"{label}: the recorded lengths do not follow lr_0 q^(passes - 1)")
        out[eps] = {"label": label, "cfg": cfg, "passes": p, "rel": rel, "median": 10.0 ** np.median(np.log10(rel), axis=0),
                    "lowest": lowest, "floor": floor, "digest": dig, "runs": runs}
    other = {eps: float(m9["C"]["soft-cond"]["deep"][label]["min_rel"]["median"]) for eps, label in LABELS.items()}
    return out, e, other


def rerun(e, cached):
    """The run (eps = 1e-7, seed RUN_SEED) of panel (b), again, with a wrapper around polar.ns5 that records the
    Frobenius norm of its input at every call and, at the first call of each pass, the largest singular value of
    its output. Follows the 'vr' branch of muon/experiments/common.run_one (no cache is read or written here)."""
    spec = e["spec"]
    if spec["loss"] != "ls":
        raise SystemExit("the rerun is written for the least-squares instance")
    prob = problems.make_matrix_ls(**{k: v for k, v in spec.items() if k != "loss"})
    optimum.solve_optimum(prob)
    n = prob.n
    cfg = dict(cached["cfg"])
    if any(isinstance(v, dict) for v in cfg.values()):
        raise SystemExit("unit-tagged parameters in the configuration: resolve them as common.resolve does")
    budget = int(round(float(e["passes"]) * n))
    algo0 = vrmuon.VRMuon(prob, n_iters=1, **cfg)
    n_it = max(1, int((budget - algo0.init_cost) / expected_cost(algo0)))
    norms, first, last = [], [], {}

    def recording_ns5(D):
        O = polar.ns5(D, eps=EPS_DEFAULT)
        if len(norms) % n == 0:                                  # first iteration of a pass
            first.append((float(np.sqrt(np.vdot(D, D))), float(polar.singular_values(O)[0])))
        norms.append(float(np.sqrt(np.vdot(D, D))))
        last["O"] = O
        return O

    cfg["orth"] = recording_ns5
    r = vrmuon.vr_muon(prob, np.zeros((prob.K, prob.d)), n_it, seed=RUN_SEED, max_grad_evals=budget, stop_dist2=0.0,
                       **cfg)
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    rel = thin(np.asarray(r["dist2_F"], dtype=float) / D0)
    ref = cached["runs"][list(e["seeds"]).index(RUN_SEED)]
    ref_rel = np.asarray(ref["trace"]["rel"], dtype=float)
    if rel.shape != ref_rel.shape or np.max(np.abs(rel / ref_rel - 1.0)) > 1e-9:
        raise SystemExit("the rerun does not reproduce the cached run")
    first.append((norms[-1], float(polar.singular_values(last["O"])[0])))     # the last iteration of the run
    first = np.array(first)
    passes = 1.0 + np.arange(len(first))                         # the table costs the first pass; the last entry
    #                                                              is the end of the run (e["passes"] passes)
    if len(first) != int(e["passes"]) or len(norms) != (len(first) - 1) * n:
        raise SystemExit("the rerun does not have one recorded iteration per pass")
    frozen = first[FROZEN_FROM - 1:, 0]
    if frozen.max() / frozen.min() > 1.05:
        raise SystemExit(f"the input of the map still moves after pass {FROZEN_FROM}")
    return {"norms": np.array(norms), "first": first, "passes": passes, "n": n, "iters": len(norms),
            "diff": float(np.max(np.abs(rel / ref_rel - 1.0))), "rel": rel, "trace_passes": np.asarray(ref["trace"]["passes"])}


def data():
    maps = data_maps()
    m9 = load_summary()
    runs, e, other = data_runs(m9)
    run = rerun(e, runs[EPS_DEFAULT])
    return {"maps": maps, "runs": runs, "other": other, "run": run, "seeds": list(e["seeds"]), "spec": e["spec"],
            "total_passes": e["passes"]}


def print_all(D):
    mp, run = D["maps"], D["run"]
    print(f"(a) M: {M_SHAPE[0]} x {M_SHAPE[1]} standard Gaussian, seed {M_SEED}, |M|_F = 1; singular values "
          + ", ".join(f"{v:.4f}" for v in mp["sv"]))
    print(f"    {len(SCALES)} scales from {SCALES[0]:g} to {SCALES[-1]:g}; ns5 from the matrix and from the scalar map: "
          f"largest relative difference {mp['worst']:.1e}")
    print("    scale s: [largest, smallest] singular value of ns5(sM) with eps = 1e-7 | with eps = 1e-30")
    for k in range(0, len(SCALES), 20):
        a, b = mp[EPS_DEFAULT][k], mp[EPS_SMALL][k]
        print(f"      {SCALES[k]:7.0e}: [{a[0]:.3e}, {a[1]:.3e}] | [{b[0]:.4f}, {b[1]:.4f}]")
    print(f"    band of p^5 over [0.01, 1]: [{mp['band'][0]:.4f}, {mp['band'][1]:.4f}]; with eps = 1e-7 the largest "
          f"singular value is below it for every s <= {mp['s_leave']:.2e}")
    print(f"    slope of p^5 at 0: {mp['slope']:.1f}; largest singular value at s = {SCALES[0]:g} over "
          f"(slope s_1 s / eps): {mp['linear_ratio']:.4f}")
    print(f"    run of seed {RUN_SEED} (eps = 1e-7) rerun: {run['iters']} iterations; relative squared distances against "
          f"the cache: largest relative difference {run['diff']:.1e}")
    print("    pass: |input of ns5|_F, largest singular value of its output (first iteration of the pass)")
    for p in (1, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 120, 200, int(run["passes"][-1])):
        k = int(p) - 1
        print(f"      {p:4d}: {run['first'][k, 0]:.3e}, {run['first'][k, 1]:.3e}"
              + ("   (last iteration of the run)" if k == len(run["first"]) - 1 else ""))
    last = run["norms"][-run["n"]:]
    print(f"    over the last pass the norm of the input stays in [{last.min():.4e}, {last.max():.4e}]")
    print(f"(b) {INSTANCE} {json.dumps(D['spec'])}, seeds {D['seeds']}, {D['total_passes']} passes")
    for eps in (EPS_DEFAULT, EPS_SMALL):
        r = D["runs"][eps]
        cfg = {k: v for k, v in r["cfg"].items() if k in ("estimator", "batch", "orth", "orth_kw", "beta", "nesterov",
                                                           "schedule", "lr_decay", "lr")}
        print(f"    eps = {eps:g}: {json.dumps(cfg)}")
        print(f"      lowest relative squared distance per seed: {', '.join(f'{v:.2e}' for v in r['lowest'])}; median "
              f"{r['floor']:.3e} (digest: {r['digest']}); on soft-cond (not drawn): {D['other'][eps]:.2e}")
        sel = [int(np.argmin(np.abs(r["passes"] - v))) for v in (1, 30, 60, 90, 120, 160, 300)]
        print("      median over the seeds at passes " + "; ".join(f"{r['passes'][j]:.0f}: {r['median'][j]:.2e}"
                                                                   for j in sel))


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
DASHED = (0, (4.0, 1.8))                     # ns5 with its default eps (as in Figure 4(a) of the paper)
DOTTED = (0, (1.0, 1.4))                     # ns5 with eps = 1e-30


def sci(v):
    """2.3e-18 -> '2.3 x 10^-18' in mathtext."""
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\cdot10^{{{int(e)}}}$"


def panel_a(ax, D, ps, b):
    ink = ps.INK
    mp, run = D["maps"], D["run"]
    ymin, ymax = 1e-3, 9.0
    ax.plot([EPS_DEFAULT, EPS_DEFAULT], [ymin, mp[EPS_DEFAULT][np.searchsorted(SCALES, EPS_DEFAULT), 1]],
            color=ink["muted"], lw=0.9, zorder=1)          # reference line in muted ink: the run line is primary ink
    for eps in (EPS_SMALL, EPS_DEFAULT):                         # bands: smallest to largest singular value
        ax.fill_between(SCALES, mp[eps][:, 1], mp[eps][:, 0], color=es.tint(ps.ORANGE, 0.20), lw=0, zorder=2)
    ax.plot(SCALES, mp["polar"][:, 0], color=ps.ORANGE, lw=2.0, zorder=3, label="exact polar factor")
    ax.plot(SCALES, mp[EPS_DEFAULT][:, 0], color=ps.ORANGE, lw=1.5, ls=DASHED, zorder=5,
            label=r"ns5, $\epsilon = 10^{-7}$ (Muon)")
    ax.plot(SCALES, mp[EPS_DEFAULT][:, 1], color=ps.ORANGE, lw=0.9, ls=DASHED, zorder=5)
    ax.plot(SCALES, mp[EPS_SMALL][:, 0], color=ps.ORANGE, lw=1.5, ls=DOTTED, zorder=4,
            label=r"ns5, $\epsilon = 10^{-30}$")
    ax.plot(SCALES, mp[EPS_SMALL][:, 1], color=ps.ORANGE, lw=0.9, ls=DOTTED, zorder=4)
    # one run of panel (b): the input of ns5 and the largest singular value of its output, pass after pass
    f = run["first"]
    b.add(3, ax.plot(f[:, 0], f[:, 1], color=ink["primary"], lw=0.9, zorder=6, label="one run of (b)"))
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", zorder=8)
    dot = dict(ls="", marker="o", ms=4.6, mfc=ink["primary"], mec=ink["surface"], mew=0.8, zorder=7)
    lead = dict(arrowstyle="-", color=ink["secondary"], lw=0.7, shrinkA=0.5, shrinkB=2.5)
    last = int(run["passes"][-1])
    # pass 60 sits at s = 1.2e-7, just right of the line s = eps: its label goes up-left so that its leader is not
    # read as a continuation of that line
    for p, off, ha, va, arrow in ((30, (0, 11), "center", "bottom", lead), (60, (-12, 11), "right", "bottom", lead),
                                  (90, (-6, 2), "right", "center", None), (last, (-6, -2), "right", "center", None)):
        x, y = f[p - 1]
        b.add(3, ax.plot([x], [y], **dot))
        b.add(3, ax.annotate(f"pass {p}", (x, y), xytext=off, ha=ha, va=va, arrowprops=arrow, **kw))
    ax.annotate(r"$s = \epsilon$", (EPS_DEFAULT, 0.12), xytext=(4, 0), ha="left", va="center", **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SCALES[0], SCALES[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2])
    ax.set_yticks([1e-2, 1e-1, 1.0])
    ax.minorticks_off()
    ax.set_xlabel(r"scale $s$ of the input $sM$  ($\|M\|_F = 1$)")
    ax.set_ylabel(r"singular values of $o(sM)$")
    b.legend(ax, loc="lower right", handlelength=1.9, borderaxespad=0.0, labelspacing=0.25, frameon=True,
             facecolor=ink["surface"], edgecolor="none", framealpha=1.0, borderpad=0.25)
    es.panel_title(ax, "(a) the map at scale $s$")


def panel_b(ax, D, ps, b):
    ink = ps.INK
    kw = dict(fontsize=es.small(), color=ink["secondary"], textcoords="offset points", linespacing=1.15)
    for eps, ls, text in ((EPS_DEFAULT, DASHED, r"Muon's $\epsilon = 10^{-7}$:" + "\nfloor "),
                          (EPS_SMALL, DOTTED, r"$\epsilon = 10^{-30}$:" + "\n")):
        r = D["runs"][eps]
        for row in r["rel"]:
            ax.plot(r["passes"], row, color=ps.ORANGE, lw=0.5, alpha=0.55, zorder=2)
        ax.plot(r["passes"], r["median"], color=ps.ORANGE, lw=1.7, ls=ls, zorder=4)
        ax.annotate(text + sci(r["floor"]), (298.0, r["floor"]), xytext=(0, 5), ha="right", va="bottom", **kw)
    # the run of panel (a) at the marked passes
    run = D["run"]
    for p, off, ha, va in ((30, (7, 3), "left", "center"), (60, (7, 3), "left", "center"),
                           (90, (-6, -7), "right", "top")):
        k = int(np.argmin(np.abs(run["trace_passes"] - p)))      # the record nearest to pass p (within 0.7 pass)
        b.add(3, ax.plot([run["trace_passes"][k]], [run["rel"][k]], ls="", marker="o", ms=4.6, mfc=ink["primary"],
                         mec=ink["surface"], mew=0.8, zorder=7))
        b.add(3, ax.annotate(f"pass {p}", (run["trace_passes"][k], run["rel"][k]), xytext=off, ha=ha, va=va, **kw))
    ax.set_yscale("log")
    ax.set_xlim(0, 300)
    ax.set_ylim(1e-33, 30.0)
    ax.set_yticks([1e-30, 1e-24, 1e-18, 1e-12, 1e-6, 1.0])
    ax.minorticks_off()
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("passes")
    es.panel_title(ax, r"(b) $\|W_t - W_\ast\|_F^2\,/\,\|W_\ast\|_F^2$")


def draw(D):
    ps = es.setup("talk")
    fig = es.figure("talk", height=2.6)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0))
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b = es.Build()
    panel_a(ax_a, D, ps, b)
    panel_b(ax_b, D, ps, b)
    b.add(2, ax_b)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"(a) one ${M_SHAPE[0]} \times {M_SHAPE[1]}$ Gaussian $M$; (b) least squares, "
                  rf"$\kappa = 10^{{{int(round(np.log10(D['spec']['kappa'])))}}}$, $b = 1$, ns5 + "
                  rf"Nesterov, $\mathrm{{lr}}_t = 0.1 \cdot 0.8^{{t/n}}$, {len(D['seeds'])} seeds"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
