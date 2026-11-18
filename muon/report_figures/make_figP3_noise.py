"""Figure P3 of the paper muon/paper_saga_to_muon.tex: what a finite-sum estimator changes inside polar steps
(finding 2). One row of three panels, all drawn by the functions of the scripts that made them first:

  (a) two floors          M3, softmax memory-limited instance (panel (a) of figR2_muon_vr);
  (b) batch one           finite-sum least squares with an intercept and skewed residuals (n = 2000, K = 2, d = 3):
                          minibatch polar steps go to the least-distances solution, the SAGA estimator to the
                          least-squares solution (panel (a) of figT2s_noise);
  (c) b x offset          batch size times the nuclear norm of the gradient at the limit point: zeros of the mean field,
                          runs with decreasing lengths, SAGA estimator (panel (b) of figT2s_noise).
(The SAGA-Muon / MISO panel of figR2_muon_vr is not repeated: the paper quotes its numbers.)

Data: muon/results/M3_summary.json, M6_summary.json (checked against muon/results/digest.md by the functions of
make_figR2_muon_vr.py) and the cache muon/theory_checks/paper_figures_data.json of make_paper_figures.py (nothing is
recomputed if the cache holds the keys noise_trace and noise_offsets). The numbers quoted in the paper are printed.

Run:    .venv/Scripts/python muon/report_figures/make_figP3_noise.py
Output: muon/figures/figP3_noise.png and .pdf
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NAME = "figP3_noise"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R2 = _load_module(os.path.join(HERE, "make_figR2_muon_vr.py"), "make_figR2_muon_vr")
PF = _load_module(os.path.join(ROOT, "muon", "theory_checks", "make_paper_figures.py"), "make_paper_figures")


def print_noise(tr, rows):
    c = np.array(tr["checkpoints"])
    print(f"(c) batch one: {len(c)} checkpoints from {c[0]:.0f} to {c[-1]:.0f} iterations; at the last one: minibatch "
          f"to LS {tr['mb_to_ls'][-1]:.4g}, minibatch to LAD {tr['mb_to_lad'][-1]:.4g}, SAGA to LS "
          f"{tr['saga_to_ls'][-1]:.4g}")
    print("(d) b, zero of the mean field, runs, SAGA estimator (nuclear norm of the gradient at the limit point):")
    for r in rows:
        print(f"    b = {int(r['b']):3d}: root {r['root_mean']:.4g} (b x {r['b'] * r['root_mean']:.3f}), runs {r['run']:.4g}"
              f" (b x {r['b'] * r['run']:.3f}), SAGA {r['saga']:.3g}")
    worst = max(abs(r["run"] - r["root_mean"]) / r["root_mean"] for r in rows)
    print(f"    largest relative gap between the runs and the zero of the mean field: {worst:.3f}")


def layout_checks(fig, axes):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    msgs = []
    for name, ax in axes:
        leg = ax.get_legend()
        lb, ab = leg.get_window_extent(rend), ax.bbox
        lines = [(l.get_xdata(), l.get_ydata()) for l in ax.get_lines() if len(l.get_xdata())]
        hit = R2._inside(ax, lb, lines, pad=2.0 * pt)
        if hit:
            msgs.append(f"({name}) the legend is on the data ({hit} sampled points)")
        if lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1 or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1:
            msgs.append(f"({name}) the legend leaves the axes")
    print("layout checks (legends inside the axes and away from the data): " + ("ok" if not msgs else "; ".join(msgs)))
    return not msgs


def main():
    A, B, Cd = R2.data_a(R2.load("M3_summary")), R2.data_b(R2.load("M6_summary")), R2.data_c(R2.load("M1_summary"))
    R2.print_a(A)
    R2.print_b(B, R2.compare_b(B), R2.same_run(A, B))
    if not R2.report_digest(R2.check_digest(A, B, Cd)):
        raise SystemExit("the figure is not drawn: the numbers above disagree with the digest")
    data = PF.load_data(["noise_trace", "noise_offsets"])
    tr, rows = data["noise_trace"], data["noise_offsets"]["rows"]
    print_noise(tr, rows)
    ps = R2._plotstyle()
    fig = ps.figure(width=ps.FIG_W, height=2.45)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.30, 1.12, 1.0))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    R2.panel_a(ax_a, A, ps)
    PF._noise_panel_trace(ax_b, tr, ps)
    ax_b.set_title("(b) batch one, least squares", loc="left")
    ax_b.set_ylim(1e-8, 40.0)                                  # narrower panel: room for the legend below the curves
    PF._noise_panel_offsets(ax_c, rows, ps)
    ax_c.set_title("(c) b × offset", loc="left")
    h, _ = ax_c.get_legend_handles_labels()                    # shorter labels for the narrower panel
    ps.legend(ax_c, handles=h, labels=["mean-field zero", "minibatch runs", "SAGA estimator"], loc="lower right",
              bbox_to_anchor=(1.0, 0.06), handlelength=1.2, handletextpad=0.4, borderpad=0.1, labelspacing=0.25)
    layout_checks(fig, [("a", ax_a), ("b", ax_b), ("c", ax_c)])
    ps.savefig(fig, NAME)


if __name__ == "__main__":
    main()
