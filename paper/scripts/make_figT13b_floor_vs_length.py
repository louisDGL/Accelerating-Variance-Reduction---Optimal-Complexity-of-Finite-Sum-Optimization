"""Figure T13b of the lecture (paper/talk/figures/figT13b_floor_vs_length): at a constant length, a table changes the
order of the floor of a polar step, not its existence. Explanatory figure, not in the paper; it draws the sentence of
item (M1) of paper/paper_saga_to_muon.tex: "with the SAGA estimator the iterates settle at a squared distance
3.8 eta^2 (softmax) or 6.0 eta^2 (least squares), whereas the floor of minibatch Muon is linear in its learning rate".

Two log-log panels with shared axes, one per instance of experiment M3 ((a) softmax, (b) least squares: the two
memory-limited instances, n = 1000, d = 50, K = 10, kappa = 1e3, alpha = 1, data seed 0): x = the fixed spectral
length eta of the step W+ = W - eta polar(omega) (SAGA table, batch one, exact polar factor; orange) or the learning
rate of Muon (b = 32, Nesterov momentum 0.95, five Newton-Schulz steps: the singular values of its step are about
0.7 to 1.2 times the learning rate, so it is not a spectral length; blue, the colour of "no table");
y = the stationary ABSOLUTE squared distance ||W - W_*||_F^2.

Protocol (the one of muon/experiments/exp_M3_floors.py, column "stationary level"): a run started at
W_0 = W_* + 1e-7 ||W_*||_F P (table of the estimator initialized there); the level is the median of the recorded
relative squared distance over the second half of the run, here multiplied by ||W_*||_F^2 (the instance is rebuilt
with the project code, muon/experiments/common.problem).
  filled marks   stored levels, muon/results/M3_summary.json, results[<instance>][<label>]["stationary"], labels
                 "SAGA est., fixed length 1e-1 | 1e-2 | 1e-3" and "Muon b=32, lr 1e-1 | 1e-2": 30 passes, median of
                 the three algorithm seeds (the range over the seeds is within the marks; it is printed);
  open marks     new runs of this script, algorithm seed 0, with common.run_one (no cache is read or written):
                 the SAGA table at eta = 3e-4, 3e-3, 3e-2, 3e-1 (30 passes); Muon at lr = 3e-2 and 3e-1 (30 passes)
                 and at lr = 1e-3 and 3e-3 with 0.3 / lr passes (300 and 100): at a small learning rate Muon needs a
                 number of passes proportional to 1 / lr to leave W_* and reach its level (30 passes are not enough
                 below lr = 1e-2). Muon at lr = 3e-4 would need 1000 passes and is not run.
  thin lines     guides, not fits and not bounds: C eta^2 with C = the median of level / eta^2 over the nine stored
                 values of the instance (3.8 and 6.0, the constants of the paper; written "floor ~ 3.8 eta^2" because
                 level / eta^2 spans 3.72-3.90 and 5.98-6.09 over the stored values, 3.78-3.87 and 5.84-6.05 over the
                 new ones), and c lr with c = the median of level / lr over the six stored values of the instance.
                 Muon's run at lr = 1e-3 (300 passes) is close to stationary only in its second half (reviewer's
                 check: level / lr = 391, 455, 444 over passes 120-180, 180-240, 240-300), so its mark may sit a few
                 per cent low.

Checks (the script stops if one fails): the stored medians against muon/results/digest.md (M3 tables, column
"stationary level"); the constants 3.8 and 6.0 of the paper against the stored levels; the protocol, by rerunning
one stored Muon run (softmax, lr 1e-2, seed 0) and comparing its level; the stationarity of every new run (medians of
the third and fourth quarters of the run within 20 %).

Status. Measured: every mark (two instances; three seeds or one). The laws "proportional to eta^2" and "proportional
to lr" are read from these marks over the ranges drawn; the script prints the fitted slopes. Proved, and not drawn:
the existence of a floor of order eta^2 at a fixed length, whatever the estimator (fact F2 of muon/notes_theory.md;
the bound of the paper is in function values, mu_op eta^2 / 8). Not comparable on this figure: the two heights at a
given abscissa (batch one against b = 32, a spectral length against a learning rate with momentum).

Deviations from the specification of the figure: two panels (one per instance) instead of one, because the marks
of the two instances overlap at every abscissa (their levels differ by factors 1.6 and 2.6 only); Muon is drawn in
blue (the colour of "no table" in the set of figures), not in grey.

The script prints every number it plots. About 55 s (16 new runs and one rerun of a stored run, two worker
processes).
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT13b_floor_vs_length.py
Output: paper/talk/figures/figT13b_floor_vs_length.pdf and .png
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

sys.path.insert(0, os.path.join(es.ROOT, "muon", "experiments"))
import common as C  # noqa: E402  (muon/experiments/common.py: problem, run_one; nothing is cached)

NAME = "figT13b_floor_vs_length"
RESULTS = os.path.join(es.ROOT, "muon", "results")
INSTANCES = (("soft-mem", "softmax", "o"), ("ls-mem", "least squares", "s"))
PAPER_C = {"soft-mem": 3.8, "ls-mem": 6.0}                  # constants of item (M1) of the paper
STORED = {"saga": (("SAGA est., fixed length 1e-1", 1e-1), ("SAGA est., fixed length 1e-2", 1e-2),
                   ("SAGA est., fixed length 1e-3", 1e-3)),
          "muon": (("Muon b={b}, lr 1e-1", 1e-1), ("Muon b={b}, lr 1e-2", 1e-2))}
NEW = {"saga": ((3e-4, 30), (3e-3, 30), (3e-2, 30), (3e-1, 30)),          # (length or learning rate, passes)
       "muon": ((1e-3, 300), (3e-3, 100), (3e-2, 30), (3e-1, 30))}
MUON_CFG = dict(beta=0.95, nesterov=True, orth="ns5")       # the configuration "Muon b=32" of exp_M3_floors.py
SEED_NEW = 0
JOBS = 2
X_LIM, Y_LIM = (2e-4, 4.5e-1), (3e-8, 3e3)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def task(spec, kind, batch, lr, passes):
    """A task of exp_M3_floors.py for the start 'near_opt' (same keys; 'passes' is 30 there)."""
    if kind == "saga":
        method, cfg = "vr", {"estimator": "saga", "length": "const", "lr": lr}
    else:
        method, cfg = "muon_sgd", dict(batch=batch, lr=lr, **MUON_CFG)
    return {"spec": spec, "method": method, "cfg": cfg, "seed": SEED_NEW, "stop_rel": 1e-30, "passes": passes,
            "start": "near_opt"}


def level(res):
    """The rule of exp_M3_floors.py for the stationary level (median over the second half of the trace), and the
    medians over the third and the fourth quarters (stationarity check)."""
    rel = np.asarray(res["trace"]["rel"], dtype=float)
    n = len(rel)
    return float(np.median(rel[n // 2:])), float(np.median(rel[n // 2:3 * n // 4])), float(np.median(rel[3 * n // 4:]))


def _digest_rows(inst):
    rows, inside = {}, False
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("### "):
                inside = line[4:].startswith(inst + " (") and "passes near W_*" in line
            elif line.startswith("## "):
                inside = False
            elif inside and line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows[cells[0]] = cells
    return rows


def _mmm(v, fmt="{:.1e}"):
    return f"{fmt.format(np.median(v))} [{fmt.format(v.min())}, {fmt.format(v.max())}]"


def data():
    with open(os.path.join(RESULTS, "M3_summary.json"), encoding="utf-8") as f:
        m3 = json.load(f)
    D = {"seeds": m3["seeds"], "inst": {}}
    tasks, index = [], []
    for key, _, _ in INSTANCES:
        spec, batch = m3["instances"][key]["spec"], m3["instances"][key]["batch"]
        prob = C.problem(spec)
        d = {"spec": spec, "batch": batch, "passes_stat": max(4, m3["instances"][key]["passes"] // 2),
             "W2": float(np.vdot(prob.W_star, prob.W_star)), "stored": {}, "new": {}}
        digest = _digest_rows(key)
        for kind, items in STORED.items():
            rows = []
            for label, x in items:
                label = label.format(b=batch)
                st = np.asarray(m3["results"][key][label]["stationary"], dtype=float)
                if digest.get(label, [None] * 3)[2] != _mmm(st):
                    raise SystemExit(f"{key}, '{label}': stationary level {_mmm(st)} in M3_summary.json, "
                                     f"{digest.get(label)} in the digest")
                rows.append({"label": label, "x": x, "rel": st, "abs": st * d["W2"]})
            d["stored"][kind] = rows
        for kind, items in NEW.items():
            for x, passes in items:
                tasks.append(task(spec, kind, batch, x, passes))
                index.append((key, kind, x, passes))
        D["inst"][key] = d
    # protocol check: one stored run again (Muon, softmax, lr 1e-2, seed 0 of M3)
    k0 = INSTANCES[0][0]
    d0 = D["inst"][k0]
    tasks.append(dict(task(d0["spec"], "muon", d0["batch"], 1e-2, d0["passes_stat"]), seed=D["seeds"][0]))
    t0 = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        res = list(ex.map(C.run_one, tasks))
    D["wall"] = time.perf_counter() - t0
    again = level(res[-1])[0]
    stored = float(d0["stored"]["muon"][1]["rel"][0])
    D["protocol"] = (again, stored)
    if abs(again / stored - 1.0) > 0.05:
        raise SystemExit(f"protocol check: Muon lr 1e-2 on {k0}, seed {D['seeds'][0]}: level {again:.3e} rerun, "
                         f"{stored:.3e} stored")
    for (key, kind, x, passes), r in zip(index, res[:-1]):
        lv, q3, q4 = level(r)
        if r["diverged"] or abs(q3 / q4 - 1.0) > 0.20:
            raise SystemExit(f"new run {key}, {kind}, {x:g}: not stationary (quarters {q3:.2e}, {q4:.2e}) or diverged")
        d = D["inst"][key]
        d["new"].setdefault(kind, []).append({"x": x, "passes": passes, "rel": lv, "abs": lv * d["W2"], "q3": q3,
                                              "q4": q4, "wall": r["wall"]})
    for key, d in D["inst"].items():
        sa, mu = d["stored"]["saga"], d["stored"]["muon"]
        d["C"] = float(np.median(np.concatenate([r["abs"] / r["x"] ** 2 for r in sa])))
        d["c"] = float(np.median(np.concatenate([r["abs"] / r["x"] for r in mu])))
        if f"{d['C']:.1f}" != f"{PAPER_C[key]:.1f}":
            raise SystemExit(f"{key}: stored levels give {d['C']:.3f} eta^2, the paper says {PAPER_C[key]} eta^2")
        for kind in ("saga", "muon"):
            x = np.array([r["x"] for r in d["stored"][kind]] + [r["x"] for r in d["new"][kind]])
            y = np.array([float(np.median(r["abs"])) for r in d["stored"][kind]] + [r["abs"] for r in d["new"][kind]])
            d[f"slope_{kind}"] = float(np.polyfit(np.log(x), np.log(y), 1)[0])
    return D


def print_all(D):
    print(f"new runs: {D['wall']:.0f} s with {JOBS} processes; protocol check (Muon lr 1e-2, {INSTANCES[0][0]}, seed "
          f"{D['seeds'][0]}): level {D['protocol'][0]:.4e} rerun, {D['protocol'][1]:.4e} stored")
    for key, name, _ in INSTANCES:
        d = D["inst"][key]
        s = d["spec"]
        print(f"{key} ({name}): {s['loss']}, n = {s['n']}, d = {s['d']}, K = {s['K']}, kappa = {s['kappa']:g}, alpha = "
              f"{s['alpha']:g}, data seed {s['seed']}; ||W_*||_F^2 = {d['W2']:.3f}; Muon batch {d['batch']}")
        for kind, unit, power in (("saga", "eta", 2), ("muon", "lr", 1)):
            print(f"  {'SAGA table, fixed length, b = 1' if kind == 'saga' else 'Muon, b = %d' % d['batch']}:")
            for r in d["stored"][kind]:
                a = r["abs"]
                print(f"    stored  {unit} = {r['x']:g}: relative level, seeds {D['seeds']}: {_mmm(r['rel'], '{:.2e}')}; "
                      f"absolute {np.median(a):.3e} [{a.min():.3e}, {a.max():.3e}]; / {unit}^{power}: "
                      f"{np.median(a) / r['x'] ** power:.3f} [{(a / r['x'] ** power).min():.3f}, "
                      f"{(a / r['x'] ** power).max():.3f}]")
            for r in d["new"][kind]:
                print(f"    new     {unit} = {r['x']:g} ({r['passes']} passes, seed {SEED_NEW}, {r['wall']:.1f} s): "
                      f"relative level {r['rel']:.2e} (third quarter {r['q3']:.2e}, fourth {r['q4']:.2e}); absolute "
                      f"{r['abs']:.3e}; / {unit}^{power}: {r['abs'] / r['x'] ** power:.3f}")
            print(f"    slope of log(level) against log({unit}), all marks: {d['slope_' + kind]:.3f}")
        print(f"  guides: {d['C']:.3f} eta^2 (paper: {PAPER_C[key]} eta^2) and {d['c']:.1f} lr (medians over the stored "
              f"values)")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    hue = {"saga": ps.ORANGE, "muon": ps.BLUE}        # SAGA table: orange; no table (Muon on minibatches): blue
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    xg = np.array(X_LIM)
    kw = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.25, zorder=6)
    for ax, (key, name, _), letter in zip(axes, INSTANCES, "ab"):
        d = D["inst"][key]
        ax.plot(xg, d["C"] * xg ** 2, color=hue["saga"], lw=0.9, zorder=2)
        ax.plot(xg, d["c"] * xg, color=hue["muon"], lw=0.9, zorder=2)
        for kind in ("saga", "muon"):
            ax.plot([r["x"] for r in d["new"][kind]], [r["abs"] for r in d["new"][kind]], ls="", marker="o", ms=6.2,
                    mfc=ink["surface"], mec=hue[kind], mew=1.5, zorder=4,
                    label="new run, one seed" if kind == "saga" else None)
            ax.plot([r["x"] for r in d["stored"][kind]], [float(np.median(r["abs"])) for r in d["stored"][kind]], ls="",
                    marker="o", ms=7.2, mfc=hue[kind], mec=ink["surface"], mew=1.0, zorder=5,
                    label=f"M3, median of {len(D['seeds'])} seeds" if kind == "saga" else None)
        es.panel_title(ax, f"({letter}) {name}")
        ax.text(0.04, 0.95, f"Muon (momentum, ns5), $b = {d['batch']}$:\nfloor $\\propto$ learning rate", transform=ax.transAxes,
                ha="left", va="top", **kw)
        ax.text(0.97, 0.05, f"SAGA table, $b = 1$:\nfloor $\\approx {d['C']:.1f}\\,\\eta^2$", transform=ax.transAxes,
                ha="right", va="bottom", **kw)
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1.0, 1e2])
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.minorticks_off()
    ax.set_ylabel(r"floor of $\|W - W_\ast\|_F^2$")
    for ax in axes:
        ax.set_xlabel(r"length $\eta$ or learning rate")
    # key of the marks (in ink: the hue says the method), in the empty band between the two laws of panel (a)
    L = ps.plt.Line2D
    grey = ink["secondary"]
    ps.legend(axes[0], handles=[L([], [], ls="", marker="o", ms=7.2, mfc=grey, mec=ink["surface"], mew=1.0),
                                L([], [], ls="", marker="o", ms=6.2, mfc=ink["surface"], mec=grey, mew=1.5)],
              labels=[f"{len(D['seeds'])} seeds (stored runs)", "1 seed (new runs)"], loc="center left",
              bbox_to_anchor=(0.0, 0.5), handlelength=1.0, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)
    w2 = [D["inst"][inst[0]]["W2"] for inst in INSTANCES]         # absolute = relative (the axis of figT13) x ||W_*||_F^2
    es.save(fig, NAME, "talk",
            note=(rf"absolute distances ($\|W_\ast\|_F^2 = {w2[0]:.0f}$, ${w2[1]:.0f}$); runs started at $W_\ast$; "
                  r"compare slopes, not heights"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
