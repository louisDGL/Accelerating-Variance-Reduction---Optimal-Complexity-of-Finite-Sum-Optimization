"""Figure T17b of the lecture, a backup slide (paper/talk/figures/figT17b_path_budget): when the no-table controls
of item (M5) of paper/paper_saga_to_muon.tex freeze, is it the estimator or the schedule? Explanatory figure, not in
the paper; it is panel (b) of the specification of figT17, drawn as a figure of its own, and it shows the correction
of the audit to (M5) (paper/audit/AUDIT.md and paper/audit/evidence/num-m5.md, finding M1): "without a table the
iterate freezes at 8 % to 34 % of the initial squared distance AT BATCH ONE; the values up to 93 % are at b = 32,
where the schedule is too short for any estimator".

The runs (experiment M9, part C, muon/experiments/exp_M9_schedule.py): the two non-square conditioning-limited
instances "soft-cond" and "ls-cond" (n = 200, d = 40, K = 8, kappa = 1e4, alpha = 2), W_0 = 0, 300 passes without
stop rule, algorithm seeds 10 to 14, steps W+ = W - lr_t ns5(.) on the geometric cooldown that was selected for
Muon's recipe WITH the table at batch one (softmax: q = 0.7, lr_0 = 0.3; least squares: q = 0.75, lr_0 = 0.3):
    lr_t = lr_0 q^(t g / n),   g = gradients per step (b for the minibatch gradient and for the SAGA table, 2 b for
    the momentum buffers MARS-M and STORM-Muon),   total path  sum_t lr_t = lr_0 / (1 - q^(g/n)).
One mark per (instance, estimator, batch size): x = total path / ||W_*||_op, y = final relative squared distance
||W - W_*||_F^2 / ||W_*||_F^2 (median of the five seeds; it starts at 1).
  blue, filled   no table, minibatch gradient through ns5 + Nesterov momentum 0.9, b = 1 and 32
                 (M9_summary.json, C[inst]["controls"]["no table (minibatch gradient), ns5 + Nesterov, b=..."]);
  blue, open     no table, MARS-M (momentum 0.95, ns5), b = 1 and 32 (C[inst]["controls"]["MARS-M, b=..."]);
                 STORM-Muon has the same path and ends within 0.015 of MARS-M: printed, not drawn;
  orange         the SAGA table on the same schedule and through the same map: at b = 1 the runs
                 C[inst]["deep"]["ns5 + Nesterov (Muon's recipe)"] of M9 (the script checks that the configuration is
                 the one of the control but for the estimator); at b = 32 NEW runs of this script with the project
                 code (muon/experiments/common.run_one, no cache read or written; seeds 10 to 14; about 25 s of
                 computation), the control that the audit ran on two seeds (0.785 and 0.574).
The vertical line is sum_t lr_t = ||W_*||_op. Proved: a run whose steps have spectral norm at most c lr_t cannot
come closer to W_* than ||W_*||_op - c sum_t lr_t in spectral norm; for the exact polar factor c = 1, and for the
five Newton-Schulz steps c = the largest value of the scalar map of ns5 on [0, 1] (computed here from
muon/lab/polar.NS5_COEFFS: 1.20). The script stops unless every mark at b = 32 has c x path < ||W_*||_op.

Checks (the script stops if one fails): the medians of the controls against muon/results/digest.md (M9 part C); the
per-seed values against the cache muon/results/M9_law.pkl; the schedule, by comparing lr_0 q^(...) with the recorded
lengths trace["eta"] of the run of seed 10 of every control; the closed form of the path against the sum over the
iterations of that run.

Status. Measured: every mark (two instances, five seeds; the orange marks at b = 32 are new runs of this script).
Proved: the reach bound. What the figure says and does not say: at b = 32 the freeze says nothing about the table
(the table freezes too); at b = 1 the path is 2.4 to 11.5 times ||W_*||_op and the runs with the table end below
3e-5 (on softmax four of five leave the schedule: this cooldown sits on the cliff of figT17), whereas the runs
without a table keep 8 % to 34 % of the initial squared distance. The no-table estimators were not tuned on their
own grid (one schedule per instance).

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT17b_path_budget.py
Output: paper/talk/figures/figT17b_path_budget.pdf and .png
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

sys.path.insert(0, os.path.join(es.ROOT, "muon", "experiments"))
import common as C  # noqa: E402  (muon/experiments/common.py: run_one; nothing is cached)

NAME = "figT17b_path_budget"
RESULTS = os.path.join(es.ROOT, "muon", "results")
INSTANCES = (("soft-cond", "softmax", "o"), ("ls-cond", "least squares", "s"))
BATCHES = (1, 32)
CONTROLS = (("minibatch", "no table (minibatch gradient), ns5 + Nesterov, b={b}", 1),     # (kind, key, g / b)
            ("mars", "MARS-M, b={b}", 2), ("storm", "STORM-Muon, b={b}", 2))
TABLE_DEEP = "ns5 + Nesterov (Muon's recipe)"
JOBS = 2
X_LIM = (0.04, 30.0)


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def task_key(task):
    """Key of a task in the caches of muon/experiments/common.py (common.task_key)."""
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def ns5_gain():
    """Largest singular value that five Newton-Schulz steps can return: the maximum over [0, 1] of the scalar map
    (the singular values of M / ||M||_F are in [0, 1])."""
    from lab import polar
    x = np.linspace(0.0, 1.0, 200001)
    return float(polar.ns_scalar_map(x).max())


def _digest_controls(inst):
    """Rows 'configuration without a table' of the instance in the section "M9 part C" of the digest."""
    rows, part, inside = {}, False, False
    with open(os.path.join(RESULTS, "digest.md"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                part, inside = line.startswith("### M9 part C"), False
            elif part and line.startswith("**"):
                inside = line.startswith(f"**{inst}**")
            elif part and inside and line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                rows[cells[0]] = cells
    return rows


def data():
    with open(os.path.join(RESULTS, "M9_summary.json"), encoding="utf-8") as f:
        m9 = json.load(f)
    with open(os.path.join(RESULTS, "M9_law.pkl"), "rb") as f:
        law = pickle.load(f)
    D = {"gain": ns5_gain(), "inst": {}}
    tasks, index = [], []
    for key, _, _ in INSTANCES:
        e = m9["C"][key]
        s, n, passes, seeds = e["spec"], e["spec"]["n"], e["passes"], e["seeds"]
        q, lr0 = e["controls_schedule"]["lr_decay"], e["controls_schedule"]["lr"]
        w_op = e["norms"]["W_star_op"]
        digest = _digest_controls(key)
        d = {"spec": s, "q": q, "lr0": lr0, "w_op": w_op, "D0": e["norms"]["D0"], "passes": passes, "seeds": seeds,
             "marks": {}}
        for b in BATCHES:
            for kind, label, mult in CONTROLS:
                label = label.format(b=b)
                c = e["controls"][label]
                g = mult * b
                if c["cfg"]["batch"] != b or (c["method"] == "vr") != (mult == 1):
                    raise SystemExit(f"{key}, {label}: unexpected configuration")
                if c["method"] == "vr":
                    if (c["cfg"]["lr_decay"], c["cfg"]["lr"], c["cfg"]["orth"]) != (q, lr0, "ns5"):
                        raise SystemExit(f"{key}, {label}: not on the schedule of the controls")
                elif c["cfg"]["schedule"] != ["geometric", q, n / (2.0 * b)] or c["cfg"]["lr"] != lr0:
                    raise SystemExit(f"{key}, {label}: not on the schedule of the controls")
                vals = np.asarray(c["values"], dtype=float)
                mine = f"{np.median(vals):.3f} [{vals.min():.3f}, {vals.max():.3f}]"
                if digest.get(label, [None, None])[1] != mine:
                    raise SystemExit(f"{key}, {label}: {mine} in M9_summary.json, {digest.get(label)} in the digest")
                runs = [law.get(task_key({"spec": s, "method": c["method"], "cfg": c["cfg"], "passes": passes,
                                          "seed": seed, "stop_rel": 0.0})) for seed in seeds]
                if any(r is None for r in runs) or not np.allclose([r["final_rel"] for r in runs], vals, rtol=1e-12):
                    raise SystemExit(f"{key}, {label}: the runs of M9_law.pkl are not those of the summary")
                # the schedule: the length of the last step before a record at passes p is lr_0 q^(p - b/n)
                tr = runs[0]["trace"]
                p, eta = np.asarray(tr["passes"], dtype=float), np.asarray(tr["eta"], dtype=float)
                ok = np.isfinite(eta)
                dev = float(np.max(np.abs(eta[ok] / (lr0 * q ** (p[ok] - b / n)) - 1.0)))
                if dev > 1e-9:
                    raise SystemExit(f"{key}, {label}: recorded lengths differ from lr_0 q^(t g / n) by {dev:.1e}")
                path = lr0 / (1.0 - q ** (g / n))
                finite = float(np.sum(lr0 * q ** (np.arange(runs[0]["iters"]) * g / n)))
                if abs(finite / path - 1.0) > 1e-9:
                    raise SystemExit(f"{key}, {label}: sum over the {runs[0]['iters']} iterations {finite}, closed "
                                     f"form {path}")
                d["marks"][kind, b] = {"label": label, "path": path, "x": path / w_op, "values": vals,
                                       "y": float(np.median(vals)), "iters": runs[0]["iters"], "eta_dev": dev,
                                       "source": "M9 part C"}
            # the SAGA table on the same schedule, through the same map
            cfg0 = e["controls"][CONTROLS[0][1].format(b=b)]["cfg"]
            cfg = dict(cfg0, estimator="saga")
            path = lr0 / (1.0 - q ** (b / n))
            if b == 1:
                deep = e["deep"][TABLE_DEEP]
                if deep["cfg"] != cfg:
                    raise SystemExit(f"{key}: the runs '{TABLE_DEEP}' are not on the schedule of the controls")
                vals = np.asarray([r["final_rel"] for r in deep["runs"]], dtype=float)
                d["marks"]["table", b] = {"label": f"SAGA table, ns5 + Nesterov, b={b}", "path": path,
                                          "x": path / w_op, "values": vals, "y": float(np.median(vals)),
                                          "on_schedule": deep["on_schedule"], "source": "M9 part C"}
            else:
                for seed in seeds:
                    tasks.append({"spec": s, "method": "vr", "cfg": cfg, "passes": passes, "seed": seed,
                                  "stop_rel": 0.0})
                    index.append((key, b))
                d["marks"]["table", b] = {"label": f"SAGA table, ns5 + Nesterov, b={b}", "path": path,
                                          "x": path / w_op, "values": [], "source": "new runs of this script"}
        D["inst"][key] = d
    t0 = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        res = list(ex.map(C.run_one, tasks))
    D["wall"] = time.perf_counter() - t0
    for (key, b), r in zip(index, res):
        if r["diverged"]:
            raise SystemExit(f"{key}: a new run with the table at b = {b} diverged")
        D["inst"][key]["marks"]["table", b]["values"].append(r["final_rel"])
    for d in D["inst"].values():
        for (kind, b), m in d["marks"].items():
            m["values"] = np.asarray(m["values"], dtype=float)
            m["y"] = float(np.median(m["values"]))
            if b == 32 and D["gain"] * m["x"] >= 1.0:
                raise SystemExit("a mark at b = 32 is not below the reach bound")
            if m["x"] < 1.0 and m["y"] < (1.0 - D["gain"] * m["x"]) ** 2 * d["w_op"] ** 2 / d["D0"]:
                raise SystemExit("a final distance is below the reach bound")
    return D


def print_all(D):
    print(f"largest singular value returned by ns5 (maximum of its scalar map on [0, 1]): {D['gain']:.4f}")
    print(f"new runs (SAGA table at b = 32): {D['wall']:.0f} s with {JOBS} processes")
    for key, name, _ in INSTANCES:
        d = D["inst"][key]
        s = d["spec"]
        print(f"{key} ({name}): {s['loss']}, n = {s['n']}, d = {s['d']}, K = {s['K']}, kappa = {s['kappa']:g}; schedule "
              f"q = {d['q']}, lr_0 = {d['lr0']}; ||W_*||_op = {d['w_op']:.3f}; {d['passes']} passes, seeds {d['seeds']}")
        for (kind, b), m in d["marks"].items():
            v = m["values"]
            print(f"    {m['label']:52s} sum lr_t = {m['path']:8.3f} = {m['x']:6.3f} ||W_*||_op; final relative "
                  f"squared distance: median {m['y']:.3g} [{v.min():.3g}, {v.max():.3g}]  ({m['source']}"
                  + (f"; {m['on_schedule']}/{len(v)} seeds on schedule" if "on_schedule" in m else "")
                  + (f"; {m['iters']} iterations, recorded lengths within {m['eta_dev']:.0e} of the schedule"
                     if "iters" in m else "") + ")"
                  + ("" if kind != "storm" else "  [not drawn]"))
    b1 = [m["y"] for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 1 and k != "table"]
    b32 = [m["y"] for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 32 and k != "table"]
    t1 = [m["values"].max() for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 1 and k == "table"]
    t32 = [m["y"] for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 32 and k == "table"]
    x1 = [m["x"] for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 1]
    x32 = [m["x"] for d in D["inst"].values() for (k, b), m in d["marks"].items() if b == 32]
    gap = max(abs(d["marks"]["storm", b]["y"] - d["marks"]["mars", b]["y"]) for d in D["inst"].values()
              for b in BATCHES)
    print(f"no table, medians: b = 1: {min(b1):.3f} to {max(b1):.3f}; b = 32: {min(b32):.3f} to {max(b32):.3f} "
          f"(the paper: 8 % to 93 %; corrected: 8 % to 34 % at batch one)")
    print(f"SAGA table: b = 1: every run ends below {max(t1):.2e}; b = 32: medians {min(t32):.3f} and {max(t32):.3f} "
          f"(audit, two seeds: 0.574 and 0.785)")
    print(f"paths in units of ||W_*||_op: b = 1: {min(x1):.2f} to {max(x1):.2f}; b = 32: {min(x32):.3f} to "
          f"{max(x32):.3f} (audit: 0.08 to 0.37); times the gain of ns5: at most {D['gain'] * max(x32):.3f}")
    print(f"largest difference between the medians of STORM-Muon and MARS-M: {gap:.4f}")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    NO, TB = ps.BLUE, ps.ORANGE                    # no table: blue; SAGA table: orange
    fig = es.figure("talk", height=2.75)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.75, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(X_LIM[0], 1.0, color=es.tint(ink["muted"], 0.16), lw=0, zorder=0)
    ax.axvline(1.0, color=ink["primary"], lw=1.0, zorder=1)

    def style(kind, marker):
        ms = 7.0 if marker == "o" else 6.3
        if kind == "minibatch":
            return dict(marker=marker, ms=ms, mfc=NO, mec=ink["surface"], mew=1.0, zorder=5)
        if kind == "mars":
            return dict(marker=marker, ms=ms - 0.8, mfc=ink["surface"], mec=NO, mew=1.5, zorder=4)
        return dict(marker=marker, ms=ms, mfc=TB, mec=ink["surface"], mew=1.0, zorder=6)

    for key, _, marker in INSTANCES:
        d = D["inst"][key]
        for (kind, b), m in d["marks"].items():
            if kind == "storm":
                continue
            ax.plot([m["x"]], [m["y"]], ls="", clip_on=False, **style(kind, marker))
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"total path $\Sigma_t\,\mathrm{lr}_t\ /\ \|W_\ast\|_{op}$")
    ax.set_ylabel(r"final $\|W - W_\ast\|_F^2\,/\,\|W_\ast\|_F^2$")
    kw = dict(fontsize=es.small(), color=ink["secondary"], linespacing=1.25, zorder=7)
    t1 = max(m["values"].max() for d in D["inst"].values() for (k, b), m in d["marks"].items()
             if b == 1 and k == "table")
    expo = int(np.floor(np.log10(t1)))
    mant = int(np.ceil(t1 / 10.0 ** expo))                       # the bound rounded up to one digit
    ax.text(0.05, 0.36, "$b = 32$: path too short,\nwith or without a table", ha="left", va="top", **kw)
    ax.text(26.0, 0.80, "$b = 1$: path long enough;\nwith the table,\n"
            + rf"below ${mant}\cdot10^{{{expo}}}$", ha="right", va="top", **kw)

    # key, in ink, in its own cell
    kx = fig.add_subplot(gs[0, 1])
    es.diagram_axes(kx)
    grey = ink["secondary"]
    rows = [(dict(marker="o", ms=7.0, mfc=NO, mec=ink["surface"], mew=1.0), "no table: minibatch\ngradient"),
            (dict(marker="o", ms=6.2, mfc=ink["surface"], mec=NO, mew=1.5), "no table: MARS-M"),
            (dict(marker="o", ms=7.0, mfc=TB, mec=ink["surface"], mew=1.0), "SAGA table, same\nschedule and map\n($b = 32$: new runs)"),
            (dict(marker="o", ms=7.0, mfc=grey, mec=ink["surface"], mew=1.0), "softmax"),
            (dict(marker="s", ms=6.3, mfc=grey, mec=ink["surface"], mew=1.0), "least squares")]
    ys = (0.93, 0.76, 0.565, 0.33, 0.23)
    for (st, text), y in zip(rows, ys):
        kx.plot([0.06], [y], ls="", **st)
        kx.text(0.16, y, text, ha="left", va="center", fontsize=es.small(), color=grey, linespacing=1.2)
    d0 = D["inst"][INSTANCES[0][0]]
    kx.text(0.0, 0.07, f"medians of {len(d0['seeds'])} seeds,\n{d0['passes']} passes from $W_0 = 0$", ha="left",
            va="center", fontsize=es.small(), color=grey, linespacing=1.2)
    es.save(fig, NAME, "talk",
            note=(rf"M9 controls, $n = {d0['spec']['n']}$, cooldown tuned for Muon's recipe at $b = 1$; steps of "
                  rf"spectral norm $\leq {D['gain']:.2f}\,\mathrm{{lr}}_t$ (ns5)"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
