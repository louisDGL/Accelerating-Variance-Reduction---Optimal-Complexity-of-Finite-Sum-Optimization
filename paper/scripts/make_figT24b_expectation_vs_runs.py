"""Figure T24b of the lecture (paper/talk/figures/figT24b_expectation_vs_runs): under independent sampling the
EXPECTED squared error of linear SAGA loses one nat per pass, but a single run loses more once every entry of the
initial table has been refreshed. Explanatory figure, not in the paper; to be shown before figure T24. It explains the
corrected item (M7) of paper/paper_saga_to_muon.tex ("in expectation", "linear SAGA", "the memory factor bounds the
mean squared error, not a single run") and the row "Memory" of Table 2.

The model (vrlab/order_model.py, an exact instance of the statements (R1) and (R3) of its docstring): n = 1000
quadratics f_i with ONE common Hessian; in the eigenbasis each mode follows the scalar recursion
    e+ = e - g (e - psi_i + mean_j psi_j),   psi_i <- e,   g = gamma lambda,   c = n g = gamma n lambda = 2,
started at e_0 = 1 with the uniform table psi_j = e_0. One panel, the squared error e_t^2 / e_0^2 at the end of each
pass (n iterations), passes 0 to 60, logarithmic axis:
  - independent sampling, expectation: E e_t^2, exact, read from results/B4.pkl ["model"]["illus"]["exact_E_e2"] and
    recomputed here with order_model.iid_moment_matrix (4 x 4 recursion of the second moments);
  - independent sampling, nine single runs (seeds 0..8): results/B4.pkl ["model"]["illus"]["runs_e2"], recomputed here
    with order_model.simulate (the index stream of vrlab.algorithms.saga for the same seed);
  - the passes at which the last entry of the initial table of each run is refreshed
    (["model"]["illus"]["all_refreshed_pass"], recomputed here): nine ticks at the top and the band they span;
  - a fresh permutation per pass, expectation: E e^2 at pass ends, exact, order_model.rr_moment_matrix(n, g) applied
    pass by pass to the initial moments (1, 1, 1, 1), first component; checked here against the mean of RR_SEEDS
    simulated reshuffled runs.
Exponents in the legend (nats per pass of the squared error): 1.0 = order_model.iid_exponent(n, c) (0.9992);
1.5 = order_model.rr_ms_exponent(n, c) (1.538); "1.5 to 1.7" = range over the nine seeds of the exponent of a single
run fitted over the passes 30-150 of longer runs of the same seeds, results/B4_summary.json
["model"]["by_c"]["2"]["n=1000"]["iid_typical"] (median 1.579, range 1.504-1.680; the paper's "about 1.6"). A range,
not the median, is written because the nine displayed runs, fitted over the passes 30-60, have a median of 1.68 (range
1.49-1.83, printed): their visible slope is closer to 1.7 than to 1.6.

Status. Proved: the two expectations (exact linear recursions of the second moments). Measured (9 seeds): the single
runs and their exponent. The picture is about linear SAGA on a sum with a common Hessian at one step (c = 2); it is not
the fixed-length recipe of the paper, and at this step a single independent run is about as fast as the expectation
under a permutation: "two nats against one" compares expectations.

Checks (the script stops if one fails): the stored expectation against its recomputation and against three values and
the exponent printed in results/digest.md (section "B4 (M): expectation against single runs"); the stored runs and
ticks against their recomputation; the reshuffled expectation against simulated runs.

Progressive build (states s1-s3): s1 the expectation under independent sampling; s2 the nine runs and the ticks;
s3 the expectation under a fresh permutation.

The script prints every number it plots.
Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT24b_expectation_vs_runs.py
Output: paper/talk/figures/figT24b_expectation_vs_runs.pdf and .png, and the states _s1 ... _s3
"""
from __future__ import annotations

import json
import os
import pickle
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_style as es  # noqa: E402

sys.path.insert(0, es.ROOT)
from vrlab import order_model  # noqa: E402
from vrlab.algorithms import _index_stream  # noqa: E402

NAME = "figT24b_expectation_vs_runs"
RESULTS = os.path.join(es.ROOT, "results")
RR_SEEDS = tuple(range(20))            # simulated reshuffled runs for the check of the exact reshuffled expectation
TOL_RR = 0.1                           # largest accepted |log(mean of the simulated runs / exact expectation)|
# results/digest.md, "B4 (M): expectation against single runs, i.i.d. indices, n = 1000, c = 2 (9 seeds)"
DIGEST_E = {10: 8.12e-08, 20: 3.72e-12, 60: 1.63e-29}
DIGEST_EXPONENT = 0.9992
DIGEST_LAST = (7.723, 5.722, 9.116)    # pass at which the last slot of the initial memory is refreshed: median, min, max


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def last_initial_entry(n, passes, seed):
    """Pass at which the last entry of the initial table is refreshed under independent sampling (seed of the run)."""
    idx = np.concatenate(list(_index_stream(np.random.default_rng(seed), n, passes * n)))
    first = np.full(n, -1)
    first[idx[::-1]] = np.arange(idx.size)[::-1]              # first occurrence of each index
    if first.min() < 0:
        raise SystemExit(f"seed {seed}: an entry of the initial table is never refreshed in {passes} passes")
    return float(first.max() + 1) / n


def data():
    with open(os.path.join(RESULTS, "B4.pkl"), "rb") as f:
        ill = pickle.load(f)["model"]["illus"]
    with open(os.path.join(RESULTS, "B4_summary.json"), encoding="utf-8") as f:
        summ = json.load(f)["model"]
    n, c = int(ill["n"]), float(ill["c"])
    g = c / n
    E_iid = np.asarray(ill["exact_E_e2"], dtype=float)
    runs = np.asarray(ill["runs_e2"], dtype=float)
    last = np.asarray(ill["all_refreshed_pass"], dtype=float)
    P = E_iid.size - 1
    seeds = list(range(runs.shape[0]))
    passes = np.arange(P + 1)

    # recomputations and checks of what is read
    Tn = np.linalg.matrix_power(order_model.iid_moment_matrix(n, g), n)
    mom, again = np.ones(4), [1.0]
    for _ in range(P):
        mom = Tn @ mom
        again.append(float(mom[0]))
    dev_E = float(np.abs(np.array(again) / E_iid - 1.0).max())
    runs_again = np.array([order_model.simulate(n, g, "iid", passes=P, seed=s)["e"][::n] ** 2 for s in seeds])
    dev_runs = float(np.abs(runs_again / runs - 1.0).max())
    last_again = np.array([last_initial_entry(n, P, s) for s in seeds])
    dev_last = float(np.abs(last_again - last).max())
    exp_iid = order_model.iid_exponent(n, c)
    if dev_E > 1e-9 or dev_runs > 1e-9 or dev_last > 1e-9:
        raise SystemExit(f"results/B4.pkl is not reproduced: expectation {dev_E:.1e}, runs {dev_runs:.1e}, "
                         f"ticks {dev_last:.1e}")
    for p, v in DIGEST_E.items():
        if abs(E_iid[p] / v - 1.0) > 5e-3:
            raise SystemExit(f"E e^2 at pass {p}: {E_iid[p]:.3e} against {v:.3e} in results/digest.md")
    if abs(exp_iid - DIGEST_EXPONENT) > 5e-5:
        raise SystemExit(f"exponent of E e^2: {exp_iid:.5f} against {DIGEST_EXPONENT} in results/digest.md")
    if np.abs(np.array([np.median(last), last.min(), last.max()]) - np.array(DIGEST_LAST)).max() > 1e-3:
        raise SystemExit("the passes of the last refresh of the initial table differ from results/digest.md")

    # a fresh permutation per pass: exact expectation at pass ends, checked against simulated runs
    R = order_model.rr_moment_matrix(n, g)
    mom, E_rr = np.ones(4), [1.0]
    for _ in range(P):
        mom = R @ mom
        E_rr.append(float(mom[0]))
    E_rr = np.array(E_rr)
    sim_rr = np.array([order_model.simulate(n, g, "rr", passes=P, seed=s)["e"][::n] ** 2 for s in RR_SEEDS])
    dev_rr = float(np.abs(np.log(sim_rr.mean(axis=0) / E_rr)).max())
    spread_rr = float(np.abs(np.log(sim_rr / E_rr)).max())
    if dev_rr > TOL_RR:
        raise SystemExit(f"reshuffled expectation: |log(simulated mean / exact)| = {dev_rr:.3f} > {TOL_RR}")
    exp_rr = order_model.rr_ms_exponent(n, c)

    typ = summ["by_c"][f"{c:g}"][f"n={n}"]["iid_typical"]
    if int(typ["n"]) != len(seeds) or list(summ["window_passes"]) != [30, 150]:
        raise SystemExit("results/B4_summary.json: unexpected seeds or window of the single-run exponents")
    half = P // 2
    fit = np.array([order_model.fit_exponent(np.log(r), burn=half) for r in runs])
    return {"n": n, "c": c, "P": P, "passes": passes, "seeds": seeds, "E_iid": E_iid, "runs": runs, "last": last,
            "E_rr": E_rr, "exp_iid": exp_iid, "exp_rr": exp_rr, "typ": typ, "window": list(summ["window_passes"]),
            "fit_displayed": fit, "fit_from": half, "dev_E": dev_E, "dev_runs": dev_runs, "dev_last": dev_last,
            "dev_rr": dev_rr, "spread_rr": spread_rr}


def print_all(D):
    sel = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60]
    sel = [p for p in sel if p <= D["P"]]
    print(f"linear SAGA on the identical-Hessian model, n = {D['n']}, c = gamma n lambda = {D['c']:g}; squared error at "
          f"pass ends")
    print("  passes                               : " + ", ".join(f"{p:>8d}" for p in sel))
    print("  independent, E e^2 (exact)           : " + ", ".join(f"{D['E_iid'][p]:8.2e}" for p in sel))
    print("  independent, median of the 9 runs    : " + ", ".join(f"{np.median(D['runs'][:, p]):8.2e}" for p in sel))
    print("  independent, smallest of the 9 runs  : " + ", ".join(f"{D['runs'][:, p].min():8.2e}" for p in sel))
    print("  independent, largest of the 9 runs   : " + ", ".join(f"{D['runs'][:, p].max():8.2e}" for p in sel))
    print("  fresh permutation, E e^2 (exact)     : " + ", ".join(f"{D['E_rr'][p]:8.2e}" for p in sel))
    last = D["last"]
    print("  last entry of the initial table refreshed at pass (seeds 0..8): " + ", ".join(f"{v:.3f}" for v in last)
          + f"; median {np.median(last):.3f}, range {last.min():.3f}-{last.max():.3f}")
    t = D["typ"]
    f = D["fit_displayed"]
    print(f"  exponents (nats per pass): independent, expectation {D['exp_iid']:.4f}; fresh permutation, expectation "
          f"{D['exp_rr']:.4f}; single independent runs, passes {D['window'][0]}-{D['window'][1]} "
          f"(B4_summary.json): median {t['median']:.3f}, range {t['min']:.3f}-{t['max']:.3f}; the nine displayed runs "
          f"fitted over the passes {D['fit_from']}-{D['P']}: median {np.median(f):.3f}, range {f.min():.3f}-{f.max():.3f}")
    print(f"  checks: stored expectation against its recomputation {D['dev_E']:.1e} (relative); stored runs "
          f"{D['dev_runs']:.1e}; stored ticks {D['dev_last']:.1e}; reshuffled expectation against the mean of "
          f"{len(RR_SEEDS)} simulated runs: max |log ratio| {D['dev_rr']:.3f} (single runs: {D['spread_rr']:.3f})")


# ----------------------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------------------
def draw(D):
    ps = es.setup("talk")
    ink = ps.INK
    IID, RR = ps.BLUE, ps.ORANGE                  # independent sampling; a fresh permutation per pass
    fig = es.figure("talk", height=2.75)
    ax = fig.add_subplot(111)
    b = es.Build()
    p = D["passes"]
    y_lo, y_hi = 1e-49, 30.0

    # s1: the expectation under independent sampling
    ax.plot(p, D["E_iid"], color=IID, lw=3.0, zorder=5,
            label=f"independent, expectation (exact): {D['exp_iid']:.1f} nat per pass")

    # s2: the nine runs, and when the last entry of the initial table of each run is refreshed
    for k, r in enumerate(D["runs"]):
        b.add(2, ax.plot(p, r, color=IID, lw=0.8, alpha=0.75, zorder=3,
                         label=(f"independent, {len(D['seeds'])} single runs: {D['typ']['min']:.1f} to "
                                f"{D['typ']['max']:.1f}"
                                if k == 0 else None)))
    last = D["last"]
    b.add(2, ax.axvspan(last.min(), last.max(), color=es.tint(ink["secondary"], 0.16), lw=0, zorder=1))
    b.add(2, ax.vlines(last, 0.955, 1.0, transform=ax.get_xaxis_transform(), color=ink["primary"], lw=1.0, zorder=6))
    b.add(2, ax.annotate("the last entry of the initial table\nis refreshed (one tick per run)", (last.max(), 1.0),
                         xycoords=("data", "axes fraction"), xytext=(6, -3), textcoords="offset points", ha="left",
                         va="top", fontsize=es.small(), color=ink["secondary"], linespacing=1.15))

    # s3: the expectation under a fresh permutation per pass
    b.add(3, ax.plot(p, D["E_rr"], color=RR, lw=2.4, zorder=4,
                     label=f"fresh permutation, expectation (exact): {D['exp_rr']:.1f}"))

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(0, D["P"])
    ax.set_xticks(range(0, D["P"] + 1, 10))
    ax.set_yticks([1.0, 1e-10, 1e-20, 1e-30, 1e-40])
    ax.minorticks_off()
    ax.set_xlabel("passes")
    ax.set_ylabel(r"squared error $e_t^2\,/\,e_0^2$")
    # anchored by its top-left corner, so that the entries do not move from one state of the build to the next
    b.legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.29), handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)
    es.save(fig, NAME, "talk", build=b,
            note=(rf"linear SAGA, $n = {D['n']}$ quadratics, one common Hessian, $\gamma n\lambda = {D['c']:g}$; "
                  f"exact expectations, {len(D['seeds'])} simulated runs"))


def main():
    D = data()
    print_all(D)
    draw(D)


if __name__ == "__main__":
    main()
