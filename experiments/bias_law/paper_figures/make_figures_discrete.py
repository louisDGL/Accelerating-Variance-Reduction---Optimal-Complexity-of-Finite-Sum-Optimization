"""Figures for the discrete-noise section.  Reads results/bias_law/figure_data/{exact_lattice.jsonl, mc_3x3.json, wor.json,
momentum_runs.jsonl} (produced by exp_exact_lattice.py, exp_mc_3x3.py, exp_wor.py, exp_momentum_discrete.py) and writes
figures/bias_law/{fig_lattice, fig_wor, fig_momentum_discrete}.pdf (+ .png previews at 200 dpi),
results/bias_law/figure_data/{fig_lattice, fig_wor, fig_momentum_discrete, table_exact}.json (the plotted numbers).
usage: make_figures_discrete.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "..", "figures", "bias_law")
RES = os.path.join(HERE, "..", "..", "..", "results", "bias_law"); DATA = os.path.join(RES, "figure_data")
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
CL = [BLUE, VERM, GREEN]                      # residue classes b = 0, 1, 2 mod 3
G33 = (3 + 2 * 0.68691) / 50
COEF12 = [-1.0119, +0.3326, -0.1775]          # results/bias_law/lattice/e02.log: sqrt(b) (b G_b - g*) for 1x2, b = 0, 1, 2 mod 3


def tag(ax, s, text="", x=0.03, y=0.97, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", va="top", ha="left", fontsize=9, **kw)
    if text:
        ax.annotate(text, xy=(x, y), xycoords="axes fraction", xytext=(15, -0.6), textcoords="offset points", va="top", ha="left", fontsize=8, **kw)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=200); plt.close(fig)


def load_exact():
    recs = {}
    for line in open(os.path.join(DATA, "exact_lattice.jsonl")):
        r = json.loads(line); recs[(r["geom"], r["b"])] = r
    out = {}
    for (g, b), r in sorted(recs.items()):
        out.setdefault(g, []).append(r)
    return out


def by_class(rows, key="m", bmax=None, bmin=None):
    res = []
    for c in range(3):
        rr = [r for r in rows if r["b"] % 3 == c and (bmax is None or r["b"] <= bmax) and (bmin is None or r["b"] >= bmin)]
        res.append((np.array([r["b"] for r in rr]), np.array([r[key] for r in rr])))
    return res


# ------------------------------------------------------------------------------------------------ FIG 1
def fig_lattice():
    ex = load_exact(); mc = json.load(open(os.path.join(DATA, "mc_3x3.json")))
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 3.6))
    plotted = {}
    # (a) sign
    ax = axs[0, 0]
    for c, (b, m) in enumerate(by_class(ex["sign"], bmax=60)):
        ax.plot(b, m, "o", color=CL[c], ms=2.6, label=[r"$b\equiv 0$", r"$1$", r"$2\ (\mathrm{mod}\ 3)$"][c])
        plotted.setdefault("a_sign", {})[c] = dict(b=b.tolist(), m=m.tolist())
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_ylim(-1.3, 1.75); ax.set_xlim(0, 61); ax.set_yticks([-1, 0, 1])
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.26), ncol=3, handletextpad=0.0, columnspacing=0.5)
    tag(ax, "(a)", "sign")
    # (b) 1x2, (c) 2x3, (d) 2x2
    for ax, g, lab, t, key in [(axs[0, 1], "1x2", r"$1\times 2$", "(b)", "b_1x2"), (axs[0, 2], "2x3", r"$2\times 3$", "(c)", "c_2x3"),
                               (axs[1, 0], "2x2", r"$2\times 2$", "(d)", "d_2x2")]:
        for c, (b, m) in enumerate(by_class(ex[g])):
            ax.plot(b, m, "o-", color=CL[c], ms=2.4 if g != "1x2" else 3, lw=0.7)
            plotted.setdefault(key, {})[c] = dict(b=b.tolist(), m=m.tolist())
        ax.axhline(ex[g][0]["gstar"], color="k", ls="--", lw=0.8)
        ax.set_xscale("log"); tag(ax, t, lab)
    axs[0, 1].set_ylim(-0.02, 0.52); axs[0, 2].set_ylim(-0.01, 0.2); axs[1, 0].set_ylim(-0.02, 0.4)
    # (e) 3x3 Monte Carlo
    ax = axs[1, 1]
    for c in range(3):
        rr = [r for r in mc["results"] if r["mod3"] == c]
        b = np.array([r["b"] for r in rr]); m = np.array([r["m"] for r in rr]); se = np.array([r["se"] for r in rr])
        ax.errorbar(b * [0.9, 1.0, 1.11][c], m, yerr=se, fmt="o", color=CL[c], ms=2.8, lw=0.8, capsize=1.5, capthick=0.6)
        plotted.setdefault("e_3x3", {})[c] = dict(b=b.tolist(), m=m.tolist(), se=se.tolist(), nsamples=[r["nsamples"] for r in rr])
    ax.axhline(G33, color="k", ls="--", lw=0.8); ax.set_xscale("log"); ax.set_ylim(-0.012, 0.26)
    tag(ax, "(e)", r"$3\times 3$ (Monte Carlo)")
    # (f) rates
    ax = axs[1, 2]
    mk = {"1x2": "o", "1x3": "s", "2x3": "^"}
    for g in ("1x2", "1x3", "2x3"):
        for c, (b, e) in enumerate(by_class(ex[g], key="err", bmin=4)):
            ok = np.abs(e) > 0
            ax.plot(b[ok], np.abs(e[ok]), mk[g], color=CL[c], ms=2.6, mfc=CL[c] if g == "1x2" else "none", mew=0.6)
            plotted.setdefault("f_rate_" + g, {})[c] = dict(b=b.tolist(), abs_err=np.abs(e).tolist())
    bb = np.array([4.0, 6000.0])
    for c in range(3):
        ax.plot(bb, abs(COEF12[c]) / np.sqrt(bb), "-", color=CL[c], lw=0.5)
    ax.plot([15, 150], [1.2e-3, 1.2e-4], "k-", lw=0.6); ax.text(30, 1.6e-4, r"$b^{-1}$", fontsize=8, ha="right")
    ax.text(2300, 0.03, r"$b^{-1/2}$", fontsize=8, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3, 7000); ax.set_ylim(5e-5, 40)
    ax.legend(handles=[Line2D([], [], marker=mk[g], color="k", ls="", ms=3, mfc="k" if g == "1x2" else "none", mew=0.6,
                              label=r"$%s\times %s$" % (g[0], g[2])) for g in mk],
              loc="upper right", bbox_to_anchor=(1.03, 0.9), ncol=3, handletextpad=0.0, columnspacing=0.6, borderaxespad=0.0)
    tag(ax, "(f)")
    for ax in axs.ravel(): ax.set_xlabel(r"$b$", labelpad=1)
    for ax in axs[:, 0]: ax.set_ylabel(r"$b\,G_b^\dagger$")
    axs[1, 2].set_ylabel(r"$|b\,G_b^\dagger-g_*|$", labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.6)
    save(fig, "fig_lattice")
    json.dump(plotted, open(os.path.join(DATA, "fig_lattice.json"), "w"), indent=1)
    # table
    table = {}
    for g, rows in ex.items():
        table[g] = {}
        for c in range(3):
            r = max((r for r in rows if r["b"] % 3 == c), key=lambda r: r["b"])
            table[g]["b=%d mod 3" % c] = dict(b=r["b"], bG=r["m"], gstar=r["gstar"], err=r["err"], sqrtb_err=r["sqrtb_err"],
                                              b_err=r["b_err"], locked_at_zero=r["locked"], dropped_mass=r["dropped"], method="exact")
    table["3x3"] = {}
    for c in range(3):
        r = max((r for r in mc["results"] if r["mod3"] == c), key=lambda r: r["b"])
        e = r["m"] - G33
        table["3x3"]["b=%d mod 3" % c] = dict(b=r["b"], bG=r["m"], se=r["se"], gstar=G33, err=e, sqrtb_err=np.sqrt(r["b"]) * e,
                                              sqrtb_err_se=np.sqrt(r["b"]) * r["se"], b_err=r["b"] * e, b_err_se=r["b"] * r["se"],
                                              method="Monte Carlo, %d samples" % r["nsamples"])
    json.dump(table, open(os.path.join(DATA, "table_exact.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ FIG 2
def fig_wor():
    w = json.load(open(os.path.join(DATA, "wor.json")))
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    x = np.linspace(0, 1, 5)
    for ax in axs:
        ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0.5, color="0.6", lw=0.5)
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_xlabel(r"sampling fraction $b/n$"); ax.set_xlim(0, 1); ax.set_ylim(-1.6, 1.6)
    A = w["populations"]["A"]; n = A["n"]; r = A["rows"]
    axs[0].plot(x, (n - 2 * x * n) / (n - 2), "k--", lw=0.9, label=r"$(n-2b)/(n-2)$")
    axs[0].errorbar([q["frac"] for q in r], [q["coef"] for q in r], yerr=[q["se"] for q in r], fmt="o", color=ORANGE, ms=3.2,
                    capsize=1.5, lw=0.8, capthick=0.6, label=r"measured, $n=400$", zorder=3)
    axs[0].plot([], [], "k:", lw=0.8, label="with replacement")
    axs[0].legend(loc="lower left", borderaxespad=0.2, handlelength=1.8)
    axs[0].set_ylabel(r"$b\,\langle G_b^\dagger,g_*\rangle/\|g_*\|^2$")
    tag(axs[0], "(a)", r"generic population, $2\times 3$", x=0.30)
    names = sorted([k for k in w["populations"] if k.startswith("B")], key=lambda s: int(s[1:]))
    axs[1].plot(x, 1 - 2 * x, "k--", lw=0.9)
    for name in names:
        B = w["populations"][name]; big = name == names[-1]
        for c in range(3):
            rr = [q for q in B["rows"] if q["mod3"] == c]
            axs[1].errorbar([q["frac"] for q in rr], [q["coef"] for q in rr], yerr=[q["se"] for q in rr], fmt="o" if big else "s",
                            color=CL[c], ms=3.2 if big else 2.8, mfc=CL[c] if big else "none", mew=0.6, capsize=1.2, lw=0.7, capthick=0.5, zorder=3)
    h = [Line2D([], [], marker="s", color="0.3", ls="", ms=3, mfc="none", mew=0.6, label=r"$n=%d$" % w["populations"][names[0]]["n"])]
    if len(names) > 1:
        h.append(Line2D([], [], marker="o", color="0.3", ls="", ms=3.2, label=r"$n=%d$" % w["populations"][names[-1]]["n"]))
    h += [Line2D([], [], marker="o", color=CL[c], ls="", ms=3, label=r"$b\equiv %d$" % c) for c in range(3)]
    axs[1].legend(handles=h, loc="lower left", borderaxespad=0.2, handletextpad=0.2, labelspacing=0.25)
    tag(axs[1], "(b)", r"lattice population, $2\times 2$", x=0.30)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig_wor")
    json.dump(w["populations"], open(os.path.join(DATA, "fig_wor.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ FIG 3
def momentum_summary():
    runs = {}
    for line in open(os.path.join(DATA, "momentum_runs.jsonl")):
        r = json.loads(line); runs.setdefault((r["kind"], r["b"], r["beta"]), []).append(r)
    rows = []
    for (kind, b, beta), rs in sorted(runs.items()):
        g = np.concatenate([r["G_run"] for r in rs]); x = np.concatenate([r["X_run"] for r in rs]); g3 = np.concatenate([r["G3_run"] for r in rs])
        beff = b * (1 + beta + beta ** 2) / (1 - beta ** 2); bvar = b * (1 + beta) / (1 - beta); n = len(g)
        k = np.cov(g, x)[0, 1] / x.var(ddof=1)                    # control variate coefficient (E x = 0 exactly)
        gc = g - k * x
        rows.append(dict(kind=kind, b=b, beta=beta, b_eff=beff, b_var=bvar, runs=n, T=rs[0]["T"],
                         val=beff * g.mean(), se=beff * g.std(ddof=1) / np.sqrt(n),
                         val_cv=beff * gc.mean(), se_cv=beff * gc.std(ddof=1) / np.sqrt(n), cv_coef=k,
                         val_third_quarter=beff * g3.mean(), se_third_quarter=beff * g3.std(ddof=1) / np.sqrt(n),
                         val_varnorm=bvar * g.mean(), se_varnorm=bvar * g.std(ddof=1) / np.sqrt(n), raw_mean_grad=g.mean()))
    return rows


def fig_momentum():
    rows = momentum_summary()
    mc = {r["b"]: r for r in json.load(open(os.path.join(DATA, "mc_3x3.json")))["results"]}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.4))
    sty = {0.0: (BLUE, "o"), 0.5: (ORANGE, "s"), 0.9: (VERM, "^")}
    ins = axs[1].inset_axes([0.50, 0.09, 0.47, 0.36])
    for ax, kind, gs in [(axs[0], "nuc", G33), (axs[1], "sign", 1 / 6), (ins, "sign", 1 / 6)]:
        ax.axhline(gs, color="k", ls="--", lw=0.9, label=r"$g_*$" if ax is axs[0] else None)
        for beta, (col, mk) in sty.items():
            rr = [r for r in rows if r["kind"] == kind and r["beta"] == beta]
            if ax is ins and beta == 0: continue
            ax.errorbar([r["b_eff"] for r in rr], [r["val_cv"] for r in rr], yerr=[r["se_cv"] for r in rr], fmt=mk, color=col,
                        ms=3.6 if ax is not ins else 2.8, lw=0.7, capsize=1.5, capthick=0.6, label=r"$\beta=%g$" % beta, zorder=3)
            if beta > 0 and kind == "nuc":
                ax.axhline(gs * (1 + beta + beta ** 2) / (1 + beta) ** 2, color=col, ls=":", lw=0.9)
        ax.set_xscale("log")
    bs = sorted({r["b"] for r in rows if r["kind"] == "nuc" and r["beta"] == 0} & set(mc))
    axs[0].plot(bs, [mc[b]["m"] for b in bs], "x", color="k", ms=5.5, mew=0.7, zorder=2, label=r"$b\,G_b^\dagger$ (direct)")
    axs[0].plot([], [], ":", color="0.4", lw=0.9, label="variance rule")
    axs[0].set_ylabel(r"$b_{\mathrm{eff}}\times$ mean gradient entry"); axs[0].set_ylim(-0.012, 0.235); axs[0].set_xlim(0.8, 330)
    axs[0].legend(loc="upper right", borderaxespad=0.0, ncol=2, columnspacing=0.8, handletextpad=0.3, labelspacing=0.25, handlelength=1.6)
    axs[1].set_ylim(-1.3, 1.45); axs[1].set_xlim(0.8, 330); axs[1].set_yticks([-1, 0, 1])
    ins.set_xlim(3.5, 330); ins.set_ylim(0.148, 0.196); ins.tick_params(labelsize=7, length=2, pad=1.5)
    for ax in axs: ax.set_xlabel(r"$b_{\mathrm{eff}}$", labelpad=1)
    tag(axs[0], "(a)", r"nuclear, $3\times 3$"); tag(axs[1], "(b)", "sign")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig_momentum_discrete")
    json.dump(dict(gstar_nuc=G33, gstar_sign=1 / 6, direct_pure_batch_3x3={b: dict(m=mc[b]["m"], se=mc[b]["se"]) for b in bs},
                   note="plotted: val_cv +- se_cv (control variate = time-averaged noise, exact mean zero); val +- se is the raw average",
                   rows=rows), open(os.path.join(DATA, "fig_momentum_discrete.json"), "w"), indent=1)


NOTES = """## Things that did NOT behave as the first-order theory alone would suggest (read before quoting)

1. Fig. 1(d), 2x2: on b = 0 mod 3 the minimiser is exactly 0 up to b = 513 and is still 35% below g_* at b = 2049
   (0.0742 vs 0.1138); sqrt(b) x error is -2.29, -2.02, -1.82 (b=1536), -1.79 (b=2049): no rate can be read off.
   On the other two classes b G_b moves along rational plateaux (1/7, 1/11, 2/17, 1/9, 1/15 at b=1536) rather than smoothly;
   convergence there is visible but not monotone.
2. Fig. 1(f): 1x3 and 2x3 decay like 1/b in every class, but the constant is class dependent and about 8-10 times larger on
   b = 0 mod 3 (2x3: b x error = -1.02 vs +0.13, -0.09; 1x3: -0.81 vs +0.23, -0.10).  The 2x3 range is only b <= 102.
3. Fig. 1(e), 3x3: b = 3, 6, 12 are locked at 0 (b = 30: 0.026, b = 99: 0.072, b = 300: 0.083).  The three points at
   b = 999..1001 are each about 2 standard errors below g_* (ratios 0.967-0.977 +- 0.015); b = 3000..3002 agree within 1 se.
4. Fig. 2(b), lattice population: the reversal of SIGN holds (no point has the wrong sign beyond 2.5 se), but the
   LINE (n-2b)/(n-2) is not followed at n = 648: every b = 0 mod 3 is locked at exactly 0, for 0.42 <= b/n <= 0.58 all
   three classes are locked at 0, and classes 1, 2 deviate from the line by up to 0.63 (rms 0.28-0.29).  At n = 10368 classes 1 and 2
   follow the line with rms deviation 0.07 (se 0.02-0.08; the largest deviations, 0.10-0.15 = 6-9 se, are at the extreme
   fractions b/n = 0.02 and 0.98, where b(n-b)/n ~ 210 is small); class 0 is still mostly locked (|c_b| <= 0.16).  This is the same
   transient locking as in Fig. 1(d) (2x2), the relevant size being b(n-b)/n.
   Caveat on the estimator: near a locked point individual batch minimisers are bimodal (some batches exactly 0, others
   not), so the mean of batch minimisers is a noisy and possibly biased (towards 0) estimate of the population minimiser;
   the class-0 values at n = 10368 (0.03-0.15) should be read as "partially released", not as precise numbers.
5. Fig. 2(a), generic population: agreement with the line within about 2 se except at small sampling fractions
   (b = 8, 25, 392, 375: +0.020 to +0.033 in absolute value above the line, 3 to 7 se) - a finite-b correction, present
   also with replacement.  g_* itself carries about 1% Monte-Carlo error.
6. Fig. 3(a): for beta = 0 the iteration reproduces the pure-batch lattice values b G_b (crosses), NOT g_*:
   0.200, 0.038, 0, 0.093, 0.069, 0, 0.088 for b = 1, 2, 3, 4, 8, 12, 16.  Exception: b = 12, beta = 0 gives 0.0122 +- 0.0002
   where the direct minimiser is exactly 0; the value decreases along the run (0.0133 third quarter, 0.0111 fourth
   quarter), i.e. a step-size artefact at the kink (one-sided slopes -0.185 / +0.034), not yet converged at T = 4e5.
   With beta > 0 all points with b_eff >= 9 are within 5% of g_* and clearly separated from the variance rule
   (0.068 / 0.066); b = 1, beta = 0.5 (b_eff = 2.3) is at 0.131.
7. Fig. 3(b): sign descent fails to converge ONLY for beta = 0 (b_eff x limit = +1, -1, 0 by residue class; the small
   offsets such as 0.9944 at b = 16 are step-size effects).  With momentum the filtered noise is no longer lattice and
   the scaled limit DOES approach 1/6: beta = 0.5: 0.437, 0.154, 0.187, 0.178, 0.171, 0.170, 0.168, 0.170 for
   b = 1, 2, 3, 4, 6, 8, 12, 16; beta = 0.9: within 3% of 1/6 from b = 2 on (still 3-5 se above for b <= 4; b = 1: 0.1765 +- 0.0006).
   Panel (b) therefore shows all three beta, the non-convergence being the beta = 0 series only (inset: beta > 0).
8. Compute: T was reduced to 4e5 steps and 64 runs per configuration (192 / 256 for beta = 0.9, b = 8 / 16) because the
   machine was shared; the wall-clock budget of 40 minutes was exceeded (two machine crashes; the without-replacement
   runs were redone and the extra batches for n = 10368 were cut short).
   The 2x3 exact range stops at b = 102 and the 2x2 one at b = 2049 for cost reasons.
"""


def slope(rows, c, bmin):
    rr = [r for r in rows if r["b"] % 3 == c and r["b"] >= bmin and r["err"] != 0 and not r["locked"]]
    return float(np.polyfit(np.log([r["b"] for r in rr]), np.log([abs(r["err"]) for r in rr]), 1)[0]), len(rr)


def numbers():
    ex = load_exact(); mc = json.load(open(os.path.join(DATA, "mc_3x3.json"))); w = json.load(open(os.path.join(DATA, "wor.json")))
    T = json.load(open(os.path.join(DATA, "table_exact.json"))); mom = json.load(open(os.path.join(DATA, "fig_momentum_discrete.json")))
    L = []; P = L.append
    P("# Numbers for the discrete-noise experiments (generated by `experiments/bias_law/paper_figures/make_figures_discrete.py`)\n")
    P("Noise law everywhere: i.i.d. entries in {-1, 2} with probabilities (2/3, 1/3): mean 0, variance 2, third cumulant 2 (lattice, span 3). "
      "`b G_b` always denotes ONE entry of b times the minimiser (the minimiser is a multiple of 11^T). All +- are one Monte-Carlo standard error.\n")
    P("Predicted limits g_* (per entry): sign 1/6 = 0.166667; 1x2 0.125; 1x3 0.1; 2x3 0.094674; 2x2 0.113774; 3x3 %.5f.\n" % G33)
    P("## Fig. 1 (`fig_lattice.pdf`) and `data/table_exact.json`\n")
    P("Exact computations (`exp_exact_lattice.py`, data `data/exact_lattice.jsonl`): exact binomial law; only approximation = truncation of the support. "
      "1xN: box with each one-dimensional tail < 1e-13 (N=2) / 1e-11 (N=3). 2x2 and 2x3: all lattice points with weight above a threshold; "
      "dropped probability mass (computed, stored per point) <= %.1e. Largest number of lattice points summed per field evaluation: %.2e (2x3, b=102).\n"
      % (max(r["dropped"] for g in ("2x2", "2x3") for r in ex[g]), max(r["npts"] for r in ex["2x3"])))
    P("Largest b per residue class:\n")
    P("| geometry | b mod 3 | b | b G_b | g_* | sqrt(b) (bG_b - g_*) | b (bG_b - g_*) |"); P("|---|---|---|---|---|---|---|")
    for g in ("sign", "1x2", "1x3", "2x3", "2x2", "3x3"):
        for c in range(3):
            r = T[g]["b=%d mod 3" % c]
            if g == "3x3":
                P("| 3x3 (MC) | %d | %d | %.5f +- %.5f | %.5f | %+.3f +- %.3f | %+.1f +- %.1f |" % (c, r["b"], r["bG"], r["se"], r["gstar"], r["sqrtb_err"], r["sqrtb_err_se"], r["b_err"], r["b_err_se"]))
            else:
                P("| %s | %d | %d | %+.6f | %.6f | %+.4f | %+.3f |" % (g, c, r["b"], r["bG"], r["gstar"], r["sqrtb_err"], r["b_err"]))
    P("")
    P("- (a) sign: b G_b = 0, +1, -1 for b = 0, 1, 2 mod 3 for EVERY b in 1..60 and for b = 300..302, 3000..3002 (exact medians, unique). Never approaches 1/6.")
    r12 = {c: [r for r in ex["1x2"] if r["b"] % 3 == c] for c in range(3)}
    P("- (b) 1x2: sqrt(b)(bG_b - g_*) at the largest b: " + ", ".join("%+.4f (b=%d)" % (r12[c][-1]["sqrtb_err"], r12[c][-1]["b"]) for c in range(3))
      + "; predicted lattice coefficients -1.0119, +0.3326, -0.1775 (b = 0, 1, 2 mod 3). Class 0 is locked at exactly 0 for b <= %d (computed multiples of 3) and released from b = %d."
      % (max(r["b"] for r in r12[0] if r["locked"]), min(r["b"] for r in r12[0] if not r["locked"])))
    for g, bmin in (("1x2", 60), ("1x3", 30), ("2x3", 30)):
        P("- (f) %s: log-log slope of |bG_b - g_*| (least squares, b >= %d): " % (g, bmin) + ", ".join("class %d: %.3f (%d pts)" % ((c,) + slope(ex[g], c, bmin)) for c in range(3)))
    for g in ("1x3", "2x3"):
        P("- %s: b(bG_b - g_*) at the largest b: " % g + ", ".join("%+.3f (b=%d)" % (T[g]["b=%d mod 3" % c]["b_err"], T[g]["b=%d mod 3" % c]["b"]) for c in range(3)))
    r23 = [r for r in ex["2x3"] if r["b"] % 3 == 0]
    P("- (c) 2x3: class 0 locked at exactly 0 for b = %s; b=12: %.6f, b=18: %.6f." % (", ".join(str(r["b"]) for r in r23 if r["locked"]), *[r["m"] for r in r23 if r["b"] in (12, 18)]))
    r22 = [r for r in ex["2x2"] if r["b"] % 3 == 0]
    P("- (d) 2x2: class 0 locked at exactly 0 for every computed multiple of 3 up to b = %d (all of 3..66, then %s); then "
      % (max(r["b"] for r in r22 if r["locked"]), ", ".join(str(r["b"]) for r in r22 if r["locked"] and r["b"] > 66))
      + ", ".join("%.6f at b=%d" % (r["m"], r["b"]) for r in r22 if not r["locked"]) + " (g_* = 0.113774).")
    d22 = {r["b"]: r["m"] for r in ex["2x2"]}
    P("- (d) 2x2, classes 1 and 2: b G_b sits on rational plateaux (kinks of the objective): 1/7 = 0.142857 for all b = 1 mod 3 in [%d, %d]; 1/11 = 0.090909 for all b = 2 mod 3 in [%d, %d]; "
      % (min(b for b, m in d22.items() if b % 3 == 1 and abs(m - 1 / 7) < 1e-6), max(b for b, m in d22.items() if b % 3 == 1 and abs(m - 1 / 7) < 1e-6 and b <= 66),
         min(b for b, m in d22.items() if b % 3 == 2 and abs(m - 1 / 11) < 1e-6), max(b for b, m in d22.items() if b % 3 == 2 and abs(m - 1 / 11) < 1e-6 and b <= 66))
      + "later values: " + ", ".join("%.6f (b=%d)" % (d22[b], b) for b in (514, 515, 769, 770, 1027, 1028, 1537, 1538) if b in d22) + ".")
    P("\n3x3, Monte Carlo (`exp_mc_3x3.py`): minimiser of the sample mean of ||m 11^T + S_b||_* with a mean-zero linear control variate, common random numbers; "
      "value = mean over independent batches (5e5 samples each) of the batch minimisers, error bar = sd of batch minimisers / sqrt(#batches).\n")
    P("| b | b mod 3 | b G_b | ratio to g_* | samples |"); P("|---|---|---|---|---|")
    for r in mc["results"]:
        P("| %d | %d | %.5f +- %.5f | %.3f +- %.3f | %.1e |" % (r["b"], r["mod3"], r["m"], r["se"], r["m"] / G33, r["se"] / G33, r["nsamples"]))
    P("\n## Fig. 2 (`fig_wor.pdf`), data `data/wor.json`\n")
    P("Coefficient c_b = b <G_b, g_*>/|g_*|^2; prediction (n-2b)/(n-2); with replacement 1. %d independent batches per point (lattice population with n = 10368: 64 batches for b <= 7775, 33 for b = 7776, 16 beyond - the run was interrupted; see the column) (A: %.0e samples each, zero of the 6-dim mean field by common random numbers; "
      "B: %.0e samples each, scalar minimiser with control variate). Population A: g_* recomputed twice by Monte Carlo, relative difference %.4f (a common scale error of that size affects all c_b).\n"
      % (w["NB"], w["samples_per_batch"]["A"], w["samples_per_batch"]["B"],
         np.linalg.norm(np.array(w["gstar_A"]) - np.array(w["gstar_A_repeat"])) / np.linalg.norm(w["gstar_A"])))
    for name in sorted(w["populations"], key=lambda s: (len(s), s)):
        pop = w["populations"][name]; rows = pop["rows"]
        P("Population %s (n = %d%s):\n" % (name, pop["n"], ", generic 2x3 (c05)" if name == "A" else ", lattice 2x2"))
        P("| b | b/n | b mod 3 | batches | measured c_b | predicted | (meas - pred)/se |"); P("|---|---|---|---|---|---|---|")
        for q in rows:
            P("| %d | %.3f | %d | %d | %+.4f +- %.4f | %+.4f | %s |" % (q["b"], q["frac"], q["mod3"], q.get("nbatch", 16), q["coef"], q["se"], q["pred"], ("%+.1f" % ((q["coef"] - q["pred"]) / q["se"])) if q["se"] > 1e-6 else "-"))
        for cls in ([None] if name == "A" else [0, 1, 2]):
            rr = [q for q in rows if cls is None or q["mod3"] == cls]
            d = np.array([q["coef"] - q["pred"] for q in rr])
            P("\n%s: rms(measured - predicted) = %.4f, max |.| = %.4f, mean = %+.4f; sign of c_b equals sign of (n - 2b) at %d of %d points."
              % ("all b" if cls is None else "class b = %d mod 3" % cls, np.sqrt((d ** 2).mean()), np.abs(d).max(), d.mean(),
                 sum(1 for q in rr if q["coef"] * q["pred"] > 0), len(rr)))
        P("")
    P("## Fig. 3 (`fig_momentum_discrete.pdf`), data `data/fig_momentum_discrete.json`\n")
    r0 = mom["rows"][0]
    P("Iteration M_t = beta M_{t-1} + (1-beta)(grad F(W_t) + xi_t), W_{t+1} = W_t - eta_t o(M_t), eta_t = (t+1)^(-3/4), T = %d steps, 3x3, F diagonal quadratic with curvatures in [0.5, 2], W_0 = 0, M_0 = 0. "
      "Quantity: b_eff x (time average over the last half of the run of the entrywise mean of grad F(W_t)), b_eff = b(1+beta+beta^2)/(1-beta^2); mean and standard error over independent runs. "
      "`raw` = plain average; `cv` (PLOTTED) = same with the time-averaged noise (exact mean zero) as a linear control variate, coefficient fitted across runs (fitted values are close to -1). "
      "`Q3`, `Q4` = raw value over the third / fourth quarter of the run (drift diagnostic). Variance-rule alternative: b_eff g_*/b_var = g_* (1+beta+beta^2)/(1+beta)^2 = %.5f (beta=0.5), %.5f (beta=0.9) for 3x3.\n"
      % (r0["T"], G33 * 1.75 / 2.25, G33 * 2.71 / 3.61))
    for kind, gs in (("nuc", G33), ("sign", 1 / 6)):
        P("%s (g_* = %.5f):\n" % ("Nuclear 3x3" if kind == "nuc" else "Entrywise sign", gs))
        P("| b | beta | b_eff | runs | raw | cv (plotted) | cv / g_* | Q3 | Q4 |"); P("|---|---|---|---|---|---|---|---|---|")
        for r in sorted([r for r in mom["rows"] if r["kind"] == kind], key=lambda r: (r["beta"], r["b"])):
            P("| %d | %g | %.2f | %d | %+.4f +- %.4f | %+.4f +- %.4f | %.3f +- %.3f | %+.4f | %+.4f |" % (r["b"], r["beta"], r["b_eff"], r["runs"], r["val"], r["se"], r["val_cv"], r["se_cv"],
              r["val_cv"] / gs, r["se_cv"] / gs, r["val_third_quarter"], 2 * r["val"] - r["val_third_quarter"]))
        P("")
    P("Direct pure-batch 3x3 minimisers b G_b (crosses in panel a): " + ", ".join("b=%s: %.4f +- %.4f" % (b, v["m"], v["se"]) for b, v in mom["direct_pure_batch_3x3"].items()) + ".\n")
    P(NOTES)
    open(os.path.join(RES, "NUMBERS_discrete.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["lattice", "wor", "momentum"]
    if "lattice" in which: fig_lattice()
    if "wor" in which: fig_wor()
    if "momentum" in which: fig_momentum()
    if "numbers" in which or len(sys.argv) == 1: numbers()
