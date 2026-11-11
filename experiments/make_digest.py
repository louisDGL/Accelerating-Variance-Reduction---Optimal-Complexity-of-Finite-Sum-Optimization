"""Build results/digest.md: a compact, human-readable digest of every *_summary.json (numbers used in the report).
Usage: python experiments/make_digest.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT = os.path.join(common.RESULTS, "digest.md")


def fmt(x):
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-3:
            return f"{x:.3g}"
        return f"{x:.4g}"
    return str(x)


def table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(fmt(v) for v in r) + " |\n"
    return s


def main():
    parts = ["# Digest of numerical results\n", "Generated from results/*_summary.json by experiments/make_digest.py.\n"]
    files = {os.path.basename(p)[:-13]: p for p in glob.glob(os.path.join(common.RESULTS, "*_summary.json"))}
    files = {k: v for k, v in files.items() if not k.endswith("_quick") and k != "GD_quick"}

    if "validation_book" in files:
        d = json.load(open(files["validation_book"], encoding="utf-8"))
        parts.append("\n## Validation (Figure 5.4 setting of the book)\n")
        rows = [(n, v["problem"]["R2"], v["problem"]["mu"], v["problem"]["kappa"]) + tuple(v["final_Fgap"][k] for k in ["SGD", "SAGA", "GD"]) + tuple(v["passes_to_1e-8"][k] for k in ["SGD", "SAGA", "GD"]) for n, v in d.items()]
        parts.append(table(rows, ["n", "R2 (per-sample L_max)", "mu", "kappa", "F-F* SGD @50", "SAGA @50", "GD @50", "passes to 1e-8 SGD", "SAGA", "GD"]))

    if "B3" in files:
        d = json.load(open(files["B3"], encoding="utf-8"))
        parts.append("\n## B3: tightness of Proposition 5.9 (n = 1000, d = 50, 5 seeds, textbook step)\n")
        parts.append("### Problems\n")
        rows = [(k, v["lam"], v["R2"], v["kappa"], v["L_F"], v["kappa_F"], v["cov_cond"], v["kappa_local"], v["mu_local_over_mu"], v["sigma_prime_min"], v["sigma_prime_mean"], v["d_eff"]) for k, v in d["problems"].items()]
        parts.append(table(rows, ["config", "lam", "R2", "kappa", "L_F", "kappa_F=L_F/mu", "cov cond", "local kappa(H*)", "lmin(H*)/mu", "min sigma'", "mean sigma'", "d_eff"]))
        parts.append("### Attribution of the log-gap (passes 2-20) and window-averaged ratios\n")
        rows = []
        for k, v in d["attribution"].items():
            sh = v["shares"]
            rows.append((k, v["speedup"], sh["I1"], sh["I3"], sh["I2"], sh["I4"], sh["I5"], v.get("r_sq_geomean"), v.get("r_coco_geomean"), v.get("r_var_geomean"), v.get("r_sc_geomean"), v.get("r_min_geomean"), v.get("r_init"), v.get("final_D")))
        parts.append(table(rows, ["config", "speed-up exact/bound", "share I1", "I3", "I2", "I4", "I5", "r_sq", "r_coco=L_eff/R2", "r_var", "r_sc=mu/mu_eff", "r_min", "r_init", "final E D_t"]))
        parts.append("### Attribution in order A of the B3 note (undo I4, I2, I3, I1, then the min I5; passes 2-20)\n")
        rows = []
        for k, v in d["attribution"].items():
            sa = v.get("shares_orderA", {})
            rows.append((k, sa.get("I4"), sa.get("I2"), sa.get("I3"), sa.get("I1"), sa.get("I5")))
        parts.append(table(rows, ["config", "share I4", "I2", "I3", "I1", "I5 (min, undone last)"]))
        parts.append("### Late window (passes 20-100) and leave-one-out exponents (per iteration, passes 2-20)\n")
        rows = []
        for k, v in d["attribution"].items():
            sl = v.get("shares_late_20_100", {})
            loo = v.get("loo_exponents_per_iter", {})
            rows.append((k, v.get("speedup_late_20_100"), sl.get("I1"), sl.get("I3"), sl.get("I2"), sl.get("I4"), sl.get("I5"), v.get("c_exact_per_iter"), v.get("c_bound_per_iter"), loo.get("I1"), loo.get("I3"), loo.get("I2"), loo.get("I4"), loo.get("I5")))
        parts.append(table(rows, ["config", "speed-up late", "late I1", "I3", "I2", "I4", "I5", "c_exact", "c_bound", "loo I1", "loo I3", "loo I2", "loo I4", "loo I5"]))

    if "B2" in files:
        d = json.load(open(files["B2"], encoding="utf-8"))
        parts.append("\n## B2: effective curvature along trajectories (n = 1000, d = 50)\n")
        rows = [(k, v["lmin_H_over_mu"], v["lmax_H_over_mu"], v["saga_mu_eff_over_mu_start"], v["saga_mu_eff_over_mu_geomean"], v["saga_mu_eff_over_mu_end"], v["gd_mu_eff_over_mu_end"], v["saga_L_eff_over_R2_geomean"], v["gd_L_eff_over_R2_end"], v["L_eff_limit_min_over_R2"], v["L_eff_limit_max_over_R2"], v["saga_energy_low1_end"], v["gd_energy_low1_end"]) for k, v in d.items()]
        parts.append(table(rows, ["config", "lmin(H*)/mu", "lmax(H*)/mu", "SAGA mu_eff/mu start", "geomean", "end", "GD mu_eff/mu end", "SAGA L_eff/R2 geomean", "GD L_eff/R2 end", "L_eff limit min/R2", "max/R2", "SAGA energy on slowest eigvec (end)", "GD (end)"]))

    if "B1" in files:
        d = json.load(open(files["B1"], encoding="utf-8"))
        parts.append("\n## B1: preconditioned SAGA (n = 1000, d = 50, kappa = 1e3, 5 seeds)\n")
        rows = [(k, v["R2_P"], v["mu_P"], v["mu_P_local"], v["kappa_P"], v["kappa_P_local"], v["max_lev"], v["mean_lev"], v["rate_emp_D_P"] * 1000, v["rate_emp_Fgap"] * 1000, v["rate_bound_global"] * 1000, v["rate_bound_local"] * 1000, v.get("r_coco_geomean"), v.get("r_sc_geomean"), v.get("r_sq_geomean"), v.get("r_var_geomean"), v.get("r_min_geomean")) for k, v in d["preconditioners"].items()]
        parts.append(table(rows, ["alpha|P", "R2_P", "mu_P global", "mu_P local", "kappa_P", "kappa_P local (F'' vs P)", "max lev", "mean lev", "emp rate/pass (D_P)", "emp rate/pass (Fgap)", "bound rate/pass (global mu_P)", "bound (local mu_P)", "r_coco", "r_sc", "r_sq", "r_var", "r_min"]))

    if "GammaDelta" in files:
        d = json.load(open(files["GammaDelta"], encoding="utf-8"))
        parts.append("\n## (gamma, Delta) inside the proof (n = 1000, kappa = 1e3, alpha = 1)\n")
        o = d["optimal_within_proof"]
        parts.append(f"Optimal within the proof: c = {fmt(o['c'])}, gamma R2 = {fmt(o['gamma'] * d['problem']['R2'])}, certified rate/iter = {fmt(o['rate'])} vs textbook {fmt(o['rate_textbook'])} (gain x{fmt(o['gain_over_textbook'])}).\n\n")
        rows = [(g, d["empirical_rates_per_pass"][g], d["certified_rates_per_pass"][g]) for g in d["empirical_rates_per_pass"]]
        parts.append(table(rows, ["gamma R2", "empirical exponent/pass", "certified exponent/pass (Delta optimised)"]))

    if "A3" in files:
        d = json.load(open(files["A3"], encoding="utf-8"))
        parts.append("\n## A3: L-Katyusha Lyapunov instrumentation (3 seeds)\n")
        rows = [(k, v["theta1"], v["eta"], v["p"], v["rho_lemma"], v["rho_exact_geomean"], v["speedup_exact_over_lemma"], v["r_var_geomean"], v.get("r_var_max"), v["max_violation_rho_exact_minus_rho_lemma"], v["final_rel_D"]) for k, v in d.items()]
        parts.append(table(rows, ["problem", "theta1", "eta", "p", "rho (Lemma 4.5)", "rho exact (geomean)", "speed-up exact/lemma", "r_var geomean", "r_var max", "max(rho_exact - rho_lemma)", "final D/D0"]))
        parts.append("### A3: additional ratios of the A3 note (centering, Bregman co-coercivity, cancellation shares)\n")
        rows = [(k, v.get("r_cen_geomean"), v.get("r_coco_lk_geomean"), v.get("c_share_median"), v.get("c_share_frac_negative"), v.get("c_share_frac_above_1"), v.get("q_cancel_median"), v.get("q_cancel_frac_ge_1")) for k, v in d.items()]
        parts.append(table(rows, ["problem", "r_cen geomean", "r_coco (Bregman) geomean", "c(t) median", "frac c<0", "frac c>1", "q(t) median", "frac q>=1"]))

    if "A4b" in files:
        d = json.load(open(files["A4b"], encoding="utf-8"))
        parts.append("\n## A4b: shared-regularizer (A1-note) and aligned instances (n = 16, eps = 1e-8, 2 seeds)\n")
        rows = [(k, v["slope_x_ge_10n"], *[fmt(x) for x in v["grads"]]) for k, v in d.items() if k != "constants"]
        parts.append(table(rows, ["instance|x-axis|method", "slope (x >= 10n)"] + [f"grads @x {i}" for i in range(max(len(v["grads"]) for k, v in d.items() if k != "constants"))]))
        rows = [(k, v["kappa_eff"], v["kappa_max_window"], v["m"], v["kappa_F"]) for k, v in d["constants"].items()]
        parts.append(table(rows, ["instance|kappa", "kappa_eff = R2/lmin(F'')", "kappa_max(m, eps)", "m", "kappa_F = L_F/mu"]))

    if "A2" in files:
        d = json.load(open(files["A2"], encoding="utf-8"))
        parts.append("\n## A2: naive acceleration (one seed)\n")
        for label, v in d.items():
            parts.append(f"### {label}\n")
            rows = [(k, r["final_rel_D"], r["grads"], r["stopped"], r.get("grid_best")) for k, r in v["runs"].items()]
            parts.append(table(rows, ["method", "final D/D0", "grads used", "stopped", "grid best"]))
            rows = [(k, g["grads"], g["stopped"], g["final_rel_D"]) for k, g in v["grid"].items()]
            parts.append(table(rows, ["naive (beta, gamma R2)", "grads to 1e-6", "stopped", "final D/D0"]))

    if "A4" in files:
        d = json.load(open(files["A4"], encoding="utf-8"))
        parts.append("\n## A4: gradient complexity vs kappa (n = 16, m = 256, eps = 1e-8, 3 seeds)\n")
        rows = [(k, v["slope_kappa_ge_10n"], *[fmt(x) for x in v["grads"]]) for k, v in d.items()]
        parts.append(table(rows, ["instance|method", "slope (kappa >= 10n)"] + [f"grads @kappa {i}" for i in range(len(next(iter(d.values()))["grads"]))]))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
