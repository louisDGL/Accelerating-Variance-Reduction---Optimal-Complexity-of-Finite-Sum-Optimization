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


def na(x, none="n/a"):
    """Placeholder for a missing (None) summary value."""
    return none if x is None else x


def ci(v, key):
    lo, hi = v.get(key, [None, None])
    return f"[{lo:.3f}, {hi:.3f}]" if isinstance(lo, float) and isinstance(hi, float) and lo == lo and hi == hi else "n/a"


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
        parts.append("\nAll attribution windows and window-averaged ratios use the instrumentation records with V_t >= 1e-18 "
                     "(floor set by the precision of eta*).\n\n")
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
        parts.append("\nloo columns: -log of the single-substitution rates of vrlab.analysis.leave_one_out (one inequality "
                     "kept measured, the others applied, no max), averaged over the window. They are not nested in the proof "
                     "chain: 'loo I1' and 'loo I2' are hybrids whose rate can exceed 1, hence the negative values; they are "
                     "diagnostics, not contraction exponents.\n\n")
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
        first = next(iter(d.values()))
        if "lambda2_over_lmin" in first:            # keys of remark_stats() in exp_B2_mu_eff.py (Rem. B2-P2-outcome, report Sec. exp-B2)
            p = first["protocol"]
            parts.append(f"\nProtocol (read from results/B2.pkl): theta_0 = 0; SAGA {p['saga_passes']} passes, a record every {p['saga_record_every_iters']} "
                         f"iterations; GD at most {p['gd_max_iters']} iterations, a record every {p['gd_record_every_iters']}. Records with "
                         f"D_t = ||theta_t - eta*||^2 < {fmt(p['floor_D'])} are discarded and 'end' is the last record above this floor. pi_min: share of "
                         "D_t on the eigenvector q_min of H* = F''(eta*) with the smallest eigenvalue lmin(H*); d_eff = 1/sum_k pi_k^2.\n\n")
            parts.append("### B2: bottom of the spectrum of H*, criterion of Rem. B2-P2-outcome, initial state theta_0 = 0 (SAGA and GD)\n")
            rows = [(k, f"{v['lambda2_over_lmin']:.6g}", v["n_eigs_within_10pct_of_lmin"], v["crit_9n_lmin_over_16R2"], v["crit_9n_mu_over_16R2"],
                     v["energy_low1_start"], na(v["d_eff_dirs_start"]), na(v["rayleigh_H_start_over_mu"]), v["Rx_dist_start"]) for k, v in d.items()]
            parts.append(table(rows, ["config", "lambda'/lmin(H*)", "# eigs of H* <= 1.1 lmin(H*)", "9n lmin(H*)/(16R2)", "9n mu/(16R2) = 9n/(16 kappa)",
                                      "pi_min(theta_0)", "d_eff(theta_0)", "Rayleigh quotient of H* at theta_0 - eta*, /mu", "R_x sqrt(D_0)"]))
            parts.append("### B2: SAGA, end state and alignment with q_min (times in passes)\n")
            rows = [(k, v["saga_end_pass"], na(v["saga_first_pass_below_floor"], "never"), v["saga_D_end"], v["saga_Rx_dist_end"],
                     v["saga_energy_low1_min"], v["saga_energy_low1_argmin_pass"], na(v["saga_first_pass_energy_low1_ge_0.99"], "never"),
                     na(v["saga_energy_low1_pass10"]), na(v["saga_d_eff_dirs_pass10"]), na(v.get("saga_d_eff_dirs_end")),
                     na(v.get("saga_d_eff_dirs_geomean")), v["saga_energy_low5_end"]) for k, v in d.items()]
            parts.append(table(rows, ["config", "end (passes)", "first record below the floor", "D_t (end)", "R_x sqrt(D_t) (end)",
                                      "min pi_min", "at pass", "first pass with pi_min >= 0.99", "pi_min at pass 10", "d_eff at pass 10",
                                      "d_eff (end)", "d_eff geomean", "share of the 5 lowest eigvecs (end)"]))
            parts.append("### B2: SAGA, effective curvature relative to mu and to lmin(H*)\n")
            rows = [(k, v["saga_mu_eff_end_over_lmin"], na(v.get("saga_mu_eff_quadratic_proxy_end_over_mu")), na(v["saga_mu_eff_quadratic_proxy_end_rel_diff"]),
                     v["saga_mu_eff_over_mu_min"], v["saga_mu_eff_over_mu_min_all_states"], v["saga_mu_eff_monotone_decreasing_fraction"],
                     na(v["saga_mu_eff_over_mu_10_passes_before_end"]), v["saga_mu_eff_last10_passes_nonincreasing_fraction"],
                     v["R2_over_saga_L_eff_geomean"]) for k, v in d.items()]
            parts.append(table(rows, ["config", "mu_eff/lmin(H*) (end)", "quadratic proxy/mu (end)", "(proxy - mu_eff)/mu_eff (end)",
                                      "min mu_eff/mu (kept records)", "min mu_eff/mu (all records)", "frac. of non-increasing steps",
                                      "mu_eff/mu 10 passes before the end", "frac. non-increasing, last 10 passes", "R2 / geomean L_eff"]))
            if "saga_last5_n_states" in first:          # window statistics and decay rates (Rem. B2-P2-outcome, item (a))
                parts.append("### B2: SAGA over the last five recorded passes before the end, and decay of D_t over the second half\n")
                parts.append("\nStates above the floor with t >= t_end - 5n. Decay: minus the least-squares slope of log D_t against "
                             "passes over the kept states with t >= t_end/2; slowest mode: 2 n gamma lmin(H*) = n lmin(H*)/(2 R2) per "
                             "pass (textbook step).\n\n")
                rows = [(k, v["saga_last5_n_states"], v["saga_last5_mu_eff_over_mu_geomean"], v["saga_last5_mu_eff_over_mu_min"],
                         v["saga_last5_mu_eff_over_mu_max"], v["saga_last5_mu_eff_over_lmin_geomean"], v["saga_last5_mu_eff_over_lmin_min"],
                         v["saga_last5_mu_eff_over_lmin_max"], na(v["saga_last5_d_eff_dirs_mean"]), na(v["saga_last5_d_eff_dirs_min"]),
                         na(v["saga_last5_d_eff_dirs_max"]), v["saga_last5_energy_low1_max"], v["saga_last5_energy_low5_mean"],
                         v["saga_decay_rate_second_half_per_pass"], v["saga_slowest_mode_rate_per_pass"], v["saga_decay_rate_over_slowest_mode"])
                        for k, v in d.items()]
                parts.append(table(rows, ["config", "# states", "mu_eff/mu geomean", "min", "max", "mu_eff/lmin(H*) geomean", "min", "max",
                                          "d_eff mean", "min", "max", "max pi_min", "mean share of the 5 lowest eigvecs",
                                          "decay of D_t per pass (2nd half)", "slowest mode per pass", "decay / slowest mode"]))
            parts.append("### B2: gradient descent (step 1/L_F; times in iterations)\n")
            rows = [(k, v["gd_end_iter"], v["gd_D_end"], v["gd_energy_low5_end"], na(v["gd_first_iter_energy_low1_ge_0.99"], "never"),
                     na(v.get("gd_d_eff_dirs_end")), v["gd_mu_eff_end_over_lmin"], v["gd_mu_eff_over_mu_min"], v["gd_mu_eff_over_mu_min_all_states"],
                     v["gd_mu_eff_monotone_decreasing_fraction"]) for k, v in d.items()]
            parts.append(table(rows, ["config", "end (iteration)", "D_t (end)", "share of the 5 lowest eigvecs (end)",
                                      "first iteration with pi_min >= 0.99", "d_eff (end)", "mu_eff/lmin(H*) (end)", "min mu_eff/mu (kept records)",
                                      "min mu_eff/mu (all records)", "frac. of non-increasing steps"]))

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
        parts.append("\nrho_lemma = max{c_Z, c_Y, c_W}: one-step factor certified by Lemma 10 of Kovalev, Horvath and Richtarik "
                     "(ALT 2020; Lemma 4.5 and Theorem 4.6 in the arXiv version 1901.08689 are its Lemma 10 and Theorem 11); the "
                     "closed-form rate rho of Theorem 11 satisfies rho <= 1 - rho_lemma. Speed-up: mean of log rho_exact over the "
                     "records with Psi_t >= 1e-10 (all seeds), divided by log rho_lemma.\n\n")
        rows = [(k, v["theta1"], v["eta"], v["p"], v["rho_lemma"], v["rho_exact_geomean"], v["speedup_exact_over_lemma"], v["r_var_geomean"], v.get("r_var_max"), v["max_violation_rho_exact_minus_rho_lemma"], v["final_rel_D"]) for k, v in d.items()]
        parts.append(table(rows, ["problem", "theta1", "eta", "p", "rho_lemma (Lemma 10)", "rho exact (geomean)", "speed-up exact/lemma", "r_var geomean", "r_var max", "max(rho_exact - rho_lemma)", "final D/D0"]))
        parts.append("### A3: additional ratios of the A3 note (centering, Bregman co-coercivity, cancellation shares)\n")
        rows = [(k, v.get("r_cen_geomean"), v.get("r_coco_lk_geomean"), v.get("c_share_median"), v.get("c_share_frac_negative"), v.get("c_share_frac_above_1"), v.get("q_cancel_median"), v.get("q_cancel_frac_ge_1")) for k, v in d.items()]
        parts.append(table(rows, ["problem", "r_cen geomean", "r_coco (Bregman) geomean", "c(t) median", "frac c<0", "frac c>1", "q(t) median", "frac q>=1"]))
        parts.append("\nSign-aware reading of q(t) (A3 note, Table of the cancellation row): q_+(t) = q(t) when <F'(x_t), x_t - w_t> > 0 "
                     "(c(t) > 0) and 0 otherwise; only then does the cancelled term have the sign a bound without the anchor would have "
                     "to absorb. Componentwise Lemma 10: max over the records of (E Psi_{t+1} - c_Z Z_t - c_Y Y_t - c_W W_t)/Psi_t "
                     "(negative = holds). Speed-up vs rho: mean log rho_exact / log(1 - rho), rho = min{sigma/(6 theta1), theta1/(2n)}.\n\n")
        rows = [(k, v.get("inner_xw_frac_positive"), v.get("q_plus_frac_ge_1"), v.get("q_cancel_frac_ge_1_given_inner_positive"), v.get("q_cancel_median_given_inner_positive"), v.get("c_share_frac_negative_given_q_ge_1"), v.get("contraction_componentwise_max"), v.get("speedup_exact_over_rho")) for k, v in d.items()]
        parts.append(table(rows, ["problem", "frac c>0", "frac q_+>=1", "frac q>=1 given c>0", "q median given c>0", "frac c<0 given q>=1", "max componentwise Lemma 10 residual/Psi", "speed-up exact/rho"]))

    if "A4b" in files:
        d = json.load(open(files["A4b"], encoding="utf-8"))
        parts.append("\n## A4b: shared-regularizer (A1-note) and aligned instances (n = 16, eps = 1e-8, 2 seeds)\n")
        parts.append("\nSlopes: least-squares fits of log(gradients) against log(x) for the per-x geometric means over the seeds "
                     "(K values of x in the fitting range), with 95% t-intervals with K - 2 degrees of freedom. Gradient counts as "
                     "in vrlab.algorithms (L-Katyusha: 3 per iteration in expectation, about 3/2 of the stored-reference "
                     "convention of the A1 note's bounds; slopes are unaffected).\n\n")
        rows = [(k, v["slope_x_ge_10n"], ci(v, "slope_ci95"), v.get("slope_window_10n_kmax"), ci(v, "slope_window_ci95"), *[fmt(x) for x in v["grads"]]) for k, v in d.items() if k != "constants"]
        parts.append(table(rows, ["instance|x-axis|method", "slope (x >= 10n)", "95% t-interval", "slope in window 10n <= kappa <= kappa_max", "95% t-interval"] + [f"grads @x {i}" for i in range(max(len(v["grads"]) for k, v in d.items() if k != "constants"))]))
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
        parts.append("\nSlopes: least-squares fits of log(gradients) against log(kappa) for the per-kappa geometric means over the "
                     "seeds, over kappa >= 10n (K values of kappa), with 95% t-intervals with K - 2 degrees of freedom. Gradient "
                     "counts as in vrlab.algorithms: L-SVRG, L-Katyusha and Nesterov+SVRG are charged 3 per iteration in "
                     "expectation, about 3/2 of the stored-reference convention of the A1 note's bounds (slopes are "
                     "unaffected). Nesterov+SVRG: best of 6 (beta, gamma) settings, selected on seed 0 on (H) and re-run and "
                     "re-selected on every seed on (R), whose instance depends on the seed.\n\n")
        rows = [(k, v["slope_kappa_ge_10n"], ci(v, "slope_ci95"), *[fmt(x) for x in v["grads"]]) for k, v in d.items()]
        parts.append(table(rows, ["instance|method", "slope (kappa >= 10n)", "95% t-interval (K - 2 d.o.f.)"] + [f"grads @kappa {i}" for i in range(len(next(iter(d.values()))["grads"]))]))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
