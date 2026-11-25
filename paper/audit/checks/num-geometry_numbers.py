"""Audit num-geometry, check 1: every number of (M4), (M3), Figure 2 and the 'Experiments' paragraph of
paper/paper_saga_to_muon.tex, recomputed from the raw summaries muon/results/M7, M1, M8, M5b (no prose, no digest).

Read-only. Run from the project root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-geometry_numbers.py
"""
import json
import math
import random
import statistics as st

R = "muon/results/"
m7 = json.load(open(R + "M7_summary.json"))
m1 = json.load(open(R + "M1_summary.json"))
m8 = json.load(open(R + "M8_summary.json"))
m5b = json.load(open(R + "M5b_summary.json"))

COV = "feature covariance, (c_l Sigma_hat + lam I)^-1"
POL = "grad F(W_0)"


def med(x):
    return st.median(x)


print("=" * 100)
print("(M4)  M7_summary.json: per-sample condition numbers R_P^2/mu_P")
print("=" * 100)
for key, inst in m7["metric"].items():
    kg, kl, n = inst["kappa_euclid_global"], inst["kappa_euclid"], inst["n"]
    print(f"{key:9s} loss={inst['loss']:7s} n={n:4d}  Euclid: global {kg:9.2f} = {kg / n:5.2f} n ; local {kl:9.2f} = "
          f"{kl / n:5.2f} n")
    for a in (POL, COV):
        rows = [r for r in inst["rows"] if r["anchor"] == a]
        g = min(rows, key=lambda r: r["kappa_P_global"])
        l = min(rows, key=lambda r: r["kappa_P"])
        nontriv = [r for r in rows if r["floor"] < 1.0]
        gn = min(nontriv, key=lambda r: r["kappa_P_global"])
        print(f"    {a[:22]:22s} best global {g['kappa_P_global']:9.2f} (f={g['floor']:g}) ratio to Euclid global "
              f"{g['kappa_P_global'] / kg:.6f} [gain {kg / g['kappa_P_global']:7.2f}] ; best global with f<1: "
              f"ratio {gn['kappa_P_global'] / kg:.4f} (f={gn['floor']:g}) ; best local {l['kappa_P']:8.2f} "
              f"(f={l['floor']:g}) Euclid local/that = {kl / l['kappa_P']:.3f}, Euclid global/that = "
              f"{kg / l['kappa_P']:.3f}")
soft = [k for k, v in m7["metric"].items() if v["loss"] == "softmax"]
allr = [(r["kappa_P_global"] / m7["metric"][k]["kappa_euclid_global"], k, r["anchor"], r["floor"])
        for k in soft for r in m7["metric"][k]["rows"]]
print("softmax, all anchors x floors: smallest ratio (global metric constant / Euclidean global) =", min(allr))
strict = [x for x in allr if not (x[2] == COV and x[3] == 1.0)]
print("softmax, same without the covariance metric at f = 1 (= a multiple of the identity):", min(strict))
print("number of softmax (anchor, floor) cells:", len(allr), "; cells with ratio < 1 - 1e-9:",
      sum(1 for x in allr if x[0] < 1 - 1e-9), "; cells with ratio == 1 (1e-9):",
      [(x[1], x[2][:12], x[3]) for x in allr if abs(x[0] - 1) < 1e-9])

print()
print("=" * 100)
print("(M3), Figure 2(b), M4 passes:  M1_summary.json (confirmation seeds)")
print("=" * 100)


def row(inst, fam, src=m1):
    for r in src[inst]["rows"]:
        if r["family"] == fam:
            return r
    raise KeyError(fam)


def show(inst, fam, src=m1, tol=None):
    r = row(inst, fam, src)
    tol = tol or f"{src[inst]['tol']:g}"
    s = r["summary"][tol]
    cp = r.get("confirm_passes")
    return (f"{fam:30s} median {s['median']:7.2f} [{s['min']:.1f}, {s['max']:.1f}] reached {s['reached']}/{s['runs']}"
            f"  best={json.dumps(r['best'])}")


FIG = ["SAGA", "B-SAGA", "IS-B-SAGA", "cov-metric B-SAGA", "cov-metric SAGA", "SAGA-Muon (joint)", "soft SAGA-Muon",
       "IS soft SAGA-Muon", "one-shot polar-metric B-SAGA", "one-shot polar-metric SAGA"]
for inst in ("soft-cond", "ls-cond", "ls-orth", "soft-sq"):
    print(f"--- {inst}: tol {m1[inst]['tol']:g}, budget {m1[inst]['passes']}, n={m1[inst]['spec']['n']} "
          f"K={m1[inst]['spec']['K']} d={m1[inst]['spec']['d']}  R2/mu_true={m1[inst]['info']['R2_over_mu_true']:.1f}"
          f" R2/lam={m1[inst]['info']['kappa']:.1f}")
    for fam in FIG:
        print("   ", show(inst, fam))
print("protocol:", m1["soft-cond"]["protocol"])

print()
print("memory-limited instances: best family of every class (all confirmation seeds reach 1e-9), batch one and b=r")
GROUPS = {}
for inst in ("soft-mem", "ls-mem"):
    tol = f"{m1[inst]['tol']:g}"
    print(f"--- {inst} tol {tol}; ln(1/tol) = {math.log(1 / m1[inst]['tol']):.2f} nats, i.e. {math.log(1 / m1[inst]['tol']):.1f}"
          f" passes at exactly e^-1 per pass")
    by = {}
    for r in m1[inst]["rows"]:
        s = r["summary"][tol]
        if s["reached"] == s["runs"]:
            by.setdefault(r["group"], []).append((s["median"], r["family"], s["min"], s["max"]))
    for g, v in by.items():
        v.sort()
        print(f"    group {g:8s}: best {v[0][1]:32s} {v[0][0]:.2f} [{v[0][2]}, {v[0][3]}]   (n families {len(v)})")
    allv = sorted(x for v in by.values() for x in v)
    print("    ten fastest:", [(round(a, 1), b) for a, b, _, _ in allv[:10]])

print()
print("=" * 100)
print("M8_summary.json: two further draws (data seeds 1, 2) + seed 0 from M1")
print("=" * 100)
ORTH = ["SAGA-Muon (joint)", "soft SAGA-Muon", "IS soft SAGA-Muon", "split SAGA-Muon"]
EUCL = ["B-SAGA", "IS-B-SAGA"]
ONE = ["one-shot polar-metric SAGA", "one-shot polar-metric B-SAGA"]
print("instances in M8:", {k: list(v.keys()) for k, v in m8.items()})
for inst in ("soft-cond", "ls-cond"):
    for sd in ("0", "1", "2"):
        if sd == "0":
            rows = {r["family"]: r for r in m1[inst]["rows"]}
        else:
            rows = {r["family"]: r for r in m8[inst][sd]["rows"]}
        tol = "1e-06"
        g = lambda f: rows[f]["summary"][tol]
        full = lambda f: g(f)["reached"] == g(f)["runs"]
        fams = [f for f in (["SAGA"] + EUCL + ["cov-metric SAGA"] + ORTH + ONE)]
        line = "  ".join(f"{f[:18]}={g(f)['median']:.1f}({g(f)['reached']}/{g(f)['runs']})" for f in fams)
        bo = min((g(f)["median"], f) for f in ORTH if full(f))
        be = min((g(f)["median"], f) for f in EUCL if full(f))
        b1 = min((g(f)["median"], f) for f in ONE if full(f))
        cov = g("cov-metric SAGA")["median"]
        every = sorted((g(f)["median"], f) for f in fams if full(f))
        print(f"--- {inst} data seed {sd}: n families {len(rows) if sd != '0' else 10}; SAGA step {rows['SAGA']['best']}")
        print("    " + line)
        print(f"    best recomputing polar: {bo}; /SAGA = {bo[0] / g('SAGA')['median']:.3f}; best Euclid no precond: {be};"
              f" ratio = {bo[0] / be[0]:.3f}")
        print(f"    best recomputing polar / cov-metric SAGA = {bo[0] / cov:.3f}; best one-shot {b1}; overall best: "
              f"{every[0]}  (second {every[1]})")
        if sd == "0":
            cb = rows["cov-metric B-SAGA"]["summary"][tol]["median"]
            print(f"    (M1 only, not among the ten families of M8: cov-metric B-SAGA = {cb})")

print()
print("=" * 100)
print("M5b_summary.json: batch one on grids of ratio 2^(1/4) (data seed 0)")
print("=" * 100)
for inst in ("soft-cond", "ls-cond", "ls-orth"):
    cell = m5b["instances"][inst]["0"]["batches"]["1"]
    for part in ("factor2", "fine", "shifted"):
        out = []
        for fam in ("SAGA", "B-SAGA", "SAGA-Muon"):
            r = cell[part]["rows"][fam]
            s = r["summary"]["1e-06"]
            out.append(f"{fam} {s['median']:.2f} [{s['min']}, {s['max']}] ({s['reached']}/{s['runs']}) {json.dumps(r['best'])}")
        print(f"{inst:9s} {part:8s}: " + " | ".join(out))
    f2, fi = cell["factor2"]["rows"], cell["fine"]["rows"]
    for fam in ("SAGA", "B-SAGA", "SAGA-Muon"):
        a, b = f2[fam]["summary"]["1e-06"]["median"], fi[fam]["summary"]["1e-06"]["median"]
        print(f"      {fam:10s} factor-2 grid {a:.2f} -> 2^(1/4) grid {b:.2f}: change {100 * (b / a - 1):+.1f}%")
    print(f"      SAGA-Muon/SAGA: factor-2 {f2['SAGA-Muon']['summary']['1e-06']['median'] / f2['SAGA']['summary']['1e-06']['median']:.3f}"
          f" -> fine {fi['SAGA-Muon']['summary']['1e-06']['median'] / fi['SAGA']['summary']['1e-06']['median']:.3f}")

print()
print("=" * 100)
print("Resolution: bootstrap SE of the median over the ten confirmation seeds, families of (M3)")
print("=" * 100)
random.seed(0)


def boot_se(x, B=4000):
    x = [v if v is not None else float("inf") for v in x]
    meds = []
    for _ in range(B):
        s = [random.choice(x) for _ in x]
        meds.append(st.median(s))
    fin = [m for m in meds if math.isfinite(m)]
    return st.pstdev(fin) / st.median(x) if fin else float("nan")


for inst in ("soft-cond", "ls-cond", "ls-orth"):
    for fam in ["SAGA", "B-SAGA", "IS-B-SAGA", "cov-metric B-SAGA", "cov-metric SAGA", "SAGA-Muon (joint)",
                "soft SAGA-Muon", "IS soft SAGA-Muon", "one-shot polar-metric B-SAGA", "one-shot polar-metric SAGA"]:
        r = row(inst, fam)
        cp = [float("inf") if (v is None) else v for v in r["confirm_passes"]]
        fin = [v for v in cp if math.isfinite(v)]
        print(f"{inst:9s} {fam:30s} median {st.median(cp):6.2f} SE(median)/median {100 * boot_se(cp):5.1f}%  "
              f"range/median {100 * (max(fin) - min(fin)) / st.median(cp):5.1f}%  n_finite {len(fin)}")
print()
print("pairwise comparisons of (M3) with a paired bootstrap over the confirmation seeds (seed noise only)")


def pair(inst, a, b, B=4000):
    xa = [float("inf") if v is None else v for v in row(inst, a)["confirm_passes"]]
    xb = [float("inf") if v is None else v for v in row(inst, b)["confirm_passes"]]
    rs = []
    for _ in range(B):
        ia = [random.randrange(len(xa)) for _ in xa]
        ib = [random.randrange(len(xb)) for _ in xb]
        rs.append(st.median([xa[i] for i in ia]) / st.median([xb[i] for i in ib]))
    rs.sort()
    return st.median(xa) / st.median(xb), rs[int(0.05 * B)], rs[int(0.95 * B)]


for inst in ("soft-cond", "ls-cond"):
    for a, b in (("IS soft SAGA-Muon", "IS-B-SAGA"), ("soft SAGA-Muon", "IS-B-SAGA"), ("soft SAGA-Muon", "B-SAGA"),
                 ("one-shot polar-metric B-SAGA", "IS-B-SAGA"), ("one-shot polar-metric B-SAGA", "cov-metric B-SAGA"),
                 ("SAGA-Muon (joint)", "B-SAGA")):
        r, lo, hi = pair(inst, a, b)
        print(f"{inst:9s} {a:30s} / {b:20s} = {r:.3f}  [5%, 95%] = [{lo:.3f}, {hi:.3f}]")

print()
print("tuning gap: for each family of (M3), relative gap between the two best grid points (score = geometric mean of")
print("the passes on the tuning seeds), and the step constants of the finalists")
for inst in ("soft-cond", "ls-cond"):
    for fam in ["SAGA", "B-SAGA", "IS-B-SAGA", "cov-metric B-SAGA", "SAGA-Muon (joint)", "soft SAGA-Muon",
                "one-shot polar-metric B-SAGA"]:
        r = row(inst, fam)
        sc = []
        for t in r["tune"]:
            s = t.get("score_refined") if t.get("score_refined") is not None else None
            if s is not None and math.isfinite(s):
                sc.append((s, json.dumps(t["point"])))
        sc.sort()
        if len(sc) >= 2:
            print(f"{inst:9s} {fam:30s} refined scores: " + " ; ".join(f"{s:.1f}" for s, _ in sc[:3])
                  + f"   gap best->2nd {100 * (sc[1][0] / sc[0][0] - 1):.1f}%   n grid points {r['n_points']}, kept {r['n_kept']}")

print()
print("=" * 100)
print("(M4) heading: what the improved constant is worth in the certified rate min{1/(3n), 3 mu_P/(16 R_P^2)}")
print("=" * 100)
for key, inst in m7["metric"].items():
    if inst["loss"] != "ls":
        continue
    n, kE = inst["n"], inst["kappa_euclid_global"]
    rate = lambda k: min(1 / (3 * n), 3 / (16 * k))
    for a in (POL, COV):
        rows = [r for r in inst["rows"] if r["anchor"] == a]
        kP = min(r["kappa_P_global"] for r in rows)
        print(f"{key:8s} {a[:18]:18s} kappa {kE:8.1f} -> {kP:7.1f} (constant x{kE / kP:6.1f});  certified rate per "
              f"iteration {rate(kE):.3e} -> {rate(kP):.3e} (x{rate(kP) / rate(kE):5.1f}); 1/(3n) = {1 / (3 * n):.3e}; "
              f"Euclidean rate memory-limited: {3 / (16 * kE) > 1 / (3 * n)}")

print()
print("ls-orth, tuning curve of the two metric SAGA families (edge flags of the selected points)")
for fam in ("one-shot polar-metric SAGA", "cov-metric SAGA"):
    r = row("ls-orth", fam)
    pts = sorted(((t["score_refined"] if t.get("score_refined") is not None else t["score"]), json.dumps(t["point"]))
                 for t in r["tune"] if t.get("score") is not None and math.isfinite(t["score"]))
    print(f"  {fam}: edge {r['edge']}; best five tuning scores: " + " | ".join(f"{s:.1f} {p}" for s, p in pts[:5]))
