"""Shared helpers of the Muon experiments: paths, problems with their optimum, a parallel task runner with an
incremental cache, the tune-then-confirm protocol and the metrics.

Protocol (used by every comparison of methods):
  * budget     a number of PASSES, i.e. of per-sample gradient evaluations divided by n (one full gradient = 1 pass);
               every method is charged with the accounting of muon/lab (algorithms.py, vrmuon.py, mvr.py);
  * metric     passes needed to bring ||W - W_*||_F^2 / ||W_0 - W_*||_F^2 below a tolerance (first recorded point,
               resolution 0.1 pass or one iteration), infinity if not reached within the budget; W_0 = 0;
  * tuning     every method family has a grid of step parameters expressed in problem units (e.g. gamma = c / R2_op).
               (0) Screening: every grid point is first run for 20% of the budget on the first tuning seed, and only
               the best third of the grid of each family (at least 4 points, ranked by the relative squared distance
               reached; diverged points are dropped) goes on. Families with at most 4 grid points, deterministic
               methods and methods whose learning-rate schedule spans the budget are not screened.
               (1) Each remaining point is run on the TUNING seeds (0, 1) with the full budget and scored by the
               geometric mean of the passes to the target tolerance (a run that misses the target counts as the
               budget times (1 + log10(best relative distance / tolerance)), so that closer misses rank first; a
               diverged run counts as infinity). The grid neighbours of the best point that the screening had dropped
               are then run too (up to 3 rounds), so that the best point is a local minimum of the score on the grid.
               (2) Refinement: the 3 best points are run on 3 more tuning seeds (2, 3, 4) and the selected point
               minimizes the geometric mean over the 5 tuning seeds;
  * report     the selected point is then run on 10 fresh CONFIRMATION seeds (10..19), never used for tuning; tables
               give the median and the range over these seeds, the number of seeds that reach the target, and the
               number that go on to the stop level. Runs stop early once the relative squared distance is 1e-3
               (tuning) or 1e-4 (confirmation) times the target tolerance (the "stop level").
Deterministic methods (full-batch) use one run per grid point. Seeds are algorithm seeds (sampling of the indices);
the problem instance has its own seed in its specification.

The cache of `run_tasks` is keyed by the task only. `code_fingerprint()` is stored with every summary, and
`audit_cache.py` recomputes a random sample of cached runs with the current code to check that they are reproduced.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pickle
import sys
import time

import numpy as np

for _stream in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
MUON = os.path.dirname(HERE)
ROOT = os.path.dirname(MUON)
for _p in (ROOT, MUON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS = os.path.join(MUON, "results")
FIGURES = os.path.join(MUON, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

TUNE_SEEDS = (0, 1)
REFINE_SEEDS = (2, 3, 4)       # extra tuning seeds for the finalists of each family
REFINE_TOP = 3                 # number of finalists
NEIGHBOUR_ROUNDS = 3           # rounds of "run the dropped grid neighbours of the best point"
CONFIRM_SEEDS = tuple(range(10, 20))
DETERMINISTIC = ("gd", "muon_full", "agd", "lbfgs")          # methods without randomness: one run per grid point
TOLS = (1e-3, 1e-6, 1e-9, 1e-12)
SCREEN_FRACTION = 0.2          # screening budget, as a fraction of the budget
SCREEN_KEEP = 1.0 / 3.0        # fraction of the grid points of a family kept after screening ...
SCREEN_MIN_POINTS = 4          # ... at least this many; families with at most this many points are not screened
SCREEN_STOP, TUNE_STOP, CONFIRM_STOP = 1e-3, 1e-3, 1e-4     # runs stop this factor below the target tolerance


# ----------------------------------------------------------------------------------------------
# problems
# ----------------------------------------------------------------------------------------------
_PROBLEMS = {}


def spec(loss, n=1000, d=50, K=10, kappa=1000.0, alpha=1.0, seed=0, **kw):
    """Specification of a problem instance (a plain dict, hashable through its JSON form)."""
    s = dict(loss=loss, n=int(n), d=int(d), K=int(K), kappa=float(kappa), alpha=float(alpha), seed=int(seed))
    s.update(kw)
    return s


def spec_tag(s):
    extra = "".join(f"_{k}{v}" for k, v in sorted(s.items())
                    if k not in ("loss", "n", "d", "K", "kappa", "alpha", "seed"))
    return f"{s['loss']}_n{s['n']}_d{s['d']}_K{s['K']}_k{s['kappa']:g}_a{s['alpha']:g}_s{s['seed']}{extra}"


def problem(s):
    """The problem of a specification, with its optimum (cached in the process)."""
    key = json.dumps(s, sort_keys=True)
    if key not in _PROBLEMS:
        from lab import optimum, problems
        kw = {k: v for k, v in s.items() if k != "loss"}
        mk = problems.make_softmax if s["loss"] == "softmax" else problems.make_matrix_ls
        prob = mk(**kw)
        optimum.solve_optimum(prob)
        _PROBLEMS[key] = prob
    return _PROBLEMS[key]


def over(unit, c):
    """A step parameter in problem units: over('R2_op', 0.25) is resolved to 0.25 / prob.R2_op at run time. Units:
    R2, R2_op, L_F, L_op (smoothness constants), 'n' (c / n) and 'one' (the plain number c)."""
    return {"over": unit, "c": float(c)}


def times(unit, c):
    """A parameter proportional to a problem constant: times('L_op', 0.5) is resolved to 0.5 * prob.L_op (used for the
    smoothness parameter L of muon_full, whose step is <G, O> / L)."""
    return {"times": unit, "c": float(c)}


def _unit(prob, u):
    return 1.0 if u == "one" else float(prob.n if u == "n" else getattr(prob, u))


def resolve(cfg, prob):
    """Replace the unit-tagged values of a configuration by numbers. The pseudo-parameter `miso_delta` = delta expands
    to the Euclidean member that keeps the iterate at the minimizer of the surrogate (theta = delta / n,
    gamma = 1 / lam, table relaxed by delta): MISO / Finito for delta = 1, dual-free SDCA for delta < 1."""
    if "miso_delta" in cfg:
        cfg = dict(cfg)
        dl = cfg.pop("miso_delta")
        if isinstance(dl, str):                    # 'sdca', 'sdca/2', '2sdca': multiples of lam n / (R2 + lam n)
            mult = {"sdca": 1.0, "sdca/2": 0.5, "2sdca": 2.0}[dl]
            dl = min(1.0, mult * prob.lam * prob.n / (prob.R2 + prob.lam * prob.n))
        dl = float(dl)
        cfg.update(theta=dl * int(cfg.get("batch", 1)) / prob.n, gamma=1.0 / prob.lam, mem_relax=dl)
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and set(v) == {"over", "c"}:
            out[k] = v["c"] / _unit(prob, v["over"])
        elif isinstance(v, dict) and set(v) == {"times", "c"}:
            out[k] = v["c"] * _unit(prob, v["times"])
        else:
            out[k] = v
    return out


def constants(prob):
    return {k: float(getattr(prob, k)) for k in ("R2", "R2_op", "L_F", "L_op", "lam", "kappa", "kappa_F", "kappa_op")}


# ----------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------
def passes_to(ge_over_n, rel, tol):
    """First recorded number of passes at which rel <= tol (inf if never)."""
    k = np.nonzero(np.asarray(rel) <= tol)[0]
    return float(ge_over_n[k[0]]) if len(k) else float("inf")


def _thin(x, m=240):
    x = np.asarray(x)
    if len(x) <= m:
        return x.tolist()
    idx = np.unique(np.round(np.linspace(0, len(x) - 1, m)).astype(int))
    return x[idx].tolist()


def run_one(task):
    """Run one task (a JSON-serializable dict) and return a compact result dict.

    task = {'spec': problem specification, 'method': 'vr' | 'gd' | 'agd' | 'lbfgs' | 'muon_full' | 'muon_sgd' | 'sgd' |
            'mvr' | 'katyusha',
            'cfg': keyword arguments of the method with unit-tagged steps (see `over`), 'passes': budget,
            'seed': algorithm seed, optional 'stop_rel' (stop when the relative squared distance is below it,
            default 1e-26), optional 'instr' (vr only: instrument every that many passes), optional
            'start': 'near_opt' (W_0 = W_* + 1e-7 ||W_*||_F P with P a fixed unit direction in the gauge subspace,
            instead of W_0 = 0; distances stay normalized by ||W_*||_F^2)}."""
    from lab import algorithms as alg, mvr, polar, vrmuon as vm
    s = task["spec"]
    prob = problem(s)
    n = prob.n
    W0 = np.zeros((prob.K, prob.d))
    if task.get("start") == "near_opt":
        P = np.random.default_rng(999).standard_normal((prob.K, prob.d))
        P = prob.remove_gauge(P)
        W0 = prob.W_star + 1e-7 * float(np.linalg.norm(prob.W_star)) * P / float(np.linalg.norm(P))
    D0 = float(np.vdot(prob.W_star, prob.W_star))
    F0 = prob.Fgap(W0)
    cfg = resolve(task["cfg"], prob)
    passes = float(task["passes"])
    budget = int(round(passes * n))
    stop = task.get("stop_rel", 1e-26) * D0
    method = task["method"]
    seed = int(task.get("seed", 0))
    gauge_polar = (lambda M: polar.polar_svd(prob.remove_gauge(M)))
    t0 = time.perf_counter()
    err = None
    try:
        if method == "vr":
            kw = {}
            if task.get("instr"):
                b = int(cfg.get("batch", 1))
                kw = dict(instrument_every=max(1, int(task["instr"] * n / b)))
            n_it = 10 ** 9
            if cfg.get("schedule", "const") != "const" or cfg.get("warmup"):
                from lab.exact import expected_cost               # a schedule must span the actual number of iterations
                algo0 = vm.VRMuon(prob, n_iters=1, **cfg)
                n_it = max(1, int((budget - algo0.init_cost) / expected_cost(algo0)))
            r = vm.vr_muon(prob, W0, n_it, seed=seed, max_grad_evals=budget, stop_dist2=stop, **kw, **cfg)
        elif method == "gd":
            r = alg.gd(prob, W0, int(passes), stop_dist2=stop, **cfg)
        elif method == "agd":
            r = alg.agd(prob, W0, int(passes), stop_dist2=stop, **cfg)
        elif method == "lbfgs":
            r = alg.lbfgs(prob, W0, int(passes), stop_dist2=stop, **cfg)
        elif method == "muon_full":
            per = 2 if cfg.get("rule") == "linesearch" else 1
            orth = cfg.pop("orth", "svd")
            r = alg.muon_full(prob, W0, int(passes // per), stop_dist2=stop,
                              orth=gauge_polar if orth == "svd" else _gauged(prob, orth), **cfg)
        elif method in ("muon_sgd", "sgd"):
            b = int(cfg["batch"])
            fn = alg.muon_sgd if method == "muon_sgd" else alg.sgd
            if method == "muon_sgd":
                cfg["orth"] = _gauged(prob, cfg.get("orth", "ns5"))
            r = fn(prob, W0, budget // b, seed=seed, stop_dist2=stop, **cfg)
        elif method == "katyusha":
            from lab import katyusha
            r = katyusha.lkatyusha(prob, W0, 10 ** 9, seed=seed, stop_dist2=stop, max_grad_evals=budget, **cfg)
        elif method == "mvr":
            b = int(cfg["batch"])
            r = mvr.mvr_muon(prob, W0, max(1, budget // (b * (2 if cfg.get("same_batch", True) else 1))),
                             seed=seed, stop_dist2=stop, max_grad_evals=budget, **cfg)
        else:
            raise ValueError(f"unknown method {method!r}")
    except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError, ZeroDivisionError) as exc:
        err = repr(exc)[:200]
        r = None
    wall = time.perf_counter() - t0
    if r is None:
        return {"error": err, "diverged": True, "final_rel": float("inf"), "min_rel": float("inf"),
                "passes_to": {f"{t:g}": float("inf") for t in TOLS}, "wall": wall, "stopped": "error"}
    ge = np.asarray(r["grad_evals"], dtype=float) / n
    rel = np.asarray(r["dist2_F"], dtype=float) / D0
    fg = np.asarray(r["Fgap"], dtype=float) / F0
    ok = np.isfinite(rel)
    diverged = bool(r["stopped"] == "diverged" or not ok.all() or rel[-1] > 1e3)
    out = {
        "passes_to": {f"{t:g}": passes_to(ge[ok], rel[ok], t) for t in TOLS},
        "final_rel": float(rel[-1]) if ok[-1] else float("inf"),
        "min_rel": float(np.nanmin(rel[ok])) if ok.any() else float("inf"),
        "final_Fgap_rel": float(fg[-1]) if np.isfinite(fg[-1]) else float("inf"),
        "passes_used": float(ge[-1]), "iters": int(r["iters"][-1]), "stopped": r["stopped"], "diverged": diverged,
        "wall": wall, "trace": {"passes": _thin(ge), "rel": _thin(rel), "Fgap_rel": _thin(fg)},
    }
    for k in ("refreshes", "fallbacks", "zbar_err", "clipped"):
        if k in r:
            out[k] = r[k]
    if "eta" in r and np.ndim(r["eta"]) == 1 and len(r["eta"]) == len(r["grad_evals"]):
        out["trace"]["eta"] = _thin(np.asarray(r["eta"], dtype=float))
    if "mem" in r and np.ndim(r["mem"]) == 1:
        out["trace"]["mem"] = _thin(np.asarray(r["mem"], dtype=float))
    if task.get("instr") and r.get("instr"):
        keys = ("t", "grad_evals", "dist2_F", "srank_E", "mu_eff_F", "mu_eff_op", "nrank_G", "L_eff_F", "nuc_grad",
                "nuc_G", "nuc_m", "grad_norm_F", "G_norm_F", "cos_G", "pair", "curv_O", "decrease", "decrease_model",
                "O_op", "O_fro2")
        out["instr"] = {k: [float(st[k]) for st in r["instr"]] for k in keys if k in r["instr"][0]}
    return out


def _gauged(prob, orth):
    """Orthogonalization by name applied after the gauge projection (softmax); the name itself for least squares."""
    if prob.loss != "softmax" or callable(orth):
        return orth
    from lab import polar
    fn = polar.get_orth(orth)
    return lambda M: fn(prob.remove_gauge(M))


# ----------------------------------------------------------------------------------------------
# parallel runner with an incremental cache
# ----------------------------------------------------------------------------------------------
def task_key(task):
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


def code_fingerprint():
    """SHA-1 of the code that determines the results (muon/lab/*.py, common.py, families.py), stored with every
    summary. The cache itself is keyed by the task only; see audit_cache.py."""
    h = hashlib.sha1()
    files = sorted(os.path.join(MUON, "lab", f) for f in os.listdir(os.path.join(MUON, "lab")) if f.endswith(".py"))
    files += [os.path.join(HERE, "common.py"), os.path.join(HERE, "families.py")]
    for path in files:
        with open(path, "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _refresh(task):
    """True if the task must be recomputed even when cached: the environment variable MUON_REFRESH holds
    semicolon-separated substrings matched against the JSON of the configuration (e.g. "tangent")."""
    pats = [q for q in os.environ.get("MUON_REFRESH", "").split(";") if q]
    if not pats:
        return False
    txt = json.dumps(task["cfg"], sort_keys=True)
    return any(q in txt for q in pats)


def _load_cache(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


_REFRESHED = set()        # keys computed in this process (MUON_REFRESH recomputes a cached run once per process)


def run_tasks(tasks, name, n_jobs=None, verbose=True):
    """Run the tasks (list of dicts for `run_one`) in parallel processes with single-threaded BLAS, reusing and
    updating the cache muon/results/<name>.pkl (key: hash of the task). Returns the list of results, in order."""
    path = os.path.join(RESULTS, f"{name}.pkl")
    cache = _load_cache(path)
    keys = [task_key(t) for t in tasks]
    todo = {}
    for k, t in zip(keys, tasks):
        if (k not in cache or (_refresh(t) and k not in _REFRESHED)) and k not in todo:
            todo[k] = t
    if todo:
        n_jobs = n_jobs or int(os.environ.get("MUON_JOBS", "4"))
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        t0 = time.perf_counter()
        if verbose:
            print(f"[{name}] {len(todo)} runs to do ({len(tasks) - len(todo)} cached), {n_jobs} processes", flush=True)
        items = list(todo.items())
        done = 0
        if n_jobs == 1:
            for k, t in items:
                cache[k] = run_one(t)
                done += 1
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(run_one, t): k for k, t in items}
                last = time.perf_counter()
                for fut in as_completed(futs):
                    cache[futs[fut]] = fut.result()
                    done += 1
                    if time.perf_counter() - last > 60:
                        _save_cache(path, cache)
                        last = time.perf_counter()
                        if verbose:
                            print(f"[{name}] {done}/{len(items)} done, {time.perf_counter() - t0:.0f}s", flush=True)
        _save_cache(path, cache)
        _REFRESHED.update(todo)
        if verbose:
            print(f"[{name}] finished {len(items)} runs in {time.perf_counter() - t0:.0f}s", flush=True)
    return [cache[k] for k in keys]


# ----------------------------------------------------------------------------------------------
# tune, then confirm
# ----------------------------------------------------------------------------------------------
def grid_points(grid):
    """All combinations of a dict of lists, as a list of dicts (a single empty dict for an empty grid)."""
    if not grid:
        return [{}]
    names = list(grid)
    return [dict(zip(names, vals)) for vals in itertools.product(*(grid[k] for k in names))]


def score(res, tol, budget):
    """Tuning score of one run (smaller is better): passes to `tol`, or a penalized budget if the target is missed."""
    if res["diverged"] or not math.isfinite(res["final_rel"]):
        return float("inf")
    p = res["passes_to"][f"{tol:g}"]
    if math.isfinite(p):
        return p
    return budget * (1.0 + max(0.0, math.log10(max(res["min_rel"], 1e-300) / tol)))


def geo_mean(vals):
    vals = [v for v in vals]
    if any(not math.isfinite(v) for v in vals):
        return float("inf")
    return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-12)))))


def make_tasks(s, family, passes, seeds, points=None, stop_rel=None):
    """Tasks of a family ({'name', 'method', 'base': cfg, 'grid': {param: [values]}}) on the instance s."""
    pts = grid_points(family.get("grid", {})) if points is None else points
    tasks = []
    for pt in pts:
        cfg = dict(family.get("base", {}))
        cfg.update(pt)
        for seed in seeds:
            t = {"spec": s, "method": family["method"], "cfg": cfg, "passes": passes, "seed": int(seed)}
            if stop_rel is not None:
                t["stop_rel"] = float(stop_rel)
            tasks.append(t)
    return pts, tasks


def _screenable(fam, n_points):
    """Screening applies to stochastic families whose short run is a prefix of the long one (no schedule spanning the
    budget) and whose grid has more than SCREEN_MIN_POINTS points."""
    if fam["method"] in DETERMINISTIC or n_points <= SCREEN_MIN_POINTS:
        return False
    base = fam.get("base", {})
    return base.get("schedule", "const") == "const" and not base.get("warmup")


def _neighbours(grid, i):
    """Flat indices (in the order of `grid_points`) of the grid neighbours of point i: one step along one parameter."""
    names = list(grid)
    if not names:
        return []
    sizes = [len(grid[k]) for k in names]
    multi = [int(v) for v in np.unravel_index(i, sizes)]
    out = []
    for a in range(len(names)):
        for dlt in (-1, 1):
            j = multi[a] + dlt
            if 0 <= j < sizes[a]:
                m2 = list(multi)
                m2[a] = j
                out.append(int(np.ravel_multi_index(m2, sizes)))
    return out


def tune_and_confirm(specs, families, passes, tol, name, tune_seeds=TUNE_SEEDS, confirm_seeds=CONFIRM_SEEDS,
                     n_jobs=None, screen=True, refine_seeds=REFINE_SEEDS):
    """The protocol of the module docstring for every (instance, family). Deterministic families use a single run
    per grid point. Returns a list of records
      {'spec', 'tag', 'family', 'method', 'base', 'best': selected grid point, 'best_score': its tuning score (over
       all its tuning seeds), 'refined': whether the refinement seeds were used, 'finalists': grid points,
       'tune': [{'point', 'score' (tuning seeds 0-1), 'score_refined' (finalists: all tuning seeds), 'scores',
                 'final_rel', 'screen_rel', 'kept' (run at full budget)}],
       'edge', 'n_points', 'n_kept', 'confirm': [results],
       'summary': {tolerance: {'median', 'min', 'max', 'q1', 'q3', 'reached', 'runs'}},
       'stop_level': {'reached', 'runs'} (confirmation runs that went on to CONFIRM_STOP x tol),
       'final_rel': {'median', 'min', 'max'}, 'diverged'}."""
    # ---- stage 0: screening on a short budget, first tuning seed
    plan0, tasks0 = [], []
    short = max(1, int(math.ceil(SCREEN_FRACTION * passes)))
    for s in specs:
        for fam in families:
            pts = grid_points(fam.get("grid", {}))
            if screen and _screenable(fam, len(pts)):
                _, tk = make_tasks(s, fam, short, (tune_seeds[0],), stop_rel=SCREEN_STOP * tol)
                plan0.append((len(tasks0), len(tk)))
                tasks0.extend(tk)
            else:
                plan0.append(None)
    res0 = run_tasks(tasks0, name + "_screen", n_jobs) if tasks0 else []
    cells = []
    it = iter(plan0)
    for s in specs:
        for fam in families:
            det = fam["method"] in DETERMINISTIC
            pts = grid_points(fam.get("grid", {}))
            p0 = next(it)
            screen_rel = [None] * len(pts)
            kept = list(range(len(pts)))
            if p0 is not None:
                r0 = res0[p0[0]:p0[0] + p0[1]]
                screen_rel = [float("inf") if (r["diverged"] or not math.isfinite(r["min_rel"])) else r["final_rel"]
                              for r in r0]
                order = sorted(range(len(pts)), key=lambda i: (screen_rel[i], i))
                n_keep = max(SCREEN_MIN_POINTS, int(math.ceil(len(pts) * SCREEN_KEEP)))
                kept = sorted(i for i in order[:n_keep] if math.isfinite(screen_rel[i])) or order[:1]
            cells.append({"s": s, "fam": fam, "det": det, "pts": pts, "screen_rel": screen_rel,
                          "seeds": (0,) if det else tuple(tune_seeds), "first": kept, "runs": {}})

    def run_points(todo):
        """Run (cell, point index, seeds) at full budget and store score and final distance per seed."""
        tasks, plan = [], []
        for cell, i, seeds in todo:
            _, tk = make_tasks(cell["s"], cell["fam"], passes, seeds, points=[cell["pts"][i]],
                               stop_rel=TUNE_STOP * tol)
            plan.append((cell, i, seeds, len(tasks), len(tk)))
            tasks.extend(tk)
        res = run_tasks(tasks, name + "_tune", n_jobs) if tasks else []
        for cell, i, seeds, lo, m in plan:
            d = cell["runs"].setdefault(i, {})
            for seed, r in zip(seeds, res[lo:lo + m]):
                d[seed] = (score(r, tol, passes), r["final_rel"])

    def score2(cell, i):                                   # score on the first tuning seeds
        return geo_mean([cell["runs"][i][sd][0] for sd in cell["seeds"]])

    def score_all(cell, i):                                # score on every tuning seed run for this point
        return geo_mean([v[0] for v in cell["runs"][i].values()])

    # ---- stage 1: full budget on the tuning seeds for the points kept by the screening
    run_points([(c, i, c["seeds"]) for c in cells for i in c["first"]])
    # ---- stage 1a: the grid neighbours of the best point that the screening dropped
    for _ in range(NEIGHBOUR_ROUNDS):
        todo = []
        for c in cells:
            best = min(sorted(c["runs"]), key=lambda i: (score2(c, i), i))
            for j in _neighbours(c["fam"].get("grid", {}), best):
                if j not in c["runs"] and (c["screen_rel"][j] is None or math.isfinite(c["screen_rel"][j])):
                    todo.append((c, j, c["seeds"]))
        if not todo:
            break
        run_points(todo)
    # ---- stage 2: refinement of the finalists on more tuning seeds
    todo = []
    for c in cells:
        ranked = sorted(sorted(c["runs"]), key=lambda i: (score2(c, i), i))
        fin = [i for i in ranked[:REFINE_TOP] if math.isfinite(score2(c, i))]
        c["refined"] = bool(refine_seeds) and not c["det"] and len(fin) >= 2
        c["finalists"] = fin if c["refined"] else ranked[:1]
        if c["refined"]:
            todo += [(c, i, tuple(refine_seeds)) for i in fin]
    run_points(todo)
    records, ctasks, cplan = [], [], []
    for c in cells:
        s, fam, pts = c["s"], c["fam"], c["pts"]
        best = min(c["finalists"], key=lambda i: (score_all(c, i), score2(c, i), i))
        tune = []
        for i, pt in enumerate(pts):
            row = {"point": pt, "score": float("inf"), "scores": [], "final_rel": [], "screen_rel": c["screen_rel"][i],
                   "kept": i in c["runs"]}
            if i in c["runs"]:
                row.update(score=score2(c, i), scores=[c["runs"][i][sd][0] for sd in c["seeds"]],
                           final_rel=[c["runs"][i][sd][1] for sd in c["seeds"]])
                if c["refined"] and i in c["finalists"]:
                    row["score_refined"] = score_all(c, i)
                    row["scores_refined"] = [v[0] for v in c["runs"][i].values()]
            tune.append(row)
        rec = {"spec": s, "tag": spec_tag(s), "family": fam["name"], "method": fam["method"],
               "base": fam.get("base", {}), "best": pts[best], "best_score": score_all(c, best), "tune": tune,
               "refined": c["refined"], "finalists": [pts[i] for i in c["finalists"]],
               "edge": _edge(pts, best, fam.get("grid", {})), "n_points": len(pts), "n_kept": len(c["runs"])}
        cseeds = (0,) if c["det"] else tuple(confirm_seeds)
        _, tk = make_tasks(s, fam, passes, cseeds, points=[pts[best]], stop_rel=CONFIRM_STOP * tol)
        cplan.append((rec, len(ctasks), len(tk)))
        ctasks.extend(tk)
        records.append(rec)
    cres = run_tasks(ctasks, name + "_confirm", n_jobs)
    for rec, lo, m in cplan:
        res = cres[lo:lo + m]
        rec["confirm"] = res
        rec["summary"] = {}
        for t in TOLS:
            v = np.array([r["passes_to"][f"{t:g}"] for r in res])
            vs = np.sort(v)
            rec["summary"][f"{t:g}"] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                                        "q1": float(vs[(len(vs) - 1) // 4]), "q3": float(vs[(3 * (len(vs) - 1) + 3) // 4]),
                                        "reached": int(np.isfinite(v).sum()), "runs": len(v)}
        f = np.array([r["final_rel"] for r in res])
        rec["final_rel"] = {"median": float(np.median(f)), "min": float(f.min()), "max": float(f.max())}
        rec["stop_level"] = {"reached": int(sum(r.get("stopped") == "stop_dist2" for r in res)), "runs": len(res)}
        rec["diverged"] = int(sum(r["diverged"] for r in res))
    return records


def _num(v):
    """Numeric value of a grid entry for ordering (the constant c of a unit-tagged value); None if not numeric."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and "c" in v:
        return float(v["c"])
    return None


def _edge(pts, best, grid):
    """Names of the grid parameters whose selected value is the smallest ('-') or the largest ('+') value of its grid
    (the optimum may be outside). Compared by value, whatever the order of the list; parameters with non-numeric
    values are not flagged. Natural boundaries (theta towards 1, miso_delta = 1) are flagged like the others."""
    out = []
    for k, vals in grid.items():
        nums = [_num(v) for v in vals]
        if len(vals) < 2 or any(x is None for x in nums):
            continue
        x = _num(pts[best][k])
        if x == min(nums):
            out.append(k + "-")
        elif x == max(nums):
            out.append(k + "+")
    return out


# ----------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(name, obj):
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    return path


def load_json(name):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_point(pt):
    """Compact text of a grid point, e.g. gamma=0.5/R2_op, theta=0.1."""
    parts = []
    for k, v in pt.items():
        if isinstance(v, dict) and set(v) == {"over", "c"}:
            parts.append(f"{k}={v['c']:g}" + ("" if v["over"] == "one" else f"/{v['over']}"))
        elif isinstance(v, dict) and set(v) == {"times", "c"}:
            parts.append(f"{k}={v['c']:g}*{v['times']}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def fmt_passes(summary, tol):
    """'median [min, max]' of the passes to `tol` over the confirmation seeds ('-' when never reached)."""
    s = summary[f"{tol:g}"]
    if s["reached"] == 0:
        return "-"

    def f(v):
        return f"{v:.1f}" if math.isfinite(v) else "inf"
    if s["runs"] == 1:
        return f(s["median"])
    return f"{f(s['median'])} [{f(s['min'])}, {f(s['max'])}]"
