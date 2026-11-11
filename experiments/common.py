"""Shared helpers for the experiment scripts: paths, result caching, problem construction, seeds."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

for _stream in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252: force UTF-8 for Greek letters
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vrlab import problems, solvers  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def result_path(name, ext="pkl"):
    return os.path.join(RESULTS, f"{name}.{ext}")


def save_result(name, obj):
    with open(result_path(name), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_result(name):
    p = result_path(name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_json(name, obj):
    with open(result_path(name, "json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def logistic_problem(n, d, kappa, alpha, seed=0, **kw):
    """Synthetic logistic problem with its optimum computed (Newton, high precision)."""
    prob = problems.make_logistic(n, d, kappa=kappa, alpha=alpha, seed=seed, **kw)
    solvers.solve_optimum(prob)
    return prob


def hard_problem(n, m, kappa, seed=0, **kw):
    prob = problems.make_hard_instance(n, m, kappa, seed=seed, **kw)
    solvers.solve_optimum(prob)
    return prob


def strip_run(run, keep_instr=True):
    """Drop heavy arrays (theta) before saving."""
    r = {k: v for k, v in run.items() if k not in ("theta",)}
    if not keep_instr:
        r.pop("instr", None)
    return r


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] {time.time() - self.t0:.1f}s", flush=True)
