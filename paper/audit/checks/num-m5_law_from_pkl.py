"""Audit num-m5, check 1: recompute the numbers of (M5) (part C of experiment M9) from the raw cache
muon/results/M9_law.pkl, independently of the summary statistics stored in M9_summary.json.

Read-only. Run from the project root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 paper/audit/checks/num-m5_law_from_pkl.py
"""
import hashlib
import json
import math
import pickle

import numpy as np

RES = "muon/results/"
m9 = json.load(open(RES + "M9_summary.json"))
cache = pickle.load(open(RES + "M9_law.pkl", "rb"))
print("runs in M9_law.pkl:", len(cache))


def key(task):
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()


for inst in ("soft-cond", "ls-cond"):
    e = m9["C"][inst]
    D0, wop, n = e["norms"]["D0"], e["norms"]["W_star_op"], e["spec"]["n"]
    print(f"\n===== {inst}: n = {n}, ||W*||_F^2 = {D0:.1f}, ||W*||_op = {wop:.2f}, passes {e['passes']}, seeds {e['seeds']}")
    for kind in ("deep", "controls"):
        for label, d in e[kind].items():
            runs = []
            for seed in e["seeds"]:
                task = {"spec": e["spec"], "method": d.get("method", "vr"), "cfg": d["cfg"], "passes": e["passes"],
                        "seed": seed, "stop_rel": 0.0}
                r = cache.get(key(task))
                assert r is not None, (label, seed)
                runs.append(r)
            cfg = d["cfg"]
            mins = np.array([r["min_rel"] for r in runs])
            fins = np.array([r["final_rel"] for r in runs])
            # minimum recomputed from the trace itself
            tmins = np.array([np.nanmin(np.asarray(r["trace"]["rel"], float)) for r in runs])
            print(f"-- [{kind}] {label}")
            print(f"   lowest rel dist2 (stored): median {np.median(mins):.3g} [{mins.min():.3g}, {mins.max():.3g}]"
                  f" | from trace: median {np.median(tmins):.3g} | final: median {np.median(fins):.3g}"
                  f" [{fins.min():.3g}, {fins.max():.3g}]")
            if kind == "controls":
                # total length of the schedule: sum_t lr_0 q^(t g / n), g gradients per step
                if d["method"] == "mvr":
                    q, per = cfg["schedule"][1], cfg["schedule"][2]          # steps per pass
                    lr0 = cfg["lr"]
                    path = lr0 / (1.0 - q ** (1.0 / per))
                else:
                    q, lr0, b = cfg["lr_decay"], cfg["lr"], cfg["batch"]
                    path = lr0 / (1.0 - q ** (b / n))
                print(f"   schedule q = {q}, lr_0 = {lr0}: total length sum_t lr_t = {path:.2f}"
                      f" = {path / wop:.3f} ||W*||_op  (W_0 = 0; a unit-spectral-norm step map can travel at most this far)")
                continue
            # widths of dist^2 / lr^2 per run
            widths_half, widths_full, lr_floor, pass_floor = [], [], [], []
            for r in runs:
                tr = r["trace"]
                p, rel, eta = (np.asarray(tr[k], float) for k in ("passes", "rel", "eta"))
                ok = np.isfinite(eta) & (eta > 0) & np.isfinite(rel)
                p, rel, eta = p[ok], rel[ok], eta[ok]
                ratio = rel * D0 / eta ** 2
                if rel.min() > 1e-12:
                    continue                                   # off schedule
                win = (rel >= 1e-12) & (rel <= 1e-6)
                level = np.median(ratio[win])
                k0 = int(np.argmax(rel <= 1e-6))
                k1 = k0 + int(np.nonzero(ratio[k0:] <= 3 * level)[0][-1])
                x = ratio[(p >= 0.5 * p[k1]) & (p <= p[k1])]
                widths_half.append(x.max() / x.min())
                # whole on-schedule part: from the first record below rel = 1e-2 to the x3 record
                ka = int(np.argmax(rel <= 1e-2))
                y = ratio[ka:k1 + 1]
                widths_full.append(y.max() / y.min())
                lr_floor.append(eta[k1])
                pass_floor.append(p[k1])
            if widths_half:
                print(f"   on schedule: {len(widths_half)}/5 | width max/min of dist2/lr2, second half of the run:"
                      f" per seed {', '.join(f'{w:.2f}' for w in widths_half)} (median {np.median(widths_half):.2f})")
                print(f"   width from rel <= 1e-2 to the floor: per seed {', '.join(f'{w:.2f}' for w in widths_full)}"
                      f" (median {np.median(widths_full):.2f})")
                print(f"   learning rate when the run leaves the law (x3 record): median {np.median(lr_floor):.3g}"
                      f" at pass {np.median(pass_floor):.0f}; machine epsilon 2.2e-16;"
                      f" sqrt(floor) = {math.sqrt(np.median(mins)):.2g}")
