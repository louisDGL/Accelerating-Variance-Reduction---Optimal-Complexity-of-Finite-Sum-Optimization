"""Audit code-fidelity, bloc (c) : petites reproductions independantes avec le code du depot (common.run_one, aucun
cache lu ni ecrit), et controle d'equite des temoins sans table de (M5).

    R1  (M5, l. 532-539)  ls-cond, graine 10, 300 passes sans regle d'arret : plancher du facteur polaire exact avec
        table (q = 0.8, lr_0 = 0.3), de la recette de Muon avec table (q = 0.8, lr_0 = 0.1), gel sans table
        (gradient de lot b = 1, ns5 + Nesterov, q = 0.75, lr_0 = 0.3).
    R2  (M2, l. 501-503)  softmax n = 1000, kappa = n : passes a 1e-6 de MISO, SAGA-Muon joint (c = 1/4), SAGA,
        graines 0, 1, 2 (memes indices).
    R3  (M6, l. 587-589)  rapport "fastest spectral / fastest control" sur ls-cond b = 16 et soft-cond b = 32, aux
        points selectionnes (M9_summary.json), graines de confirmation 10..19.
    R4  equite des temoins sans table : longueur totale du refroidissement contre ||W_*||_op ; la TABLE elle-meme a
        b = 32 sur le refroidissement des temoins ; le gradient de lot b = 1 sur un refroidissement plus lent.

Lancer depuis la racine du projet :
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python3 paper/audit/checks/code-fidelity_repro.py R1|R2|R3|R4|R4b
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "muon", "experiments"))
import common as C                                                    # noqa: E402
from common import over                                               # noqa: E402

SOFT_COND = C.spec("softmax", 200, 40, 8, 1e4, 2.0)
LS_COND = C.spec("ls", 200, 40, 8, 1e4, 2.0)
part = sys.argv[1] if len(sys.argv) > 1 else "R1"


def run(spec, method, cfg, passes, seed, stop_rel=0.0):
    t0 = time.perf_counter()
    r = C.run_one({"spec": spec, "method": method, "cfg": cfg, "passes": passes, "seed": seed, "stop_rel": stop_rel})
    r["_wall"] = time.perf_counter() - t0
    return r


def recipe(est, b, orth, q, lr0, nesterov, **kw):
    cfg = dict(estimator=est, batch=b, length="const", orth=orth, schedule="geometric", lr_decay=q, lr=lr0, **kw)
    if nesterov:
        cfg.update(beta=0.9, nesterov=True)
    return cfg


if part == "R1":
    print("R1 : ls-cond, graine 10, 300 passes, sans regle d'arret (papier : 2.3e-29 ; pres de 1e-18 ; 8 %)")
    for label, cfg in (("polaire exact + table, q=0.8, lr0=0.3", recipe("saga", 1, "svd", 0.8, 0.3, False)),
                       ("ns5 + Nesterov + table, q=0.8, lr0=0.1", recipe("saga", 1, "ns5", 0.8, 0.1, True)),
                       ("ns5 + Nesterov SANS table (b=1), q=0.75, lr0=0.3", recipe("sgd", 1, "ns5", 0.75, 0.3, True))):
        r = run(LS_COND, "vr", cfg, 300, 10)
        print(f"   {label:52s} min_rel {r['min_rel']:.3e}  final_rel {r['final_rel']:.3e}  [{r['_wall']:.0f}s]",
              flush=True)

elif part == "R2":
    s = C.spec("softmax", 1000, 50, 10, 1e3, 1.0)
    print("R2 : softmax n = 1000, kappa = n ; passes a 1e-6 (papier : MISO 11.9, joint 11.8, SAGA 20.2)")
    for label, cfg in (("MISO", dict(estimator="saga", placement="euclid", miso_delta=1.0)),
                       ("SAGA-Muon joint, c = 1/4", dict(estimator="saga", gamma=over("R2_op", 0.25))),
                       ("SAGA, c = 1/4", dict(estimator="saga", placement="euclid", gamma=over("R2", 0.25)))):
        v = []
        for seed in (0, 1, 2):
            r = run(s, "vr", cfg, 40, seed, stop_rel=1e-8)
            v.append(r["passes_to"]["1e-06"])
        print(f"   {label:28s} {v}  mediane {np.median(v):.1f}", flush=True)

elif part == "R3":
    summ = json.load(open(os.path.join(ROOT, "muon", "results", "M9_summary.json")))
    for key, spec, b in (("ls-cond", LS_COND, 16), ("soft-cond", SOFT_COND, 32)):
        cell = summ["A"][key]["cells"][str(b)]
        cmp_ = cell["compare"]
        out = {}
        for role in ("spectral", "control"):
            row = cell["rows"][cmp_[role]["family"]]
            cfg = dict(estimator="saga", batch=b, length="const", orth=row["orth"], **row["best"])
            if row["nesterov"]:
                cfg.update(beta=0.9, nesterov=True)
            v = [run(spec, "vr", cfg, 300, seed, stop_rel=1e-10)["passes_to"]["1e-06"] for seed in range(10, 20)]
            out[role] = float(np.median(v))
            print(f"   {key} b={b} {role:8s} {cmp_[role]['family']:40s} {row['best']}  mediane {out[role]:.1f} "
                  f"({int(np.isfinite(v).sum())}/10)  [resume : {cmp_[role]['median']:.1f}]", flush=True)
        print(f"   {key} b={b} : rapport rejoue {out['spectral'] / out['control']:.3f}  "
              f"[resume {cmp_['ratios']['spectral/control']:.3f}]", flush=True)

elif part == "R4":
    print("R4 : longueur totale du refroidissement des temoins sans table, sum_t lr_t = lr_0 / (1 - q^(1/periode))")
    for key, spec, (q, lr0) in (("soft-cond", SOFT_COND, (0.7, 0.3)), ("ls-cond", LS_COND, (0.75, 0.3))):
        prob = C.problem(spec)
        wop = float(np.linalg.norm(prob.W_star, 2))
        n = spec["n"]
        print(f" == {key} : q = {q}, lr_0 = {lr0}, ||W_*||_op = {wop:.2f}")
        for b in (1, 32):
            for name, per in (("gradient de lot", n / b), ("MARS-M / STORM (2b par pas)", n / (2.0 * b))):
                path = lr0 / (1.0 - q ** (1.0 / per))
                print(f"    b = {b:2d} {name:28s} sum lr_t = {path:7.2f} = {path / wop:5.2f} ||W_*||_op"
                      + ("   <-- W_* HORS D'ATTEINTE quel que soit l'estimateur" if path < wop else ""))
        # la table elle-meme, a b = 32, sur le refroidissement des temoins
        for label, cfg in (("SAGA + table, ns5 + Nesterov, b = 32", recipe("saga", 32, "ns5", q, lr0, True)),
                           ("SAGA + table, polaire exact, b = 32", recipe("saga", 32, "svd", q, lr0, False)),
                           ("sans table, ns5 + Nesterov, b = 32", recipe("sgd", 32, "ns5", q, lr0, True))):
            v = [run(spec, "vr", cfg, 300, seed)["final_rel"] for seed in (10, 11, 12)]
            print(f"    {label:40s} distance relative finale {np.round(v, 4)}", flush=True)

elif part == "R4b":
    print("R4b : gradient de lot b = 1 SANS table (ns5 + Nesterov 0.9), 300 passes, graine 10, refroidissements "
          "plus lents que celui de la recette avec table")
    for key, spec in (("ls-cond", LS_COND), ("soft-cond", SOFT_COND)):
        for q, lr0 in ((0.75, 0.3), (0.9, 0.3), (0.95, 0.3), (0.98, 0.3), (0.98, 0.1)):
            r = run(spec, "vr", recipe("sgd", 1, "ns5", q, lr0, True), 300, 10)
            print(f"    {key} q = {q:4.2f}, lr_0 = {lr0}: final_rel {r['final_rel']:.3e}  min_rel {r['min_rel']:.3e} "
                  f"[{r['_wall']:.0f}s]", flush=True)

if part == "R2b":
    # (M2) : "SAGA: 20.2" est SAGA au pas du livre gamma = 1/(4 R^2). Meme instance, memes graines (0, 1, 2), pas
    # regle par le protocole du papier dans M1 (soft-mem : gamma = 0.5 / R^2), et pas voisins.
    s = C.spec("softmax", 1000, 50, 10, 1e3, 1.0)
    print("R2b : softmax n = 1000, kappa = n ; SAGA euclidien, passes a 1e-6 selon le pas (graines 0, 1, 2)")
    for c in (0.25, 0.35, 0.5, 0.7, 1.0):
        v = [run(s, "vr", dict(estimator="saga", placement="euclid", gamma=over("R2", c)), 40, seed,
                 stop_rel=1e-8)["passes_to"]["1e-06"] for seed in (0, 1, 2)]
        print(f"   SAGA, gamma = {c:4.2f} / R^2 : {v}  mediane {np.median(v):.1f}", flush=True)
    for c in (0.0625, 0.125, 0.25, 1.0):
        v = [run(s, "vr", dict(estimator="saga", gamma=over("R2_op", c)), 40, seed,
                 stop_rel=1e-8)["passes_to"]["1e-06"] for seed in (0, 1, 2)]
        print(f"   SAGA-Muon joint, gamma = {c:6.4f} / R2_op : {v}  mediane {np.median(v):.1f}", flush=True)
