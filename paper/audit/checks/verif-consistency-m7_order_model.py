"""Counter-check of finding C-04 (dimension consistency): what the identical-Hessian model says on the gain of a fresh
permutation per pass, at the batch sizes where (M7) quotes it. Reads muon/results/M9_summary.json (part D) and
M9b_summary.json (parts D and L); writes nothing.
    PYTHONDONTWRITEBYTECODE=1 python3 paper/audit/checks/verif-consistency-m7_order_model.py   (from the project root)
"""
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ld = lambda f: json.load(open(os.path.join(ROOT, "muon", "results", f)))                          # noqa: E731
m9, m9b = ld("M9_summary.json"), ld("M9b_summary.json")
blocks = {"soft-mem": m9["D"]["soft-mem"], "ls-mem": m9["D"]["ls-mem"], "model": m9b["D"]}
med = {}
print("rho_star (smallest factor of lr^2 per pass followed by >= 4 of 5 seeds) and median passes to 1e-9 at the "
      "fastest rho where every seed reaches it")
for inst, blk in blocks.items():
    for b in ("1", "5", "20", "50"):
        sc = blk["cells"][b]["schemes"]
        i, r = sc["iid"], sc["reshuffle"]
        med[(inst, b)] = (i["fastest"]["median"], r["fastest"]["median"])
        print(f"  {inst:8s} b = {b:>2s}: rho_star {i['rho_star']:5g} -> {r['rho_star']:5g} "
              f"({math.log(1 / i['rho_star']):.2f} -> {math.log(1 / r['rho_star']):.2f} nats); passes "
              f"{i['fastest']['median']:5.1f} [{i['fastest']['min']:.1f}, {i['fastest']['max']:.1f}] -> "
              f"{r['fastest']['median']:5.1f} [{r['fastest']['min']:.1f}, {r['fastest']['max']:.1f}] "
              f"({100 * (r['fastest']['median'] / i['fastest']['median'] - 1):+.0f}%)")
print("passes of the instances over those of the model (independent / permutation)")
for inst in ("soft-mem", "ls-mem"):
    for b in ("1", "5", "20", "50"):
        print(f"  {inst:8s} b = {b:>2s}: {med[(inst, b)][0] / med[('model', b)][0]:.2f} / "
              f"{med[(inst, b)][1] / med[('model', b)][1]:.2f}")
print("independent batches with m extra refreshes per step: rho_star next to the memory factor (1 - (b+m)/n)^(n/b)")
for inst, blk in blocks.items():
    for b in ("20", "50"):
        row = [(sc["memory_factor"], sc["rho_star"]) for name, sc in blk["cells"][b]["schemes"].items()
               if sc["order"] == "iid"]
        print(f"  {inst:8s} b = {b}: " + "; ".join(f"memory {m:.3f}, rho_star {r:g}" for m, r in row))
print("model, long runs at batch one (part L): largest x = log(1/rho) followed after 80 passes")
for nk, c in m9b["L"]["cells"].items():
    print(f"  {nk:8s}: " + ", ".join(f"{o} {v['level_end']['x_star']}" for o, v in c["orders"].items()))
