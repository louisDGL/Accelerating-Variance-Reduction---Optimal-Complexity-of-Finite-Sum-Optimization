"""Remove the entries that stopped on 'max_time' from a cached kappa sweep so that re-running the experiment script
(incremental) recomputes them, e.g. with a larger budget: VRLAB_MAX_TIME=2400 python experiments/exp_A4b_instances.py.
Usage: python experiments/retry_timeouts.py A4 [A4b ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

for tag in sys.argv[1:]:
    res = common.load_result(tag)
    if res is None:
        print(tag, "no cache")
        continue
    bad = [k for k, s in res["status"].items() if s == "max_time"]
    for k in bad:
        res["grads"].pop(k, None)
        res["status"].pop(k, None)
    common.save_result(tag, res)
    print(f"{tag}: removed {len(bad)} max_time entries: {bad}")
