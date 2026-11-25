"""Counter-check of finding G2 (num-geometry), orthogonalized side: does a step grid of ratio 2^(1/4) also speed up the
best method recomputing the polar factor on the third softmax draw (soft SAGA-Muon, eps_innov = 0.01, gamma =
0.5 / R2_op in M8: 27.45 passes), enough to restore the ratio 0.20 against SAGA at a resolved step (81.4 passes)?

Uses the repository's code (common.run_one, family 'soft SAGA-Muon' of families.py: estimator='saga', orth='soft').
Mini-protocol: steps 0.5 * 2^(k/4) / R2_op, k = -3..3, eps_innov fixed at the value selected by M8 ; selection by the
median on tuning seeds 0-4, confirmation on seeds 10-19. Read-only on the repository.

Run: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python3 paper/audit/checks/verif-num-geometry_soft_fine.py [data_seed] [eps_innov] [c_centre]
"""
import os
import statistics as st
import sys
import time

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'muon', 'experiments'))
sys.path.insert(0, os.path.join(ROOT, 'muon'))
import common as C  # noqa: E402

data_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
centre = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
TOL = 1e-6
Q = 2 ** 0.25
spec = C.spec('softmax', 200, 40, 8, 1e4, 2.0, seed=data_seed)


def run(c, seed, budget=100):
    r = C.run_one({'spec': spec, 'method': 'vr',
                   'cfg': dict(estimator='saga', orth='soft', eps_innov=eps, gamma=C.over('R2_op', c)),
                   'passes': budget, 'seed': seed, 'stop_rel': 0.5 * TOL})
    return r['passes_to']['1e-06']


t0 = time.process_time()
print('soft SAGA-Muon, softmax n=200 data seed %d, eps_innov = %g, passes to 1e-6' % (data_seed, eps))
tune = {}
for k in range(-3, 4):
    c = centre * Q ** k
    p = [run(c, s) for s in (0, 1, 2, 3, 4)]
    tune[k] = st.median(p)
    print('   gamma = %.4f/R2_op %s tuning seeds 0-4: median %s  [%s]' % (
        c, '(M8 point)' if k == 0 else '          ', tune[k], ', '.join('%.1f' % v for v in p)), flush=True)
kb = min(tune, key=tune.get)
for label, k in (('M8 point', 0), ('2^(1/4) grid', kb)):
    c = centre * Q ** k
    p = [run(c, s) for s in range(10, 20)]
    print('confirmation seeds 10-19, %-12s gamma = %.4f/R2_op : median %.2f  [%.1f, %.1f]  reached %d/10' % (
        label, c, st.median(p), min(p), max(p), sum(v < float('inf') for v in p)), flush=True)
print('(CPU %.0f s)' % (time.process_time() - t0))
