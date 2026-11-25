"""consistency (a)/(d): the repeated numbers of Finding 1 (3.2-220; 3.2-3.7; 19-220; 77-96%; 39-44%; 32-37%; 0.6%)
from results/B3_summary.json, and their sensitivity to the order in which the inequalities are undone
(field shares_orderA, the alternative order stored by the experiment). Read-only.
Usage (from project root): python3 paper/audit/checks/consistency_b3_gap.py
"""
import json
b3 = json.load(open('results/B3_summary.json'))
A = b3['attribution']
rows = []
for key, v in A.items():
    kap = float(key.split(',')[0].split('=')[1]); al = int(key.split('=')[-1])
    rows.append((kap, al, v))
rows.sort(key=lambda r: (r[0], r[1]))
print(f"{'kappa':>8s} a  speedup |  order of the paper: I1    I2    I3    I4    I5   I2+I4 | other order (A): I1    I2    I4    I5   I2+I4")
for kap, al, v in rows:
    s, sa = v['shares'], v['shares_orderA']
    print(f"{kap:8.0f} {al}  {v['speedup']:7.2f} | {s['I1']:22.3f} {s['I2']:5.3f} {s['I3']:5.3f} {s['I4']:5.3f} {s['I5']:5.3f} {s['I2']+s['I4']:6.3f} |"
          f" {sa['I1']:19.3f} {sa['I2']:5.3f} {sa['I4']:5.3f} {sa['I5']:5.3f} {sa['I2']+sa['I4']:6.3f}")
sp = [v['speedup'] for _, _, v in rows]
print('speedup: min %.2f max %.2f' % (min(sp), max(sp)))
lo = [v['speedup'] for k, a, v in rows if k <= 100]; hi = [v['speedup'] for k, a, v in rows if k == 1e5]
print('kappa<=1e2: %.2f-%.2f ; kappa=1e5: %.1f-%.1f' % (min(lo), max(lo), min(hi), max(hi)))
big = [(v['shares']['I2'] + v['shares']['I4'], v['shares_orderA']['I2'] + v['shares_orderA']['I4']) for k, a, v in rows if k >= 1e4]
print('kappa>=1e4, I2+I4: paper order %.3f-%.3f ; other order %.3f-%.3f' % (min(b[0] for b in big), max(b[0] for b in big), min(b[1] for b in big), max(b[1] for b in big)))
small = [(v['shares'], v['shares_orderA']) for k, a, v in rows if k <= 100]
for nm in ('I1', 'I2', 'I5'):
    print(f"kappa<=1e2, {nm}: paper order %.3f-%.3f ; other order %.3f-%.3f" % (min(s[0][nm] for s in small), max(s[0][nm] for s in small), min(s[1][nm] for s in small), max(s[1][nm] for s in small)))
print('max I3 share (paper order): %.4f' % max(v['shares']['I3'] for _, _, v in rows))
