"""Cross-checks of the exact constants against the Monte-Carlo values of results/c02_constants.json and the closed form for 2x2."""
from common import *
rows = {(r['K'], r['d']): r for r in map(json.loads, open(os.path.join(DATA, 'fig_constants_rows.jsonl')))}
mc = {(r['K'], r['d']): r['q'] for r in json.load(open(os.path.join(RES, 'checks', 'c02_constants.json')))}
print('q22 exact formula (3 sqrt2 - 2)/7 =', (3 * 2 ** .5 - 2) / 7, ' wish:', rows[(2, 2)]['q'])
for s in sorted(rows):
    r = rows[s]; ok = r['lower'] < r['a1_plus_a2'] < r['upper'] and r['a1'] < 0 < r['a2']
    print(s, f"q={r['q']:.8f}", f"MC {mc[s]:.4f}" if s in mc else '', f"scaled={r['scaled']:.5f}", 'bounds+signs OK' if ok else 'VIOLATION', f"dps={r['dps']} chk={r['identity_check']:.0e} {r['seconds']:.0f}s")
