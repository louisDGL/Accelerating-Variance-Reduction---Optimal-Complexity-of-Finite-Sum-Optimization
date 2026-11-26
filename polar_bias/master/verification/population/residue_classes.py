"""Lattice effect at order 1/v: exact b*m for the lattice population {-1,2}^3 along the residue classes of b mod 3.
(The batch sum lives on -b(1,1,1) + 3 Z^3, so 0 is an atom iff 3 | b.)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from median_r3 import exact_law_lattice, law_to_points, geomed
gst = 0.1
print("    n     b   b%3      v      b*m (3 coords equal)   -(n-2b)/(n-2) g*     diff     diff*v    h*w0/J (atom at 0)")
# (a third group n=432, b=214..218 near half sampling needs a 5 GB array and ~10 min per case; not run)
for n, bs in [(216, [52, 53, 54, 55, 56]), (432, [106, 107, 108, 109, 110])]:
    for b in bs:
        P = exact_law_lattice(n, b); X, w = law_to_points(P, b)
        m, gn, _ = geomed(X, w)
        assert np.ptp(m) < 1e-9
        v = b * (n - b) / n; pred = -(n - 2 * b) / (n - 2) * gst
        i0 = np.argmin(np.linalg.norm(X, axis=1)); w0 = w[i0] if np.linalg.norm(X[i0]) < 1e-12 else 0.0
        print(f"{n:5d} {b:5d} {b % 3:4d} {v:8.2f}   {b * m[0]:+.6f}   (|grad| {gn:.0e})    {pred:+.6f}      {b * m[0] - pred:+.6f}  {(b * m[0] - pred) * v:+.3f}     {math.sqrt(v) * w0 / 0.3761264:.4f}")
