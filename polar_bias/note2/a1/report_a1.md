# Report A1 — q_{K,d} > (K-1)(d-1)/(2(K+d+1)), i.e. a_1 < 0

The statement, its reformulations and the proof are written up as Theorem "a_1 < 0" and the remark that follows it
in `../paper/discrete_bias.pdf` (Section 8). This file records how to reproduce the computer-assisted steps.

Run from this folder with `../../.venv/bin/python`:

| script | role |
|---|---|
| `wish.py` | exact / high-precision `E tr W^s` for real `s` (de Bruijn Pfaffian formula) |
| `fitmom.py`, `exactmom.py` | polynomials `E tr W^k`, `k <= 8` (stored in `mom_polys.pkl`) |
| `final_checks.py` | `NUM(K,d)`, positivity of shifted coefficients, K = 2 closed form, non-concavity of `r` |
| `finite.py` | exact rational Hermite certificates for the 52 remaining sizes (smallest bound 0.0323 at (3,21)) |
| `scan.py`, `taylor.py`, `cert.py`, `positivity.py` | exploration (which moment orders suffice) |

Independent re-verification (different code, Wick enumeration of pairings): `../checks/v01_wick_moments.py 8` then
`../checks/v02_a1_proof.py`; output in `../results/v02_a1.log`.

Status: PROVED for all K, d >= 2 (K = 2 by hand; K >= 3 computer-assisted, exact integer arithmetic).
Not proved: concavity of `r(p) = E tr W^{(p+2)/2} / E tr W^{p/2}` on [0,4] (numerical, 74 sizes);
`(K-1)/2 - q > (K-1)(K+2)/(16 d)` (numerical). Disproved: concavity of `r` on (0, infinity).
