# theory/bias_law/notes — discrete noise, the rank-deficient variety, and the sign of a_1

Second investigation (2026-11-25), sequel to `theory/bias_law/notes/polar_bias.pdf` (not modified).

| Path | Content |
|---|---|
| `paper/discrete_bias.tex`, `.pdf` | the note: theorems, proofs, exact computations, originality assessment |
| `research_log.md` | failed routes, intermediate discoveries, conjectures, directions not pursued |
| `literature_report2.md` | literature search (sources, what was read in the original and what was not) |
| `audit_report.md`, `audit_checks.py` | independent audit of the first note |
| `referee_report2.md` | independent referee pass on the new note |
| `a1/` | computer-assisted proof of `a_1 < 0` (see `a1/report_a1.md`) |
| `checks/e01_exact_lattice.py` | exact medians for lattice noise (sign, 1x2, 1x3, 2x2, 2x3) |
| `checks/e02_lattice_term.py` | predicted `b^{-3/2}` lattice term in R^2 |
| `checks/v01_wick_moments.py`, `v02_a1_proof.py` | independent verification of the `a_1` proof |
| `results/` | logs of the runs quoted in the note |

Run (from `checks/`): `../../.venv/bin/python e01_exact_lattice.py {sign|1x2|1x3|2x2|2x3} [b1,b2,...] [truncation]`.
