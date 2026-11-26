# polar_bias/ — the first-order bias law of stochastic orthogonalized steps

(This folder was developed under the working name `Paul/`; it was renamed when it was integrated in the repository.)

Research note extending `paper/paper_saga_to_muon.tex` (Finding 2: through the polar map, noise moves the fixed point).
Written 2026-11-25. Everything of this direction lives in this folder; nothing outside it was modified.

## Later material (same folder)

| Path | Content |
|---|---|
| `note2/` | second note, `note2/paper/discrete_bias.pdf`: discrete (finite-sum) noise, the rank-deficient variety, computer-assisted proof of `a_1 < 0` (`note2/a1/`, including `wish.py`, exact Wishart moments); see `note2/README.md` |
| `master/` | master theorem document `master/master.pdf` (statements in `shared/`, one proof file per theorem in `proofs/`, independent verification scripts and referee reports in `verification/`) |
| `papers/icml/twomoment.tex`, `.pdf` | paper draft "Where Does Muon Converge? The Two-Moment Law of the Bias of Orthogonalized Stochastic Steps" |
| `papers/colt/discrete.tex`, `.pdf` | paper draft "Edgeworth without Cramér: the bias of orthogonalized stochastic steps under discrete noise" |
| `papers/experiments/` | scripts of the figures of the two drafts (`figA_*`, `figB_*`, `digits_*`, `figC_plot.py`, `make_numbers.py` for the two-moment paper; `exp_*`, `make_figures_discrete.py` for the discrete one) |
| `papers/figures/` | figures (PDF + PNG), plotted data (`data/`), and `NUMBERS_twomoment.md`, `NUMBERS_discrete.md`: every quoted number with its Monte-Carlo error, including what did not behave as predicted |

The drafts in `papers/` and some scripts of `papers/experiments/` were still being edited when this snapshot was taken.

## Direction chosen

The manuscript proves that stochastic polar steps converge to a point whose gradient is minus a *nuclear median* of the
noise, and gives a *formal* `1/b` expansion of that offset for i.i.d. entries, with a constant `a_{K,d}/c_{K,d}` known
only by Monte Carlo. Of the three candidate directions (general bias theory, momentum, matrix geometry), the one with the
most structure to find turned out to be the intersection of A and C: **the bias operator can be computed exactly**. The
note is built around that theorem, with the general expansion as its foundation and momentum, parameter-dependent noise
and sampling without replacement as extensions.

## Results (`paper/polar_bias.pdf`, 11 pages, proofs included)

1. **Cornish-Fisher law for norm medians (Theorem 4).** For any norm and any noise law satisfying Cramér's condition,
   the minimizers of the noise-smoothed dual norm are at `g*/b + o(1/b)`, `g* = -(1/6) J^{-1} T[K_3]`, with `J`, `T` the
   second and fourth derivatives at 0 of the Gaussian-smoothed dual norm. This makes the formal statement of the
   repository rigorous, for sign, normalized and polar steps at once.
2. **Two-moment theorem (Theorem 6), the central result.** For isotropic covariance and polar steps, the offset is
   `(a_1 E[z ||z||_F^2] + a_2 E[z z^T z]) / (6 sigma^2 b)`: of the `C(Kd+2, 3)` components of the third cumulant, only
   two cubic matrix moments matter. `a_1`, `a_2` are explicit rational functions of `K`, `d` and of one number
   `q_{K,d} = E sum_{i!=j} s_i s_j/(s_i+s_j) / E sum_i s_i` of a Gaussian matrix. Consequences:
   - the constant of the manuscript is `a/c = 3(3+2q)/((K+2)(d+2))`, with `9/((K+2)(d+2)) < a/c < 3/(max(K,d)+2)` for
     every size: the "factor of order max(K,d)" below sign descent is now a theorem with its constant;
   - in the proportional limit `max(K,d) a/c -> theta(gamma)` in `[12/5, 3]` (`12/5` for square matrices);
   - the fourth derivative of the smoothed nuclear norm is negative in every direction;
   - for rank-one per-sample noise, orthogonalization *multiplies* the offset of Frobenius normalization by a factor in
     `(1, min(K,d))`;
   - neither sign descent nor polar steps dominate: explicit noise laws that bias one and not the other.
3. **Parameter-dependent noise (Theorem 12).** All zeros of the mean field near a nondegenerate critical point are at
   `W* + H^{-1} g*/b + o(1/b)`. The formula predicts, with no fitted parameter, the plateau measured in Figure 3(c) of
   the manuscript: 0.314 predicted, 0.308 measured at `b = 32`.
4. **Momentum (Theorems 15 and 16).** Polar steps on an exponential moving average, with fixed `beta` and Robbins-Monro
   lengths, converge almost surely to minus a nuclear median of the stationary filtered noise; its first-order bias is
   that of batch size `b (1+beta+beta^2)/(1-beta^2)`, not the variance-based `b (1+beta)/(1-beta)`.
5. **Sampling without replacement (Proposition 18).** Exact complement duality `Med(xi_{n-b}) = -(b/(n-b)) Med(xi_b)`;
   the offset vanishes at `b = n/2` and changes sign beyond.

What is *not* proved is listed in Section 9 of the note and in `novelty_ledger.md` (sign of `a_1` for all sizes, discrete
finite-sum noise, `beta -> 1` at fixed `b`, anisotropic closed forms; three quoted theorems were not read in their
original source).

## Files

| Path | Content |
|---|---|
| `paper/polar_bias.tex`, `.pdf` | the note: statements, proofs, tables |
| `novelty_ledger.md` | what was already in the repository, closest published results, what is added, limits of the search |
| `literature_agent_report.md` | raw report of the literature search (sources read, queries, verdicts) |
| `checks/lib.py` | polar factor, zeros of mean fields with common random numbers, constants, general predictor |
| `checks/c01_two_constants.py` | invariant tensors, Gram matrix, Monte-Carlo check of the two-invariant structure |
| `checks/c02_constants_q.py` | table of `q`, `a_1`, `a_2`; Stein identity; Marchenko-Pastur limit |
| `checks/c03_general_law.py` | first-order law on five noise laws (i.i.d., rank-one, co-skewed, one factor, Kronecker) |
| `checks/c04_momentum.py` | effective batch with momentum (frozen iterate) and coupled dynamics |
| `checks/c05_without_replacement.py` | complement duality and sign change at `b = n/2` |
| `checks/c06_state_dependent.py` | population least squares with parameter-dependent noise |
| `checks/c07_paper_instance.py` | prediction of the plateau of Figure 3(c) of the manuscript |
| `checks/review_verify.py` | independent re-derivation of the algebra and constants of Theorem 6 (referee pass) |
| `results/*.log`, `*.json` | outputs of the runs quoted in the note (`c03_a..g`: one law or batch range each) |

## Running

```bash
python3 -m venv polar_bias/.venv && polar_bias/.venv/bin/pip install numpy scipy sympy mpmath matplotlib scikit-learn
cd polar_bias/checks
../.venv/bin/python c01_two_constants.py mc      # 2 min
../.venv/bin/python c02_constants_q.py           # 5 min
../.venv/bin/python c03_general_law.py iid       # laws: iid rank1 coskew factor kron; 5-10 min each
BS='[(256,1_000_000),(1024,600_000)]' TAG=_large ../.venv/bin/python c03_general_law.py kron
../.venv/bin/python c04_momentum.py frozen       # then: coupled (about 10 min)
../.venv/bin/python c05_without_replacement.py
../.venv/bin/python c06_state_dependent.py
../.venv/bin/python c07_paper_instance.py
cd ../paper && pdflatex polar_bias && pdflatex polar_bias
```

Monte-Carlo zeros of mean fields carry errors of 1-10% at the sample sizes used (stated next to each number in the logs).
