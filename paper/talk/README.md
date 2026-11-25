# Figures for a lecture on the paper

Explanatory figures for a lecture-style presentation of `paper/paper_saga_to_muon.pdf`. They are not in the paper
(five companion figures for the paper itself are in `paper/figures/figE*` and in the optional appendix, see
`paper/README.md`). Each figure carries one message, is drawn at its final size for a 16:9 slide, and most are
revealed in several states.

| Path | Content |
|---|---|
| `figures/figT*.pdf`, `.png` | the figures (vector PDF for LaTeX, PNG at 300 dpi for other slide tools); `name_s1`, `name_s2`, ... are the states of a progressive build, `name` is the complete figure |
| `talk_figures.tex`, `.pdf` | a catalogue on 16:9 beamer slides: one frame per figure in the order of the lecture, the frame title is the message, builds are overlays; the four figures of the paper are included at their place. It is a set of slides to pick from, not a finished talk |
| `../scripts/make_figT*.py` | one script per figure (a few scripts draw two); `../scripts/explain_style.py` is the shared style |

## Using a figure

* Size: at most 5.90 x 2.75 in, to be placed without scaling on a 160 mm x 90 mm slide (beamer:
  `\documentclass[aspectratio=169,11pt]{beamer}`, then `\includegraphics{figT05_bound_vs_run}`); the smallest text is
  8 pt at that size. The background is the surface colour of the project's figures (`#FCFCFB`); `talk_figures.tex`
  sets the slide background to the same colour.
* Builds: `\includegraphics<1>{name_s1}`, `\includegraphics<2>{name_s2}`, ... on one frame (the states superpose
  exactly); the macro `\buildframe{title}{name}{N}` of `talk_figures.tex` does it.
* Each figure has a scope line at its bottom left (instance, seeds, grid, what is proved): keep it on the slide.
* Colours follow the entity across the set: orange = a SAGA table inside polar steps, the polar (spectral) map, a
  fresh permutation per pass; blue = no table (whatever the length rule), Euclidean SAGA, independent sampling;
  grey = controls and context (exact gradient, control maps; the map of `figT17` uses a neutral ramp); red / yellow =
  the certified and the a-posteriori bound of Figure 1; the colours of I1-I5 are those of Figure 1(c). Text is always
  in ink. Two things to know: SGD keeps the red of the project's figures in `figT02`, and the two figures of the
  paper's appendix that sit next to Figure 4 (`figE3` (b), `figE5`) keep its grey for runs without a table.
* Notation: the minimizer is written with an asterisk, as in the text of the paper (the four figures of the paper use
  a star); `eta` is a step length except in `eta_*`, the minimizer of the vector problems, as in the paper.

Rebuild a figure from the project root (single-threaded BLAS, as everywhere in the project), then the catalogue:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figT10_sign_median.py
cd paper/talk && pdflatex talk_figures && pdflatex talk_figures
```

Every script states in its docstring what the figure shows, its toy or its data (file and keys), the status of each
element (proved, measured, illustration) and its build states; it prints every number it draws and stops if a number
it relies on disagrees with the repository or with the paper. No script takes more than a minute.

## The figures, in the order of the lecture

Thirty-six figures (and four variants of the roadmap), numbered in the order of the paper. The catalogue follows
this order with one exception: `figT23` (ages of the table) is shown before `figT19`, whose statement uses the ages.

Kind: **toy** = an exact low-dimensional instance of the statement, computed by the script; **data** = results of
the repository, redrawn; **new** = a small new computation with the project code (stated on the figure);
**diagram** = objects and formulas of the paper laid out.

### The objects

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT01_roadmap` (3 states; `_r1` ... `_r4`: one row in focus) | The rate of SAGA rests on four ingredients; a Muon-type step changes each of them | Table 2 (corrected wording), plan of the lecture | diagram |
| `figT02_saga_table` (3) | SGD and SAGA are both unbiased; only SAGA's error vanishes at the minimizer | Section 2.1; the two properties of the estimator | toy: ridge logistic regression in the plane, n = 50 |
| `figT03_polar_factor` (3) | The polar factor keeps the singular directions and sets every singular value to one; it is the steepest step for the spectral norm; ns5 approximates it | Section 2.1, F1 | toy: one 2 x 2 and one 8 x 40 matrix |
| `figT04_recipe_pipeline` (3) | SAGA, Muon and the recipe (4) on one skeleton: estimator, map, step | Sections 2.1, 2.3, Eq. (4) | diagram (inset computed) |

### Finding 1: the geometry

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT05_bound_vs_run` (3) | At kappa = 1e4 SAGA gains 11.4 orders of magnitude in 100 passes, its certificate a factor 0.15 | (C1), (C3); Figure 1(a) as a build | new: 5 runs with `vrlab` |
| `figT06_proof_chain` (3) | The certified rate is an exact identity followed by five inequalities; two of them insert global constants | Section 2.2, Eq. (1), (C2) | diagram + data (B3) |
| `figT06b_gap_shares` (2) | Shares of the logarithmic gap in the two orders of attribution: about nine tenths for I2 + I4 at kappa >= 1e4 on spectrum j^-1, order-dependent at kappa = 1e2 | (C2), limitations | data (B3) |
| `figT06c_curvature_far` (3) | The global constant lambda is approached only far away, where no run goes | (C3), hypothesis of Theorem 1 | toy: ridge logistic regression in the plane, n = 100, kappa = 1e4 |
| `figT07_two_identities` (4) | Along runs: exact contraction = certified x a slack below 8/3 x the curvature met | Eq. (1), (C2) | new (same runs as T05) + data (B3) |
| `figT07b_polar_identity` (2) | The same anatomy for the nuclear-scaled polar step, on a quadratic | Eq. (2), "the same anatomy on quadratics" | new: exact gradients on the least-squares instance |
| `figT08_worst_case` (3) | On its worst case the certificate has the right order; on the logistic instance SAGA converges at gamma R^2 = 2 and stalls at 4 | (C4), corrected | toy (exact second moments) + data (GammaDelta) + rerun |
| `figT09_nfl_plane` (4) | Theorem 1 in the plane: a norm reweights directions; it cannot change the range of curvature of one direction | Theorem 1 | toy of `figE2` |
| `figT09b_nfl_norms` (3) | Condition numbers in five norms and a scan: the global one never improves, the local one can; on a quadratic loss the matched norm gives one | Theorem 1, (C5), (M4) | toy of `figE2` |
| `figT09c_nfl_bands` (4) | Theorem 1 over all directions: a norm moves the waviness of the band from one edge to the other and keeps its height | Theorem 1, "a norm can only trade smoothness against strong convexity" | toy of `figT06c` and its least-squares twin |

### Finding 2: the estimator

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT10_sign_median` (4) | A sign step on a sampled gradient computes a median; a SAGA table brings the mean back | Theorem 2(b), Proposition 3(c)-(d) for K = d = 1 | toy: n = 201 scalars |
| `figT11_polar_median` (4) | In the plane, polar steps go to the geometric median of the data; with a table, to the minimizer | Theorem 2, Proposition 3 for K x d = 2 x 1 | toy of `figE3` |
| `figT12_offset_scaling` (3) | The offset of a minibatch is of order 1/b: exact in dimension one, formal for i.i.d. entries, measured on the finite sum | Proposition 4, Figure 3(c) | toy + new Monte Carlo + data |
| `figT13_two_floors` (4) | Two floors, one from the length and one from the noise: only a vanishing length with a table removes both (relative squared distances) | (M1), Figure 3(a) | data (the runs of Figure 3(a)) |
| `figT13b_floor_vs_length` | At a constant length a table changes the order of the floor (length squared against length), not its existence (absolute squared distances: the factor to `figT13` is on the scope line) | (M1) | data (the runs of Figure 3(a)) + new runs |
| `figT14_estimator_error` (3) | With a table the error of the estimator vanishes with the iterate; without one it stays at its variance at the minimizer | Proposition 3(c), (M5) | new: exact enumeration along a cached run |

### Fixed lengths on a cooldown, fed by a table (Section 3.3)

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT14b_cooldown_toy` (4) | One schedule, three estimators, same seeds: with a table the iterate follows the cooldown, without one it freezes | (M5), Proposition 5(c) | toy: 200 planar points |
| `figT15_schedule_law` (3) | At batch one, with a table the squared distance follows lr_t^2 down to the floor of the map | (M5), Figure 4(a) | data (M9, part C) |
| `figT15b_schedule_band` (2) | The same runs divided by lr_t^2: a flat band until the floor | (M5), Proposition 5(c) | data (M9, part C) |
| `figT16_same_schedule` (2) | Same map, momentum, schedule and seeds: only the estimator differs | (M5), corrected | data (M9, part C) |
| `figT17_cooldown_map` (2) | Two ways for a cooldown to fail: too short a path, or too fast (a cliff in q) | (M5), limitations | data (M9, part B) |
| `figT17b_path_budget` | A control without a table tests the estimator only if its path could reach the minimizer: yes at batch one, no at b = 32 | (M5), corrected | data (M9, part C) + new runs |
| `figT18_ns5_scale` (3) | Newton-Schulz has a scale: its constant eps sets a floor near 1e-18 | (M5), Proposition 5(c) | toy + data (M9, part C) |
| `figT19_one_step_map` (3) | One inequality behind Proposition 5(b): a floor at a constant length, an invariant interval on a slow cooldown | Proposition 5(b) | exact plots of the map of the proof |
| `figT19b_certified_cooldown` (3) | Proposition 5(b) in action: the certified cooldown, its bound and the runs under it | Proposition 5(b) | toy of `figT14b` |

### Batch size and memory

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT20_estimator_rank` (3) | At batch one the estimator is close to rank one; with 16 or 32 samples it is not | (M6), (M2), F1 | new diagnostic + data (M9, part A) |
| `figT21_minibatch_ratios` (3) | Batches of 16 and 32: passes of the spectral variant against the controls and against (B-)SAGA, cell by cell, with the resolution of the grid | (M6), Figure 4(b) | data (M9, part A) |
| `figT22_grid_cliff` (3) | Why every ratio comes with its tuning grid: the fast steps end at a cliff | (M3), (M6), corrected | data (M5b) |
| `figT23_memory_ages` (3) | An entry keeps its error until it is refreshed: ages of the table under three sampling orders | Proposition 5(b), (M7) | toy: exact laws and one draw |
| `figT24b_expectation_vs_runs` (3) | Linear SAGA: one nat per pass is the rate of the mean, not of a run | (M7), corrected | data (B4) + exact recursion |
| `figT24_order_rates` (3) | Sampling order: what is proved for linear SAGA, what the instances gain | (M7), Figure 4(c) | exact rates + data (B4, M9, M9b) |

### What is proved, what is run

| Figure | Message | Explains | Kind |
|---|---|---|---|
| `figT25_proofs_and_runs` (2) | Each certificate holds in a regime at a measured distance from the runs, row by row | Section 4.1 | data + constants of the paper and of the audit |

## What the figures do not say

The scope lines carry the hypotheses; a few points must also be said aloud.

* The toys in dimension one and two (`figT06c`, `figT09*`, `figT10`, `figT11`, `figT14b`, `figT19b`) are exact
  instances of the statements, where the polar factor of a vector is its normalization: their numbers are toy
  numbers, not results of the paper, and they are not evidence about the spectral map.
* Theorem 1 is drawn on two different planar toys: `figT09`, `figT09b` (n = 40, kappa_F = 9, the toy of `figE2`) and
  `figT06c`, `figT09c` (n = 100, kappa_F = 1670). Say it when the slides follow each other.
* `figT20` (a) is a new diagnostic on one instance; rank and passes are juxtaposed, not linked by a proof. Its panel
  (b) shows the polar factor without momentum, whereas `figT21` uses the fastest spectral variant of each cell.
* `L_eff` has two meanings, as in the paper: the cocoercivity met along runs of SAGA (`figT05`-`figT07`) and the
  constant `L_op + 10 A L_max` of Proposition 5(b) (`figT19`, `figT19b`).
* "Minibatch gradient" is the paper's name for the estimator without a table, also at batch one (`figT15`-`figT17b`).
* The shares of `figT06`, `figT06b` are fractions of a difference of exponents on passes 2-20 and depend on the order
  of attribution, which is why both orders are drawn.
* `figT07b`: the factor of about 4 holds in the tail of the run on one least-squares instance; over the whole run the
  factor is larger; nothing is measured for the polar step on logistic or softmax losses.
* `figT15`-`figT17b`: the geometric form of the cooldown and the momentum 0.9 are choices of the paper (Muon's
  reference runs cool linearly); the controls without a table run at the schedule tuned for the recipe.
* `figT21`, `figT22`: the comparators with minibatches are SAGA, B-SAGA and normalized or sign steps, not Euclidean
  methods in general; passes do not count the factorization of each step.
* `figT24`, `figT24b`: "one nat per pass, up to two" are rates of the expected squared error of linear SAGA with a
  common Hessian; at b = 20 and 50, where the instances save 36-49 % of the passes, the model gains nothing.
* Nothing is proved at the cooldowns that are run: Proposition 5(b) certifies cooldowns 1e4 to 1e6 times slower
  (`figT19b`, `figT25`).
