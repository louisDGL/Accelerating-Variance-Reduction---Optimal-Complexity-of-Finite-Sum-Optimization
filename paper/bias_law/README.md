# Two papers

Both are anonymous drafts built from the shared statements and proofs in `theory/bias_law/` (`shared/`, `proofs/`).

| paper | file | venue format | content |
|---|---|---|---|
| I | `icml/twomoment.tex` | ICML 2026 (`icml2026.sty`) | where Muon-type iterations converge (any norm, any noise, momentum); first-order law `g*/b_eff`; two-moment theorem, `a1 < 0 < a2`; corollaries; experiments incl. real data |
| II | `colt/discrete.tex` | COLT 2026 (`colt2026.cls`) | the law without Cramér's condition: class X(eps), codimension mechanism, Laplacian mechanism, nuclear norm in every size but 1x1, failure for sign descent, momentum uniformly in beta, sampling without replacement |

Build each with `pdflatex X && bibtex X && pdflatex X && pdflatex X` from its own folder. Each paper cites the other as
an anonymous companion submission (`companionI`, `companionII` in `refs.bib`); replace before submission.

Where the rest lives (paths from the repository root):

- `figures/bias_law/` – figures (PDF + PNG).
- `results/bias_law/` – `NUMBERS_discrete.md`, `NUMBERS_twomoment.md` (every number quoted in the text), `figure_data/`
  (plotted data), `figure_logs/`, `checks/` and `lattice/` (logs of the numerical checks).
- `experiments/bias_law/` – scripts: `paper_figures/` (figures of the two papers), `checks/` (c01–c07), `lattice/`
  (exact lattice computations, Wick moments), `a1/` (computer-assisted proof of `a1 < 0`), `verification/` and
  `referee_note2/` (scripts of the independent reviews). Memory note: run one script at a time;
  `paper_figures/digits_problem.py __main__` uses a 4-process pool and large Gaussian batches, and running several
  experiment scripts concurrently has exhausted a 36 GB machine. `digits_frozen.py` and `digits_state.py` are
  single-process and light.
- `colt/tab_exact.tex` is generated from `results/bias_law/figure_data/table_exact.json`.
- `bibliography/bias_law_literature_1.md`, `_2.md` – literature reports.
