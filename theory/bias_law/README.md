# Master theorem document

`master.pdf` (build: `pdflatex master && bibtex master && pdflatex master && pdflatex master`) consolidates every
result of the three notes (`notes/polar_bias`, `notes/discrete_bias`, and the note `orbital-bias`, copied
in `verification/orbital-bias_source.tex`) in one notation, with complete proofs, a dependency map (Section 3) and a
verification record (last section).

Single source of truth, shared with the two papers in `paper/bias_law/`:

| file | content |
|---|---|
| `shared/macros.tex` | math macros |
| `shared/statements.tex` | every definition / assumption / theorem body, as `\Stmt...` macros |
| `shared/refs.bib` | bibliography |
| `proofs/p_*.tex` | lemmas and proofs, one module per topic |
| `verification.tex` | verification record (quoted theorems, computer-assisted steps, reviews, numerical checks, gaps) |
| `notes/` | the two original research notes (`polar_bias`, `discrete_bias`) with their referee and audit reports, research log and novelty ledger |
| `verification/` | referee reports; their scripts are in `experiments/bias_law/verification/` (`referee_momentum.md`, `referee_orbital.md`, `referee_population.md`, `referee_final_colt.md`, `referee_final_icml.md`) |

A statement or proof must be edited in `shared/` or `proofs/` only; the master document and both papers then pick it up.

Open items before submission are listed in the "Known gaps and non-claims" and "What rests on quoted theorems"
subsections of the verification record. The most important: the exact form of Theorem 20.1 of Bhattacharya–Rao used
in Lemma `lem:edge` has not been re-read in the original.
