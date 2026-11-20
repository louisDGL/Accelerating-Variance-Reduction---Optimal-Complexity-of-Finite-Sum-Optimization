# Archive

Earlier deliverables of the project, superseded by the paper [`../paper/paper_saga_to_muon.pdf`](../paper/README.md)
and moved here unchanged on 2026-11-20 (sources and PDFs). Scripts that are not needed to reproduce the current results
(tests, checks, explorations) are archived separately in the local, unversioned `_archive/` folder.

| Folder | Content |
|---|---|
| `report/` | the final report of the course: `main.pdf` (192 pages: main body in `sections/`, the five theory notes of `theory/` as appendices) and `synthesis.pdf` (its Section 4 alone). `make_restated.py` copies the statements of the notes into `restated/` (generated files: edit the note, not them) and `build.py` runs everything in order. |
| `muon/` | the write-ups of the Muon side study: the theory notes `notes_spectral_theory` (proofs of Theorems 1-2 and Propositions 3-4 of the paper), the condensed paper, the short report and the extended report; see [`muon/README.md`](muon/README.md) |

## Rebuild

**Report.** Its sources include `../theory/`, `../figures/` and `../bibliography/` by explicit relative paths, so it is
built from the root of the project, where it used to live:

```bash
cp -r archive/report report        # temporary copy at the root (not versioned)
python report/build.py             # make_restated.py, then main.pdf and synthesis.pdf
rm -rf report
```

Checked on 2026-11-20 in a copy of this layout: `main.pdf` rebuilds with the same 192 pages; `synthesis.pdf` needs the
LaTeX package `needspace`, which was not installed on the machine of the check.

**Muon write-ups.** See [`muon/README.md`](muon/README.md) (a search path for the figures, no copy needed).
