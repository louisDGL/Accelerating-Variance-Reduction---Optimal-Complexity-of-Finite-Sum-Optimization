"""Figure E4 of the appendix of explanatory figures of the paper (paper/figures/figE4_recipe_pipeline): SAGA, Muon and
the recipe of Section 3.3 on one skeleton, estimator -> map -> step. Explanatory figure; it adds no result to the paper.

It is the complete diagram of the lecture figure figT04_recipe_pipeline, drawn in paper mode by the same drawing
function (`draw_diagram` of make_figT04_recipe_pipeline.py, whose docstring gives the content of every station, the
sources of the formulas and the checks). In short:

  three lanes   SAGA (Section 2.1: table, identity, constant step), Muon as published by Jordan et al. (Nesterov
                momentum of the minibatch gradient, five Newton-Schulz steps ns5, a step of fixed length whose learning
                rate is cooled down) and the recipe (4) (SAGA estimator on a batch, exact polar factor or ns5, fixed
                length on a geometric cooldown, the geometric form being a choice of the paper);
  colours       the station that holds a table is tinted: blue for Euclidean SAGA, orange when it feeds a polar step;
  under them    the two other length rules of the paper for the polar step (nuclear-scaled, clipped) and an inset
                computed by the script: for one 8 x 40 standard Gaussian matrix omega = U diag(s) V^T (seed in
                make_figT04_recipe_pipeline.py), the values s_k / s_1 before the map, u_k^T polar(omega) v_k (all 1)
                and u_k^T ns5(omega) v_k after it, with `polar_svd` and `ns5` of muon/lab/polar.py. The values after
                ns5 are computed in two ways (matrix routine, scalar quintic) and the script stops if they disagree.

Status. Definitions (the lanes) and an illustration on one draw of an exact property of the two maps (the inset).
The script prints every number it plots. Nothing is read from the results of the project.

Run:    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/scripts/make_figE4_recipe_pipeline.py
Output: paper/figures/figE4_recipe_pipeline.pdf and .png
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figT04_recipe_pipeline as T04  # noqa: E402  (the data and the drawing function; importing it draws nothing)

NAME = "figE4_recipe_pipeline"


def main():
    D = T04.data()
    T04.print_all(D)
    T04.draw(D, "paper", NAME)


if __name__ == "__main__":
    main()
