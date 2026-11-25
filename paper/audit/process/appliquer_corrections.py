#!/usr/bin/env python3
"""Construit la version corrigee du papier a partir de l'original, sans y toucher.

Usage (depuis la racine du depot) :
    python3 paper/audit/process/appliquer_corrections.py <fichier .tex de sortie> [niveau]

Chaque correction est un remplacement de chaine exact, qui doit apparaitre une et une seule fois dans
paper/paper_saga_to_muon.tex ; le script s'arrete sinon. `niveau` : 1 = corrections des erreurs confirmees
(defaut), 2 = y ajoute les precisions mineures marquees niveau 2.
Le patch paper/audit/corrections.patch est le diff entre l'original et la sortie de niveau 1.
"""
import sys

SRC = "paper/paper_saga_to_muon.tex"

# (identifiant du constat, niveau, ancien texte, nouveau texte)
FIXES = [
    # ---- resume
    ("anatomy-abstract", 1,
     r"global constants; the nuclear-scaled",
     r"global constants; on quadratics the nuclear-scaled"),
    ("recipe-abstract", 1,
     r"\emph{Muon's own recipe.} Steps of fixed length on a geometric cooldown of the learning rate, fed by a",
     r"\emph{Muon's own recipe.} Steps of fixed length on a cooldown of the learning rate, here geometric, fed by a"),
    ("abstract-058-093", 1,
     r"map needs $0.58$--$0.93$ of the passes of a normalized step, and a fresh permutation per pass saves $36$--$49\%$ of the",
     r"map needs fewer passes than a normalized step ($0.58$--$0.93$, resolved to about $20\%$), and a fresh permutation per pass saves $36$--$49\%$ of the"),
    # ---- introduction
    ("intro-prop5", 1,
     r"other estimators freeze (Proposition~\ref{prop:schedule}); with minibatches",
     r"other estimators freeze \finding{M5} (Proposition~\ref{prop:schedule} at slower cooldowns); with minibatches"),
    # ---- tableau 1
    ("hu-pep", 2,
     r"\cite{taylor2017smooth,hu2017unified}; one-point curvature conditions",
     r"\cite{taylor2017smooth} or quadratic constraints \cite{hu2017unified}; one-point curvature conditions"),
    ("anatomy-table1", 1,
     r"the same anatomy for the polar step \nat{p}. \\",
     r"the same anatomy for the polar step on quadratics \nat{p}. \\"),
    ("m2-table1", 1,
     r"on a cooldown \nat{e}; at batch one the nuclear-scaled polar step of SAGA behaves as MISO \nat{e}. \\",
     r"on a cooldown \nat{e}; at batch one and untuned steps the nuclear-scaled polar step of SAGA behaves as MISO \nat{e}. \\"),
    # ---- section 2
    ("recipe-grid", 1,
     r"Section~\ref{sec:recipe} uses the same protocol, with step grids of ratio $2^{1/4}$.",
     r"Section~\ref{sec:recipe} uses the same protocol, with constant steps on grids of ratio $2^{1/4}$ and schedules tuned on $q$ in steps of $0.05$ or $0.1$ and half-decades of $\mathrm{lr}_0$ (resolving about $20\%$)."),
    # ---- (C2)
    ("c2-cD", 2,
     r"\eqref{eq:saga-repr} I1 and I2 together contribute at most a factor $8/3$",
     r"$c_D$ of \eqref{eq:saga-repr} I1 and I2 together contribute at most a factor $8/3$"),
    ("c2-families", 1,
     r"I2 and I4 carry $64$--$98\%$ of the gap, whose size depends",
     r"I2 and I4 carry $64$--$98\%$ of the gap for $\kappa\ge10^4$, whose size depends"),
    # ---- anatomie du pas polaire
    ("ray-factor4", 1,
     r"factor $4$ on the ray that attracts generic runs \cite{shulgin2026quadratic}; a large gap needs a global constant far",
     r"factor $4$ in the tail of generic runs (observed; on the flattest ray, \cite{shulgin2026quadratic}); a lasting large gap needs a global constant far"),
    # ---- (C4)
    ("c4-step", 1,
     r"whereas the observed exponent is $4.5$ times the best certificate and SAGA still converges at $\gamma R^2=4$, eight times",
     r"whereas the observed exponent is $4.4$ times the best certificate and SAGA still converges at $\gamma R^2=2$, four times"),
    ("c4-lower", 1,
     r"constant step needs at least $\frac{\kappa-1}4\log\frac1{2\varepsilon}$ iterations",
     r"constant step, started at $\theta_0=(r,r)$, needs at least $\frac{\kappa-1}4\log\frac1{2\varepsilon}$ iterations"),
    # ---- apres le theoreme 1
    ("sidford", 1,
     r"$\kop\ge\min(K,d)$ \cite{sidford2018coordinate}.",
     r"$\kop\ge\min(K,d)$ (as for $\ell_\infty$ \cite[Lemma~A.1]{sidford2018coordinate})."),
    # ---- (M3)
    ("m3-saga-fine", 1,
     r"on grids of ratio $2^{1/4}$ B-SAGA needs $27.1$ /",
     r"on grids of ratio $2^{1/4}$ SAGA needs $49.8$ / $83.5$, B-SAGA $27.1$ /"),
    ("m3-ratio", 1,
     r"SAGA and $0.78$--$1.08$ of those of the best Euclidean method without preconditioner;",
     r"SAGA on factor-$2$ grids and $0.78$--$1.08$ of those of the best Euclidean method without preconditioner;"),
    # ---- theoreme 2 et proposition 3
    ("smd-otherwise", 1,
     r"by $\Med(\xi)=\{0\}$; otherwise the limit gradient is an \emph{offset}",
     r"by $\Med(\xi)=\{0\}$; if $0\notin\Med(\xi)$ the limit gradient is an \emph{offset}"),
    ("prop3d-unreg", 1,
     r"(d) (Batch one on least squares.)",
     r"(d) (Batch one on unregularized least squares.)"),
    # ---- (M1)
    ("m1-absolute", 1,
     r"with the SAGA estimator the iterates settle at a",
     r"with the SAGA estimator the iterates settle at an absolute"),
    ("m1-storm", 1,
     r"a momentum buffer as in MARS-M \cite{liu2025marsm} keeps a floor).",
     r"a STORM buffer \cite{cutkosky2019momentum} keeps a floor)."),
    ("m1-only", 2,
     r"Only a finite-sum estimator reaches rounding precision, with a length",
     r"Only the SAGA estimator reaches rounding precision (the loopless-SVRG one misses the targets), with a length"),
    # ---- (M2)
    ("m2-title", 1,
     r"\item[(M2)] \emph{At batch one the nuclear-scaled polar step of SAGA is MISO in disguise.}",
     r"\item[(M2)] \emph{At batch one and untuned steps the nuclear-scaled polar step of SAGA is MISO in disguise.}"),
    ("m2-indices", 1,
     r"exactly. With the same indices the joint SAGA-Muon needs",
     r"exactly. With the same indices and untuned steps the joint SAGA-Muon needs"),
    ("m2-fails", 1,
     r"softmax with $\kappa=n$; SAGA: $20.2$) and fails where MISO fails ($\kappa\gg n$): it is not a spectral method.",
     r"softmax with $\kappa=n$; SAGA: $20.2$, tuned $11.4$) and fails where MISO fails ($\kappa=10n$) unless its step is tuned \finding{M3}: untuned, it is not a spectral method."),
    # ---- section 3.3
    ("recipe-linear", 1,
     r"it keeps steps of fixed length and cools the learning rate down. We",
     r"it keeps steps of fixed length and cools the learning rate down (linearly in its reference runs; geometrically here). We"),
    ("m5-freeze-1", 1,
     r"Without a table the same schedule freezes the iterate at $8\%$ to $93\%$ of the initial squared distance, for the",
     r"Without a table the schedule selected for Muon's recipe freezes the iterate at $8\%$ to $34\%$ of the initial squared distance at batch one, for the"),
    ("m5-freeze-2", 1,
     r"minibatch gradient ($b=1$, $32$) as for MARS-M",
     r"minibatch gradient as for MARS-M"),
    ("prop5-init", 1,
     r"$\phi_i$ the iterate at which entry $i$ was last replaced; at iteration $t$ a",
     r"$\phi_i$ the iterate at which entry $i$ was last replaced ($\phi_i=W_0$ at first); at iteration $t$ a"),
    ("prop5-orders", 1,
     r"cooldown is $3.3\cdot10^4$ to $8.9\cdot10^4$ times slower, in $\log(1/q)$, than $q=0.8$, and",
     r"cooldown for permutation orders is $3.3\cdot10^4$ to $8.9\cdot10^4$ times slower, in $\log(1/q)$, than $q=0.8$, and"),
    ("fig4-same-cooldown", 2,
     r"fastest control map on the same cooldown,",
     r"fastest control map on the same grid of cooldowns,"),
    ("m6-same-cooldown", 2,
     r"cooldown, and without momentum the exact polar factor needs",
     r"grid of cooldowns, and without momentum the exact polar factor needs"),
    ("m7-upto", 1,
     r"single runs under independent sampling lose about $1.6$ (measured, $n=1000$). On such a sum the recipe follows",
     r"single runs under independent sampling lose up to about $1.6$ (measured, $n=1000$). On such a sum the recipe at batch one follows"),
    ("m7-mse", 1,
     r"bounds the mean error, not a single run.",
     r"bounds the mean squared error, not a single run."),
    ("m7-model-control", 1,
     r"with a permutation at $b=20$ and $50$; why they lose more is not established.",
     r"with a permutation at $b=20$ and $50$, where the model itself gains nothing ($\rho_\star=0.22$ both ways); why they lose more is not established."),
    # ---- tableau 2
    ("anatomy-table2", 1,
     r"The same exact anatomy \eqref{eq:polar-repr}; no norm",
     r"The same exact anatomy on quadratics \eqref{eq:polar-repr}; no norm"),
    ("minibatch-table2-a", 1,
     r"reproduced by Euclidean devices at batch one \finding{M3} but not with",
     r"reproduced by Euclidean devices at batch one \finding{M3}, not by (B-)SAGA with"),
    ("table2-ref", 1,
     r"The limit gradient is minus a nuclear median of the noise (Proposition~\ref{prop:offset}). \\",
     r"The limit gradient is minus a nuclear median of the noise (Theorem~\ref{thm:smd}; size: Proposition~\ref{prop:offset}). \\"),
    ("m2-table2", 1,
     r"or a cooldown \finding{M5}; at batch one, MISO \finding{M2}. \\",
     r"or a cooldown \finding{M5}; at batch one and untuned steps, MISO \finding{M2}. \\"),
    ("memory-table2", 1,
     r"At most one nat per pass under independent sampling \finding{M3}, two under a permutation (note B4). &",
     r"In expectation, at most one nat per pass under independent sampling \finding{M3}, two under a permutation (linear model, note B4). &"),
    # ---- discussion et conclusion
    ("disc-1e6", 1,
     r"$10^4$ to $10^5$ times slower than in \finding{M5};",
     r"$10^4$ to $10^6$ times slower than in \finding{M5};"),
    ("concl-norm", 1,
     r"and no fixed change of geometry repairs the",
     r"and no change of norm repairs the"),
    ("concl-schedule", 1,
     r"converges to rounding error where the same schedule freezes without one.",
     r"converges to rounding error where such a schedule freezes without one."),
    ("concl-minibatch", 1,
     r"with minibatches the spectral map pays, and a fresh permutation per pass, which lifts the",
     r"with minibatches the spectral map is ahead of (B-)SAGA and normalized steps, and a fresh permutation per pass, which lifts the"),
    ("concl-memory", 1,
     r"memory rate of SAGA from $e^{-1}$ to nearly $e^{-2}$, pays where memory limits.",
     r"expected memory rate of linear SAGA from $e^{-1}$ to nearly $e^{-2}$, pays where memory limits."),
]


# Reglages de mise en page seulement (aucun mot change), pour revenir a 12 pages ; patch separe.
LAYOUT = {
    "floatsep": (r"\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}", r"\setlength{\textfloatsep}{7pt plus 2pt minus 2pt}"),
    "figwidth": (r"\includegraphics[width=0.97\textwidth]", r"\includegraphics[width=0.92\textwidth]"),
    "bibsep": (r"\setlength{\itemsep}{0.5pt}", r"\setlength{\itemsep}{0pt}"),
    "figwidth90": (r"\includegraphics[width=0.97\textwidth]", r"\includegraphics[width=0.90\textwidth]"),
}


def main():
    out = sys.argv[1]
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    skip = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()
    tex = open(SRC, encoding="utf-8").read()
    applied = []
    for key, lvl, old, new in FIXES:
        n = tex.count(old)
        if n != 1:
            sys.exit(f"ERREUR {key}: {n} occurrence(s) de l'ancien texte")
        if lvl > level or key in skip:
            continue
        tex = tex.replace(old, new)
        applied.append(key)
    if len(sys.argv) > 4:
        for key in sys.argv[4].split(","):
            old, new = LAYOUT[key]
            if tex.count(old) < 1:
                sys.exit(f"ERREUR mise en page {key}")
            tex = tex.replace(old, new)
            applied.append("layout:" + key)
    open(out, "w", encoding="utf-8").write(tex)
    print(f"{len(applied)} corrections appliquees (niveau {level}) -> {out}")
    print(" ".join(applied))


if __name__ == "__main__":
    main()
