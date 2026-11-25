#!/usr/bin/env python3
"""Audit 'novelty' : controles reproductibles (lecture seule).

1. Seuil kappa < 4.08 de la ligne 609 du papier : ((kappa-1)/(kappa+1))^2 < e^{-1}
   (facteur par passe de DIAG, Mokhtari et al., Corollaire 3, mis au carre) <=> kappa < (sqrt(e)+1)/(sqrt(e)-1).
2. Travaux proches identifies par l'audit ou par les verifications du depot : sont-ils cites dans le papier
   (.tex / .bib) et figurent-ils dans les registres du depot ?

Lancer depuis la racine du projet :  python3 paper/audit/checks/novelty_close_works.py
"""
import math
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
tex = (ROOT / "paper" / "paper_saga_to_muon.tex").read_text(encoding="utf-8")
bib = (ROOT / "paper" / "paper_saga_to_muon.bib").read_text(encoding="utf-8")
records = {
    "muon/literature_check.md": (ROOT / "muon" / "literature_check.md").read_text(encoding="utf-8"),
    "muon/findings.md": (ROOT / "muon" / "findings.md").read_text(encoding="utf-8"),
    "bibliography/literature_review.md": (ROOT / "bibliography" / "literature_review.md").read_text(encoding="utf-8"),
    "archive/muon/notes_spectral_theory.tex": (ROOT / "archive" / "muon" / "notes_spectral_theory.tex").read_text(encoding="utf-8"),
}

# 1. seuil 4.08
se = math.sqrt(math.e)
kappa_star = (se + 1) / (se - 1)
print("1. seuil : (sqrt(e)+1)/(sqrt(e)-1) = %.4f ; verification ((k-1)/(k+1))^2 a k=4.08 : %.4f, e^-1 = %.4f"
      % (kappa_star, ((4.08 - 1) / (4.08 + 1)) ** 2, math.exp(-1)))

# 2. travaux proches : (etiquette, motifs a chercher)
works = [
    ("Singh-Mishra-Raut 2607.12360 (plancher eta^2, cooldown)", ["2607.12360", "Singh"]),
    ("CacheMuon 2606.16371 (transformee polaire en cache)", ["2606.16371", "CacheMuon"]),
    ("Auger-Hansen 1310.7697 (chaine de Markov normalisee)", ["1310.7697", "Auger"]),
    ("Davis-Drusvyatskiy 2512.04299 (critere par pas, rang nucleaire)", ["2512.04299", "Drusvyatskiy.*spectral|When do spectral"]),
    ("Xie et al. 2205.02273 (SAGA, normes non euclidiennes)", ["2205.02273"]),
    ("Song et al. 2006.06783 (clipping = Huberisation, GLM)", ["2006.06783"]),
    ("Negiar et al. 2002.11860 (table SAG dans un pas LMO)", ["2002.11860", "Negiar|N\\\\'egiar"]),
    ("Schaipp et al. 2402.12828 (clipping et mediane)", ["2402.12828", "Schaipp"]),
    ("Beretti et al. 2611.15825 (taux locaux de Muon, 11 nov. 2026)", ["2611.15825", "Beretti"]),
    ("Ying-Yuan-Sayed 1708.01383 (SAGA sous permutation)", ["1708.01383"]),
    ("Do-Dereich-Jentzen 2608.04607", ["2608.04607"]),
    ("Li-Tang-Zou 2602.11557", ["2602.11557"]),
]
print("2. travaux proches : cite dans le papier (.tex/.bib) ? present dans les registres du depot ?")
for label, pats in works:
    in_paper = any(re.search(p, tex) or re.search(p, bib) for p in pats)
    where = [name for name, txt in records.items() if any(re.search(p, txt) for p in pats)]
    print("   %-68s papier: %-3s registres: %s" % (label, "oui" if in_paper else "NON", ", ".join(where) if where else "aucun"))

# 3. occurrences de 'not found' dans le papier (numeros de ligne)
print("3. lignes du papier contenant 'not found' :")
for i, line in enumerate(tex.splitlines(), 1):
    if re.search(r"[Nn]ot found", line):
        print("   l.%d: %s" % (i, line.strip()[:150]))
