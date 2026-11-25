# Audit - dimension `consistency` (coherence interne et rigueur de redaction)

Objet : `paper/paper_saga_to_muon.tex` (700 lignes, 12 pages), PDF `paper/paper_saga_to_muon.pdf`.
Etat : TERMINE (2026-11-20). Les numeros de ligne sont ceux de l'outil Read sur le `.tex`.

Scripts de controle (lecture seule, lances depuis la racine du depot, un seul processus, BLAS mono-thread) :

| Script | Objet |
|---|---|
| `paper/audit/checks/build_consistency/` | copie de `paper/` et compilation (pdflatex, bibtex, pdflatex x2) |
| `paper/audit/checks/consistency_bibkeys.py` | cles citees contre entrees `.bib`, etiquettes, appels des figures et tableaux |
| `paper/audit/checks/consistency_b3_gap.py` | chiffres repetes du constat 1 et sensibilite a l'ordre de retrait (`results/B3_summary.json`) |
| `paper/audit/checks/consistency_m1_rows.py` | passes par famille et par instance (`muon/results/M1_summary.json`) : (M2), (M3), (M4), (M5) |
| `paper/audit/checks/consistency_m5_schedules.py` | cooldowns compares dans (M5) et figure 4(a) (`M9_summary.json`, partie C) |
| `paper/audit/checks/consistency_m6_controls.py` | variantes comparees dans (M6) et figure 4(b) (`M9_summary.json`, partie A) |
| `paper/audit/checks/consistency_m7_order.py` | (M7), figure 4(c), 36-49 % (`M9_summary.json` partie D, `M9b_summary.json`) |

## 1. Perimetre couvert

(a) repetitions de chiffres et d'affirmations : table complete, section 3. (b) etiquettes de statut et existence des
preuves : section 4. (c) sur-affirmations : section 5 et constats. (d) Limitations : section 6. (e) notations :
section 7. (f) compilation : section 2. (g) lecture du PDF page par page : section 8. (h) reproductibilite et README :
section 9. Les constats detailles sont en section 10, ce qui n'a pas ete verifie en section 11.

Bilan : 0 blocker, 4 major, 18 minor, 5 nit. Les enonces de tete (titre, resume, theoremes, propositions) sont
coherents entre eux ; les quatre constats majeurs portent sur le corps de (M2), (M5), (M6)/tableau 2 et
(M7)/tableau 2/conclusion, ou une phrase dit plus que ce que les donnees du depot portent.

## 2. (f) Compilation

| Controle | Resultat |
|---|---|
| Code de sortie | 0, aucune erreur |
| Pages | 12 (`pdfinfo`), format letter |
| References ou citations non resolues | aucune (aucun `LaTeX Warning` dans le `.log`) |
| Avertissements BibTeX | aucun |
| Overfull hbox/vbox | aucun |
| Underfull hbox | 5, tous dans la bibliographie en deux colonnes et le paragraphe Reproducibility (badness 1377 a 5985), sans effet visible |
| Patch de `\thebibliography` | applique (pas de message `PATCH OF THEBIBLIOGRAPHY FAILED`) |
| Cles citees / entrees `.bib` | 55 / 55, aucune cle citee absente, aucune entree non citee, aucun doublon |
| Etiquettes | 30, aucune dupliquee, aucun renvoi vers une etiquette inexistante |
| PDF livre | texte `pdftotext` identique a celui du PDF recompile : le PDF livre correspond au `.tex` |

Verdict : OK.

## 3. (a) Table des repetitions

Colonnes : affirmation ; occurrences ; ecart constate (renvoi au constat).

| # | Affirmation | Occurrences (lignes) | Ecart |
|---|---|---|---|
| R1 | exposant exact = 3.2 a 220 fois l'exposant certifie | 95 (resume), 150 (tableau 1), 290-291 (C1 : 3.2-3.7 pour kappa <= 1e2, 19-220 a kappa = 1e5), fig. 1(b) | aucun. `B3_summary.json` : 3.17 a 220.39 ; 3.17-3.71 ; 18.95-220.39 |
| R2 | 77-96 % de l'ecart sur I2 et I4, kappa >= 1e4 | 95-96, 150-151, 292-293 | identique trois fois (recalcule : 76.9-96.1 %). Mais (C3), l. 303, donne 59-96 % pour les deux memes inegalites retirees en premier : le lien n'est pas fait (C-13). "logarithmic" absent du resume et du tableau 1 |
| R3 | aucune norme n'ameliore le pire cas | titre 82, 93-94, 97-98, 127, 156-157, 278, 334-339, 626, 645, 675-676 | resume, tableau 1, theoreme : "condition number(s)", "vanishes along a sequence" ; titre, introduction, tableau 2 : "the worst case" ; conclusion : "no fixed change of geometry", plus large que "norm" (C-05) |
| R4 | meme anatomie pour le pas polaire | 96-97, 151 [p], 309-315, 626, 675 | "for a quadratic" seulement l. 226 et 313 ; "(not measured for the polar step)" seulement l. 315 (C-12) |
| R5 | descente miroir stochastique, mediane nucleaire | 99-101, 129-130, 164-165, 395-407, 631-632, 677 | resume : "decreasing deterministic lengths" ; theoreme : somme infinie, carres sommables ; conclusion : "with decreasing lengths" (C-06). Tableau 2 renvoie a la Proposition 4 (C-07) |
| R6 | offset d'ordre 1/b, formel | 101, 165-166 [f], 449-458, 671 | aucun |
| R7 | estimateur a somme finie : le minimiseur est le seul point fixe | titre 82-83, 101-102, 130-131, 171, 390, 425-427, 635, 678 | coherent ; "identifies a fixed point, not its attraction" seulement l. 443 ; le titre dit "variance reduction" sans "finite-sum" (C-22) |
| R8 | la recette converge jusqu'a l'arrondi, les autres estimateurs gelent | 102-105, 131-133, 178, 532-540, 636, 678-679 | "the same schedule" (538, 679) ne correspond pas aux donnees (C-02) ; l'introduction cite la Proposition 5 pour l'enonce mesure (C-08) |
| R9 | plancher pres de 1e-18 pour ns5 | 103-104, 535-537, fig. 4(a) | aucun (9.8e-19, 1.1e-18) |
| R10 | Proposition 5 : ages bornes, cooldowns plus lents ; 3.3e4-8.9e4 ; 1e4-1e5 | 105, 178-179, 578-579, 648-649 | coherent ; ordre d'echantillonnage non precise (C-15) |
| R11 | 0.58-0.93 | 105-106 ("of a normalized step"), 588-589 ("of the fastest control on the four conditioning-limited instances") | aucun (0.579-0.928 ; le controle le plus rapide est toujours la normalisation de Frobenius avec moment de Nesterov) |
| R12 | 36-49 % de passes en moins | 106-107, 640 | absent du corps : (M7) donne 16.7-20.0 puis 9.1-12.2 (C-16). Recalcule : 36.1, 37.7, 43.5, 49.2 % |
| R13 | a batch un les dispositifs euclidiens font aussi bien | 128, 157-158, 374, 627-628, 679-680 | coherent ; le tableau 2 ajoute "but not with minibatches", non porte pour la metrique de covariance (C-03) |
| R14 | MISO a batch un | 172 ("behaves as MISO [e]"), 499-503 ("is MISO in disguise"), 636 | intitule plus fort que l'etiquette ; "fails where MISO fails" contredit (M3) (C-01) |
| R15 | plafond memoire : un nat par passe, deux avec permutation | 382-384, 597-603, 638, 680-681 | "in expectation" en (M3) ; absent du tableau 2 et de la conclusion ; "linear SAGA, common Hessian, n to infinity" seulement en (M7) (C-04) ; (M3) depasse le "plafond" (C-09) |
| R16 | cibles 1e-6, 1e-9 "where memory limits" | 262-263, fig. 2, 382, fig. 4, 540, 599 | les chiffres de (M2) sont a 1e-6 sur une instance limitee par la memoire, sans le dire (C-01) |
| R17 | protocole : 5 graines de reglage, 10 nouvelles | 263-265, 285, 362, 524-528, 533, 668-669 | l. 265 "the same protocol" faux pour la loi de (M5) et pour (M7) (C-14) |
| R18 | passes dans les metriques figees | 372-373 (13.3 ; SAGA 27.1 ; covariance 13.1), 375-378, 540-542 (12.9-24.4 ; 13.5-16.1) | "covariance metric" = SAGA non amorti en (M4), B-SAGA en (M5) : 13.1 hors de 13.5-16.1 (C-11) |
| R19 | factorisations non comptees | 262, 594, 669 | aucun |
| R20 | ecarts a kappa = 1e3 : 5.2 (C3), 17 % et 4.5 (C4) | 305-306, 319-320 | coherent : 5.16 / 1.17 = 4.4 |
| R21 | B-SAGA : 33.8 / 31.9 (pas de facteur 2), 27.1 / 31.9 (2^(1/4)) | 377-379 | coherent avec `M1_summary.json` (33.8, 31.9) |
| R22 | passes sur les deux instances non carrees | 375-378, fig. 2(b) | coherent : 69.55 / 98.7 ; 22.05-29.0 / 24.35-43.55 ; 15.05-33.8 / 16.05-31.9 |
| R23 | sans table : 8 a 93 % de la distance initiale | 538 | coherent (0.083 a 0.931) mais au cooldown regle de la recette, pas a q = 0.8 (C-02) |
| R24 | distance carree absolue ou relative | 263 (relative), 494 ("3.8 eta^2", absolue), 495-496 (relative), fig. 3(a) (relative) | melange non signale (C-10) |

## 4. (b) Etiquettes de statut et existence des preuves

Les etiquettes [p] [m] [e] [f] n'apparaissent que dans le tableau 1 (14 occurrences de `\nat`, lignes 138-180 et 269).
La legende du tableau 1 renvoie a la section 2.4 pour les etiquettes, mais la section 2.4 ne definit que [p]
("Proofs of [p] claims are in the main report or the theory notes") : [m], [e], [f] ne sont definis que dans la
legende elle-meme (nit, C-24).

| Element [p] | Enonce dans le papier | Preuve | Verdict |
|---|---|---|---|
| "the same anatomy for the polar step" (tableau 1, l. 151) | eq. (2), l. 226-232, paragraphe l. 309-315 | identite citee (Shulgin et al. eq. 17, Jawanpuria et al. eq. 8) ; `archive/muon/notes_spectral_theory.tex`, Proposition "exact one-step representation" (l. 355-378, etiquetee [K] dans la note). Bornes J dans [mu_op, L_op] et mu_PL <= J refaites a la main (Cauchy-Schwarz dans le produit scalaire de H) | OK, pour les quadratiques seulement (C-12) |
| Theoreme 1 (tableau 1, l. 156-157) | l. 334-340 | preuve complete dans le papier, l. 341-347 ; aussi `archive/muon/notes_spectral_theory.tex` l. 248 et rapport etendu l. 418 | OK (existence ; le detail releve d'une autre dimension) |
| Theoreme 2 (l. 164-165) | l. 395-407 | "Proof sketch" l. 408-416 ; preuve complete `archive/muon/notes_spectral_theory.tex`, theoreme `thm:smd`, l. 868-900 (memes enonces (a) a (c)) | OK ; l'esquisse ne dit pas ou est la preuve complete (C-18) |
| Proposition 3(d) (l. 165) | l. 427-431 | preuve l. 438-439 ; note archivee `prop:lad`, l. 938-957 | OK ; hypothese "sans ridge" implicite (C-19) |
| Proposition 3(c) (l. 171) | l. 425-427 | preuve l. 436-438 ; note archivee `thm:median`(e), l. 918-936 | OK |
| Proposition 5 (l. 178-179) | l. 547-566 | "Proof sketch" l. 567-575 avec renvoi "Details: theory notes of the Muon study, F12-F14" ; `muon/notes_theory.md` F12 (l. 525-563), F13 (l. 565-693, la forme close L_op + 10 A L_max et la condition mu_op <= L_max y sont), F14 (l. 695-738) : enonces et preuves presents | OK |
| eq. (1), "the main report proves" (l. 216) | l. 217-221 | `theory/B3_prop59_tightness.tex`, Proposition `prop:B3-representation`, l. 222-244, incluse dans `archive/report/main.tex` (l. 68) | OK |
| (C4) borne inferieure, facteur 3 et 4/3 (l. 318-324) | texte | `theory/B3_prop59_tightness.tex`, `prop:B3-lower` (l. 406-417), `prop:B3-optimum`, `rem:B3-conclusion` | OK ; point de depart omis (C-20) |
| (C5) "every P has kappa_P >= kappa" (l. 327) | texte | `theory/B1_preconditioned_saga.tex`, `prop:B1-nofreelunch` (l. 780-790) | OK |
| (M7) "(proved)" : min{2c,1} et min{2, log((2-g)/g)} (l. 600-603) | texte | `theory/B4_sampling_order.tex`, `prop:B4-iid-limit` (l. 83-104) et `prop:B4-rr-ms` (l. 305-330) ; B4 est dans `theory/main.pdf` (pas dans `archive/report/main.pdf`), comme le dit le papier | OK ; les deux limites ne sont pas dans le meme regime (c fixe contre g fixe) et portent sur l'esperance (C-04) |
| Discussion, F15 (l. 649-650) | texte | `muon/notes_theory.md` F15 (l. 740-792) : gamma <= 1/(14 n L_max), 1400 et 350 fois sous les pas regles | OK |
| Proposition 3(a), (b) | l. 423-425 | preuve l. 434-436 ; exemple l. 393 refait a la main : psi minimal en g = 1, valeurs du champ moyen -1, -0.6, -0.2, 0.4, 1 | OK |

Aucun element etiquete [p] n'est sans enonce ni sans preuve. Deux enonces prouves ne sont qu'esquisses dans le papier
(Theoreme 2, Proposition 5) ; les preuves completes existent dans des fichiers du depot cites au paragraphe
Reproducibility.

## 5. (c) Sur-affirmations : formulations de tete

| Formulation | Couverture | Verdict |
|---|---|---|
| Titre "no norm repairs the logistic worst case" | Theoreme 1 : toute norme, modeles ridge dont la courbure de perte s'annule le long d'une suite ; "worst case" = constante globale de conditionnement, comme le dit le resume | OK avec cette lecture |
| Titre "variance reduction acts on the fixed point" | Proposition 3(c) pour SAGA et SVRG ; les variantes a moment (MARS-M, STORM) gardent un plancher selon (M1), (M5) | OK si "finite-sum" (C-22) |
| Resume "the nuclear-scaled polar step has the same anatomy" | formule exacte sur quadratiques ; non mesure pour le pas polaire | reserve manquante (C-12) |
| Resume "decreasing deterministic lengths" ; conclusion "with decreasing lengths" | Theoreme 2(b) : somme infinie, carres sommables | imprecis (C-06) |
| Resume "we prove this law for tables of bounded age, at slower cooldowns" | Proposition 5(b) | OK |
| Resume "0.58-0.93 of the passes of a normalized step" | (M6), quatre instances, b = 16 et 32 | OK |
| Resume "saves 36-49% of the passes where memory limits (batches of 20 and 50)" | (M7), deux instances, cinq graines | OK comme mesure ; attribution au plafond memoire non portee par le modele (C-04) |
| Intitule (M2) "is MISO in disguise" ; "fails where MISO fails" | experience M6 a pas non regles, 3 graines, cible 1e-6 | contredit par (M3) a pas regles (C-01) |
| Intitule (M5) "With a table the cooldown converges; without one it freezes the iterate" | partie C de M9 | les deux moities ne sont pas mesurees au meme cooldown (C-02) |
| Intitule (M6) "Minibatches: where the spectral map pays" ; tableau 2 "but not with minibatches" | comparaison avec controles normalises, SAGA, B-SAGA au meme batch | metrique de covariance non lancee en minibatch (C-03) |
| Intitule (M7) "a fresh permutation per pass lifts the memory ceiling" ; conclusion "lifts the memory rate of SAGA from e^{-1} to nearly e^{-2}" | mesure sur deux instances a b = 20, 50 ; preuve pour SAGA lineaire, hessienne commune, esperance, n infini | (C-04) |
| Conclusion "no fixed change of geometry repairs the worst case" | Theoreme 1 : normes | plus large que le theoreme (C-05) |
| "Muon's own recipe" | ns5 + Nesterov 0.9, longueurs fixes, cooldown geometrique regle ; une matrice, modeles convexes | la forme geometrique du cooldown est un choix de l'etude : non verifie contre la source (section 11) |

## 6. (d) Section Limitations (l. 667-672)

| Limite reelle | Couverte ? |
|---|---|
| objectifs fortement convexes, donnees synthetiques, taille moderee | oui (l. 667) |
| ordre de retrait des inegalites | oui, sans chiffre ; le depot contient l'autre ordre (C-13) |
| nombre d'instances et de tirages des donnees | oui (l. 668-669) ; (M1), (M2) non cites |
| nombre de graines | non dans Limitations ; donne en 2.3 et dans les legendes ; (M2) repose sur 3 graines et (M5 loi), (M7) sur 5 sans graines de confirmation (C-14) |
| cout des factorisations | oui (l. 669 ; aussi l. 262, 594), sans chiffre |
| reglage, falaise en q | oui (l. 671) ; lr_0 fixe a partir de la norme de W_* dans (M7) non dit (C-14) |
| hypotheses du Theoreme 1 | oui ("global constants") ; "normes" seulement non dit (C-05) |
| hypotheses du Theoreme 2 | oui (bruit additif, longueurs deterministes, pas de moment) ; facteur polaire exact et non Newton-Schulz non dit (C-21) |
| Proposition 4 formelle | oui |
| Proposition 5 loin des pas utilises | oui (l. 578-581, 647-650) |
| modele de (M7) a courbure unique | oui ; le fait que ce modele ne montre aucun gain de la permutation a b = 20, 50 n'est pas dit (C-04) |
| memoire de la table (un gradient par echantillon) | non (C-21) |
| anatomie du pas polaire : quadratiques seulement, non mesuree | dans le corps (l. 315), pas dans Limitations (C-12) |

## 7. (e) Definitions et notations

| Symbole | Sens (lignes) | Verdict |
|---|---|---|
| L_eff | contrepartie de cocoercivite de SAGA, fonction de theta (215, 219, 222, 284, 295, 303) ; constante L_op + 10 A L_max de la Proposition 5 (556-557, 572-573) | meme macro, deux objets (C-17) |
| rho | longueur ecretee min{rho, gamma norme nucleaire} (257) ; rho = 1 - mu_op/L_eff par iteration (557, 573) ; rho_star, facteur de lr^2 par passe (597, 599) | trois sens (C-17). S'y ajoutent q (par passe, sur lr, eq. 4) et r (par iteration, sur eta, l. 563) : quatre parametrages du cooldown |
| c_i | borne de courbure sup lambda_max (202, 237, 338, 345) ; matrice de rang un dans la preuve de la Proposition 5 (568-569) | conflit (C-17) ; aussi c innovation (254), c = gamma n lambda (600), c_D, c_M (218), c_N (337), c_{K,d} (454), c_l (548) |
| g | fonction convexe (337, 344) ; g_k valeurs propres de la metrique (242) ; g_j gradient par echantillon (428) ; g_i perte par echantillon (549) ; g = gamma lambda (602) ; scalaire (393) | cinq sens (C-17) |
| delta | poids de Lyapunov 3 gamma^2 (194, 218) ; probabilite (559-560) | conflit local, sans risque (C-17) |
| D | distance carree (193, 218) ; D_F Bregman (398) ; D^2F derivee seconde (199) ; D_t = F(W_t) - F_* (553) | conflit (C-17) |
| eta | eta_* minimiseur de SAGA (190, 193, 213) ; eta, eta_t longueur du pas polaire (204, 396, 557) | conflit (C-17) |
| kappa | R^2/mu (190) ; kappa_3 troisieme cumulant (451) ; kappa nominal du generateur en (M2) (503) ; "kappa < 4.08" de [41] non precise (609) | acceptable ; (C-17) |
| theta, vartheta | iterer de SAGA ; amortissement de l'innovation (254-255, 552) | coherent |
| gamma, lr, eta | gamma pas ou facteur nucleaire ; lr_t (eq. 4) = eta_t de la Proposition 5 | coherent (K <= d sur toutes les instances, facteur max(1, K/d)^(1/2) = 1) |
| mu_eff | SAGA seulement | coherent |
| barre c, Sigma chapeau | l. 240, 328 | jamais definis (C-17) |
| L dans l'eq. (2) | l. 227-231 | parametre libre introduit sans phrase ("W^+ = W - ... O/L"), puis "with L = L_op" : acceptable |
| b | resume l. 101, tableau 1 ; defini seulement dans les Propositions 4 et 5 | acceptable |
| W_*, F_* | l. 227, 404 | non definis explicitement ; usage standard |
| L_N(g), mu_N(F) | definis pour F seul (198-199), employes comme fonctionnelles (201, 337-339) | extension implicite, acceptable |
| "fixed point" | zero du champ moyen ou limite (bruit minibatch) ; etat qu'aucun indice ne deplace (SAGA) | deux sens, a lire avec l. 443 |

## 8. (g) Lecture du PDF page par page

| Page | Contenu | Remarques |
|---|---|---|
| 1 | titre, resume, introduction | rien a signaler |
| 2 | tableau 1, sections 2.1, 2.2 | tableau lisible, pas de debordement |
| 3 | eq. (1), (2), sections 2.3, 2.4 | rien a signaler |
| 4 | figure 1, (C1) a (C4) | figure 1 placee au-dessus du titre "3 Results" ; appelee l. 291, 294, 304 |
| 5 | figure 2, (C5), Theoreme 1, (M4), debut de (M3) | panneau (a) : pour "softmax cond." et "softmax square" le marqueur plein de l'identite semble masque par le marqueur ouvert (valeurs presque egales) ; lisible |
| 6 | fin de (M3), section 3.2, Theoreme 2, Proposition 3 | rien a signaler |
| 7 | figure 3, Proposition 4, controles numeriques | l'axe de (a) est relatif alors que (M1) cite 3.8 eta^2 en absolu (C-10) |
| 8 | figure 4, (M1), (M2), debut de 3.3 | la figure 4 (section 3.3) est placee au-dessus de (M1) et (M2), qui renvoient a la figure 3 (nit, C-25) ; panneau (b) : les annotations "7/10" et "9/10" sont en gris, sans indiquer la serie (variante spectrale, d'apres `M9_summary.json`) |
| 9 | fin de (M5), Proposition 5, (M6), debut de (M7) | rien a signaler |
| 10 | tableau 2, fin de (M7), section 4 | le tableau 2 coupe l'item (M7) (nit, C-25) |
| 11 | fin de 4.3, Reproducibility, references | rien a signaler |
| 12 | references [22] a [55] | rien a signaler |

Chaque figure et chaque tableau est appele dans le texte (figure 1 : 3 appels ; figure 2 : 2 ; figure 3 : 2 dans le
corps plus la legende ; figure 4 : 3 ; tableaux 1 et 2 : 2 chacun). Aucune formule tronquee, aucun debordement de
marge. Panneau 4(c) : sur le modele, les losanges pleins (permutation) sont au-dessus des losanges vides (independant)
a b = 20 et 50, ce que le texte ne commente pas (C-04).

## 9. (h) Reproductibilite et README

| Chemin ou affirmation | Existe / exact ? |
|---|---|
| `archive/report/main.pdf` | oui |
| `muon/notes_theory.md` (F1-F16 ; F12-F16) | oui, F1 a F16 presents |
| `archive/muon/notes_spectral_theory.pdf`, `archive/muon/report_spectral_muon_new.pdf` | oui |
| `theory/main.pdf` contient la note B4 | oui (`theory/main.tex` l. 25) |
| `muon/findings.md`, `muon/results/digest.md`, `results/findings.md` (section 9 = B4) | oui |
| `exp_M1`, `exp_M2`, `exp_M3`, `exp_M5b`, `exp_M6`, `exp_M7`, `exp_M8`, `exp_M9`, `exp_M9b`, `exp_B3`, `exp_B4` | oui (`muon/experiments/`, `experiments/`) |
| `paper/scripts/make_figR1_classical.py`, `make_figP2_geometry.py`, `make_figP3_noise.py`, `make_figP4_recipe.py` | oui |
| liste des experiences complete ? | non : manquent `exp_B1` (chiffres de C5), `exp_B2` (C3), `exp_A3` (Katyusha, C3), `exp_gamma_delta` (C4), `exp_M5` ("grids of factor 2" de M6), `muon/theory_checks/` (Proposition 4, figure 3(b)-(c)) (C-23) |
| README (l. 4, 20, 27-30) : 55 entrees toutes citees, 12 pages, 4 figures | oui |
| README (l. 50) : "(M1)-(M4) ... `muon/findings.md` (Sections 5.1-5.10)" | les sections 5.3 a 5.6, 5.8, 5.9 n'existent pas dans le fichier (seuls 5.1, 5.2, 5.7, 5.10 a 5.13 ; marqueurs `<!-- M1M2 -->`, `<!-- M7M8 -->`) (C-23) |
| README (l. 53-54) : "it changed no statement and no number" | le mot "absolute" de (M1) a disparu a la condensation (C-10) |
| "The labels (C1)-(C5) and (M1)-(M4) are those of the extended report" (l. 269-270) | le rapport etendu a (C1) a (C6) et (M1) a (M4), memes sujets, intitules de (C4), (M1), (M3), (M4) reformules (nit, C-24) |

## 10. Constats

### C-01 (major) - (M2), l. 502-503 : chiffres a pas non regles presentes sans reserve, et "fails where MISO fails" contredit par (M3)

Citation : "With the same indices the joint SAGA-Muon needs the passes of MISO where MISO converges ($11.8$ against
$11.9$, softmax with $\kappa=n$; SAGA: $20.2$) and fails where MISO fails ($\kappa\gg n$): it is not a spectral method."

Preuve. `muon/results/M6_summary.json` : `tol = 1e-06`, 3 graines, 40 passes, parametres fixes ("SAGA (Euclidean),
c = 1/4" : passes 20.1, 20.2, 20.2 ; MISO : 12.0, 11.9, 11.5 ; "joint SAGA-Muon, c = 1/4" : 11.8, 13.2, 11.7).
L'instance est la meme que `soft-mem` de `M1_summary.json` (softmax, n = 1000, K = 10, d = 50, kappa = 1000, graine 0),
ou SAGA regle atteint 1e-6 en 11.4 passes et 1e-9 en 17.1. Le journal `muon/findings.md` l. 403-404 l'ecrit :
"Its speed-up over SAGA at textbook steps [...] is MISO's, and tuned SAGA recovers it". De plus la section 2.3
(l. 263) annonce la cible 1e-9 "where memory limits" et dix graines : (M2) n'y obeit pas et ne le dit pas.
Pour kappa >> n : dans M6 (kappa/n = 10, 40 passes) rien n'atteint 1e-6, SAGA euclidien compris ; mais sur les
instances limitees par le conditionnement du papier (R^2/mu de 9n a 50n), a pas regles, `M1_summary.json` donne pour
SAGA-Muon joint 29.0, 43.55 (9/10), 28.1, 20.3 passes et pour MISO 42.7, 120.65, jamais (0/10), 29.15. (M3), l. 375-377,
cite lui-meme 29.0 / 43.5 pour la methode jointe. La phrase "fails where MISO fails ($\kappa\gg n$)" n'est donc vraie
qu'au pas de manuel transfere.

Correction proposee (l. 502-503) :
`With the same indices and untuned textbook-type steps (three seeds, target $10^{-6}$) the joint SAGA-Muon needs the passes of MISO where MISO converges ($11.8$ against $11.9$, softmax with $\kappa=n$; SAGA at $\gamma=1/(4R^2)$: $20.2$, tuned: $11.4$) and stalls where MISO stalls ($\kappa\gg n$); tuned, it converges there \finding{M3}: at batch one it acts as a Euclidean method, not a spectral one.`

Confiance : haute.

### C-02 (major) - (M5), l. 538, et conclusion, l. 678-679 : "the same schedule" n'est pas le cooldown des courses avec table

Citation : "Without a table the same schedule freezes the iterate at $8\%$ to $93\%$ of the initial squared distance" ;
conclusion : "converges to rounding error where the same schedule freezes without one".

Preuve. `M9_summary.json`, partie C (`consistency_m5_schedules.py`). Avec table : facteur polaire exact a q = 0.8,
lr_0 = 0.1 (softmax) ou 0.3 (moindres carres), 5/5 graines, plancher 2.8e-29 et 2.3e-29 ; recette de Muon a q = 0.8,
lr_0 = 0.1, 5/5, plancher 9.8e-19 et 1.1e-18. Sans table (`controls_schedule`) : seulement q = 0.7 (softmax) et
q = 0.75 (moindres carres), lr_0 = 0.3, le point regle de la recette de Muon. A ce cooldown-la, la recette avec
table ne reste sur le cooldown que pour 1 graine sur 5 en softmax (distance relative mediane 1.3e-7) et 4 sur 5 en
moindres carres. Aucune course sans table n'existe a q = 0.8. La legende de la figure 4(a) est exacte ("the schedule
selected for Muon's recipe ($q=0.75$, lr_0 = 0.3)") ; c'est le corps du texte et la conclusion qui disent "the same".
Le gel sans table n'est pas en doute (estimateur qui ne s'annule pas en W_*), mais la comparaison "au meme cooldown"
n'a pas ete mesuree.

Correction proposee (l. 538) :
`Without a table the schedule selected for Muon's recipe ($q=0.7$ or $0.75$, $\mathrm{lr}_0=0.3$) freezes the iterate at $8\%$ to $93\%$ of the initial squared distance, for the`
et (l. 679) : `geometric cooldown, converges to rounding error where such a schedule freezes without one.`

Confiance : haute.

### C-03 (major) - Tableau 2, l. 627-628 ; discussion, l. 645-646 ; conclusion, l. 680 : "not with minibatches" n'est montre que contre SAGA et B-SAGA

Citation : "A change of norm buys trajectory properties, reproduced by Euclidean devices at batch one \finding{M3} but
not with minibatches \finding{M6}." ; "with minibatches the spectral map pays".

Preuve. Les familles lancees en minibatch (`M9_summary.json` partie A, `M5b_summary.json`) sont : SAGA, B-SAGA,
SAGA-Muon a echelle nucleaire, et les pas de longueur fixe (polar, ns5, frob, sign, avec ou sans Nesterov). Ni la
metrique de covariance ni l'echantillonnage d'importance, qui sont les dispositifs euclidiens de (M3), n'y figurent.
Or, en passes (l'unite du papier), la metrique de covariance a batch un fait deja mieux que la variante spectrale la
plus rapide a b = 16 ou 32 sur trois instances sur quatre : soft-cond 15.05 contre 22.4 / 25.7 ; ls-cond 16.05 contre
27.7 (9/10) / 30.8 ; ls-orth 13.15 contre 16.6 / 17.4 ; soft-sq 13.5 contre 13.6 / 13.2 (egalite)
(`M1_summary.json`, `consistency_m6_controls.py`). Le corps de (M6) est exact sur ses comparateurs ; c'est la
generalisation du tableau 2, de la discussion et de la conclusion qui depasse la mesure.

Correction proposee (l. 627-628) :
`A change of norm buys trajectory properties, reproduced by Euclidean devices at batch one \finding{M3}; with minibatches the spectral map beats normalized steps and (B-)SAGA \finding{M6}, the covariance metric not being run there. \\`
et, en fin de (M6), l. 594 : `Passes do not count the factorization of each step, and the covariance metric was not run with minibatches (at batch one it needs $13.1$--$16.1$ passes on these instances).`

Confiance : haute sur les faits, moyenne sur la severite.

### C-04 (major) - (M7), l. 600-606 ; tableau 2, l. 638 ; conclusion, l. 680-681 : le gain de la permutation est attribue au plafond memoire de la table, que le modele ne montre pas a b = 20, 50

Citations : "Linear SAGA explains part of it [...] On such a sum the recipe follows $1.2$--$1.3$ nats of lr^2 per pass
with independent indices and $1.7$--$1.9$ with a permutation" ; tableau 2 : "At most one nat per pass under independent
sampling \finding{M3}, two under a permutation (note B4)." ; conclusion : "a fresh permutation per pass, which lifts the
memory rate of SAGA from $e^{-1}$ to nearly $e^{-2}$, pays where memory limits."

Preuve. (i) `M9b_summary.json`, partie D (`consistency_m7_order.py`) : sur le modele, rho_star vaut 0.22 avec les deux
ordres a b = 20 et b = 50, et les passes vers 1e-9 passent de 8.7 a 9.4 (b = 20) et de 7.0 a 9.1 (b = 50) avec la
permutation, soit plus lent. Les valeurs 1.2-1.3 et 1.7-1.9 nats viennent de la partie L, a batch un (non dit dans le
papier). Le journal `muon/findings.md` l. 829-838 conclut : "On the model a fresh permutation per pass [...] changes
nothing at b = 20 and 50 (0.22 both ways) [...] The large gain of reshuffling measured in part D is therefore not a
property of the table alone". La figure 4(c) le montre (losanges pleins au-dessus des vides) sans que le texte le
dise. Le gain de 36-49 % est donc mesure aux tailles de batch ou le modele de la table n'en predit aucun.
(ii) Les taux "un nat" et "deux nats" sont prouves pour l'erreur quadratique moyenne de SAGA lineaire sur n
quadratiques de hessienne commune, n tendant vers l'infini (`theory/B4_sampling_order.tex`, `prop:B4-iid-limit` a c
fixe, `prop:B4-rr-ms` a g fixe), et la note ecrit l. 340 "Not proved: that a typical run decays at the rate of E e^2".
(M7) dit lui-meme que les courses individuelles sous tirage independant perdent environ 1.6 nat par passe : "At most
one nat per pass" sans "in expectation" (tableau 2) le contredit, et "the memory rate of SAGA" (conclusion) generalise
un enonce de modele.

Corrections proposees. (M7), l. 605-606 :
`batches but $1.0$ to $1.3$ times with a permutation at $b=20$ and $50$; on the model itself the order changes nothing at these batch sizes ($\rho_\star=0.22$ both ways), so the gain on the instances is not a property of the table alone, and why they lose more under independent batches is not established. A permutation`
Tableau 2, l. 638 : `In expectation, at most one nat per pass under independent sampling \finding{M3}, up to two under a permutation (linear model, note B4). &`
Conclusion, l. 680-681 : `normalized steps do as well; with minibatches the spectral map pays, and a fresh permutation per pass, which in a linear model lifts the expected memory rate of SAGA from $e^{-1}$ to nearly $e^{-2}$, saves passes where memory limits.`

Confiance : haute sur les faits, moyenne sur la severite.

### C-05 (minor) - Conclusion, l. 675-676 (et commentaire d'en-tete, l. 2-3) : "no fixed change of geometry" plus large que le Theoreme 1

Citation : "no fixed change of geometry repairs the worst case on ridge logistic and softmax models".
Probleme : le Theoreme 1 couvre les normes. Une geometrie de Bregman fixe n'est pas une norme : avec la fonction de
reference h = F, mu_h(F) = 1 et L_h(f_i) <= n (car Hess f_i <= n Hess F), donc le rapport par echantillon est au plus
n, quel que soit kappa. Le papier cite lui-meme les methodes de Bregman (dragomir2021fast) pour "SAGA in a fixed
metric". Titre, resume et theoreme disent bien "norm".
Correction proposee : `global constants, as does the exact formula of the nuclear-scaled polar step, and no change of norm repairs the`
Confiance : haute.

### C-06 (minor) - Resume, l. 99 ; conclusion, l. 677 : "decreasing lengths" n'est pas l'hypothese du Theoreme 2

Citation : "With additive noise and decreasing deterministic lengths, stochastic polar steps [...] converge to a point
where the gradient is minus a nuclear median" ; "with decreasing lengths, polar steps converge to minus a nuclear
median of the noise".
Probleme : le Theoreme 2(b) demande une somme infinie et des carres sommables. Le cooldown geometrique de la
section 3.3 est decroissant, deterministe, et (M5) montre qu'il gele l'iterer : pris a la lettre, le resume est
contredit par le papier. La conclusion perd en plus "additive" et "deterministic".
Correction proposee (l. 99) : `With additive noise and deterministic lengths that are square-summable but not summable, stochastic polar`
(l. 677) : `precision: with such lengths and additive noise, polar steps converge to a point where the gradient is minus`
Confiance : haute.

### C-07 (minor) - Tableau 2, l. 632 : renvoi a la Proposition 4 pour un enonce du Theoreme 2

Citation : "The limit gradient is minus a nuclear median of the noise (Proposition~\ref{prop:offset})."
Probleme : c'est le Theoreme 2(b), prouve ; la Proposition 4 ne donne que la taille de l'offset, et elle est formelle.
Correction proposee : `The limit gradient is minus a nuclear median of the noise (Theorem~\ref{thm:smd}(b); size: Proposition~\ref{prop:offset}). \\`
Confiance : haute.

### C-08 (minor) - Introduction, l. 132-133 : la Proposition 5 citee pour un enonce mesure

Citation : "fixed lengths on a geometric cooldown, fed by a SAGA table, converge to rounding error where other
estimators freeze (Proposition~\ref{prop:schedule})".
Probleme : l'erreur d'arrondi et le gel des autres estimateurs sont (M5), etiquete [e] dans le tableau 1 ; la
Proposition 5 certifie D_t <= D_0 rho^t pour des cooldowns 1e4 a 1e5 fois plus lents et ne dit rien des autres
estimateurs.
Correction proposee : `other estimators freeze \finding{M5}, which Proposition~\ref{prop:schedule} certifies at slower cooldowns; with minibatches the spectral map then pays, and a fresh`
Confiance : haute.

### C-09 (minor) - (M3), l. 382-384 : "near the ceiling" alors que les meilleures familles depassent un nat par passe

Citation : "the best family of every class needs $16$--$18$ (softmax) and $17$--$21$ passes (least squares) to
$10^{-9}$, near the ceiling of SAGA under independent sampling, $e^{-1}$ per pass in expectation".
Probleme : 1e-9 represente 20.7 nats ; 16 a 18 passes (premiere passe d'initialisation de la table comprise) font
1.15 a 1.30 nat par passe, au-dessus du "plafond". L'explication (le facteur memoire borne l'erreur moyenne, pas une
course) n'arrive qu'en (M7), l. 604-605. `M1_summary.json`, soft-mem, 1e-9 : 15.75 a 18.05 passes selon la famille.
Correction proposee : `near the ceiling of the mean error of SAGA under independent sampling, $e^{-1}$ per pass (single runs are faster, \finding{M7}), since stale entries keep their error.`
Confiance : haute.

### C-10 (minor) - (M1), l. 494 : "squared distance 3.8 eta^2" est absolue, tout le reste est relatif

Citation : "with the SAGA estimator the iterates settle at a squared distance $3.8\,\eta^2$ (softmax) or $6.0\,\eta^2$
(least squares)".
Preuve : `M3_summary.json`, longueur fixe 1e-2, softmax : distance relative 4.6e-7, soit 3.8e-4 = 3.8 eta^2 apres
multiplication par la norme carree de W_* (826.97) ; moindres carres : 5.24e-7 x 1147 = 6.0e-4. La figure 3(a) trace
la distance relative (plateau a 4.6e-7) et la meme phrase donne ensuite 1.2e-1 et 4.8e-2, relatifs. La version
anterieure du papier disait "an absolute squared distance" (citee dans `muon/findings.md` l. 891) : le mot est tombe
a la condensation, alors que le README affirme qu'elle n'a change aucun enonce.
Correction proposee : `squared distance` remplace par `absolute squared distance` (l. 494).
Confiance : haute.

### C-11 (minor) - (M4), l. 372-373, contre (M5), l. 541-542 : "covariance metric" designe deux familles

Citation : "(SAGA $27.1$, covariance metric $13.1$)" ; "more than in the covariance metric ($13.5$--$16.1$)".
Preuve : `M1_summary.json`, instance `ls-orth` (une des quatre instances limitees par le conditionnement) :
cov-metric SAGA 13.15, cov-metric B-SAGA 13.9. La plage 13.5-16.1 de (M5) est celle de B-SAGA dans la metrique de
covariance (15.05, 16.05, 13.9, 13.5) ; le 13.1 de (M4) est SAGA non amorti. Le lecteur trouve 13.1 hors de la plage
annoncee pour la meme metrique.
Correction proposee : (l. 372) `(SAGA $27.1$, SAGA in the covariance metric $13.1$)` ; (l. 541-542) `more than damped SAGA in the covariance metric ($13.5$--$16.1$) but for a tie`.
Confiance : haute.

### C-12 (minor) - Resume, l. 96-97 ; tableau 1, l. 151 ; tableau 2, l. 626 : "the same anatomy" sans "on quadratics"

Citation : "the nuclear-scaled polar step has the same anatomy".
Probleme : l'eq. (2) est une identite pour une quadratique (l. 226) ; l'ecart y vaut environ 4 (l. 313-314) et rien
n'est mesure pour le pas polaire sur pertes logistiques ("not measured for the polar step", l. 315). Place apres les
chiffres 3.2-220 et 77-96 % de la regression logistique, "the same anatomy" laisse croire a une mesure analogue.
Correction proposee (l. 96-97) : `comes from the two inequalities that replace the curvature met along the run by global constants; on quadratics the exact formula of the nuclear-scaled polar step has the same anatomy.`
Confiance : haute.

### C-13 (minor) - (C2) contre (C3) et Limitations, l. 667-668 : 77-96 % et 59-96 % sont la meme part dans deux ordres

Citations : "I2 and I4 [...] carry $77$--$96\%$ of the logarithmic gap" (l. 292-293) ; "Rewritten with mu_eff and
L_eff, Proposition 5.9 recovers $59$--$96\%$ of the logarithmic gap for $\kappa\ge10^4$" (l. 303) ; "the shares of
(C2) depend on the order in which the inequalities are undone" (l. 667-668).
Preuve : `results/B3_summary.json`, champ `shares_orderA` (ordre I4, I2, I3, I1, I5 de la note B3), recalcule par
`consistency_b3_gap.py` : part de I2 + I4 pour kappa >= 1e4 : 76.9-96.1 % dans l'ordre du papier, 59.0-96.4 % dans
l'ordre A (minimum a kappa = 1e4, alpha = 0). Le 59-96 % de (C3) est donc la meme part dans l'autre ordre, sans que
le papier le dise. Pour kappa <= 1e2 la dependance est totale : I2 passe de 31.7-36.5 % a 0 %, I5 de 7-22 % a
56-61 %, I1 reste a 39-44 %. La phrase de (C2) "I1 (39-44 %) and I2 (32-37 %) do" ne tient que dans l'ordre choisi.
Correction proposee (l. 667-668) : `the order in which the inequalities are undone (undoing I4 and I2 first gives the $59$--$96\%$ of \finding{C3}, and for $\kappa\le10^2$ moves the share of I2 to I5), \finding{M3}--\finding{M7} rest on two to four instances and one to three`
Confiance : haute.

### C-14 (minor) - Section 2.3, l. 264-265 : "Section 3.3 uses the same protocol" inexact pour la loi de (M5) et pour (M7)

Citation : "Section~\ref{sec:recipe} uses the same protocol, with step grids of ratio $2^{1/4}$."
Preuve : `M9_summary.json`. Parties A et B : `tune_seeds` 0-1, `refine_seeds` 2-4, `confirm_seeds` 10-19, conforme.
Partie C (loi de M5) : cinq graines 10-14 aux points choisis. Partie D (M7) : cinq graines 0-4, pas de graines de
confirmation, "fastest = the rho with the smallest median passes to 1e-09 among those where every seed reaches it"
(selection et mesure sur les memes graines), et "lr_0 = 4 ||W_*||_op (1 - q^(b/n)) / (1 - 1e-05)" : la longueur
initiale est fixee a partir de la norme du minimiseur exact. Le 36-49 % du resume vient de cette partie. Les
legendes de la figure 4 donnent bien "five seeds" ; l'effet (17.9 contre 9.1 passes) depasse largement la dispersion
entre graines, donc la conclusion n'est pas en cause, seulement la description du protocole.
Correction proposee : `Section~\ref{sec:recipe} uses the same protocol for the passes to $10^{-6}$, with step grids of ratio $2^{1/4}$; the long runs of \finding{M5} and \finding{M7} use five seeds, \finding{M7} with $\mathrm{lr}_0$ set from $\nop{W_\ast}$.`
Confiance : haute.

### C-15 (minor) - Apres la Proposition 5, l. 578-579 : le facteur 3.3e4-8.9e4 vaut pour les ordres cyclique et permute, pas pour le tirage independant de (M5)

Citation : "on the two non-square conditioning-limited instances its cooldown is $3.3\cdot10^4$ to $8.9\cdot10^4$ times
slower, in $\log(1/q)$, than $q=0.8$".
Preuve : `muon/notes_theory.md`, F13, point (1), l. 657-661 : 8.9e4 et 6.5e4 pour l'ordre permute, 4.5e4 et 3.3e4 pour
l'ordre cyclique. Les courses de (M5) tirent des indices independants (pas de cle `order` dans la partie C ; F13
point (2), l. 662-663 : "independent sampling"), cas ou la Proposition 5(b) ne vaut qu'avec grande probabilite sur un horizon
fini et avec un A plus grand d'un facteur log(nT/delta). L'ecart reel est donc plus grand que celui annonce.
Correction proposee : `cooldown is $3.3\cdot10^4$ to $8.9\cdot10^4$ times slower, in $\log(1/q)$, than $q=0.8$ (cyclic and permuted orders; more for the independent indices of \finding{M5}), and along the fast runs the error`
Confiance : haute.

### C-16 (minor) - Resume, l. 106-107, et tableau 2, l. 640 : "36-49 %" n'apparait pas dans les resultats

Probleme : (M7) donne "from $16.7$--$20.0$ to $9.1$--$12.2$", dont les bornes donneraient 27 a 54 %. Le chiffre du
resume est exact par couple (instance, b) : 49.2, 37.7 (softmax, b = 20, 50), 43.5, 36.1 % (moindres carres).
Correction proposee (l. 599) : `and $50$, moves $\rho_\star$ to $0.10$--$0.18$ and the passes to $10^{-9}$ from $16.7$--$20.0$ to $9.1$--$12.2$ ($36$--$49\%$ fewer). Linear`
Confiance : haute.

### C-17 (minor) - Notations surchargees et symboles non definis

Detail en section 7. Les conflits qui peuvent tromper : `\LEff` (l. 215 contre l. 556), `\rho` (l. 257, 557, 597),
`c_i` (l. 202 contre l. 568-569), `g`, `g_j`, `g_i`, `g_k` (l. 242, 337, 428, 549, 602), `\eta_\ast` minimiseur
contre `\eta` longueur. Non definis : `\bar c` et `\hat\Sigma` (l. 240, 328).
Corrections proposees (minimales) : Proposition 5 : `let $L_A=\Lop+10AL_{\max}$, $\rho=1-\muop/L_A$` et `L_A` a la
place de `\LEff` aux l. 557, 572, 573 ; preuve (a) : `$\omega-\nabla F(W)=\sum_iw_ie_i$ [...] $e_i=(\nabla\ell_i(Wx_i)-\nabla\ell_i(\phi_ix_i))x_i^\top$ [...] $\nnuc{e_i}\le L_i\nop{W-\phi_i}$` ;
l. 257 : `clipped ($\min\{\bar\eta,\gamma\nnuc\omega\}$)` ; l. 240 : ajouter la definition de `\hat\Sigma` et de
`\bar c` (a fournir par les auteurs : la valeur de `\bar c` n'a pas ete retrouvee dans le code).
Confiance : haute.

### C-18 (minor) - Theoreme 2, l. 408-416 : "Proof sketch" sans renvoi vers la preuve complete

Probleme : la Proposition 5 dit ou sont les details (F12-F14) ; le Theoreme 2, etiquete [p], non. La preuve complete
est dans la note archivee (`archive/muon/notes_spectral_theory.tex`, section 5, theoreme "stochastic polar steps are
stochastic mirror descent"), que la section 2.4 ne distingue pas de `muon/notes_theory.md` ("the theory notes").
Correction proposee (fin de l. 415) : `minimizers. (c): [...] \cite{balles2020geometry}. Details: archived theory notes, Section~5.`
Confiance : haute.

### C-19 (minor) - Proposition 3(d), l. 427 : hypothese "sans ridge" implicite

Citation : "(d) (Batch one on least squares.) Let $g_j(W)=(Wx_j-y_j)x_j^\top$".
Probleme : le modele du papier (GLM) est regularise ; avec le terme lambda W dans l'estimateur, g_j n'est plus de rang
un et l'enonce tombe. La legende de la figure 3 dit "Unregularized" ; la note archivee dit "(no ridge inside the
polar factor)". Le tableau 1 ("at batch one on least squares") herite de l'omission.
Correction proposee : `(d) (Batch one on unregularized least squares.) Let`
Confiance : haute.

### C-20 (minor) - (C4), l. 321-323 : borne inferieure sans point de depart

Citation : "SAGA with any constant step needs at least $\frac{\kappa-1}4\log\frac1{2\varepsilon}$ iterations to bring
$\E\norm{\theta_t}^2$ below $\varepsilon\norm{\theta_0}^2$".
Probleme : `theory/B3_prop59_tightness.tex`, `prop:B3-lower`, suppose theta_0 = (r, r). Contre-exemple pour l'enonce
tel qu'ecrit : theta_0 = (0, 1), table initialisee aux gradients en theta_0, gamma = 1/R^2 : avec n composantes
identiques SAGA est la descente de gradient et theta_1 = 0.
Correction proposee : `constant step, started at $\theta_0=(r,r)$, needs at least`
Confiance : haute sur le fait ; la severite depend de la convention de lecture d'une borne inferieure.

### C-21 (minor) - Limitations, l. 667-672 : deux reserves absentes

Probleme : (i) la table stocke un gradient par echantillon (un residu de taille K pour ces modeles lineaires
generalises), ce qui borne la portee de la question "Can Muon's own recipe use a finite-sum estimator?" ; (ii) les
Theoremes 2 et la Proposition 5(a)-(b) sont pour le facteur polaire exact, alors que Muon utilise Newton-Schulz
(seule la Proposition 5(c) et le texte l. 536-537 en parlent).
Correction proposee (l. 669) : `draws of the data, counts are in gradients, not factorizations, and the table stores one residual per sample. Theorem~\ref{thm:nfl} concerns global constants, not`
et (l. 670) : `Theorem~\ref{thm:smd} assumes additive noise, deterministic lengths, the exact polar factor and no momentum;`
Confiance : moyenne (jugement de portee).

### C-22 (nit) - Titre, l. 82-83 : "variance reduction" vaut pour les estimateurs a somme finie

Probleme : le papier appelle "variance-reduced versions" les variantes a moment (l. 120) et montre qu'elles gardent
un plancher ((M1) l. 496, (M5) l. 539). "variance reduction acts on the fixed point" ne vaut que pour SAGA et SVRG
(Proposition 3(c)) ; la conclusion ecrit d'ailleurs "finite-sum variance reduction".
Correction proposee : aucune (le resume qualifie aussitot) ; variante possible du titre : `finite-sum variance reduction acts on the fixed point`.
Confiance : moyenne.

### C-23 (minor) - Paragraphe Reproducibility, l. 686-688, et README : liste d'experiences incomplete, sections absentes

Preuve : `results/digest.md` : les chiffres de (C5) viennent de "B1: preconditioned SAGA", ceux de (C4) de
"(gamma, Delta) inside the proof", ceux de Katyusha en (C3) de "A3", ceux de mu_eff de "B2" ; "on grids of factor 2"
en (M6) vient de `exp_M5_batch.py` ; les chiffres de la Proposition 4 et de la figure 3(b)-(c) de
`muon/theory_checks/`. Aucun n'est dans la liste. README : "muon/findings.md (Sections 5.1-5.10)" alors que le
fichier n'a que 5.1, 5.2, 5.7, 5.10 a 5.13 (`grep '^### 5\.'`).
Correction proposee (l. 687-688) : `\nolinkurl{experiments/} and \nolinkurl{muon/experiments/} (\texttt{exp\_M1}--\texttt{exp\_M3}, \texttt{exp\_M5}, \texttt{exp\_M5b}, \texttt{exp\_M6}--\texttt{exp\_M9b}, \texttt{exp\_A3}, \texttt{exp\_B1}--\texttt{exp\_B4}, \texttt{exp\_gamma\_delta}) and \nolinkurl{muon/theory_checks/}, figures from \nolinkurl{paper/scripts/}. A large`
Confiance : haute.

### C-24 (nit) - Section 2.4 et legende du tableau 1

La legende renvoie a la section 2.4 pour les etiquettes, qui ne definit que [p]. "The labels (C1)-(C5) and (M1)-(M4)
are those of the extended report" : le rapport etendu a aussi (C6), fondu ici dans (C3).

### C-25 (nit) - Placement des flottants

Figure 4 au-dessus de (M1) et (M2) (page 8) ; tableau 2 au milieu de l'item (M7) (page 10). Annotations "7/10",
"9/10" de la figure 4(b) sans indication de serie.

### C-26 (nit) - "pays", "where memory limits"

"the spectral map pays" peut se lire "paie un cout" ; "pays off" leve l'ambiguite (l. 133, 179, 584, 680). "where
memory limits" est employe dans le resume avant sa definition (l. 261).

### C-27 (nit) - Arrondis des demi-dixiemes

69.55 est ecrit 69.5, 27.15 ecrit 27.1, 13.15 ecrit 13.1, mais 15.05 ecrit 15.1, 16.05 ecrit 16.1, 24.35 ecrit 24.4
(arrondi du flottant) : ecart d'au plus 0.05 passe, sans consequence.


## 11. Non verifie

- Le detail des preuves (Theoremes 1 et 2, Propositions 3 a 5, notes B1, B3, B4, F12 a F15) : seule l'existence des
  enonces et des preuves a ete controlee, plus quelques verifications a la main (exemple de la l. 393, bornes de J et
  mu_PL, contre-exemple de C-20).
- Les attributions bibliographiques et les mentions "not found" : hors perimetre.
- "Muon's own recipe" : que le cooldown geometrique et le moment de Nesterov 0.9 correspondent a la recette publiee
  de Muon n'a pas ete controle contre la source.
- Les chiffres non repetes ((C5), (C3) Katyusha, (C2) "eight data sets", (M3) "16-18" et "17-21", "0.20-0.37",
  "0.78-1.08", Proposition 4 et controles numeriques, (M6) b = 4) n'ont pas ete recalcules : seuls les chiffres
  repetes ou en tension l'ont ete.
- Les scripts des figures n'ont pas ete relances (ils ecrivent dans `paper/figures/`, en lecture seule ici).
  `results/B3.pkl` est absent du depot : la figure 1 (fichiers dates du 18 novembre, script modifie le 20) ne peut
  pas etre regeneree sans relancer `exp_B3_tightness.py`. L'affirmation du README sur l'arret des scripts des
  figures 2 et 3 en cas de desaccord avec le digest n'a pas ete testee.
- La valeur de `\bar c` dans la metrique de covariance n'a pas ete retrouvee dans le code.
