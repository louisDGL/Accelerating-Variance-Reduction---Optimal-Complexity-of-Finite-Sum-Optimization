# Audit code-fidelity : fidélité du code au papier et équité des comparaisons

Statut : TERMINÉ (20 novembre 2026). Papier audité : `paper/paper_saga_to_muon.tex` (commit `9ea8ee7`).
Rien n'a été modifié hors de `paper/audit/` ; aucun cache de `muon/results/` n'a été lu en écriture
(les reproductions passent par `common.run_one`, qui n'écrit rien).

## 1. Périmètre couvert

- (a) Algorithmes, lus en entier : `muon/lab/polar.py`, `vrmuon.py`, `algorithms.py`, `maps.py`, `mvr.py`, `katyusha.py`,
  confrontés aux lignes 196-206, 253-262, 508-515 et à l'énoncé de la Proposition 5 (l. 547-553).
- (b) Protocole : `muon/experiments/common.py`, `families.py`, `exp_M1_headtohead.py`, `exp_M9_schedule.py`,
  `exp_M6_miso.py`, `exp_M3_floors.py`, `exp_M8_dataseeds.py`, en-têtes de `exp_M5b_batch_fine.py` et
  `exp_M9b_order_model.py` ; sélections et drapeaux de bord dans `M1_summary.json`, `M9_summary.json`.
- (c) Reproductions avec le code du dépôt : (M5) planchers et gel, (M2) MISO contre SAGA-Muon joint, (M6) deux
  rapports, (M3) trois cellules de confirmation de M1 ; contrôles d'équité des témoins sans table.
- (d) Partie SAGA : `vrlab/algorithms.py` (saga, ordres), `vrlab/instrumentation.py`, `vrlab/bounds.py`,
  `vrlab/analysis.py` (chaîne d'attribution), `experiments/exp_B3_tightness.py` (plan).

Scripts (tous sous `paper/audit/checks/`, sorties `.out` à côté) :
`code-fidelity_lab_units.py` (47 contrôles unitaires), `code-fidelity_repro.py` (parties R1, R2, R2b, R3, R4, R4b),
`code-fidelity_saga_exact.py` (12 contrôles).

## 2. Verdict d'ensemble

Le code fait ce que le papier décrit : aucun bogue trouvé qui changerait un chiffre, et toutes les valeurs rejouées
sont retrouvées au chiffre près. Les problèmes sont d'équité et de description :

| id | sévérité | lignes | objet |
|---|---|---|---|
| CF-1 | major | 538-540 (échos 104, 178, 523, 679) | les témoins sans table à b = 32 ne peuvent pas atteindre W_* quel que soit l'estimateur : la table elle-même gèle sur ce refroidissement |
| CF-2 | major | 501-503 | (M2) : « SAGA: 20.2 » est SAGA au pas du livre, non réglé ; réglé par le protocole du papier sur la même instance, SAGA demande 11.4 passes, comme MISO et la méthode jointe |
| CF-3 | minor | 263-265 | « tuned on five seeds », « Section 3.3 uses the same protocol » : protocole à trois étages ; (M1), (M2), les planchers de (M5) et (M7) sont des plans à paramètres fixes |
| CF-4 | minor | 257-260 | deux ingrédients réglés non décrits : epsilon de la carte « soft » (multiple de la norme de l'innovation), exposant 1/2 de l'échantillonnage d'importance |
| CF-5 | nit | 514-515 | le contrôle « signe » est reprojeté sur la jauge pour softmax (sans effet : le contrôle le plus rapide est toujours la normalisation de Frobenius) |
| CF-6 | nit | 262, 511 | conventions non dites : la passe initiale de la table est comptée ; méthodes publiées sans table tirées avec remise et moment 0.95 |

Constats d'autres auditeurs que ma lecture du code confirme sans les redoubler : grilles de la recette (num-m6m7 F1 :
`families.py` l. 271, 283-293), « on the same cooldown » (num-m6m7 F5), grille de facteur 2 de SAGA dans M1
(num-geometry G2 : `families.py` l. 48, dix points, quatre ou cinq gardés après criblage).

## 3. Affirmations contrôlées

### 3.1 Algorithmes (a)

| Affirmation du papier | Verdict | Preuve |
|---|---|---|
| l. 513 : « Muon's five Newton--Schulz steps (ns5, normalization constant $\epsilon=10^{-7}$) » | OK | `polar.py` l. 61, 157-181 : coefficients (3.4445, -4.7750, 2.0315), X/(‖X‖_F + 1e-7), 5 itérations, transposition si K > d. Égal à une réécriture indépendante de la routine publique à 1e-15 près sur 8x40, 20x20, 40x8 ; sortie dans [0.69, 1.12] ; non invariant d'échelle avec 1e-7, invariant avec 1e-30 (`lab_units`). Toutes les expériences sont en float64 (`bf16=False`). |
| l. 197-198 : polar(ω) = UV^T, polar(0) = 0 | OK | `polar.polar_svd` (seuil relatif 1e-15) ; pour softmax la factorisation est faite dans le sous-espace 1^T W = 0 (`maps.sharp(basis=...)`), ce qui est la définition du papier restreinte aux s > 0. |
| l. 253-256 et Prop. 5 : ω = m + ϑc, m = z̄ + λW, table remplacée par le gradient au point courant | OK | `vrmuon.py` l. 455-470 (pieces), 564-566 (step), 740-758 (commit). Comparé pas à pas à une table explicite de matrices sur 25 itérations, softmax et moindres carrés, b = 1 et 4, ϑ = 1 et 0.3 : écart 3e-15. La table ne contient que la partie perte (régulariseur exact), comme l'énonce la Proposition 5. |
| l. 204-205, 257 : longueur fixe, nucléaire γ‖ω‖_*, écrêtée min{ρ, γ‖ω‖_*} | OK | `vrmuon._eta` l. 499-504 ; ‖S‖_op vérifié dans les trois cas. Le facteur max(1, K/d)^(1/2) vaut 1 sur les six instances (K ≤ d). |
| l. 255-256 : ϑ = 1/n, γ = 1/λ est MISO | OK | `common.resolve` l. 127-134 ; sur 40 pas W = -z̄/λ à 2e-16 près (minimiseur du substitut). |
| l. 241-242, 258 : métrique polaire figée en W_0, C = V diag(g)V^T + r(I - VV^T), g_k = ‖G_0‖_*/max(s_k, f s_1) | OK | `vrmuon.polar_metric` l. 651-671, `metric_every = 10**9` ; l'ancre est le gradient complet en W_0 ; pas (c/R_P²) ω C avec R_P² = max_i c x_i^T C x_i + λ‖C‖_op, recalculé à 1e-10 ; non rafraîchie après 3 passes. |
| l. 259 : B-SAGA (innovation amortie), échantillonnage d'importance | OK, réserve CF-4 | pas γ(m + ϑc) (`step` l. 563-564) ; p_i ∝ L_i^a, innovation pondérée par 1/(n p_i), estimateur sans biais à 1e-16. Exposant sélectionné : a = 1/2. |
| l. 514 : moment de Nesterov 0.9 | OK | `families.recipe_family` l. 315 ; `step` l. 568-571 : D = (1-β)(ω + β buf), identique à la forme `lerp` de Muon (vérifié à 1e-18 avec la carte frob). |
| l. 511 : lr_t = lr_0 q^{tb/n} | OK | `vrmuon._lr_t` l. 478-480 ; témoins MARS-M/STORM : lr_0 q^{2bt/n} (`exp_M9_schedule.py` l. 497-499), soit la même décroissance par passe de gradients. |
| l. 515 : contrôles ω/‖ω‖_F et signe | OK, réserve CF-5 | `polar.frob_normalize`, `sign_map` ; même boucle, même table, mêmes grilles de q. |
| l. 539 : MARS-M, STORM-Muon | OK (code conforme à sa propre description) | `mvr.py` l. 84-90 : C = G + g β/(1-β)(G - G_prev sur le même lot), écrêtage de Frobenius à 1, M = βM + (1-β)C ; g = 0.025 (MARS-M), g = 1 sans écrêtage (STORM : M = G + β(M - G_prev)) ; β = 0.95, ns5. Non confronté aux articles sources (voir § 6). |

### 3.2 Protocole (b)

| Élément | Verdict | Preuve |
|---|---|---|
| Critère : distance au carré relative 1e-6 (1e-9 où la mémoire limite) | OK | `common.run_one` l. 188, 245 : ‖W - W_*‖_F² / ‖W_*‖_F² (W_0 = 0) ; `passes_to` = premier point enregistré sous le seuil, un enregistrement tous les n/(10b) pas (0.1 passe). Cibles par instance : `exp_M1_headtohead.py` l. 30-38. |
| Coûts en passes | OK, convention non dite (CF-6) | gradients par échantillon / n ; passe initiale de la table comptée (`vrmuon.py` l. 356, 960) ; 0 au départ pour les méthodes sans table ; MARS-M même lot : b puis 2b par pas. |
| Lots | OK | méthodes à table : b indices distincts (`_draws` l. 904) ; « fresh permutation per pass » : `rng.permutation(n)` à chaque passe, coupée en lots consécutifs (l. 872-878), vérifié. |
| Graines | OK, réserve CF-3 | réglage 0-4, confirmation 10-19 (`common.py` l. 62-66 ; champ `protocol` des résumés). |
| Six instances, « R²/μ de l'ordre de n » et « de 9n à 50n » | OK | `M1_summary.json` : R²/μ = 0.86n, 0.48n (mémoire) ; 48.3n, 36.5n, 8.7n, 18.4n (μ au minimiseur ; pour les moindres carrés c'est la constante exacte, pour softmax la constante globale donne 50n et 20n). |
| Équité spectral / contrôle dans (M6) | OK | mêmes grilles de q ; grilles de lr_0 propres à chaque carte et aucune sélection des contrôles avec moment sur un bord (`M9_summary.json` ; les seuls bords, `lr_decay+` et une fois `lr+`, concernent frob et sign sans moment). Le contrôle le plus rapide est frob (b = 1) ou frob + Nesterov (b = 16, 32) dans les 12 cellules. |
| Équité recette / SAGA, B-SAGA dans (M6) | OK (prudent) | les références euclidiennes sont sur des grilles 2^(1/4) (25 et 115 points), plus fines que celles de la recette. |
| Figure 4(b), références à b = 1 | OK | `make_figP4_recipe.py` l. 116-122 : M5b (grilles 2^(1/4)), la partie A de M9 ne les ayant pas lancées. |
| Témoins sans table de (M5) | Problème | CF-1. |
| (M2), protocole | Problème | CF-2. |

### 3.3 Reproductions (c)

| Résultat du papier | Rejoué | Verdict |
|---|---|---|
| l. 535 : plancher 2.3e-29 (moindres carrés), polaire exact avec table, q = 0.8, lr_0 = 0.3 | graine 10 : 3.793e-29 (= maximum des cinq graines du JSON, médiane 2.27e-29) | OK |
| l. 536 : « near 1e-18 » pour la recette de Muon à q = 0.8, lr_0 = 0.1 | graine 10 : 1.318e-18 | OK |
| l. 538 : gel sans table à 8 % (b = 1, moindres carrés) | graine 10 : 8.30e-2 | OK |
| l. 502-503 : 11.8 contre 11.9 ; SAGA 20.2 | graines 0, 1, 2 : joint [11.8, 13.2, 11.7], MISO [12.0, 11.9, 11.5], SAGA au pas du livre [20.1, 20.2, 20.2] | chiffres OK ; comparaison : CF-2 |
| (M6) rapport spectral/contrôle, ls-cond b = 16 et soft-cond b = 32 | 27.7 (9/10) / 38.9 = 0.712 ; 25.7 / 30.5 = 0.843 | OK (identique au résumé) |
| (M3) soft-cond : SAGA 69.5, joint 29.0, B-SAGA 33.8 | dix graines de confirmation, identiques graine par graine au cache | OK |

Remarque sur les caches : `M9_summary.json` et `M9b_summary.json` portent l'empreinte du code actuel
(`40df8a19...`) ; `M1`, `M7`, `M8` une empreinte antérieure (`76badfd4...`), `M2`, `M3`, `M6` aucune. Les trois
cellules de M1 et les trois configurations de M6 rejouées avec le code actuel redonnent les valeurs du cache à
l'identique : les ajouts ultérieurs (refroidissement, ordres, cartes de contrôle) n'ont pas changé ces exécutions.

### 3.4 Partie SAGA (d)

| Affirmation | Verdict | Preuve |
|---|---|---|
| Pas « textbook » γ = 1/(4R²), R² = max_i(‖x_i‖²/4 + λ), δ = 3γ² | OK | `vrlab/bounds.py` l. 10-13, `problems.py` l. 96, 215 ; recalculé (`saga_exact`). |
| l. 190-192 : SAGA | OK | réécriture indépendante, mêmes indices : écart 3e-15 après 637 itérations. |
| l. 222-224 : « Enumerating the $n$ indices gives every conditional expectation along a run exactly » | OK | `saga_state_stats` contre la force brute (les n pas possibles, Lyapunov recalculé de zéro) : V et E[V_+] égaux à 1e-12 en relatif, en métrique euclidienne et en métrique P dense. Précision : l'instrumentation est faite tous les n/10 pas (`exp_B3_tightness.py` l. 43), et l'espérance de G suppose z̄ = moyenne des z (resynchronisée chaque passe). |
| Identité (1) | OK | avec μ_eff, L_eff, r_sq, r_var mesurés : écart relatif 3e-16. |
| l. 224 : ordre I5, I4, I2, I3, I1 | OK | `analysis.attribution` / `cumulative_chain` (ordre B) ; fenêtre passes 2-20, cinq graines, n = 1000, d = 50 (`exp_B3_tightness.py` l. 20-24, 215). |
| Ordres d'échantillonnage de la note B4 | OK | 'rr' : permutation nouvelle par passe ; 'so' : une seule ; l'instrumentation exacte est refusée hors tirage indépendant. |

## 4. Détail des constats

### CF-1 (major) : à b = 32 le « gel sans table » mesure le refroidissement, pas l'estimateur

- Lieu : l. 538-540 ; échos l. 104 (résumé), 178 (tableau 1), 522-523 (légende), 679.
- Papier : « Without a table the same schedule freezes the iterate at $8\%$ to $93\%$ of the initial squared distance,
  for the minibatch gradient ($b=1$, $32$) as for MARS-M \cite{liu2025marsm} and STORM-Muon
  \cite{cutkosky2019momentum}: the cooldown outruns an estimator that does not vanish at $W_\ast$. »
- Ce que fait le code (`exp_M9_schedule.py` l. 481-499) : les témoins tournent sur le point sélectionné pour la
  recette de Muon AVEC table à lot un (softmax : q = 0.7, lr_0 = 0.3 ; moindres carrés : q = 0.75, lr_0 = 0.3), y
  compris à b = 32. Ce n'est pas le refroidissement des planchers cités dans la même phrase (q = 0.8).
- Problème : à b = 32 une passe ne compte que n/b = 6.25 pas (3.1 pour MARS-M et STORM, qui coûtent 2b), et la
  longueur totale Σ_t lr_t = lr_0/(1 - q^{1/période}) vaut 5.4 et 6.7 (2.8 et 3.4), soit 0.08 à 0.37 ‖W_*‖_op :
  W_* est hors d'atteinte quel que soit l'estimateur. Vérification directe (`code-fidelity_repro.py R4`) : la table
  SAGA elle-même, à b = 32 sur ce refroidissement, gèle à 0.76-0.79 (softmax) et 0.51-0.57 (moindres carrés) de la
  distance initiale, contre 0.82-0.83 et 0.62-0.63 sans table. Les valeurs à b = 32 (0.63 à 0.93, la moitié haute de
  « 8 % à 93 % ») ne disent donc rien de la table. Le dépôt le sait pour ses propres familles : les grilles de lr_0
  croissent avec b « because a pass has n / b steps » (`families.py` l. 286).
- Ce qui tient : à lot un le gel est réel et robuste. Sur le refroidissement sélectionné : 8 % à 34 % (médianes de
  `M9_summary.json`, C/*/controls). En ralentissant le refroidissement du gradient de lot (ns5 + Nesterov, 300
  passes, graine 10, `R4b`) : 5.0 %, 4.3 %, 4.0 % à q = 0.9, 0.95, 0.98 (moindres carrés), 11.6 %, 10.6 %, 10.2 %
  (softmax) : l'itéré reste à 4-10 % de la distance initiale, ce qui s'accorde avec le décalage du Théorème 2 à lot un.
- Réserve voisine : sur softmax, au refroidissement des témoins (q = 0.7, lr_0 = 0.3), la recette de Muon avec table
  ne reste sur la loi que pour 1 graine sur 5 (plancher médian 1.3e-7, de 5e-13 à 3e-5 ; `M9_summary.json`,
  C/soft-cond/deep). Le contraste reste de six ordres de grandeur, mais « converges to rounding error where the same
  schedule freezes » (l. 679) n'est vrai au même refroidissement que pour les moindres carrés.
- Correction proposée : voir le JSON (restreindre à b = 1 et nommer le refroidissement).

### CF-2 (major) : (M2), « SAGA: 20.2 » est un concurrent non réglé

- Lieu : l. 501-503.
- Papier : « With the same indices the joint SAGA-Muon needs the passes of MISO where MISO converges ($11.8$ against
  $11.9$, softmax with $\kappa=n$; SAGA: $20.2$) ».
- Ce que fait le code (`exp_M6_miso.py` l. 35-46, 61-63) : paramètres fixes, non réglés, trois graines (0, 1, 2),
  40 passes ; SAGA y est au pas du livre γ = 1/(4R²), la méthode jointe à γ = 1/(4 R2_op).
- Problème : l'instance est exactement l'instance softmax limitée par la mémoire de M1 (même spécification). Le
  protocole du papier y sélectionne γ = 1/(2R²) pour SAGA, qui atteint alors 1e-6 en 11.4 passes
  (`M1_summary.json`, soft-mem, SAGA, 10/10 graines de confirmation). Rejoué avec les indices de M6
  (`code-fidelity_repro.py R2b`) : 20.2 passes à γR² = 0.25, 14.8 à 0.35, 11.4 à 0.5, 10.8 à 0.7, 11.4 à 1. SAGA
  réglé fait donc jeu égal avec MISO (11.9) et la méthode jointe (11.8) : le nombre de passes ne distingue pas
  « se comporte comme MISO » de « se comporte comme SAGA » sur cette instance, et « SAGA: 20.2 » suggère un écart
  qui tient au pas.
- Ce qui discrimine vraiment, dans les mêmes données : (i) l'insensibilité au pas de la méthode jointe, 12.0, 12.0,
  11.8, 12.2 passes pour γ R2_op = 1/16, 1/8, 1/4, 1 (rejoué), là où SAGA passe de 20.2 à 10.8 ; (ii) les moindres
  carrés à κ = n : joint 18.1 contre MISO 18.9, SAGA au pas du livre 12.9 (`M6_summary.json`) ; (iii) l'échec commun
  à κ = 10n.
- Correction proposée : voir le JSON.

### CF-3 (minor) : phrase de protocole

- Lieu : l. 263-265. « Each family is tuned on five seeds and run on ten fresh ones […] Section~\ref{sec:recipe} uses
  the same protocol, with step grids of ratio $2^{1/4}$. »
- Code (`common.tune_and_confirm`) : criblage de toute la grille sur une graine à 20 % du budget (on garde le
  meilleur tiers, au moins quatre points) ; deux graines (0, 1) sur les points gardés et leurs voisins ; cinq graines
  (0-4) pour les trois finalistes seulement ; dix graines de confirmation (10-19). C'est le cas de M1, M5b, M8 et des
  parties A et B de M9.
- Hors de ce protocole : (M1) et (M2) (`exp_M3_floors.py`, `exp_M6_miso.py` : paramètres fixes, trois graines) ; les
  planchers de (M5) (partie C : cinq graines 10-14, points repris de la partie B) ; (M7) (partie D : cinq graines 0-4,
  lr_0 fixé par la règle Σ lr_t = 4‖W_*‖_op, « le refroidissement le plus rapide » choisi et rapporté sur les mêmes
  cinq graines, sans graines fraîches). Pour les grilles de la recette, voir num-m6m7 F1.

### CF-4 (minor) : deux ingrédients réglés absents de la description

- Lieu : l. 257-260.
- Carte « soft » : dans la comparaison de (M3) (`families.py` l. 133-136, `vrmuon.py` l. 491-494, 573-574)
  ε = ε_innov ‖ϑc‖_F change à chaque pas (ε_innov réglé dans {0.003, …, 3} : un second paramètre), et la longueur est
  γ⟨ω, O⟩ ; c'est seulement dans l'expérience des planchers que ε est une constante (0.4, 0.04) avec une longueur
  fixe (`exp_M3_floors.py` l. 47-50). Le papier écrit un ε sans dire lequel.
- Échantillonnage d'importance : cité avec \cite{xiao2014proximal} (p_i ∝ L_i) ; le code règle l'exposant dans
  {1/2, 1} et retient 1/2 sur les six instances (`M1_summary.json`, bord `sampling-`), et « IS » de la figure 2(b)
  (IS-B-SAGA) est fixé à 1/2 (`families.py` l. 62).

### CF-5 (nit) : contrôle « signe » sur softmax

- Lieu : l. 514-515. Pour softmax le signe est reprojeté, O - (1/K)11^T O (`polar.sign_map(center=True)`,
  `vrmuon.py` l. 307-308) : ses entrées ne sont plus ±1. Sans effet sur les rapports de (M6), le contrôle le plus
  rapide étant la normalisation de Frobenius dans les douze cellules.

### CF-6 (nit) : conventions de comptage et de tirage non dites

- Lieu : l. 262, 511. Les passes comptent la passe initiale qui remplit la table (1 passe), l'exposant tb/n de (4)
  non ; les 36-49 % de (M7) sont calculés passes initiales comprises (hors passe initiale : 38-52 %).
- Les méthodes publiées sans table (`muon_sgd`, `mvr_muon`) tirent leurs lots AVEC remise (`_batch_stream`, 91 % des
  lots de 32 parmi 200 contiennent un doublon) et utilisent le moment 0.95, alors que la recette et son témoin
  « gradient de lot » tirent sans remise avec le moment 0.9. Sans effet mesurable sur les conclusions (n = 1000 dans
  M3 ; cellules b = 32 de M5 déjà hors d'atteinte, CF-1).

## 5. Ce qui est correct et mérite d'être dit au jury

- Le code de la recette est exactement l'équation (4) et la Proposition 5 ; ns5 est la routine publique, ε compris.
- Les comparaisons de (M6) contre SAGA et B-SAGA sont prudentes (concurrents sur des grilles plus fines).
- Les nombres de passes incluent le coût de la table ; les méthodes sans table ne le paient pas.
- L'énumération « exacte » de la partie SAGA est exacte à la précision machine, identité (1) comprise.
- Les résultats sont déterministes et rejouables : treize configurations rejouées (M1, M6, M9), toutes identiques au cache.

## 6. Non vérifié

- Définitions de MARS-M et de STORM-Muon (g = 0.025, écrêtage à 1, β = 0.95) et moment par défaut de Muon contre
  les articles et le code publics : non ouverts (dimension cite-muon) ; ns5 a été comparé à une réécriture de
  mémoire de la routine publique.
- Les étages de criblage et de réglage n'ont pas été rejoués (seules des cellules de confirmation l'ont été) ; M5b
  (fenêtres, grille décalée), M8, M9 partie D et M9b n'ont pas été rejoués.
- `muon/lab/exact.py`, `muon/lab/instrumentation.py` (rang nucléaire 1.07), le simulateur `model_run` de
  `exp_M9b_order_model.py`, `vrlab/order_model.py`, l'instrumentation de L-Katyusha et les exécutions
  préconditionnées de (C5) : non audités ligne à ligne.
- Scripts de figures `paper/scripts/` : seul le choix des références de la figure 4(b) a été regardé.
- R4b porte sur une graine et 300 passes : il borne le gel sans table à lot un par l'expérience, pas par une preuve.
