# Audit num-m6m7 : chiffres expérimentaux de (M6) et (M7)

Statut : version finale (2026-11-20). Auditeur de la dimension `num-m6m7`.

## 1. Périmètre couvert

- (M6) et (M7), `paper/paper_saga_to_muon.tex` l. 584-609 ; légende de la figure 4 (b)-(c), l. 524-528 ; résumé
  l. 105-107 ; tableau 2, lignes Geometry (pour le renvoi à M6) et Memory, l. 624-640 ; Limitations l. 667-669 ;
  phrase de protocole l. 262-265 (résolution de 10 %, grilles de raison 2^(1/4)), à laquelle (M6) est confronté.
- Sources brutes relues : `muon/results/M9_summary.json` (parties A, B, D), `M9b_summary.json` (parties D, L),
  `M5b_summary.json`, `results/B4b_summary.json` ; `muon/results/digest.md` (M5, M5b, M9, M9b) ; code
  `muon/experiments/exp_M9_schedule.py`, `families.py`, `common.py` ; figure `paper/figures/figP4_recipe.png` et son
  script `paper/scripts/make_figP4_recipe.py`.
- Scripts de contrôle (tous dans `paper/audit/checks/`, à lancer depuis la racine) :
  `num-m6m7_recompute_M6_partA.py`, `num-m6m7_recompute_M5b.py`, `num-m6m7_recompute_M7.py`,
  `num-m6m7_grid_resolution_M6.py`, `num-m6m7_finer_grid_M6.py` (petite reproduction avec le code du dépôt ; sorties
  dans `num-m6m7_finer_grid_M6.out.txt`).

## 2. Résumé

Tous les chiffres de (M6) et (M7) se retrouvent dans les JSON, à l'arrondi près, sauf un (« 1.16-1.36 », qui mélange
deux grilles). Les problèmes ne sont pas des erreurs de transcription mais de portée :

1. Les familles « recette » de (M6) ne sont pas réglées sur des grilles de raison 2^(1/4), contrairement à ce que dit
   la l. 264-265 : q sur {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} et lr_0 par demi-décades. Un cran de lr_0 déplace le nombre
   de passes de 13 à 30 %.
2. Conséquence mesurée : en raffinant la grille (q par pas de 0.025, lr_0 par pas de 10^(1/8)) avec le code du dépôt
   et les mêmes graines de confirmation, le rapport « spectral / contrôle » passe de 0.58 à 0.69 (ls-cond, b = 32) et
   de 0.84 à 0.95 (soft-cond, b = 32) ; il tient sur ls-cond b = 16 (0.71), soft-cond b = 16 (0.76) et ls-orth b = 16
   (0.93). La fourchette « 0.58-0.93 » du résumé n'est donc pas robuste ; les comparaisons aux références
   euclidiennes (0.41-0.89) et au pas à échelle nucléaire (0.81-1.04) le sont.
3. « Reproduced by Euclidean devices at batch one but not with minibatches » (tableau 2) : en mini-lots, seuls SAGA et
   B-SAGA ont été lancés ; la métrique de covariance, meilleure méthode à lot un, ne l'a pas été.
4. (M7) : chiffres exacts ; réserves manquantes (lots 1 et 5 : 15-41 % ; sur le modèle à hessiennes identiques la
   permutation n'apporte rien à b = 20, 50 ; deux critères différents « four of five seeds » / « every seed »).

## 3. Contrôles, affirmation par affirmation

### 3.1 (M6), partie A de M9 (`num-m6m7_recompute_M6_partA.py`)

Numérateur : plus petite médiane (10 graines de confirmation 10..19, passes jusqu'à 1e-6) parmi polar, polar+N, ns5,
ns5+N. Dénominateurs : plus petite médiane parmi frob, frob+N, sign, sign+N ; parmi SAGA et B-SAGA (grilles 2^(1/4)) ;
SAGA-Muon à échelle nucléaire (grille 2^(1/4)). Instances : soft-cond, ls-cond (n = 200, 8x40), ls-orth, soft-sq
(n = 500, 20x20), un seul tirage des données (graine 0).

| Affirmation (papier) | Recalcul | Verdict |
|---|---|---|
| b = 1 : « 0.92-1.06 of the passes of the fastest control » | 0.937 (soft-cond, polar+N 7/10 sur frob), 0.924, 1.055, 0.993 | OK ; réserve : sur soft-cond la variante retenue perd 3 graines sur 10 ; avec les lignes 10/10 le rapport est 1.21 |
| b = 1 : « exact polar factor needs 1.16-1.36 times the passes of the Frobenius normalization » | lignes de la partie A (même grille) : 1.223, 1.245, 1.387, 1.322 → 1.22-1.39. Le « 1.16-1.36 » s'obtient avec le polar de la partie B (q par pas de 0.05) sur le frob de la partie A (q par pas de 0.1) : 1.223, 1.245, 1.362, 1.159 | Problème (mineur, constat F4) |
| b = 16, 32 : « the spectral maps need no momentum while the controls do » | variante spectrale la plus rapide sans momentum dans 8 cellules sur 8 ; contrôle le plus rapide = frob+N dans 8/8 ; frob sans momentum 1.3 à 3.7 fois plus lent que frob+N, ou cible non atteinte (ls-cond b = 32) | OK |
| « 0.58-0.93 of the passes of the fastest control » | 0.805, 0.843 (soft-cond) ; 0.712, 0.579 (ls-cond) ; 0.928, 0.894 (ls-orth) ; 0.775, 0.826 (soft-sq) | chiffres retrouvés ; non robustes à la grille (constats F1, F2) |
| « 0.41-0.89 of those of the faster of SAGA and B-SAGA » | 0.722, 0.706, 0.542, 0.408, 0.885, 0.843, 0.808, 0.748 (B-SAGA partout) | OK, robuste (dénominateur sur grille 2^(1/4) ; le numérateur bouge de moins de 6 % en raffinant) |
| « 0.81-1.04 of those of the nuclear-scaled SAGA-Muon » | 0.964, 0.868, 0.991, 0.812, 1.012, 1.038, 1.014, 0.990 | OK |

Bootstrap sur les graines de confirmation (intervalle 5-95 %) : les rapports « spectral / contrôle » ont une largeur
de 2 à 11 points (par ex. 0.805 [0.78, 0.82], 0.712 [0.67, 0.78], 0.928 [0.91, 0.96]) : le bruit des graines est
faible ; ce qui limite la résolution est la grille de réglage, pas les graines.

### 3.2 Résolution de la grille des recettes (`num-m6m7_grid_resolution_M6.py`)

- `families.py` l. 271 : `COOLDOWNS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]` ; l. 283-293 : `recipe_lr_grid` =
  demi-décades (0.03, 0.1, 0.3, 1, 3, 10, 30, 100). En-tête de `exp_M9_schedule.py` l. 17-19 : « grid of q times
  half-decades of lr_0 […]. Same-batch references with constant steps on 2^(1/4) grids ». Seules les trois références
  à pas constant sont sur des grilles de raison 2^(1/4).
- Les scores de réglage (clé `tune` de chaque ligne) suivent la loi de l'horaire : à q fixé, un cran de lr_0 (x 3.16)
  coûte ln(10^(1/2))/ln(1/q) passes, soit 2.3 (q = 0.6), 3.2 (q = 0.7), 5.2 (q = 0.8), 10.9 (q = 0.9) ; mesuré, par
  ex. soft-cond b = 16, polar, q = 0.7 : 22.3, 25.6, 28.7 passes pour lr_0 = 10, 30, 100 ; frob+N, q = 0.8 : 26.6,
  32.8, 36.9, 41.8. Le cran inférieur de lr_0 échoue dans presque toutes les cellules : le point retenu est le plus
  petit lr_0 de la grille qui converge, à une distance inconnue (entre 0 et un cran) du seuil.
- Un cran de lr_0 vaut 13 à 30 % de la médiane de la famille (16 cas : 14, 19, 20, 17, 19, 13, 17, 21, 19, 18, 30, 17,
  17, 18, 17, 20 %) ; un cran de q (0.7 → 0.8) change le taux de 60 %. La résolution « about 10 % » (l. 264) ne
  s'applique donc pas aux comparaisons entre recettes.

### 3.3 Reproduction sur grille plus fine (`num-m6m7_finer_grid_M6.py`)

Code du dépôt (`common.run_one`, `families.recipe_family`), aucun cache modifié. Le script reproduit d'abord
exactement les médianes du papier aux points retenus (soft-cond b = 32 : polar 25.72 [24.84, 27.08], frob+N 30.52
[29.80, 32.68], identiques à `M9_summary.json`). Puis, pour les deux variantes comparées par le papier dans la cellule,
q de q_papier - 0.1 à q_papier par pas de 0.025 et lr_0 de 10^(1/8) x à 10^(-5/8) x lr_0 du papier ; sélection sur
les graines 0-2, confirmation des 9 meilleurs points sur les graines 10..19, on garde le meilleur point dont les dix
graines atteignent la cible.

| Cellule | Papier (spectral / contrôle = rapport) | Grille fine, règle 10/10 | Rapport |
|---|---|---|---|
| soft-cond, b = 16 | polar 22.40 / frob+N 27.84 = 0.805 | polar 21.24 (q = 0.7, lr_0 = 7.5) / frob+N 27.84 (inchangé) | 0.763 |
| soft-cond, b = 32 | polar 25.72 / frob+N 30.52 = 0.843 | polar 25.24 (q = 0.775, lr_0 = 4.0) / frob+N 26.52 (q = 0.775, lr_0 = 7.5) | 0.952 (0.997 avec la règle « plus petite médiane » du bloc `compare`, frob+N 25.32 à 6/10) |
| ls-cond, b = 16 | polar 27.72 (9/10) / frob+N 38.92 = 0.712 | polar 26.76 (q = 0.8, lr_0 = 1.33, 10/10) / frob+N 37.52 (q = 0.8, lr_0 = 5.6) | 0.713 |
| ls-cond, b = 32 | polar 30.76 / frob+N 53.16 = 0.579 | polar 30.68 (q = 0.775, lr_0 = 4.0) / frob+N 44.44 (q = 0.85, lr_0 = 4.0) | 0.690 |
| ls-orth, b = 16 (`--small`) | ns5 16.65 / frob+N 17.94 = 0.928 | inchangé (raffinement réduit : 3 valeurs de q, 5 de lr_0) | 0.928 |

Non refait (budget CPU) : ls-orth b = 32 (0.894), soft-sq b = 16 (0.775), soft-sq b = 32 (0.826), et les cellules
b = 1.

Lecture : la carte spectrale garde un avantage net sur l'instance moindres carrés non carrée (0.69-0.71) et sur
softmax à b = 16 (0.76) ; à soft-cond b = 32 et ls-orth b = 16 (et, non refait, ls-orth b = 32 à 0.89) l'écart est
dans la résolution de 10 % que le papier s'impose. La borne basse « 0.58 » est un artefact de la grille de q du
contrôle (q = 0.9 retenu faute de 0.85).

### 3.4 (M6), M5b (`num-m6m7_recompute_M5b.py`)

Rapport = médiane de SAGA-Muon à échelle nucléaire / plus petite médiane de SAGA, B-SAGA atteignant la cible sur les
dix graines ; grilles de raison 2^(1/4) (« fine »), tirage 0.

| Affirmation | Recalcul | Verdict |
|---|---|---|
| softmax, b = 4, 16 : 0.74, 0.75 | 0.743, 0.749 | OK |
| moindres carrés, b = 4, 16 : 0.74, 0.55 | 0.742, 0.547 | OK |
| b = 1 : 0.91, 1.04 | 0.906, 1.044 | OK |
| ls-orth, b = 1, 4 : 1.30, 1.11 | 1.297, 1.111 | OK (non cité : 0.875 à b = 16, 0.712 à b = 64 sur la même instance) |
| grilles de facteur 2, b = 16 : « 1.7 to 2.7 times slower » | M5 : 2.707 (soft-cond), 1.740 (ls-cond) | OK |

Robustesse : grille décalée d'un demi-pas : 0.807 (au lieu de 0.743), 0.730, 0.716, 0.482, 0.920, 0.959, 1.479 (au lieu
de 1.297), 1.124 : même signe partout sauf b = 1 sur ls-cond (1.04 / 0.96, égalité) ; écarts jusqu'à 14 % entre les
deux grilles. Autres tirages des données à b = 16 : 0.727, 0.694 (softmax), 0.618, 0.505 (moindres carrés) : cohérents
avec 0.75 et 0.55. C'est le résultat le mieux établi de (M6).

### 3.5 (M7), partie D de M9 et M9b (`num-m6m7_recompute_M7.py`)

Instances soft-mem, ls-mem (n = 1000, 10x50, un tirage), facteur polaire exact, sans momentum, graines 0..4, lr_0 non
réglé (longueur totale 4 ||W_*||_op), grille de rho de raison 1.15 à 1.35.

| Affirmation | Recalcul | Verdict |
|---|---|---|
| rho_star = 0.27-0.43, b = 5 à 50, lots indépendants (4 graines sur 5) | soft-mem : 0.368, 0.32, 0.27 ; ls-mem : 0.43, 0.368, 0.32 | OK (b = 1 exclu : 0.43 et 0.61) |
| facteur mémoire (1-b/n)^(n/b) ≈ e^-1 | 0.358 à 0.368 | OK |
| permutation, b = 20, 50 : rho_star 0.10-0.18 | 0.10, 0.135 ; 0.18, 0.18 | OK |
| passes 16.7-20.0 → 9.1-12.2 | soft-mem : 17.9 → 9.1, 16.7 → 10.4 ; ls-mem : 20.0 → 11.3, 19.1 → 12.2 | OK |
| « 36-49 % » (résumé, tableau 2) | cellule par cellule : 49.2, 37.7, 43.5, 36.1 % | OK (ne se déduit pas des seules bornes, mais bien cellule par cellule) |
| « about 1.6 » (SAGA linéaire, n = 1000) | `results/B4b_summary.json`, typical/n=1000/c=2 : 1.634 ± 0.012 (12 graines) ; c'est le maximum sur les pas essayés (0.95 à 1.61 ailleurs) | OK ; « up to about 1.6 » serait exact (nit) |
| recette sur le modèle : 1.2-1.3 nats (indépendant), 1.7-1.9 (permutation) | partie L, lot un, 80 passes, x_star : 1.2, 1.3, 1.3 et 1.9, 1.8, 1.7 pour n = 100, 1000, 10000 | OK ; pour n = 1000 seul : 1.2-1.3 et 1.7-1.8 ; à 40 passes : 1.6-2.0 |
| instances / modèle : 1.7 à 2.7 (indépendant) | 1.73, 2.22 (b = 1) ; 1.88, 2.20 ; 2.06, 2.30 ; 2.39, 2.73 (b = 50) | OK (sur b = 1 à 50) |
| 1.0 à 1.3 avec permutation, b = 20, 50 | 0.97, 1.20, 1.14, 1.34 | OK |

« four of five seeds » (l. 596) et « every seed » (l. 527) : ce sont deux critères différents pour deux quantités
différentes, tous deux conformes aux données. rho_star = plus petit rho où au moins 4 graines sur 5 restent sur
l'horaire jusqu'à la fin du parcours (C ≤ 100 x le niveau de référence) ; les passes sont lues au « fastest rho »,
le rho de plus petite médiane parmi ceux où les cinq graines atteignent 1e-9. Les deux rho coïncident dans 7 des 8
cellules citées (exception : ls-mem, b = 50, indépendant : rho_star = 0.32, passes lues à rho = 0.368) ; dans trois
cellules indépendantes (soft-mem b = 20, 50 ; ls-mem b = 20) les passes viennent d'un horaire que 4 graines sur 5
seulement suivent jusqu'au bout. Cohérent, mais le texte ne le dit pas (constat F6).

Solidité de « 36-49 % » : étendues des graines disjointes (indépendant [16.0, 22.6], permutation [9.1, 12.2]) ; l'écart
vaut 3 à 5 crans de la grille de rho. Hors bruit. Réserves : un seul tirage des données, cinq graines, sélection du
rho et lecture des passes sur les mêmes graines, lr_0 non réglé.

Hors des lots cités : b = 1 : 38.5 % (soft-mem), 15.0 % (ls-mem) ; b = 5 : 41.2 %, 19.3 %, et rho_star inchangé sur
ls-mem (0.43 → 0.43). L'en-tête de `exp_M9_schedule.py` (l. 33-34) dit lui-même « b in {5, 1} on request, where the
order effect is not reliable ». Instance de contrôle limitée par le conditionnement (soft-cond) : 23.8 % à b = 20,
0.9 % à b = 50, cohérent avec « where memory limits ».

Modèle à hessiennes identiques (M9b, partie D, n = 1000) : la permutation aide à b = 1 (rho_star 0.32 → 0.18, passes
10.5 → 7.1) et, pour rho_star seulement, à b = 5 (0.32 → 0.18 ; passes 9.4 → 9.1) ; à b = 20 et 50, les lots où le papier chiffre le gain, elle n'apporte
rien : rho_star 0.22 → 0.22, passes 8.7 → 9.4 (+8 %) et 7.0 → 9.1 (+30 %). Avec des lots indépendants le modèle suit
déjà rho_star = 0.22 (1.5 nat par passe), au-delà de e^-1. Les médianes du modèle sont des minima sur une grille
bruitée : b = 50, indépendant : 7.0 avec des graines à [15.2, 7.0, 14.7, 5.2, 6.5] et des voisins de grille à 16.1 et
12.6 ; le « 2.7 » de « 1.7 to 2.7 » repose sur ce dénominateur.

Réponse à la question posée : « a fresh permutation per pass lifts the memory ceiling » est établi au-delà du modèle,
sur les deux instances limitées par la mémoire, à b = 20 et 50 (et, plus faiblement, à b = 1 et 5) ; c'est sur le
modèle lui-même, aux lots 20 et 50, qu'il ne l'est pas. L'explication par SAGA linéaire (1 nat → 2 nats) ne vaut qu'à
lot un (partie L).

### 3.6 Figure 4 (b) et (c)

Lue avec l'outil Read et confrontée au script (`data_b` lit le bloc `compare` de la partie A, et M5b « fine » pour
SAGA/B-SAGA à b = 1 ; `data_d` lit `fastest` des parties D de M9 et M9b).

- (b) softmax : spectral 21.45 (7/10), 22.4, 25.7 ; contrôle 22.9, 27.8, 30.5 ; SAGA/B-SAGA 27.1, 31.0, 36.4. Moindres
  carrés : 26.8, 27.7 (9/10), 30.8 ; 29.0, 38.9, 53.2 ; 31.9, 51.2, 75.4. Conforme à l'image, annotations 7/10 et 9/10
  comprises. « (median of ten seeds; k/10) » conforme.
- (c) valeurs de la figure = `fastest` (18.2 → 11.2, 23.3 → 19.8, 10.5 → 7.1 à b = 1 ; … ; 16.7 → 10.4, 19.1 → 12.2,
  7.0 → 9.1 à b = 50). Conforme ; la figure montre honnêtement que, sur le modèle, la permutation est plus lente à
  b = 20 et 50. « at the fastest cooldown that every seed follows to the target (median of five seeds) » conforme à la
  définition de `fastest`.
- Réserve : « fastest control map on the same cooldown » (l. 525) : le contrôle a son propre (q, lr_0) réglé ; pour les
  quatre cellules mini-lots tracées, q diffère dans deux (softmax b = 16 : 0.7 contre 0.8 ; moindres carrés b = 32 :
  0.8 contre 0.9) (constat F5).

### 3.7 Limitations (l. 667-669)

« (M3)-(M7) rest on two to four instances and one to three draws of the data » : (M6) partie A = 4 instances, 1 tirage ;
M5b = 3 instances, 3 tirages pour b = 16 et 64 sur deux instances ; (M7) = 2 instances (+ 1 contrôle + le modèle),
1 tirage, 5 graines. Exact. À noter : les deux chiffres du résumé reposent chacun sur un seul tirage.
« counts are in gradients, not factorizations » : exact ((M6) l. 594 le redit ; à b = 16, n = 200 : 12.5
factorisations par passe).

## 4. Constats

### F1 (major) : les recettes de (M6) ne sont pas réglées sur des grilles de raison 2^(1/4)
- Lieu : l. 262-265 (et, par renvoi, l. 584-594).
- Papier : « differences below about $10\%$ are not resolved. Section~\ref{sec:recipe} uses the same protocol, with
  step grids of ratio $2^{1/4}$. »
- Problème : seules SAGA, B-SAGA et SAGA-Muon à pas constant sont sur des grilles 2^(1/4). Les huit familles
  « recette » sont réglées sur q ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} x demi-décades de lr_0 ; un cran de lr_0 vaut 13 à
  30 % des passes, un cran de q jusqu'à 60 % du taux.
- Preuve : `muon/experiments/families.py` l. 271, 283-293 ; `exp_M9_schedule.py` l. 17-19 ; `M9_summary.json`
  A/<instance>/q_grid et cells/<b>/lr_grids ; `num-m6m7_grid_resolution_M6.py`.
- Correction proposée : voir le JSON (phrase de protocole).

### F2 (major) : « 0.58-0.93 » (résumé, (M6)) n'est pas robuste à la grille
- Lieu : l. 105-106 ; l. 587-589 ; intitulé l. 584 ; l. 133, 179, 646, 680.
- Papier : « With minibatches of $16$ or $32$ samples the spectral map needs $0.58$--$0.93$ of the passes of a
  normalized step ».
- Problème : sur une grille quatre fois plus fine, mêmes graines, même code : 0.579 → 0.690 (ls-cond b = 32),
  0.843 → 0.952 (soft-cond b = 32), 0.805 → 0.763, 0.712 → 0.713, 0.928 → 0.928. Sur les cinq cellules refaites la
  fourchette est 0.69-0.95, dont deux dans la résolution de 10 %.
- Preuve : `num-m6m7_finer_grid_M6.py` et `.out.txt`.
- Portée : l'intitulé « where the spectral map pays » reste étayé contre SAGA/B-SAGA (0.41-0.89, robuste) et, contre le
  contrôle normalisé, sur 3 des 5 cellules refaites.

### F3 (major) : « but not with minibatches » sans les dispositifs euclidiens préconditionnés
- Lieu : l. 627-628 (tableau 2) ; l. 645-646 ; l. 679-680.
- Papier : « A change of norm buys trajectory properties, reproduced by Euclidean devices at batch one \finding{M3}
  but not with minibatches \finding{M6}. »
- Problème : en mini-lots les références euclidiennes sont SAGA et B-SAGA seulement. La métrique de covariance
  (meilleure méthode à lot un : 15.1 / 16.1 passes, « the best method on least squares for every draw », l. 380-381),
  l'échantillonnage préférentiel et la métrique polaire figée n'ont pas été lancés à b > 1. À titre indicatif la
  recette spectrale la plus rapide à b = 16 demande 22.4 / 27.7 passes.
- Preuve : `M9_summary.json` coverage/A/<instance>/<b>/references = ['B-SAGA', 'SAGA', 'SAGA-Muon (joint)'] ;
  `M5b_summary.json` about/families.

### F4 (minor) : « 1.16-1.36 » mélange deux grilles
- Lieu : l. 586-587. Sur la même grille (partie A) : 1.22-1.39. Preuve : § 3.1.

### F5 (minor) : « on the same cooldown »
- Lieu : l. 585-586 et l. 525. Le contrôle a son propre (q, lr_0) ; q diffère dans 5 des 8 cellules mini-lots
  (identique dans les 4 cellules b = 1).

### F6 (minor) : deux critères pour rho_star et pour les passes
- Lieu : l. 596-599, l. 527. Voir § 3.5.

### F7 (minor) : gain de la permutation aux lots 1 et 5, et sur le modèle aux lots 20 et 50
- Lieu : l. 598-606. Voir § 3.5.

### F8 (minor) : à lot un, la variante spectrale « la plus rapide » perd des graines
- Lieu : l. 585. soft-cond b = 1 : polar+N 7/10 ; avec les lignes 10/10, 1.21 au lieu de 0.94. Va dans le sens du
  papier (« nothing above is specific to Muon ») mais la fourchette 0.92-1.06 dépend de cette règle.

### F9 (nit) : « about 1.6 » est un maximum sur les pas ; « 1.7-1.9 » mélange n = 100, 1000, 10000
- Lieu : l. 603-604.

## 5. Non vérifié

- Raffinement de grille pour ls-orth b = 32, soft-sq b = 16 et 32, et pour les cellules b = 1 (0.92-1.06, 1.16-1.36).
- Énoncés théoriques cités dans (M7) (min{2c, 1}, min{2, log((2-g)/g)}, note B4) : hors périmètre numérique.
- Les pickles bruts (`M9_*_confirm.pkl`, `M9_order.pkl`) n'ont pas été rouverts : les JSON portent les valeurs par
  graine, et la reproduction directe de deux points (soft-cond b = 32) redonne les médianes au centième.
- Effet d'un second tirage des données sur (M6) partie A et sur (M7) : aucune donnée dans le dépôt.
- Les citations de (M7) (Finito, Ying et al., Park et Ryu, Mokhtari et al., « κ < 4.08 ») : autre dimension.
