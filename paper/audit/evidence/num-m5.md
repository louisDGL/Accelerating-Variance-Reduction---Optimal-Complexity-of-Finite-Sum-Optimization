# Audit num-m5 : recette (4), constat (M5), figure 4(a), chiffres attachés à la Proposition 5

Etat : TERMINE (2026-11-20). Papier audité : `paper/paper_saga_to_muon.tex` (version « revised 20 November 2026 »).

## 1. Périmètre couvert

- Recette (4) et constat (M5), lignes 508-544.
- Légende de la figure 4, partie (a), lignes 520-523 ; image `paper/figures/figP4_recipe.png` ; script
  `paper/scripts/make_figP4_recipe.py`.
- Remarques après la Proposition 5, lignes 578-581 (« 3.3e4 to 8.9e4 », « 2.3 to 3.5 »).
- Lignes 648-650 (« 1e4 to 1e5 times slower », « 350 to 1400 times below the tuned steps »).
- Résumé, lignes 102-105 ; tableau 1, ligne 5 (lignes 173-180).

Hors périmètre (non traité) : preuve de la Proposition 5, (M6), (M7), parties (b) et (c) de la figure 4, bibliographie.

## 2. Sources et scripts

Données brutes lues : `muon/results/M9_summary.json` (parties B et C), `muon/results/M9_law.pkl` (110 runs de la
partie C), `muon/results/M1_summary.json`, `muon/results/M5b_summary.json`. Code lu : `muon/lab/polar.py`,
`muon/lab/vrmuon.py` (pas, moment, schedule), `muon/lab/mvr.py`, `muon/experiments/exp_M9_schedule.py`,
`muon/experiments/families.py`, `muon/experiments/common.py`, `_archive/muon/tests/test_facts.py`
(`test_F13_limits_on_the_benchmark`, origine des chiffres des lignes 578-581).

Scripts de contrôle (tous dans `paper/audit/checks/`, lancés depuis la racine du projet, un seul processus, BLAS
mono-thread, chacun sous 3 minutes) :

| script | objet | durée |
|---|---|---|
| `num-m5_law_from_pkl.py` | recalcul des chiffres de (M5) depuis `M9_law.pkl` ; longueur totale des schedules | 2 s |
| `num-m5_no_table_fairness.py A` | reproduction de 12 runs du cache ; table SAGA à b = 32 sur le schedule des contrôles | 77 s |
| `num-m5_no_table_fairness.py B1 <instance>` | gradient minibatch sans table, b = 1, grille (q, lr_0) | 112 s et 147 s |
| `num-m5_no_table_fairness.py B1M` | MARS-M sans table, b = 1, grille (q, lr_0) | 100 s |
| `num-m5_no_table_fairness.py B32` | b = 32 : minibatch, MARS-M, table SAGA, grille (q, lr_0) | 67 s |
| `num-m5_certified_constants.py` | 3.3e4-8.9e4 et 350-1400 depuis les constantes des instances | 1 s |
| `num-m5_independent_recipe.py` | réimplémentation indépendante de la recette (4) : planchers, mécanisme de ns5 | 152 s |
| `num-m5_estimator_error_ratio.py` | rapport erreur de l'estimateur / gradient, 6 graines | 12 s |

## 3. Verdict par affirmation

### 3.1 Recette (4), lignes 508-515

| affirmation | verdict | preuve |
|---|---|---|
| `lr_t = lr_0 q^{tb/n}` | OK | `muon/lab/vrmuon.py:478-480` : `self.lr * self.lr_decay ** (t * self.batch / self.n)` ; pour MARS-M / STORM-Muon (2b gradients par pas) `schedule=["geometric", q, n/(2b)]`, `exp_M9_schedule.py:497-499` : même facteur q par passe de gradients |
| « Muon's five Newton--Schulz steps (ns5, normalization constant ε = 1e-7) » | OK | `muon/lab/polar.py:157` : `def ns5(M, steps=5, coeffs=NS5_COEFFS, eps=1e-7, bf16=False)`, coefficients (3.4445, -4.7750, 2.0315), division par `||X||_F + eps` (ligne 173) |
| « with or without Nesterov momentum 0.9 (Muon's recipe: ns5 with momentum) » | OK (réserve N1) | `muon/experiments/families.py:314-315` : `dict(beta=0.9, nesterov=True)` ; `families.py:332` : la recette de Muon de la partie B est `recipe_family("ns5", True, 1, ...)` ; tampon `buf = beta buf + omega`, `D = (1-beta)(omega + beta buf)` (`vrmuon.py:568-571`), algébriquement la forme `lerp` du code de Muon |
| « two controls that are not orthogonalizations, ω/‖ω‖_F and the coordinatewise sign » | OK | `polar.py:234-255` |
| « q and lr_0 are tuned » | OK | partie B : q ∈ {0.55, …, 0.9} (pas 0.05), lr_0 ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1} (`families.py:272, 331`), protocole `tune_and_confirm` |

### 3.2 Constat (M5), lignes 532-544

| affirmation | verdict | preuve (valeur recalculée) |
|---|---|---|
| « batch one, 300 passes, five seeds, the two non-square conditioning-limited instances » | OK | `exp_M9_schedule.py:87-88` : `LAW_ON = ("soft-cond", "ls-cond")` (8 x 40, n = 200), graines 10-14, `passes=300`, `batch=1` |
| « the ratio varies by a factor 3.2 to 5.7 along a run » | imprécis (m3) | `width_median` des six configurations 5/5 sur le schedule : 3.82, 5.69, 3.96 (softmax), 4.12, 3.34, 3.16 (moindres carrés) : 3.16 à 5.69. Ce sont des médianes sur les graines, sur la seconde moitié de la course ; par graine 2.37 à 8.31 |
| planchers « 2.8e-29 (softmax) and 2.3e-29 (least squares) for the exact polar factor » | OK | `M9_law.pkl` : médianes 2.85e-29 [2.34e-29, 4.10e-29] et 2.27e-29 [1.43e-29, 3.79e-29] ; minimum recalculé sur la trace identique |
| « the rounding floor » | OK (plausible) | distance au carré RELATIVE (`common.py:245`, axe de la figure). sqrt(2.8e-29) = 5.3e-15 = 24 epsilon machine ; la course quitte la loi quand lr_t vaut 1.1e-15 (softmax, passe 145) et 3.7e-16 (moindres carrés, passe 155), l'espacement des flottants à la taille des coefficients de W_* (médiane \|W_*\| = 2.45 et 1.41). La réimplémentation indépendante, à l'arithmétique plus naïve, a un plancher 1e-28 à 2e-27 : niveau dépendant de l'implémentation, comme attendu d'un plancher d'arrondi |
| « near 1e-18 for Muon's recipe at q = 0.8, lr_0 = 0.1 » | OK | médianes 9.77e-19 [7.7e-19, 2.85e-18] et 1.09e-18 [1.00e-18, 1.43e-18], 5/5 graines ; réimplémentation indépendante : 1.2e-18 à 1.5e-18 et 0.93e-18 à 1.03e-18 |
| « a floor of ns5, whose constant ε breaks the scale invariance » ; « with ε = 1e-30: 1.5e-29 and 4.1e-30 » | OK | médianes 1.50e-29 et 4.06e-30 ; dans la réimplémentation indépendante le plancher 1e-18 disparaît avec ε = 1e-30 (1.3e-28 à 3.2e-27) ; quand la course quitte la loi, ‖D_t‖_F = 1.1e-11 et 7.9e-11, très en dessous de ε = 1e-7 : ns5 divise par ‖D‖_F + ε et sa sortie se contracte |
| « Without a table the same schedule freezes the iterate at 8% to 93% … (b = 1, 32) … MARS-M and STORM-Muon » | chiffres OK, interprétation fausse à b = 32 (M1) ; « the same schedule » ambigu et concurrents non réglés (m1) | médianes : softmax 0.256, 0.340, 0.341 (b = 1), 0.823, 0.931, 0.931 (b = 32) ; moindres carrés 0.083, 0.129, 0.143 (b = 1), 0.629, 0.816, 0.813 (b = 32). Voir section 4 |
| « 13.4–25.5 passes to 1e-6 on the four conditioning-limited instances » | OK | recette de Muon réglée : 21.75, 25.50, 16.65, 13.45 (soft-cond, ls-cond, ls-orth, soft-sq) |
| « damped SAGA in a frozen polar metric (12.9–24.4) » | OK (réserve m6) | `B/<inst>/M1/rows/one-shot polar-metric B-SAGA` : 22.05, 24.35, 13.95, 12.90 |
| « more than in the covariance metric (13.5–16.1) but for a tie » | OK | `cov-metric B-SAGA` : 15.05, 16.05, 13.90, 13.50 ; égalité sur soft-sq (13.45 contre 13.50) |
| « next to a cliff in q » ; « lose confirmation seeds on three instances (7/10, 9/10, 9/10) » | OK | graines atteignant 1e-6 : 7/10, 9/10, 9/10, 10/10 ; scores de réglage : soft-cond à lr_0 = 0.3, 20.0 passes à q = 0.7 et cible manquée à q = 0.65 ; ls-cond 25.3 à q = 0.75, 98.5 à q = 0.7 |
| « the fixed point q = 0.8, lr_0 = 0.1 avoids (21.3–26.4 passes) » | OK | 23.35, 26.45, 25.30, 21.30, 10/10 partout |

### 3.3 Figure 4(a) et sa légende, lignes 520-523

OK. Le script trace, pour `ls-cond`, la médiane (en log) sur les cinq graines de la distance au carré relative :
polaire exacte avec table (q = 0.8, lr_0 = 0.3, le point sélectionné : `C/ls-cond/points`), recette de Muon avec table
au point fixe (q = 0.8, lr_0 = 0.1) avec ε = 1e-7 et ε = 1e-30, et sans table le schedule sélectionné pour la recette
de Muon (q = 0.75, lr_0 = 0.3 : `C/ls-cond/controls_schedule`) sur le gradient minibatch et MARS-M à b = 1. L'image
montre les paliers attendus : environ 2e-29 (polaire, à partir de la passe 155), 1e-18 (ns5, à partir de la passe 90),
4e-30 (ε = 1e-30), 0.08 et 0.13 (sans table). La légende dit correctement que les courbes sans table sont à un autre
schedule que les courbes avec table.

### 3.4 Lignes 578-581 et 648-650

| affirmation | verdict | preuve |
|---|---|---|
| « its cooldown is 3.3e4 to 8.9e4 times slower, in log(1/q), than q = 0.8 » | chiffre OK, borne d'âge non dite (m2) | définition : log(1/0.8) / log(1/q_cert), q_cert = ρ^{n/2}, ρ = 1 - μ_op/L_eff, L_eff = L_op + 10 A L_max. Constantes recalculées depuis X : soft-cond μ_op = λ = 5.0e-5, L_op = 0.0751, L_max = 0.5 ; ls-cond μ_op = λ + λ_min = 1.371e-4, L_op = 0.1502, L_max = 1. A = n (cyclique) : 4.46e4 et 3.26e4 ; A = 2n - 1 (permutation) : 8.90e4 et 6.50e4. L'intervalle va du cyclique sur ls-cond à la permutation sur soft-cond. Pour les indices indépendants de (M5), A = ⌈n log(nT/δ)⌉ = 3860 (T = 300 n, δ = 0.05) : 8.6e5 et 6.3e5 |
| « the error of the estimator is 2.3 to 3.5 times the gradient in nuclear norm » | ordre de grandeur OK, intervalle d'une seule graine (m4) | notes : [2.69, 3.45] et [2.32, 3.42], une course par instance (graine 10, polaire, lr_0 = 0.1, 7 états). Boucle indépendante, 6 graines x 7 états : [2.40, 4.19] (soft-cond), [2.26, 4.37] (ls-cond, lr_0 = 0.1), [2.00, 3.51] (ls-cond, lr_0 = 0.3) |
| « at cooldowns 1e4 to 1e5 times slower than in (M5) » | même réserve (m2) | (M5) utilise des indices indépendants (pas de clé `order` dans les configurations, défaut `iid`) : 6e5 à 9e5 |
| « γ ≤ 1/(14 n L_max), 350 to 1400 times below the tuned steps » | OK sur les deux instances, portée non dite (m5) | 1/(14 n L_max) = 3.574e-4 / R2_op ; pas réglés de « SAGA-Muon (joint) » dans M1 : 0.5 / R2_op (soft-cond), 0.125 / R2_op (ls-cond) : rapports 1399 et 350. γ L_op = 5.4e-5 ≤ 1 |

### 3.5 Résumé (lignes 102-105) et tableau 1, ligne 5

- « follow the schedule down to rounding error with the exact polar factor (near 1e-18 with Muon's Newton--Schulz
  map, whose normalization constant sets that floor) » : OK (section 3.2), sur deux instances à batch un.
- « where minibatch and momentum-based estimators freeze » : couvert par les contrôles de (M5) à b = 1, et confirmé
  par mes contrôles avec schedules réglés (section 4.2) : sans table l'itéré gèle entre 0.25 % et 10 % de la distance
  initiale au carré, contre 1e-18 ou moins avec la table. Qualitativement établi ; les niveaux « 8 % à 93 % » du
  texte ne le sont pas (M1, m1).
- « we prove this law for tables of bounded age, at slower cooldowns » : cohérent avec la Proposition 5(b) (preuve
  hors périmètre) ; « slower » vaut 3e4 à 9e5 selon l'ordre de tirage (m2).
- Tableau 1, ligne 5, « With a table the cooldown converges, without one it freezes [e]; certified for tables of
  bounded age, at slower cooldowns [p] » : étiquettes de statut correctes.

## 4. Détail des constats

### M1 (major) : à b = 32, le gel « sans table » vient de la longueur du schedule, pas de l'estimateur

Lignes 538-540 : « Without a table the same schedule freezes the iterate at $8\%$ to $93\%$ of the initial squared
distance, for the minibatch gradient ($b=1$, $32$) as for MARS-M and STORM-Muon: the cooldown outruns an estimator
that does not vanish at $W_\ast$. »

Les contrôles sans table tournent au point sélectionné pour la recette de Muon AVEC table à batch un (soft-cond :
q = 0.7, lr_0 = 0.3 ; ls-cond : q = 0.75, lr_0 = 0.3). La longueur totale de ce schedule est
Σ_t lr_t = lr_0 / (1 - q^{g/n}) pour g gradients par pas. Avec W_0 = 0 et un pas de norme spectrale au plus
1.2 lr_t (ns5), l'itéré ne peut pas s'éloigner de 0 de plus que cette somme :

| contrôle | Σ_t lr_t, soft-cond (‖W_*‖_op = 35.82) | Σ_t lr_t, ls-cond (‖W_*‖_op = 18.22) |
|---|---|---|
| minibatch, b = 1 | 168.4 = 4.70 ‖W_*‖_op | 208.7 = 11.46 ‖W_*‖_op |
| MARS-M, STORM-Muon, b = 1 | 84.3 = 2.35 ‖W_*‖_op | 104.4 = 5.73 ‖W_*‖_op |
| minibatch, b = 32 | 5.41 = 0.15 ‖W_*‖_op | 6.67 = 0.37 ‖W_*‖_op |
| MARS-M, STORM-Muon, b = 32 | 2.78 = 0.08 ‖W_*‖_op | 3.41 = 0.19 ‖W_*‖_op |

A b = 32 aucun estimateur, même exact, ne peut atteindre W_* sur ce schedule. Contrôle direct
(`num-m5_no_table_fairness.py A`, code du dépôt, graines 10 et 11) : la TABLE SAGA à b = 32 sur le même schedule gèle
elle aussi.

| b = 32, schedule des contrôles | sans table (minibatch) | avec table SAGA |
|---|---|---|
| soft-cond (q = 0.7, lr_0 = 0.3) | 0.819, 0.830 | 0.785, 0.785 |
| ls-cond (q = 0.75, lr_0 = 0.3) | 0.629, 0.622 | 0.574, 0.573 |

(Le même script reproduit à trois chiffres douze valeurs du cache : à b = 1, 0.263, 0.279, 2.96e-05, 2.16e-12, 0.083,
0.0807, 4.56e-18, 3.55e-18 ; à b = 32 sans table, 0.819, 0.830, 0.629, 0.622.)

Conséquences : la moitié haute de l'intervalle « 8 % à 93 % » (63 % à 93 %, les six valeurs à b = 32) ne dit rien
de la table ; l'explication « the cooldown outruns an estimator that does not vanish at W_* » est fausse pour ces
valeurs ; la mention « (b = 1, 32) » suggère une robustesse en taille de batch qui n'est pas mesurée. Restreint à
b = 1, l'intervalle est 8 % à 34 %.

### m1 (minor) : « the same schedule » ; concurrents sans table évalués à un seul schedule, non réglés

Réponse à la question d'équité : non, les concurrents sans table n'ont pas été réglés sur une grille (q, lr_0). Ils
sont évalués à un seul point, celui sélectionné pour la recette de Muon avec table à batch un
(`exp_M9_schedule.py:482, 494-499`), qui est au bord de la falaise de cette méthode. Deux points de rédaction :

1. « the same schedule » n'a pas le bon antécédent. Les phrases précédentes donnent les planchers avec table à
   q = 0.8 (polaire : lr_0 = 0.1 et 0.3 ; recette de Muon : lr_0 = 0.1). Les contrôles sans table sont à q = 0.7 / 0.75,
   lr_0 = 0.3. A ce schedule-là, sur softmax, la recette AVEC table quitte elle-même le schedule pour 4 graines sur 5
   (plus basse distance relative au carré : 3.0e-5, 2.2e-12, 1.3e-7, 5.2e-13, 1.7e-5 ; `C/soft-cond/deep`). La
   légende de la figure 4(a) est correcte sur ce point ; le texte de (M5) ne l'est pas.
2. MARS-M et STORM-Muon consomment 2b gradients par pas : au même q par passe ils font deux fois moins de pas, donc
   un refroidissement deux fois plus rapide par itération qu'un schedule déjà au bord de la falaise.

Contrôle de robustesse (une graine, 300 passes, code du dépôt ; `num-m5_no_table_fairness.py B1, B1M, B32`) :
distance relative au carré finale au schedule du papier, puis au meilleur point d'une petite grille
q ∈ {sélectionné, 0.9, 0.95, 0.98}.

| estimateur sans table | soft-cond : papier → meilleur de la grille | ls-cond : papier → meilleur de la grille |
|---|---|---|
| minibatch, b = 1 (lr_0 ∈ {0.03, 0.1, 0.3}) | 0.263 → 0.102 | 0.083 → 0.038 |
| MARS-M, b = 1 (lr_0 ∈ {0.1, 0.3}) | 0.355 → 0.025 | 0.149 → 0.014 |
| minibatch, b = 32 (lr_0 ∈ {0.3, 1, 3, 10}) | 0.82 → 0.0031 | 0.63 → 0.0025 |
| MARS-M, b = 32 (lr_0 ∈ {0.3, 1, 3, 10}) | 0.93 → 0.0083 | 0.82 → 0.0060 |
| table SAGA, b = 32 (même grille) | 0.79 → 7.9e-20 | 0.57 → 2.0e-19 |

Lecture : l'intitulé « without one it freezes the iterate » et le résumé tiennent qualitativement (sans table,
0.25 % à 10 % ; avec table, le plancher de ns5), y compris à b = 32 une fois le schedule réglé. Les niveaux
« 8 % à 93 % » sont ceux d'un schedule choisi pour une autre méthode, et baissent d'un facteur 2 à 300 après un
réglage minimal. Mes contrôles n'ont qu'une graine et pas de graines de confirmation ; la dispersion entre graines
des contrôles du dépôt est inférieure à 20 %.

### m2 (minor) : 3.3e4-8.9e4 et « 1e4 to 1e5 » valent pour les ordres cyclique et permuté, pas pour le tirage de (M5)

Lignes 578-579 et 648-649. L'intervalle recalculé est exact mais mélange deux bornes d'âge de la Proposition 5(b) :
3.3e4 = ordre cyclique (A = n) sur ls-cond, 8.9e4 = permutation fraîche (A = 2n - 1) sur soft-cond. Les courses de
(M5) tirent des indices indépendants ; la borne d'âge correspondante de la proposition, A = ⌈n log(nT/δ)⌉, donne
8.6e5 (soft-cond) et 6.3e5 (ls-cond) pour T = 300 passes et δ = 0.05 (9.3e5 et 6.8e5 pour δ = 0.01), avec probabilité
1 - δ seulement. La ligne 649 compare donc à (M5) un certificat qui ne porte pas sur son tirage ; l'écart réel est
environ vingt fois plus grand (A/n = 19.3). L'erreur va dans le sens défavorable au papier (l'écart théorie /
pratique est sous-estimé), d'où la sévérité mineure.

Note : le papier définit μ_N comme « the best constants » (ligne 198). Le chiffre 3.3e4 utilise bien la meilleure
constante sur ls-cond (λ + λ_min(Σ̂) = 1.371e-4), pas `prob.mu_op = lam` du laboratoire, qui donnerait 4.5e4.

### m3 (minor) : « varies by a factor 3.2 to 5.7 along a run »

Ligne 534. 3.2 et 5.7 sont le minimum et le maximum, sur six configurations, de la MEDIANE sur cinq graines du rapport
max/min de dist²/lr², mesuré sur la SECONDE MOITIE (en passes) de la partie de la course qui suit la loi
(`exp_M9_schedule.py:463-466, 525-526`). Par graine le rapport va de 2.37 à 8.31. Depuis une distance relative de
1e-2 jusqu'au plancher, la recette de Muon au point fixe varie d'un facteur 20 à 44 sur softmax (3.9 à 9.3 sur
moindres carrés) : le transitoire initial n'est pas dans le chiffre du papier. La recette de Muon à son point
sélectionné (6.6 et 7.6 sur la seule graine de softmax qui suit le schedule) est exclue de l'intervalle.

### m4 (minor) : « 2.3 to 3.5 » repose sur une course par instance

Lignes 579-580. Source : `_archive/muon/tests/test_facts.py::test_F13_limits_on_the_benchmark`, une course par
instance (graine 10, polaire exacte, q = 0.8, lr_0 = 0.1 sur les deux instances, états aux passes 2, 6, …, 26),
moyenne sur l'indice de ‖ω_j - ∇F‖_* / ‖∇F‖_*. Avec une boucle indépendante et six suites d'indices
(`num-m5_estimator_error_ratio.py`) l'intervalle est [2.40, 4.19] sur soft-cond et [2.26, 4.37] sur ls-cond
([2.00, 3.51] à lr_0 = 0.3, le point de la figure 4(a)). L'affirmation qualitative (l'erreur dépasse le gradient, d'un
facteur 2 à 4) est confirmée ; les deux décimales ne sont pas robustes à la graine. Ce test est dans `_archive/`,
que le paragraphe de reproductibilité (lignes 683-690) ne cite pas.

### m5 (minor) : « 350 to 1400 times below the tuned steps » ne vaut que pour soft-cond et ls-cond, grilles de facteur 2

Ligne 650. Recalcul (`num-m5_certified_constants.py`) : 1399 (soft-cond) et 350 (ls-cond) avec les pas de M1 (grilles
de facteur 2). Avec les pas de M5b (grilles 2^{1/4}, ceux que (M6) cite pour la même méthode) : 1176 et 416. Sur les
six instances de M1 : 217 (ls-mem) à 3493 (soft-sq). La phrase ne dit pas de quelles instances il s'agit.

### m6 (minor) : les comparateurs de la phrase « At batch one the recipe needs … » viennent de M1 (grilles de facteur 2)

Lignes 540-542. Les passes 12.9-24.4 et 13.5-16.1 sont lues dans `M1_summary.json` (`exp_M9_schedule.py:400, 420-427` :
« factor-2 step grids »), avec des paramètres sélectionnés au bord de leur grille sur chaque instance (`theta+`,
`metric_floor+` ou `metric_floor-`), alors que la recette est réglée sur 48 points (q, lr_0). La ligne 264-265 dit
que la section 3.3 utilise des grilles de rapport 2^{1/4} : ce n'est vrai ni de ces comparateurs ni de lr_0
(demi-décades). « about as many » couvre des écarts par instance de -1 %, +5 %, +19 % (ls-orth : 16.65 contre 13.95)
et +4 %.

### n1 (nit) : « Muon's recipe » avec un moment 0.9

Ligne 514. Le dépôt décrit lui-même le Muon publié avec un moment 0.95 (`families.py:241` : « Keller Jordan's Muon:
momentum 0.95, Nesterov, 5 Newton-Schulz steps ») ; la recette de la section 3.3 utilise 0.9. Le papier écrit 0.9
explicitement, donc rien de faux ; à signaler seulement si un lecteur lit « Muon's recipe » comme « les valeurs par
défaut de Muon ». Non vérifié sur le code public de Muon.

### n2 (nit) : préciser « relative »

Lignes 533-535. Les planchers sont des distances au carré relatives (divisées par ‖W_*‖_F², `common.py:245` ; l'axe
de la figure 4(a) le dit). Le texte de (M5) écrit « the squared distance ».

### n3 (nit, en bordure du périmètre) : renvoi à la Proposition 5 dans l'introduction

Lignes 131-133 : « fixed lengths on a geometric cooldown, fed by a SAGA table, converge to rounding error where
other estimators freeze (Proposition~\ref{prop:schedule}) ». La proposition ne porte ni sur l'erreur d'arrondi ni sur
le gel des autres estimateurs, et certifie des refroidissements 3e4 à 9e5 fois plus lents : c'est (M5), étiqueté [e],
qui porte cette phrase. Le résumé et le tableau 1 séparent correctement les deux statuts.

## 5. Non vérifié

- Valeurs par défaut du code public de Muon (ε = 1e-7, cinq pas, moment) : non ouvert ; seul le code du dépôt est lu.
- Fidélité des contrôles MARS-M (g = 0.025, clip 1, β = 0.95) et STORM-Muon (g = 1) aux articles cités.
- Preuve de la Proposition 5 et du fait F15 (autre dimension).
- Les 110 runs de `M9_law.pkl` n'ont pas tous été relancés : douze le sont (valeurs identiques à trois chiffres), les autres sont
  recoupés par la réimplémentation indépendante (mêmes planchers à un facteur 1.5 près pour ns5).
- Mes contrôles de réglage (section 4, m1) : une graine, grille de 8 à 16 points, pas de graines de confirmation.
- La loi dist² = C lr_t² n'est mesurée par le dépôt que sur soft-cond et ls-cond (le papier le dit) ; rien sur
  ls-orth, soft-sq à 300 passes.
