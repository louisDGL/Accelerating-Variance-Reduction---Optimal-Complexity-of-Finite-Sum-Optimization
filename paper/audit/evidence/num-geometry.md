# Audit num-geometry : (M4), (M3), figure 2, "Geometries", "Experiments", tableau 1 ligne 2

Objet : `paper/paper_saga_to_muon.tex`, l. 367-384 ((M4), (M3)), l. 359-363 (legende de la figure 2), l. 236-242
("Geometries"), l. 253-265 ("Experiments"), l. 156-158 (tableau 1, ligne 2, "Added here").

## 1. Perimetre couvert et methode

- Chaque chiffre a ete relu dans les resumes bruts `muon/results/M7_summary.json`, `M1_summary.json`,
  `M8_summary.json`, `M5b_summary.json` (medianes recalculees a partir des `confirm_passes` par graine), pas dans la
  prose. Script : `paper/audit/checks/num-geometry_numbers.py`.
- Les constantes R_P^2/mu_P de (M4) et de la figure 2(a) ont ete recalculees de facon independante a partir des
  donnees des six instances (X, etiquettes, lambda, W_*) avec les formules du paragraphe "Geometries", sans utiliser
  `VRMuon.polar_metric` ni `exp_M7_checks.py` (hessienne softmax reconstruite). Script :
  `paper/audit/checks/num-geometry_constants.py`. Ecart relatif maximal avec M7 sur les 96 cellules : 7.5e-14.
- Petite reproduction (code du depot, `common.run_one`, 2 a 3 graines de confirmation) sur le troisieme tirage de
  l'instance softmax (graine de donnees 2), a des pas situes entre les points de la grille de facteur 2. Script :
  `paper/audit/checks/num-geometry_repro_draw2.py` (parties `saga`, `bsaga`, `isbsaga`).
- Figure `paper/figures/figP2_geometry.png` lue, script `make_figP2_geometry.py` et `panel_c` /`data_c` de
  `make_figR2_muon_vr.py` relus.
- Non ouverts : les pickles, `M1_pass*_summary.json` (le resume final `M1_summary.json` suffit).
- Ecart a la regle de lecture seule, a signaler : les deux scripts qui importent le code du depot
  (`num-geometry_constants.py`, `num-geometry_repro_draw2.py`) ont laisse Python reecrire des caches de bytecode
  `muon/lab/__pycache__/*.pyc` (fichiers ignores par git). Aucun fichier source, resultat ou fichier suivi n'a ete
  modifie (`git status` : seul `paper/audit/` apparait). Les scripts posent maintenant
  `sys.dont_write_bytecode = True` avant l'import.

## 2. Affirmations controlees

| Affirmation (papier) | Lignes | Verdict | Preuve |
|---|---|---|---|
| "best ratio to the Euclidean one exactly $1$" | 369 | OK, mais trivial (constat G1) | M7 : 96 cellules softmax (ancre x plancher), 0 sous 1 ; les 3 cellules a 1.000000 sont la metrique de covariance a f = 1, qui est un multiple de l'identite |
| "local constants are up to $17.5$ times smaller (covariance metric: $551$ against $9659$)" | 370 | OK | soft-cond : 9658.69 / 550.546 = 17.544 (f = 0.01) ; autres instances 2.39 (soft-mem), 7.91 (soft-sq) ; recalcul independant identique |
| "improves by $1.6$ to $103$ in the polar metric" | 371 | OK | 476.14/299.61 = 1.589 ; 7296.5/1613.6 = 4.522 ; 4363.6/42.42 = 102.86 |
| "($7.6$ to $163$ in the covariance metric)" | 371 | OK | 7.631 ; 163.16 ; 110.79 |
| "$R_P^2/\mu_P=42$ against $R^2/\mu=4364$" | 372 | OK | ls-orth : 42.424 et 4363.575 (recalcul independant) |
| "needs $13.3$ passes (SAGA $27.1$, covariance metric $13.1$)" | 372-373 | OK | M1 ls-orth, 10/10 graines : 13.30 ; 27.15 ; 13.15 (cov-metric SAGA) ; SAGA inchange sur grille fine (27.15) |
| "SAGA needs $69.5$ / $98.7$ passes" | 375 | chiffre exact, limite par la grille (constat G2) | M1 : 69.55 / 98.70 ; M5b grille 2^(1/4) : 49.75 / 83.55 |
| "$22.1$--$29.0$ / $24.4$--$43.5$" | 376 | OK | soft-cond : 22.05 (one-shot polar B-SAGA), 23.85 (soft), 29.00 (joint) ; ls-cond : 24.35, 29.80, 43.55 |
| "one seed of the joint method misses the target on least squares" | 377 | OK | SAGA-Muon (joint) ls-cond : 9/10, max = inf |
| "$15.1$--$33.8$ / $16.1$--$31.9$" | 377-378 | OK | cov-metric B-SAGA 15.05 / 16.05 ; IS-B-SAGA 25.70 / 27.60 ; B-SAGA 33.80 / 31.90 |
| "on grids of ratio $2^{1/4}$ B-SAGA needs $27.1$ / $31.9$" | 378-379 | OK | M5b, b = 1, partie fine : 27.10 / 31.90 |
| "$0.20$--$0.37$ of the passes of SAGA" | 379-380 | chiffre exact, artefact de grille (constat G2) | M8 + M1 : 0.342, 0.369, 0.202 (softmax) ; 0.286, 0.301, 0.296 (moindres carres) |
| "$0.78$--$1.08$ of those of the best Euclidean method without preconditioner" | 380 | chiffre exact ; borne basse non resolue (constat G3) | 0.926, 0.858, 0.780 ; 1.024, 1.079, 0.987 |
| "the covariance metric is the best method on least squares for every draw, and on softmax for one draw of three" | 380-381 | OK parmi les dix familles de M8 (constat G6) | ls : 16.55, 15.95, 16.30 (meilleur) ; softmax : 18.05 (meilleur), 26.9 (battu par 23.4), 35.3 (battu par 27.45) |
| "$16$--$18$ (softmax) and $17$--$21$ passes (least squares) to $10^{-9}$" | 382 | OK | meilleures familles des six classes VR du digest : 15.8-18.4 et 17.1-20.6 ; par groupe JSON : 15.75 / 16.35 et 17.15 / 18.45 (L-BFGS, plein lot : 23 et 25, hors "every class") |
| "near the ceiling ... $e^{-1}$ per pass" | 382-383 | imprecis (constat G7) | 1e-9 = 20.7 nats ; les medianes sont sous 20.7 passes |
| "two memory-limited ($R^2/\mu$ of the order of $n$)" | 261 | OK | soft-mem : R2/lambda = 1.00 n (local 0.86 n) ; ls-mem : 0.48 n |
| "four conditioning-limited ($R^2/\mu$ from $9n$ to $50n$; two square)" | 261-262 | OK | ls-orth 8.73 n ; soft-sq 20 n ; ls-cond 36.5 n ; soft-cond 50 n (mu globale : lambda pour softmax, lambda_min(H) pour moindres carres) |
| "$n\in\{200,500,1000\}$, $K\times d\in\{8\times40,10\times50,20\times20\}$" | 260 | OK | specs de M1 |
| "exact minimizers" | 260-261 | OK | gradient recalcule en W_* : 5e-18 a 6e-17 |
| "relative squared distance $10^{-6}$ ($10^{-9}$ where memory limits)" | 262-263 | OK | `tol` de M1 : 1e-9 (soft-mem, ls-mem), 1e-6 ailleurs |
| "tuned on five seeds and run on ten fresh ones" | 263 | OK | protocole JSON : graines 0-1, 3 finalistes sur 2-4, confirmation 10-19 (criblage a 20 % du budget sur une graine) |
| "ten families also on two further draws of the data" | 264 | OK, sur 2 instances seulement (constat G9) | M8 : 10 familles, soft-cond et ls-cond, graines de donnees 1 et 2 |
| "differences below about $10\%$ are not resolved" | 264 | sous-estime (constat G4) | bruit de graines : oui ; raffinement de grille : -15 a -28 % |
| "$\gamma(m+\vartheta c)$ with $\vartheta=1/n$, $\gamma=1/\lambda$ is MISO" | 255-256 | OK | algebre : theta - (1/lambda)(zbar + lambda theta + (1/n)(grad g_j - z_j)) = -(1/lambda) zbar_nouveau |
| Formules de R_P^2, mu_P, metrique polaire, metrique de covariance | 236-242 | OK | conformes a `vrmuon.py` (l. 41-61, 651-671) et au recalcul independant (7.5e-14) |
| Legende 2(a) : "Filled: global constants (certified); open: $\mu_P$ at the optimum (each at its best $f$); one marker when they coincide" | 359-361 | OK (nit G8) | script : min sur f separement pour global et local ; un seul marqueur si egalite a 1e-9 |
| Legende 2(b) : "median and range over ten seeds; arrow: a seed misses the target" | 361-363 | OK | `data_c` / `panel_c` : mediane, min, max des 10 `confirm_passes` de M1, fleche si max = inf ; 7 familles conformes au texte |
| Tableau 1 : "on least squares a polar metric frozen at $W_0$ does \nat{e}" | 157 | OK (constante), voir G5 | gains 1.6 / 4.5 / 103 de la constante certifiee |
| Tableau 1 : "at batch one, tuned Euclidean devices match orthogonalization \nat{e}" | 157-158 | OK a la resolution des grilles (constat G3) | rapports 0.78-1.08 ; M1 tirage 0, six instances : 0.93-1.11 |
| Intitule (M4) | 367 | suit des donnees (3 instances softmax : aucun gain, theoreme ; 3 instances moindres carres : gain de la constante), reserve G5 | voir G5 |
| Intitule (M3) | 374 | suit des donnees a la resolution des grilles, sur 2 instances x 3 tirages (et 6 instances au tirage 0) ; reserve G3 | voir G3 |

Definitions verifiees dans les donnees : "the best method recomputing the polar factor" = minimum sur {SAGA-Muon
(joint), soft SAGA-Muon, IS soft SAGA-Muon, split SAGA-Muon} (les deux familles a metrique gelee sont exclues) ; c'est
soft SAGA-Muon ou IS soft SAGA-Muon dans les six cellules (application "soft", pas le facteur polaire exact). "the best
Euclidean method without preconditioner" = minimum sur {B-SAGA, IS-B-SAGA} ; c'est IS-B-SAGA dans les six cellules.

## 3. Constats

### G2 (major) : les passes de SAGA et le rapport "0.20--0.37" sont fixes par la grille de pas de facteur 2

- Lignes 375 et 379-380 ; figure 2(b), ligne "SAGA".
- Papier : "SAGA needs $69.5$ / $98.7$ passes" ; "the best method recomputing the polar factor needs $0.20$--$0.37$ of
  the passes of SAGA".
- Probleme. SAGA a un seul parametre, regle sur une grille de facteur 2 ; ses deux meilleurs points de grille
  different de 98 % (scores de reglage 69.5 / 137.9 sur soft-cond, 99.1 / 195.9 sur ls-cond). La grille 2^(1/4) du
  depot lui-meme (M5b, b = 1, meme instance, memes graines de confirmation) donne 49.75 passes (gamma = 5.66/R2) au
  lieu de 69.55 sur soft-cond (-28 %) et 83.55 (gamma = 2.38/R2) au lieu de 98.70 sur ls-cond (-15 %). Le papier cite
  la valeur a grille fine pour B-SAGA seulement. La borne 0.20 vient du troisieme tirage softmax, ou M8 retient
  gamma = 2/R2 (136.1 passes) au lieu de 4/R2 (69.5 et 67.9 sur les deux autres tirages) : un cran de grille.
- Preuve. `M5b_summary.json`, instances/soft-cond/0/batches/1/fine/rows/SAGA : mediane 49.75 [49.3, 50.6], 10/10 ;
  ls-cond : 83.55 [82.8, 87.8]. Reproduction (`num-geometry_repro_draw2.py saga`, graines 10 et 11, tirage 2) :
  gamma = 2/R2 : 136.0, 136.2 (identique a M8) ; 2.38/R2 : 114.6 ; 2.83/R2 : 96.5 ; 3.36/R2 : 81.4 ; 4/R2 : 73.5 et
  111.0. A pas resolus le rapport vaut donc environ 27.45/81.4 = 0.34 sur ce tirage (et non 0.20), et sur le premier
  tirage 23.8/49.75 = 0.48 (softmax) et 28.25/83.55 = 0.34 (moindres carres) ; a grilles fines des deux cotes (M5b,
  SAGA-Muon joint / SAGA) : 0.49 et 0.40.
- Consequence : le gain de l'orthogonalisation face a SAGA est d'un facteur 2 a 3, pas 3 a 5.
- Correction proposee (l. 378-380) : voir le JSON (`G2`).

### G3 (minor) : la borne 0.78 n'est pas une difference resolue, et telle qu'ecrite elle contredit "reproduce"

- Lignes 374, 379-381 ; tableau 1 l. 157-158.
- Papier : "orthogonalization buys a trajectory effect that Euclidean devices reproduce" ; "$0.78$--$1.08$".
- Probleme. Sur le troisieme tirage softmax la meilleure methode orthogonalisee (soft SAGA-Muon, 27.45) devance
  toutes les methodes euclidiennes de M8, metrique de covariance comprise (IS-B-SAGA 35.2, cov-metric SAGA 35.3,
  B-SAGA 35.6), de 22 %, au-dela de la resolution annoncee de 10 % ; sur le deuxieme tirage l'ecart est de 14 % face
  aux methodes sans preconditionneur. Lue avec la resolution du papier, cette cellule dit que les dispositifs
  euclidiens ne reproduisent pas le gain. La reproduction montre que l'ecart tient en partie a la grille de facteur 2
  du cote euclidien (les trois familles euclidiennes perdent un cran sur ce tirage : 4 -> 2 et 16 -> 8).
- Preuve. `M8_summary.json` soft-cond/2. Reproduction (`... isbsaga`, graines 10-12, tirage 2) : IS-B-SAGA
  theta = 0.2 : gamma = 8/R2 : 35.5 ; 9.51/R2 : 31.7 ; 11.31/R2 : 30.8 ; 13.45/R2 : 31.5 ; 16/R2 : 47.6. Le rapport
  passe de 0.78 a environ 27.45/30.8 = 0.89 (cote orthogonalise non raffine, 3 graines). B-SAGA ne gagne rien
  (35.7-35.9 a 8/R2, pire au-dela).
- L'intitule tient donc a environ 10 % pres, mais le papier doit dire que les bornes de l'intervalle sont a la
  resolution d'une grille de facteur 2.

### G4 (minor) : "differences below about 10% are not resolved" ne couvre que le bruit de graines

- Lignes 263-264.
- Probleme. Le 10 % vient de l'erreur-type bootstrap des medianes sur dix graines (recalculee ici : 0.1 a 8.7 % pour
  les familles de (M3) avec toutes les graines a la cible). Le raffinement de la grille de pas deplace les medianes
  de bien plus : a b = 1, M5b donne SAGA -28.5 % / -15.3 %, B-SAGA -19.8 % / 0 %, SAGA-Muon -15.3 % / -23.5 %
  (soft-cond / ls-cond) et SAGA-Muon -18.3 % sur ls-orth. Les comparaisons de (M3), toutes sur grilles de facteur 2,
  ne sont donc pas resolues a 10 %.
- Preuve : `num-geometry_numbers.py`, bloc "M5b".

### G1 (minor) : "best ratio ... exactly 1" est vrai par construction

- Lignes 368-370 ; figure 2(a), marqueurs pleins bleus des trois instances softmax.
- Probleme. Le rapport 1 est atteint par la metrique de covariance au plancher f = 1, ou toutes les valeurs propres
  sont egalisees : c'est un multiple de l'identite. Le marqueur plein "covariance metric" de la figure 2(a) sur
  softmax est donc l'identite. Le resultat informatif est que toute metrique testee distincte de l'identite est
  strictement moins bonne : covariance a f < 1 : 1.14 a 4.46 ; metrique polaire : au mieux 1.419, 1.068, 1.001 (a
  f = 1), 1.20 a 3.35 pour f < 1.
- Preuve : `num-geometry_constants.py` (lignes marquees "metric is a multiple of the identity") ; `vrmuon.py` l. 655.

### G5 (minor) : "certifies a gain" porte sur la constante, pas toujours sur le taux certifie

- Lignes 367, 371.
- Probleme. Le taux certifie est min{1/(3n), 3 mu_P/(16 R_P^2)}. Sur l'instance limitee par la memoire (gain de
  constante 1.6) le taux euclidien est deja 1/(3n) : le taux certifie ne change pas. Sur l'instance carree le gain de
  103 de la constante vaut 15.5 sur le taux certifie (plafond 1/(3n)) ; sur ls-cond 4.5.
- Preuve : `num-geometry_numbers.py`, dernier bloc "(M4) heading".

### G6 (minor) : "the covariance metric is the best method" compare des familles inegalement equipees

- Lignes 380-381.
- Probleme. M8 contient cov-metric SAGA (non amortie) mais pas cov-metric B-SAGA, alors que la metrique polaire y
  figure en versions SAGA et B-SAGA. Sur le premier tirage softmax l'amortissement fait passer la metrique de
  covariance de 18.05 a 15.05 passes. "on softmax for one draw of three" vaut donc pour la version non amortie
  seulement (tirage 1 : 26.9 contre 23.4 ; tirage 2 : 35.3 contre 27.45).

### G7 (minor) : "near the ceiling" alors que les medianes sont sous le plafond

- Lignes 381-383.
- Probleme. A exactement e^-1 par passe, 1e-9 demande 20.7 passes ; les meilleures familles en demandent 15.8 a 18.4
  (softmax) et 17.1 a 20.6 (moindres carres), passe initiale comprise : elles vont plus vite que le "plafond". La
  phrase de (M7) ("bounds the mean error, not a single run") le dit, pas celle de (M3).

### G8 (nit) : figure 2(a), marqueur certifie de l'identite masque

- Sur soft-cond et soft-sq le marqueur plein de l'identite (10000) est recouvert par le marqueur ouvert (9659, 9214),
  dessine au-dessus (`zorder` 4 contre 3) : on ne voit qu'un marqueur "local", alors que la legende reserve le
  marqueur unique aux moindres carres et que la constante euclidienne certifiee est la reference du texte.

### G9 (nit) : "two further draws of the data" ne concerne que deux instances

- Ligne 264. M8 ne contient que soft-cond et ls-cond (pour ls-orth seuls les caches de criblage et de reglage du
  tirage 1 existent, sans confirmation).

## 4. Non verifie

- Effet d'une grille fine sur les familles orthogonalisees des tirages 1 et 2 (seul le cote euclidien a ete raffine,
  2 a 3 graines) ; le rapport 0.89 du constat G3 est donc indicatif.
- Reglage complet de M1 et M8 non relance ; pickles et `M1_pass*_summary.json` non ouverts.
- Le "plafond" e^-1 par passe comme theoreme (note B4) : non redemontre ici.
- Attributions de B-SAGA, de l'echantillonnage preferentiel et de la metrique de covariance : autre dimension.
