# Audit math-nfl : Théorème 1 et géométrie des pas polaires

Papier audité : `paper/paper_saga_to_muon.tex` (numéros de ligne de l'outil Read). Auditeur indépendant, dimension
`math-nfl`. Date : 2026-11-20. Rapport final.

Bilan : **0 blocker, 1 major (E2), 4 minor, 1 nit.** Le Théorème 1 (énoncé, hypothèses, preuve, exemple, conséquences) est
exact ; l'identité (2) et ses encadrements sont exacts ; `kappa_op >= min(K,d)` est vrai pour toute F. Les constats
portent sur des formulations qui disent plus que ce qui est démontré (résumé et tableau 1 pour « same anatomy »,
« ray that attracts generic runs », « no fixed change of geometry »), et sur une attribution (Sidford–Tian).

## 1. Périmètre couvert

(a) définitions l. 196-206, l. 117, l. 200-201 ; (b) Théorème 1 l. 334-347 ; (c) remarques l. 349-354 ;
(d) identité (2) l. 226-232 ; (e) paragraphe l. 309-315, résumé l. 96-97, étiquette [p] du tableau 1 l. 151, conclusion
l. 674-676 ; (f) (M4) l. 367-369 et métrique polaire l. 236-242.
Sources lues : `archive/muon/notes_spectral_theory.tex` (sections 1-3, Théorème « mad », Remarque « headtail »,
questions ouvertes), `muon/notes_theory.md` (F10, F11), `archive/muon/report_spectral_muon_new.tex` (Théorème nfl),
`muon/lab/vrmuon.py` (`polar_metric`), `muon/results/M7_summary.json`.
Sources externes ouvertes : Sidford–Tian arXiv:1808.01278v2 (PDF, texte extrait) ; Hird–Livingstone JMLR 26 (PDF,
texte extrait) ; Shulgin et al. arXiv:2610.30280v1 (HTML, 100 000 premiers caractères sur 227 000, via l'outil de
lecture qui résume) ; Jawanpuria et al. arXiv:2610.27747v1 (HTML, 100 000 premiers caractères sur 114 000, idem).

Scripts (tous dans `paper/audit/checks/`, lancés depuis la racine, un fil, < 90 s chacun) :
`math-nfl_identity2.py`, `math-nfl_muN.py`, `math-nfl_kop.py`, `math-nfl_ls_tail.py`, `math-nfl_ls_head.py`,
`math-nfl_polar_metric.py`.

## 2. Affirmations contrôlées : verdicts

### (a) Définitions

| Affirmation | Lignes | Verdict | Preuve du contrôle |
|---|---|---|---|
| `polar`, normes, `mu_N`, `L_N`, `kappa_N`, (GLM), `c_i`, pas polaire fixe / à longueur nucléaire | 196-206 | OK | Cohérent avec les notes (l. 91-125 de `notes_spectral_theory.tex`) et avec l'usage dans le Théorème 1 et (2). `c_i` vaut 1/2 (softmax), 1/4 (logistique), 1 (moindres carrés). |
| « without momentum this is steepest descent for the spectral norm » | 117 | OK (nuance) | argmin de `<G,D> + (L/2)||D||_op^2` = `-(||G||_*/L) polar(G)` : la direction de Muon sans moment est celle de la plus forte pente spectrale ; la longueur de Muon est fixe (version « normalisée »), ce que le papier distingue l. 204-205. Pas un constat. |
| « variance-reduced methods are governed by the per-sample ratios L_N(f_i)/mu_N(F) » | 200-201 | problème mineur (A1) | Établi pour les normes euclidiennes (Prop. 5.9 dans une métrique P, F11 vérifié) ; pour une norme non euclidienne aucun théorème du papier ni référence citée ne donne ce rapport. |
| `R_P^2 = max_i c_i x_i^T P^{-1} x_i + lam ||P^{-1}||_op`, `mu_P = mu_P(F)` pour `||. P^{1/2}||_F` | 236-238 | OK | `R_P^2` majore `max_i L_N(f_i) = max_i lmax(c_i P^{-1/2}x_i x_i^T P^{-1/2} + lam P^{-1})` ; vérifié numériquement (`math-nfl_polar_metric.py` : 32.6003 >= 32.4917, etc.). |

### (b) Théorème 1

Démonstration refaite ligne à ligne.

1. `D^2F(W)[D,D] = (1/n) sum_i (D x_i)^T hess l_i(W x_i) (D x_i) + lam ||D||_F^2 >= lam ||D||_F^2 >= (lam/c_N^2) N(D)^2`
   (convexité des pertes, puis `N <= c_N ||.||_F`). Donc `mu_N(F) >= lam/c_N^2`. OK.
2. Un maximiseur `D_0` de `N/||.||_F` existe (dimension finie). `D^2F(W_t)[D_0,D_0] <= max_i ||hess l_i(W_t x_i)||_op
   (1/n) sum_i ||D_0 x_i||^2 + lam ||D_0||_F^2 -> lam ||D_0||_F^2 = (lam/c_N^2) N(D_0)^2`. Donc `mu_N(F) = lam/c_N^2`,
   et `mu_F(F) = lam`. OK.
3. Pour g convexe, `D^2g >= 0` et `N(D)^2 <= c_N^2 ||D||_F^2` donnent `D^2g[D,D]/N(D)^2 >= D^2g[D,D]/(c_N^2||D||_F^2)` ;
   sup sur (W, D) : `L_N(g) >= L_F(g)/c_N^2`. OK (la convexité de g est bien nécessaire et supposée).
4. Conséquences : `kappa_N(F) = L_N(F)/mu_N(F) >= (L_F(F)/c_N^2)/(lam/c_N^2) = kappa_F(F)` ; de même pour `f_i`. OK.
5. `L_F(f_i) = c_i ||x_i||^2 + lam` : majoration par `||D x_i|| <= ||D||_F ||x_i||`, approchée le long de
   `e x_i^T/||x_i||` avec `W x_i = z` arbitraire (possible dès que `x_i != 0` ; si `x_i = 0` la formule reste vraie,
   `L_F(f_i) = lam`). OK.
6. Exemple « logistic and softmax losses with nonzero x_i: W_t = tW_1 for almost every W_1 » : softmax,
   `hess = diag(p) - pp^T -> 0` dès que `W_1 x_i` a un maximum unique ; l'ensemble des `W_1` pour lesquels deux
   coordonnées de `W_1 x_i` coïncident pour un i est une réunion finie d'hyperplans (propres car `x_i != 0`), de mesure
   nulle. Logistique (K = 1) : `l'' = s(1-s) -> 0` dès que `W_1 x_i != 0` pour tout i. La courbure de **toutes** les
   pertes tend donc vers 0 (max sur un nombre fini d'indices). Les deux réserves de l'énoncé (« nonzero x_i »,
   « almost every ») sont nécessaires : voir les contre-exemples numériques ci-dessous. OK.

Contrôle numérique (`math-nfl_muN.py`, softmax K=4, d=6, n=30 et logistique K=1, d=6, n=30, lam = 0.01 ; six normes :
spectrale, l_infini, `||D P^{1/2}||_F`, Kronecker `||S^{1/2} D P^{1/2}||_F`, spectrale préconditionnée
`||D P^{1/2}||_op`, l_1) :

- `max_i lmax(hess l_i(t W_1 x_i))` : softmax 4.6e-1, 3.4e-1, 6.2e-6, 1.3e-55 pour t = 1, 10, 100, 1000 ; logistique
  2.5e-1, 1.5e-1, 2.9e-65, 0 pour t = 1, 1e2, 1e4, 1e6. Contre-exemples : deux lignes égales de `W_1` (softmax) :
  reste à 0.50 ; `x_1 = 0` : reste à 0.25.
- `D^2F(tW_1)[D_0,D_0]/N(D_0)^2` rapporté à `lam/c_N^2` : tend vers 1.000000 pour les six normes (softmax : 5.61, 1.78,
  1.000011, 1.000000 pour la norme spectrale ; mêmes limites pour les autres).
- Sur 2000 couples (W, D) aléatoires + raffinement local par norme, le quotient de courbure ne descend jamais sous
  `lam/c_N^2` (rapport minimal entre 1.000000 et 1.0077).
- `L_N(F)` (minorant numérique) `>= L_F(F)/c_N^2` pour les six normes ; rapport par échantillon
  `L_N(f_i)/mu_N(F) >= (c_i||x_i||^2+lam)/lam` (588.8 softmax, 271 logistique) : vérifié, égalité pour la norme
  spectrale dans la direction de rang un.
- `L_F(f_i)` : quotient en `(W, e x_i^T/||x_i||)` = 5.8880724672 = `c_i||x_i||^2 + lam` (softmax) ; 2.7098313407
  (logistique).

Contrôle sur les données du dépôt (`muon/results/M7_summary.json`, bloc `metric`) : sur les trois instances softmax,
le rapport `kappa_P_global/kappa_euclid_global` de toutes les métriques testées (16, 16 et 64 lignes) est >= 1, minimum
1.000000000000 (métrique de covariance, plancher f = 1, multiple de l'identité), maximum 4.1 à 14 ; sur les trois
instances moindres carrés il descend à 0.131, 0.0061, 0.0090. Conforme au Théorème 1 et à (M4) « best ratio ...
exactly 1 ».

Verdict (b) : **OK**, énoncé, hypothèses, preuve et exemple.

### (c) Remarques l. 349-354

| Affirmation | Verdict | Preuve du contrôle |
|---|---|---|
| « It covers the spectral norm and its preconditioned versions, the l_infty norm of sign descent and Kronecker-factored metrics » | OK | Ce sont des normes (P, S définies positives) ; `c_N` = 1, `lmax(P)^{1/2}`, 1, `(lmax(S)lmax(P))^{1/2}`. Vérifié numériquement pour les six normes ci-dessus. |
| « functions that no linear preconditioner improves exist in every dimension [Proposition 7, hird2025quantifying] » | OK | JMLR 26 (2025), Proposition 7 (texte du PDF) : « There exist distributions satisfying Assumption 6 with arbitrary condition number kappa >= 1 in arbitrary dimension d >= 1 for which any non-orthogonal linear preconditioner will cause the condition number to increase. » |
| « The hypothesis is needed (for ridge least squares ||Delta Sigma^{1/2}||_F, Sigma the Hessian, has condition number one) » | OK | `D^2F[D,D] = tr(D Sigma D^T) = ||D Sigma^{1/2}||_F^2` avec `Sigma = Sigma_hat + lam I` : `mu_N = L_N = 1`. L'hypothèse est violée (`hess l_i = I`) et la conclusion `kappa_N >= kappa_F = lmax(Sigma)/lmin(Sigma)` aussi. |
| « kappa_op >= min(K,d) » | OK (vrai pour toute F) | Preuve par cadres (notes, Prop. « frame », refaite) : pour `H = hess F(W)`, cadres orthonormés `u_k, v_k`, `h_k = q(u_k v_k^T) >= mu_op` ; `Q_eps = sum eps_k u_k v_k^T` est de norme spectrale 1 et `E_eps q(Q_eps) = sum_k h_k`, donc `L_op >= sum_k h_k >= r mu_op`. Autre preuve, sans C^2 : restreindre F à `W* + U diag(s) V^T`, `s` dans `[-1,1]^r` (norme spectrale = norme l_infini de s) ; Lemme A.1 de Sidford–Tian donne une étendue >= `mu_op r/2`, la régularité au minimiseur une étendue <= `L_op/2`. Numérique (`math-nfl_kop.py`) : 210 opérateurs SPD aléatoires, tailles 2x2 à 6x4, y compris proches de l'identité : `kappa_op/r >= 1.0000015` partout, `= 1` exactement pour `H = Id` ; force brute 2x2 : min 1.0000088. |
| référence `[sidford2018coordinate]` pour cette borne | problème mineur (C1) | La source contient l'analogue l_infini sur l'étendue (introduction : « it is a folklore result that any function strongly-convex over [-1,1]^n in the l_infty norm has range at least n/2, which we show in Appendix A.1 » ; Lemme A.1), pas un énoncé sur le conditionnement ni sur la norme spectrale. |

### (d) Identité (2), l. 226-232

- Identité : `F(W - eta O) = F(W) - eta ||G||_* + (eta^2/2) <O,HO>` exactement (quadratique, `<G,O> = ||G||_*`) ; avec
  `eta = ||G||_*/L` et `||G||_*^2 = 2 mu_PL (F - F*)` on obtient (2). OK.
- `J` dans `[mu_op, L_op]` : `||O||_op = 1`. OK.
- `mu_PL >= mu_op` : forte convexité pour la norme spectrale et dualité `|<G,D>| <= ||G||_* ||D||_op` donnent
  `F* >= F(W) - ||G||_*^2/(2 mu_op)`. `mu_PL <= J` : `F* <= min_eta F(W - eta O) = F(W) - ||G||_*^2/(2J)`. OK.
- Facteur certifié : le membre de droite de (2) décroît en `mu_PL` et croît en `J` (pour `J <= L`), donc son maximum
  sur `[mu_op, .] x [., L_op]` est en `mu_PL = mu_op`, `J = L_op` : `1 - 1/kappa_op`. OK.
- Numérique (`math-nfl_identity2.py`) : 720 cas (hessiennes unilatérales, Kronecker, SPD générales ; six tailles ;
  quatre valeurs de L ; erreurs de rang plein et de rang un) : écart max 2.8e-15 ; `mu_PL <= J` violé 0 fois ; marges
  des quatre encadrements >= -1.2e-15 ; certifié - exact >= -8e-16.
- Références : Eq. (17) de Shulgin et al. lue (HTML) : `f(X+)/f(X) = 1 - alpha(2 - alpha) p(X)`, `alpha = gamma tr(H)`,
  `p = ||HX||_*^2/(tr(H) tr(HXX^T))`, qui est (2) avec `J = tr H`, `alpha = J/L`, `p = mu_PL/J`. Eq. (8) de Jawanpuria
  et al. lue (HTML) : identité exacte `f(X_{t+1}) = f(X_t) - alpha||G||_*^2 + (alpha^2/2)||G||_*^2 tr(U^T H U)`. OK.
  Théorème 3 de Jawanpuria et al. : facteur `1 - u(2-u) lmin/tr(H)`, `u = alpha tr(H)` : c'est `1 - 1/kappa_op` quand
  `L_op = tr(H)` (K >= d dans l'orientation du papier) ; pour K < d le raffinement `tr_n(H)` est dans leur annexe C
  (nit D1).

Verdict (d) : **OK**.

### (e) « The same anatomy for polar steps », l. 309-315 ; résumé l. 96-97 ; [p] du tableau 1

| Affirmation | Verdict | Preuve du contrôle |
|---|---|---|
| « the exact exponent is, to first order, (mu_PL/mu_op)(2 - J/L_op) times the certified one » | OK | `-log(1-x) ~ x` avec `x = (mu_PL/L_op)(2 - J/L_op)`, certifié `~ mu_op/L_op`. Numérique : produit des deux marges = 4.00 contre rapport exact 4.02 (20x20, kappa_op = 461) ; l'écart au premier ordre n'est visible que pour `kappa_op` petit (2x2, kappa_op = 3 : 2.67 contre 5.42), ce que couvre « to first order ». |
| « the smoothness slack 2 - J/L_op in [1,2) gains at most a factor 2 » | OK | `J` dans `[mu_op, L_op]`. Remarque : pour les moindres carrés unilatéraux avec K >= d et gradient de rang plein, `O^T O = I_d` donc `J = tr Sigma = L_op` et cette marge vaut exactement 1 (mesuré : 1.000). |
| « as I1–I2 at most 8/3 » | OK | Dans (1), le facteur `1 - (L_eff/R^2)(r_sq/4 + 3/8)` vaut 3/8 au pire cas et au plus 1. |
| « only the sharpness slack mu_PL/mu_op in [1, kappa_op] is unbounded » | OK | `mu_PL` dans `[mu_op, J]`, `J <= L_op`. |
| « On least squares ... the gap is about a factor 4 on the ray that attracts generic runs [shulgin2026quadratic] » | problème (E2, major) | Le facteur 4 sur le rayon invariant de la direction la plus plate est exact (`(1-2p_d)^2` contre `1-p_d`). Que les courses génériques y soient attirées n'est ni démontré ni affirmé par la source : c'est une observation numérique, et les notes du dépôt démontrent l'instabilité linéaire de ces rayons dans le cas commutant. De plus le facteur 4 ne vaut que dans la queue. Détail ci-dessous. |
| résumé « the nuclear-scaled polar step has the same anatomy » ; tableau 1 « the same anatomy for the polar step [p] » | problème mineur (E1) | Ce qui est prouvé : l'identité (2) **sur les quadratiques** (publiée) et les deux encadrements. Rien n'est mesuré pour le pas polaire sur la logistique ou le softmax (le papier le dit l. 315). Détail ci-dessous. |

**Ce qui est réellement prouvé pour « the same anatomy » [p].** (i) Sur une quadratique, le taux certifié
`1 - 1/kappa_op` est la formule exacte (2) évaluée aux pires valeurs de deux rapports dépendant de l'état ; (ii) la
marge de régularité est bornée par 2 ; (iii) la marge `mu_PL/mu_op` n'est bornée que par `kappa_op`. C'est un énoncé
de structure, immédiat à partir d'une identité publiée (Shulgin et al., Eq. (17) ; Jawanpuria et al., Eq. (8)) ; les
notes du dépôt le classent « [K] for (LS); immediate in general » (`notes_spectral_theory.tex` l. 355). Ne sont pas
prouvés ni mesurés : la taille de l'écart et sa répartition le long de courses du pas polaire sur un modèle
logistique ou softmax (la phrase du résumé suit immédiatement les chiffres 3.2–220 et 77–96 % mesurés pour SAGA sur
la logistique). L'identité s'étend pourtant sans effort à toute F de classe C^2 : avec
`J~ = 2 int_0^1 (1-s) D^2F(W - s eta O)[O,O] ds` dans `[mu_op, L_op]` et `mu_PL` dans `[mu_op, L_op]` (inégalités de
PL duales), on a exactement `(F(W+)-F*)/(F(W)-F*) = 1 - (mu_PL/L)(2 - J~/L)` (formule de Taylor avec reste intégral ;
vérifié sur un softmax ridge, `math-nfl_identity2.py` (iv), écart max 4.0e-15). Énoncée ainsi, la structure serait
prouvée aussi pour les modèles du Constat 1.

**Le facteur 4 sur les moindres carrés.** `math-nfl_ls_tail.py` (départs gaussiens, 5 graines, 8 tailles, 4 spectres,
160 courses) : exposant exact / certifié dans la seconde moitié des courses : K >= d : min 4.01, médiane 4.05, max
5.42 ; K < d (dont 8x40 et 10x50, formes des deux instances non
carrées du papier) : min 3.22, médiane 3.79, max 5.31. « about a factor 4 » est donc correct pour la queue.
Mais (`math-nfl_ls_head.py`, 3 graines) l'écart moyen sur une course entière depuis un départ générique est bien plus
grand : 33 à 541 jusqu'à `D/D_0 = 1e-2`, 6.9 à 13.9 jusqu'à 1e-6 (la cible du papier), 5.2 à 6.9 jusqu'à 1e-12, avec
`mu_PL/mu_op` de 21 à 3343 au premier pas (de l'ordre de `kappa_op`). À courbure (hessienne) constante, l'écart
n'est donc « about 4 » qu'asymptotiquement ; la phrase suivante (« a large gap needs a global constant far below the
curvature met ») n'est juste que si « curvature met » désigne `mu_PL` (qui dépend de la direction), pas la hessienne.
Statut de l'attraction : `notes_spectral_theory.tex` l. 800-801 (« Which of these rates a run follows is a numerical
matter [N], because the rays are unstable »), Théorème « mad » (c) l. 699-701 (« For d >= 3 these rays are linearly
unstable ») et question ouverte (2) l. 1210-1214 (« Is the flattest ray attracting for the non-commuting iteration,
and why? »). Dans mes courses l'alignement avec la direction la plus plate n'est d'ailleurs pas systématique
(cosinus médian 0.71 pour 20x20, spectre 1/k). La source citée (HTML, premiers 100 000 caractères) ne contient pas
d'énoncé d'attraction depuis des départs génériques ; son Corollaire 5 donne le facteur
`(((d-1)kappa-1)/((d-1)kappa+1))^2 = (1-2p_d)^2` pour une configuration construite.

### (f) (M4) et métrique polaire

| Affirmation | Lignes | Verdict | Preuve du contrôle |
|---|---|---|---|
| métrique polaire `omega -> omega (V diag(g) V^T + r(I - VV^T))`, `g_k = ||G_0||_*/max(s_k, f s_1)` | 240-242 | OK | Identique à `muon/lab/vrmuon.py`, fonction `polar_metric` et à F10. Sans plancher, `G_0 C = ||G_0||_* polar(G_0)` (écart 1.4e-16 et 1.9e-15, `math-nfl_polar_metric.py`). |
| « Computed once, the polar metric makes the method SAGA in a fixed metric, certified with R_P^2, mu_P » | 367-368 | OK | Preuve de F11 refaite (étapes (1) à (6) : identité de la distance pondérée, centrage de la mémoire, cocoercivité de `f^_j` avec `R_P^2`, cocoercivité des parties perte, `G >= mu_P ||.||^2_{C^-1}`) ; pour `c = 1/4`, `k = 3` le facteur est `1 - min{1/(3n), 3mu_P/(16R_P^2)}`. Numérique : espérance conditionnelle exacte de la fonction de Lyapunov le long de 600 états, softmax et moindres carrés, planchers 0, 0.1, 0.3, 1 : `max E[V+]/(rho V)` entre 0.9949 et 0.9972 (<= 1). Réserves déjà dans F11 et non contredites par le papier : la quantité contractée est la distance pondérée par `C^{-1}` ; le certificat vaut pour `gamma <= 1/(2R_P^2)`, pas pour les pas réglés. |
| (M4) « On softmax no metric tested improves the global constant (... exactly 1) » | 369 | OK | `M7_summary.json` : minimum 1.000000000000 sur les trois instances softmax. Petit softmax du script : `kappa_P/kappa` = 2.94, 2.94, 2.11, 1.47 pour f = 0, 0.1, 0.3, 1 (tous >= 1). |
| (C5) « for ridge logistic regression every P has kappa_P >= kappa » | 327 | OK | Théorème 1 avec `N = ||. P^{1/2}||_F` et `R_P^2 >= max_i L_N(f_i)` ; ou directement `x^T P^{-1} x >= lmin(P^{-1}) ||x||^2`. |

## 3. Constats

### E1 (minor) — « same anatomy » : résumé et tableau 1 sans la réserve « sur les quadratiques »
- Lignes : 96-97, 151 (et 674-675).
- Papier : « the nuclear-scaled polar step has the same anatomy » ; « the same anatomy for the polar step [p] ».
- Problème : l'étiquette [p] couvre une identité publiée valable sur les quadratiques et deux encadrements
  élémentaires. Dans le résumé, la phrase suit des mesures faites sur la logistique et laisse entendre que la même
  répartition de l'écart est établie pour le pas polaire ; le corps du texte dit lui-même « not measured for the polar
  step » (l. 315).
- Correction minimale : ajouter « on quadratics » (résumé, tableau). Correction qui renforce : énoncer (2) pour toute F
  C^2 avec la courbure intégrée (voir `proposed_fix` dans le JSON), ce qui rend l'énoncé de structure vrai pour les
  modèles du Constat 1.

### E2 (major) — « the ray that attracts generic runs » : statut numérique, et facteur 4 valable dans la queue seulement
- Lignes : 313-315.
- Problème : (i) l'attraction n'est pas démontrée (notes : numérique, rayons linéairement instables dans le cas
  commutant, question ouverte) et la référence placée en fin de phrase ne l'établit pas ; (ii) sur une course
  entière l'écart est de 7 à 14 jusqu'à 1e-6 et de l'ordre de `kappa_op` au début, à hessienne constante : la suite
  de la phrase (« a large gap needs a global constant far below the curvature met ») ne tient que si « curvature
  met » désigne `mu_PL`, alors que la phrase vient de dire que « the curvature is constant ».
- Preuves : `math-nfl_ls_tail.py`, `math-nfl_ls_head.py`, notes l. 699-701, 800-801, 1210-1214.
- Sévérité : classé major au titre de l'attribution et de la sur-affirmation (un statut numérique, que les notes du
  dépôt donnent comme question ouverte, est présenté comme un fait dans un paragraphe étiqueté [p]). Le chiffre 4
  lui-même est correct pour la queue des courses ; la correction tient en une phrase.

### C1 (minor) — `kappa_op >= min(K,d)` attribué à Sidford–Tian
- Ligne : 354.
- Problème : la source contient l'analogue l_infini sur l'étendue (Lemme A.1), dont la borne spectrale se déduit en
  trois lignes ; elle ne contient ni conditionnement ni norme spectrale. Le rapport étendu disait correctement « the
  matrix form of a folklore fact on l_infty » ; la condensation a perdu la nuance.

### B1 (minor) — « no fixed change of geometry repairs the worst case » dépasse le Théorème 1
- Lignes : 675-676 (et intitulé de ligne du tableau 1, l. 152).
- Problème : le Théorème 1 porte sur les normes. Une géométrie de Bregman fixe (régularité et forte convexité
  relatives à une fonction de référence h) n'est pas couverte, et la conclusion y est fausse : avec `h = F`, F est
  1-régulière et 1-fortement convexe relativement à h (conditionnement relatif 1) ; le SAGA de Bregman de
  [dragomir2021fast], cité l. 153 et 326, travaille dans ce cadre. Le titre et le résumé disent « no norm », ce qui
  est exact.

### A1 (minor) — « variance-reduced methods are governed by the per-sample ratios L_N(f_i)/mu_N(F) »
- Lignes : 200-201.
- Problème : vrai pour les normes euclidiennes (métrique P) ; pour une norme générale c'est une convention de
  comparaison, sans théorème à l'appui (la Proposition 5 du papier certifie d'ailleurs avec
  `mu_op/(L_op + 10 A L_max)`, pas avec ce rapport).

### D1 (nit) — `[Theorem 3, jawanpuria2026constant]` pour `1 - 1/kappa_op`
- Ligne : 232.
- Le Théorème 3 donne `1 - lmin/tr(H)` (au pas `1/tr H`) pour la quadratique unilatérale ; c'est `1 - 1/kappa_op` si
  K >= d ; pour K < d (`L_op` = somme des K plus grandes valeurs propres) il faut leur annexe C (Remarque 2). Lu à
  travers un outil qui résume : confiance moyenne.

## 4. Non vérifié

- Corollaire 5 et annexes de Shulgin et al. (au-delà des 100 000 premiers caractères du HTML) : l'absence d'énoncé
  d'attraction n'est établie que sur la partie lue et sur les notes du dépôt.
- Numéros exacts d'équations et de théorèmes de Shulgin et al. et de Jawanpuria et al. : lus à travers un outil qui
  résume la page, pas sur le texte brut.
- Carlson et al. 2015, Bernstein–Newhouse 2024, Lau et al. 2025 (l. 117, 205-206) : non rouverts ; l'énoncé
  mathématique (plus forte pente spectrale) est vérifié, pas le contenu des sources.
- Chiffres de (M4) hors du rapport global softmax (1.6–103, 7.6–163, 13.3 passes, etc.) : hors périmètre.
- Observation hors périmètre, pour l'auditeur de la bibliographie : (M1) l. 491-492 cite
  `[Lemma 4, jawanpuria2026constant]` pour `max(F(W_t),F(W_{t+1})) - F* >= mu_op eta^2/8` ; d'après la lecture du
  HTML, le Lemme 4 borne `max(||E_t||_F, ||E_{t+1}||_F)` par la demi-longueur du pas (énoncé sur les distances) ; la
  forme en valeurs de F s'en déduit par forte convexité (notes, Théorème « steps » (c)). À confirmer sur le texte.

## 5. Corrections proposées (texte LaTeX, minimal)

- **E1** (minor, l. 96-97, 151 (and 674-675), confiance high) :

  ```latex
  Abstract (l. 96-97): replace ``the nuclear-scaled polar step has the same anatomy'' by ``on quadratics, the exact one-step formula of the nuclear-scaled polar step has the same structure''. Table 1 (l. 151): ``the same structure for the polar step on quadratics \nat{p}''. Optional strengthening, which makes the statement hold for the models of Finding 1 (l. 226-227): ``For $F$ of class $C^2$, $G=\nabla F(W)\ne0$, $O=\polar(G)$, $\eta=\nnuc G/L$, $J=2\int_0^1(1-s)D^2F(W-s\eta O)[O,O]\,ds\in[\muop,\Lop]$ ($=\inner{O}{HO}$ for a quadratic with Hessian $H$) and $\muPL=\nnuc G^2/(2(F(W)-F_\ast))\in[\muop,\Lop]$ (in $[\muop,J]$ for a quadratic), the nuclear-scaled step $W^+=W-\eta O$ satisfies exactly''.
  ```

- **E2** (major, l. 313-315, confiance high) :

  ```latex
  On least squares the Hessian is constant and the gap tends to about a factor $4$ in the tail of generic runs, its value on the invariant ray of the flattest direction \cite{shulgin2026quadratic} (observed, not proved; the head gains more, $\muPL/\muop$ starting near $\kop$); a lasting large gap needs a global constant far below the curvature met, as for logistic and softmax losses (not measured for the polar step).
  ```

- **C1** (minor, l. 354, confiance high) :

  ```latex
  and the spectral geometry has its own cost, $\kop\ge\min(K,d)$ for every $F$ (the matrix form of a folklore fact on $\ell_\infty$ \cite[Lemma~A.1]{sidford2018coordinate}).
  ```

- **B1** (minor, l. 675-676 (and row title l. 152), confiance high) :

  ```latex
  no fixed norm repairs the worst case on ridge logistic and softmax models
  ```

- **A1** (minor, l. 200-201, confiance medium) :

  ```latex
  in a Euclidean norm, variance-reduced methods are governed by the per-sample ratios $L_N(f_i)/\mu_N(F)$, which we compare for every norm
  ```

- **D1** (nit, l. 232, confiance medium) :

  ```latex
  \cite[Theorem~3 and Appendix~C]{jawanpuria2026constant}
  ```

