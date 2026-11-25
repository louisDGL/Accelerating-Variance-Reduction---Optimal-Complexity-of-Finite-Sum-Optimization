# Audit math-smd : théorie des pas polaires stochastiques

Papier : `paper/paper_saga_to_muon.tex` (numéros de ligne de l'outil Read). Date : 2026-11-20.
Sources lues : `archive/muon/notes_spectral_theory.tex` (l. 322-347 et 832-956, preuves complètes),
`archive/muon/report_spectral_muon_new.tex` (l. 255-292, 439-496), `muon/notes_theory.md` (grep).
Scripts (lancés depuis la racine du dépôt, un seul fil, chacun < 3 min CPU) :

| script | objet | durée |
|---|---|---|
| `paper/audit/checks/math-smd_thm2a_inequality.py` | Théorème 2(a) : identité des trois points, lissité, inégalité de sous-gradient, inégalité (a), espérances exactes (bruit à support fini) + Monte-Carlo | 97 s |
| `paper/audit/checks/math-smd_thm2_limit.py scalar` / `2x2` / `pmI` | limites du Théorème 2 (scalaire, 2x2, ±I_2, Cauchy) | 13 s / 46 s / 19 s |
| `paper/audit/checks/math-smd_prop3.py d1` / `rest` | Proposition 3 (a)-(d), d = 1, existence/unicité, SVRG | 15 s / 75 s |
| `paper/audit/checks/math-smd_pl_bounds.py` | les deux inégalités de la ligne 415 | < 5 s |

## 1. Périmètre couvert

(a) l. 244-249 ; (b) l. 389-393 ; (c) Théorème 2, l. 395-416, énoncés et esquisse de preuve étape par étape ;
(d) l. 418-420 ; (e) Proposition 3, l. 422-440, et remarques l. 442-447 ; plus les reprises de ces énoncés dans le
résumé (l. 99-102), l'introduction (l. 128-131), le Tableau 1 (l. 164-166, 171), le Tableau 2 (l. 629-636) et la
conclusion (l. 676-678). Hors périmètre (non traité) : Proposition 4, chiffres des « Numerical checks » et de la
Figure 3, attributions bibliographiques, mentions « was not found ».

## 2. Verdict d'ensemble

Aucun constat blocker ni major. Les énoncés (a), (b), (c) du Théorème 2 et les items (a) à (d) de la Proposition 3
sont exacts tels qu'écrits ; chaque étape de l'esquisse de preuve a été redémontrée à la main et l'inégalité (a) a été
vérifiée avec des espérances exactes. Trois réserves mineures (une phrase de commentaire qui dépasse ce que le théorème
prouve, la partie SVRG de l'item (c) dont la preuve écrite ne traite que la table, un renvoi du Tableau 2) et deux
remarques de forme.

## 3. Affirmations contrôlées

### (a) Lignes 244-249

| affirmation | verdict | preuve du contrôle |
|---|---|---|
| « No moment is required » ; `psi_xi` bien définie | OK | intégrande bornée : `| ||G+xi||_* - ||xi||_* | <= ||G||_*` (inégalité triangulaire inverse) ; `polar` est borélienne et bornée (limite simple de `M(M^T M + eps I)^{-1/2}`), donc `Phi_xi` est défini sans moment. Contrôle : bruit de Cauchy 2x2, la récurrence converge vers `W_*` (gradient moyen de queue 0.0010 contre 0.367 au départ). |
| « psi_xi is convex and 1-Lipschitz for ||.||_* » | OK | espérance de fonctions convexes ; `|psi(G)-psi(G')| <= E| ||G+xi||_* - ||G'+xi||_* | <= ||G-G'||_*`. |
| « Phi_xi(G) in d psi_xi(G) » | OK | point par point `polar(M) in d||.||_*(M)` : pour `M = U diag(s) V^T` (SVD compacte, `s > 0`), `d||M||_* = {UV^T + Z : U^T Z = 0, Z V = 0, ||Z||_op <= 1}` contient `UV^T` ; en `M = 0` le sous-différentiel est la boule spectrale, qui contient `polar(0) = 0`. On prend l'espérance de l'inégalité de sous-gradient (tout est borné). Numérique : `min <Phi(G), G-G'> - (psi(G)-psi(G'))` >= 0 à l'arrondi près (-9e-16) sur 4 lois dont atomes de rang déficient et atome `G + xi = 0`. |
| « with equality to grad psi_xi(G) when G + xi has full rank almost surely » | OK | à rang plein `min(K,d)` le sous-différentiel de `||.||_*` est réduit à `{UV^T}` (si `K <= d`, `U` est carrée donc `U^T Z = 0` force `Z = 0`) ; convergence dominée (quotients bornés par `||Delta||_*`) : les dérivées directionnelles de `psi_xi` sont linéaires. Numérique 2x2, entrées Exp(1)-1 : au minimiseur, `||Phi(G^dag)||_F = 1.6e-3` sur un échantillon indépendant (bruit MC 1e-3). |
| définition `Med(xi) = argmin_C E[||xi - C||_* - ||xi||_*]` | OK | c'est la médiane géométrique d'une loi dans un espace normé (forme renormalisée qui évite tout moment), ici pour la norme nucléaire ; cohérent avec l'usage « nuclear median ». |

### (b) Lignes 389-393

| affirmation | verdict | preuve |
|---|---|---|
| « arg min psi_xi = -Med(xi) » | OK | `psi_xi(G) = E[||xi - (-G)||_* - ||xi||_*]`. |
| « nonempty, convex and compact » | OK | coercivité SANS moment : soit `R` avec `p = P(||xi||_* <= R) > 1/2` ; sur cet événement l'intégrande vaut au moins `||G||_* - 2R`, ailleurs au moins `-||G||_*`, d'où `psi_xi(G) >= (2p-1)||G||_* - 2pR`. Continuité (1-lipschitzienne) + coercivité : argmin non vide, fermé, borné ; convexe car `psi_xi` l'est. |
| « it contains every zero of Phi_xi » | OK | `0 = Phi_xi(G) in d psi_xi(G)`. |
| « consists of zeros of Phi_xi where G + xi has full rank almost surely » | OK | différentiabilité en `G` avec gradient `Phi_xi(G)`, nul en un minimiseur. |
| contre-exemple scalaire `P(xi=1)=0.4`, `P(xi=-1)=0.6` : « psi_xi is minimal at g=1 and |Phi_xi| >= 0.2 » | OK | `psi(g) = 0.4|g+1| + 0.6|g-1| - 1`, pentes -1, -0.2, +1 : minimum unique en `g = 1`. `Phi(g) = 0.4 sign(g+1) + 0.6 sign(g-1)` vaut -1, -0.6 (en -1), -0.2, 0.4 (en 1), 1. Grille de 600 001 points + {-1, 1} : argmin 1.0000, `min |Phi| = 0.200`. Simulation : `G_T` moyen 0.992, moyenne de queue 0.990 (prédit 1 ; le champ moyen n'a pas de zéro mais l'itérée converge bien vers `-Med`). |

### (c) Théorème 2 (l. 395-416), étape par étape

| étape | verdict | détail |
|---|---|---|
| hypothèses de l'énoncé | OK | `F` est `C^2` sur tout l'espace (implicite : les constantes `mu_op, L_op` sont définies l. 198-199 par `D^2F`). `xi_t` i.i.d. « independent of the past » : seul sert `Loi(xi_t | passé) = Loi(xi)`. Longueurs déterministes : nécessaires (cf. l. 446-447). Mesurabilité : `polar` borélienne. Intégrabilité : `||W_t - W_0||_op <= somme des eta_s`, donc `D_F(W_t, W^dag)` est borné à `t` fixé ; aucune hypothèse de variance. Bruit additif indépendant de l'itérée : indispensable (sinon `psi` dépend de `W`), et annoncé (résumé l. 99, limites l. 670). |
| « grad F is a bijection with inverse grad F^* » | OK | `mu_op > 0` : `grad F` fortement monotone et continu sur un espace de dimension finie, donc bijectif ; `F^*` est partout différentiable et `grad F^* = (grad F)^{-1}`. `W^dag = grad F^*(G^dag)` existe donc pour tout `G^dag`. |
| lecture miroir `grad F^*(G^+) = grad F^*(G) - eta O` | OK | réécriture de `W^+ = W - eta O` ; `O` est un sous-gradient stochastique de `psi_xi` en `G`. |
| identité des trois points `D_{F^*}(G^dag,G_{t+1}) = D_{F^*}(G^dag,G_t) - eta_t <O_t, G_t - G^dag> + D_{F^*}(G_t,G_{t+1})` | OK | recalcul : avec `h = F^*`, `D_h(u,x^+) - D_h(u,x) - D_h(x,x^+) = <grad h(x^+) - grad h(x), x - u> = -eta <O, x - u>`. Signes et ordre des arguments corrects. Numérique : résidu relatif max 7e-15 (quadratiques 1x1, 2x2, 2x3, et `F` non quadratique 3x3). |
| `D_{F^*}(G',G) = D_F(W,W')` | OK | `F^*(G) = <W,G> - F(W)` ; substitution. D'où `D_{F^*}(G^dag,G_t) = D_F(W_t,W^dag)` et `D_{F^*}(G_t,G_{t+1}) = D_F(W_{t+1},W_t)`. |
| `D_F(W_{t+1},W_t) <= L_op eta_t^2 / 2` | OK | lissité spectrale et `||polar(.)||_op <= 1` (0 si l'argument est nul). Numérique : `max D_F(W^+,W) - L_op eta^2/2` vaut 4e-14 (1x1) et 2e-13 (2x2), c'est-à-dire 0 à l'arrondi dans les cas quadratiques carrés où il y a égalité (`tr(O S O^T) = tr S` pour `O` orthogonale), et -2e-10 (2x3), -4e-7 (non quadratique 3x3). `L_op` exact dans les tests (somme des `min(K,d)` plus grandes valeurs propres ; `tr S + c d^2` pour le cas non quadratique). |
| `E<O_t, G_t - G^dag> = <Phi_xi(G_t), G_t - G^dag> >= psi_xi(G_t) - psi_xi(G^dag)` | OK | indépendance de `xi_t` et de `W_t`, puis inégalité de sous-gradient. |
| énoncé (a) | OK | les quatre lignes ci-dessus. Vérifié avec espérances EXACTES (lois à 2-7 atomes) sur 2000 à 4000 triplets `(W_t, eta, loi)` par cas : `max (LHS - RHS) <= 0` dans les 4 cas (1.6e-15 à l'arrondi dans le cas scalaire, où l'inégalité est une égalité ; -1e-8, -7e-7, -3e-6 ailleurs). L'inégalité vaut d'ailleurs pour tout point de référence `W'` (la minimalité de `G^dag` ne sert pas) : vérifié aussi. Monte-Carlo, loi continue (entrées Exp(1)-1, `F` non quadratique 2x2, N = 2e5) : score z minimal de RHS - LHS sur 12 configurations = +259. |
| borne moyennée de (a) | OK | espérance totale, somme télescopique, `D_F >= 0`, puis Jensen pour `psi_xi` convexe. `bar G_T` est la moyenne pondérée des GRADIENTS (pas le gradient d'une itérée moyenne), comme écrit. Suppose `W_0` déterministe (sinon mettre `E D_F(W_0,W^dag)`), sans conséquence. |
| (b) : Robbins-Siegmund | OK | `V_t = D_F(W_t,W^dag) >= 0`, `E[V_{t+1}|F_t] <= V_t - eta_t(psi(G_t) - min psi) + L_op eta_t^2/2`, `somme eta_t^2 < infini` : `V_t` converge p.s. et `somme eta_t (psi(G_t) - min psi) < infini` p.s. ; avec `somme eta_t = infini`, `liminf psi(G_t) = min psi`. |
| (b) : coercivité de `psi_xi` sans moment | OK | voir (b) ci-dessus ; elle sert à l'existence de `G^dag` (argmin compact non vide). Le caractère borné de `W_t` vient de `D_F(W_t,W^dag) >= (mu_op/2)||W_t - W^dag||_op^2`. |
| (b) : ensemble dénombrable dense de minimiseurs | OK | le point d'accumulation `G'` est aléatoire, alors que Robbins-Siegmund donne un ensemble négligeable par `G^dag` : on prend `{G_k}` dénombrable dense dans l'argmin (compact). Pour tout minimiseur `G'`, `D_F(W_t,W') - D_F(W_t,W_k) = const(k) + <G_k - G', W_t>`, donc `limsup - liminf` de `D_F(W_t,W')` est au plus `2||G_k - G'||_* sup_t ||W_t||_op`, arbitrairement petit : `D_F(W_t,W')` converge, vers 0 puisqu'il tend vers 0 le long de la sous-suite ; forte convexité : `W_t -> W'`, `G_t -> G'`. L'argument est correct et complet (notes l. 895-899). |
| (c) : `2 mu_op (F - F_*) <= ||grad F||_*^2 <= 2 L_op (F - F_*)` | OK | forte convexité : `F(W') >= F(W) - ||G||_* r + mu_op r^2/2 >= F(W) - ||G||_*^2/(2 mu_op)` ; lissité le long de `-polar(G)` avec `t = ||G||_*/L_op`. Numérique : sur 3000 quadratiques le rapport `||G||_*^2 / (2 (F-F_*))` reste dans `[mu_op, L_op]`, bornes atteintes (1.000000 des deux côtés) ; non quadratique : dans [1.18 mu_op, 0.95 L_op]. |
| (c) : conclusion | OK | `G_t -> G' in -Med(xi)` ; si `Med = {0}`, `W_t -> grad F^*(0) = W_*` ; si `0` n'est pas dans `Med` (compact), `lim F(W_t) - F_* = F(W') - F_* >= ||G'||_*^2/(2 L_op) >= min_C ||C||_*^2/(2 L_op) > 0`. La limite existe puisque `W_t` converge. |

Simulations de la limite (pas `eta_t = eta_0 (1 + t/t_0)^{-3/4}`) :

- Scalaire, `xi = Exp(1) - 1` (centré, asymétrique). Prédiction : gradient limite `1 - log 2 = 0.3069`, l'unique zéro
  de `E sign(g + xi)`. Moyennes de queue sur 4000 chaînes, T = 30000 : 0.3056 (`F = w^2/2`), 0.3044
  (`F = w^2/2 + 0.8 log cosh(w - 0.5)`), 0.3011 (`F = 2w^2 + sqrt(1 + (w+1)^2)`) ; écart <= 0.006, de l'ordre de
  `L eta_T`. Sous « `E xi = 0` » on attendrait 0. Le gradient limite ne dépend pas de `F`, comme le dit le théorème.
- Loi à deux points centrée `P(xi=2)=1/3`, `P(xi=-1)=2/3` (`E xi = 0`, `Med = {-1}`, `E|xi| = 4/3`) : moyenne de queue
  0.995 (prédit 1).
- 2x2, entrées i.i.d. Exp(1)-1 (`E xi = 0`, rang plein p.s.). Minimiseur de `psi_xi` sur 1e6 tirages :
  `G^dag = [[0.2114, 0.2113], [0.2122, 0.2139]]` (multiple de `1 1^T` aux fluctuations près, comme l'impose la
  symétrie), `||G^dag||_* = 0.425` ; champ moyen en `G^dag` sur échantillon indépendant : 1.6e-3 (bruit 1e-3), alors
  que `||Phi(0)||_F = 0.253`. Runs (2000 chaînes, T = 20000) : gradient moyen de queue
  `[[0.2098, 0.2113], [0.2102, 0.2111]]` (quadratique) et `[[0.2097, 0.2109], [0.2094, 0.2108]]` (non quadratique,
  `sech^2 = 1.0` au point limite, donc courbure non quadratique active) ; distance à `G^dag` 0.0038 et 0.0045 contre
  `||G^dag||_F = 0.424`. La distance moyenne des chaînes à `G^dag` décroît avec `eta_t` (0.20 en T/8, 0.09 en T).
  Bornes : `F(W^dag) - F_* = 0.0919` (quadratique), 0.0543 (non quadratique), entre `||G^dag||_*^2/(2 L_op)` = 0.036
  et 0.017 respectivement (Théorème 2(c)) et `||G^dag||_*^2/(2 mu_op) = 0.312 <= (E||xi||_*)^2/(2 mu_op) = 7.79`
  (Proposition 3(b)).

### (d) Lignes 418-420

| affirmation | verdict | preuve |
|---|---|---|
| « The theorem replaces E xi = 0 by Med(xi) = {0} » | OK | condition suffisante de convergence vers `W_*` (Théorème 2(c)) ; aucune des deux conditions n'implique l'autre (Cauchy : `Med = {0}` sans espérance ; Exp(1)-1 : centré mais `Med = {log 2 - 1}`), ce que les simulations confirment. |
| « otherwise the limit gradient is an offset that no schedule of this class removes » | problème mineur (constat S1) | prouvé seulement si `0` n'appartient pas à `Med(xi)`. |

### (e) Proposition 3 (l. 422-440) et remarques (l. 442-447)

| affirmation | verdict | preuve |
|---|---|---|
| (a) loi symétrique : `0 in Med(xi)` ; `Med = {0}` si `0` est dans le support | OK | `psi_xi` paire et convexe, donc minimale en 0 ; `psi(G) - psi(0) = (1/2) E[||xi+G|| + ||xi-G|| - 2||xi||]`, intégrande `>= max(0, 2||G|| - 2||xi||)` (deux inégalités triangulaires), strictement positive sur `{||xi||_* < ||G||_*}`, de probabilité > 0 si 0 est dans le support. Numérique (gaussien 2x2, 2e5 tirages, 20 directions) : minoration ponctuelle vérifiée (min 4e-16). La condition de support est nécessaire : pour `xi = ±I_2`, `psi(S) = psi(0)` pour toute `S` symétrique de norme spectrale < 1. |
| (b) `E xi = 0` : `||C||_* <= E||xi||_*`, puis `F(W^dag) - F_* <= (E||xi||_*)^2/(2 mu_op)` | OK | `C = E[C - xi]`, Jensen, puis minimalité de `C` contre 0 ; l'intégrabilité est implicite dans `E xi = 0`. 40 lois centrées à support fini (1x1 à 3x3) : `E||xi-C|| - ||C|| >= -4e-15`, `E||xi|| - E||xi-C|| >= 1e-3`. La borne sur `F` utilise la première inégalité de la l. 415. |
| (c) SAGA : point fixe sans décalage, unique | OK | en `(W_*, table exacte)`, `omega_j = grad F(W_*) = 0` pour tout `j` et `polar(0) = 0` ; réciproque : table inchangée par tout indice, donc `z_i = f_i'(W)`, `omega_j = grad F(W)`, nul seulement en `W_*` (longueur > 0). Numérique : `max_j |omega_j| = 8e-17`. |
| (c) « or SVRG » | problème mineur (constat S2) | vrai si le rafraîchissement du point de référence fait partie de la dynamique ; la preuve écrite ne traite que la table et la parenthèse « that no index moves » ne convient pas à SVRG. |
| (d) `polar(r x^T) = r x^T/(|r||x|) = grad_W ||Wx - y|| / ||x||` | OK | SVD de rang un ; écart 2e-16 ; différences finies : 8e-11. Pour `r = 0`, `polar(0) = 0` est un sous-gradient (minimum du terme). |
| (d) « is the stochastic subgradient method on Lambda » ; convergence p.s. vers un minimiseur | OK | `E_j polar(g_j(W))` est dans `d Lambda(W)`, sous-gradients de norme de Frobenius <= 1 ; analyse classique (Robbins-Siegmund + ensemble dénombrable dense). EXISTENCE d'un minimiseur : toujours (`Lambda` ne dépend que de `W` restreint à `vect{x_i}`, où `(1/n) somme ||W x_i||/||x_i||` est une norme, d'où coercivité sur le quotient). UNICITÉ : non en général (`K = d = 1`, `x = (1,1)`, `y = (0,1)` : tout `w` de [0,1] minimise ; plan de rang déficient : ensemble de minimiseurs non borné) ; la proposition dit bien « a minimizer ». |
| (d) « in general not the least-squares solution » | OK | n = 60, K = 2, d = 3, ordonnée à l'origine, résidus asymétriques : `||W_LAD - W_LS||_F = 0.243`, `Lambda(W_LAD) = 0.3897 < Lambda(W_LS) = 0.4134`, `||grad Lambda(W_LAD)|| = 6e-11` (zéro du champ moyen). Runs (1500 chaînes, T = 40000, `eta_t = 0.3(1+t/200)^{-3/4}`) : moyenne de queue à 0.0136 de `W_LAD` et 0.246 de `W_LS` ; `Lambda(moyenne) - min Lambda = 4.0e-5` contre `Lambda(W_LS) - min Lambda = 2.4e-2`. |
| l. 443 « For d = 1 the limit in (d) is the geometric median of the y_i/x_i » | OK (remarque S4) | `||w x_i - y_i||/|x_i| = ||w - y_i/x_i||` : identité vérifiée (écart 0) ; argmin de `Lambda` = médiane géométrique par Weiszfeld (écart 2e-12), différent des moindres carrés ([1.153, -0.198] contre [0.909, -0.272]). Le pas est alors exactement `w + eta (Y - w)/||Y - w||` avec `Y = y_j/x_j`. |
| l. 446-447 « With lengths proportional to the sampled gradient the step is plain stochastic gradient descent, which converges to the least-squares solution » | OK | `||r x^T||_* = |r||x|`, donc `gamma ||g_j||_* polar(g_j) = gamma g_j`. Run témoin (mêmes données, `gamma_t = 0.3 eta_t`) : moyenne de queue à 0.0004 de `W_LS` et 0.243 de `W_LAD`. La convergence suppose `gamma_t` décroissant (implicite) ; « deterministic lengths are needed » se lit « longueur indépendante de l'échantillon courant ». |
| l. 442-443 « (b) is the mean-median inequality in a normed space; (c) identifies a fixed point, not its attraction » | OK | (b) : `|moyenne - médiane| <= E|X - médiane| <= E|X - moyenne|`, même preuve ; rapport `|C|/E|xi|` jusqu'à 0.98 pour une loi à deux points, constante 1 non améliorable. (c) : réserve exacte (pas d'énoncé de convergence). |

Reprises ailleurs dans le papier : résumé l. 99-102, introduction l. 128-131 et Tableau 1 l. 164-166 sont conformes au
Théorème 2 et à la Proposition 3 (l'hypothèse de bruit additif est rappelée dans le résumé et dans les limites l. 670).

## 4. Constats

### S1 (minor) — l. 418-419, « otherwise » dépasse le Théorème 2(c)

- Phrase : « The theorem replaces $\E\xi=0$ by $\Med(\xi)=\{0\}$; otherwise the limit gradient is an \emph{offset} that
  no schedule of this class removes. »
- Problème : « otherwise » couvre `Med(xi) != {0}`, alors que le Théorème 2(c) ne prouve un gradient limite non nul que
  si `0` n'appartient pas à `Med(xi)`. Dans le cas intermédiaire `0 in Med(xi) != {0}` (loi symétrique dont le support
  évite 0, que la Proposition 3(a) laisse justement ouverte), le théorème donne seulement la convergence vers un point
  aléatoire de `-Med(xi)`, qui peut être 0.
- Preuve : `xi = ±I_2` avec probabilité 1/2, `F = ||W||_F^2/2`, `W_0 = 0`. `psi_xi(S) = 0 = min` pour toute `S`
  symétrique avec `||S||_op <= 1` (`psi(0.5 I) = psi(-0.9 I) = 0`, `psi(1.2 I) = 0.4`). La récurrence reste sur `w I`
  avec `w_{t+1} = w_t ∓ eta_t` : limite aléatoire (2000 chaînes : moyenne +0.017, écart-type 0.54), ni nulle ni
  déterminée par la loi du bruit. Ici il y a bien un décalage presque sûr, mais le théorème ne le dit pas, et le « no
  schedule removes » n'est établi que pour `0` hors de `Med(xi)`.
- Correction proposée : `The theorem replaces $\E\xi=0$ by $\Med(\xi)=\{0\}$; if $0\notin\Med(\xi)$ the limit gradient is an \emph{offset} that no schedule of this class removes.`
- Confiance : élevée.

### S2 (minor) — l. 425-427 et 436-438, item (c) pour SVRG

- Phrase : « For a SAGA or SVRG estimator, [...] such estimators have a fixed point without offset (iterate and memory
  that no index moves), and it is the only one. » ; preuve : « conversely a table that no index changes has
  $z_i=f_i'(W)$, hence $\omega_j=\nabla F(W)$, which vanishes only at $W_\ast$. »
- Problème : la réciproque n'est écrite que pour une table. Pour SVRG la mémoire (point de référence) n'est déplacée
  par aucun indice mais par le rafraîchissement ; avec la définition entre parenthèses prise à la lettre, l'unicité est
  fausse pour SVRG, et elle est vraie dès que le rafraîchissement est compté dans la dynamique.
- Preuve : composantes de même hessienne, `f_j(w) = (1/2)(w - a_j)^T H (w - a_j)` : pour TOUT point de référence
  `w~`, `omega_j = f_j'(w) - f_j'(w~) + grad F(w~) = grad F(w)` pour tout `j`. Donc `(W_*, w~)` n'est déplacé par aucun
  indice, quel que soit `w~` (numérique : `max_j |omega_j| = 1.8e-15` avec `||w~ - W_*|| = 3.31`). Sur des données
  génériques (moindres carrés ridge) ce n'est pas le cas (`max_j |omega_j| = 4.59` en `W_*` avec une autre référence).
  La conclusion utile reste vraie dans tous les cas : si aucun indice ne déplace l'itérée, `0 = (1/n) somme_j omega_j =
  grad F(W)` (estimateur sans biais), donc `W = W_*` ; et une mémoire que sa mise à jour laisse inchangée est celle
  prise en `W_*` (table : `z_j = f_j'(W_*)` ; référence rafraîchie : `w~ = W_*`).
- Correction proposée (preuve, l. 436-438) : `(c) With $z_i=f_i'(W_\ast)$, $\omega_j=\nabla F(W_\ast)=0$ and $\polar(0)=0$; conversely, if no index moves the iterate, $0=\frac1n\sum_j\omega_j=\nabla F(W)$, so $W=W_\ast$, and a memory that its own update leaves unchanged (table entry, or reference point when refreshed) is the one at $W_\ast$.`
  et, l. 426-427, `(iterate and memory that no update moves)`.
- Confiance : élevée sur le fait, moyenne sur la gravité (lecture naturelle correcte).

### S3 (minor) — l. 632 (Tableau 2), renvoi vers la Proposition 4

- Phrase : « The limit gradient is minus a nuclear median of the noise (Proposition~\ref{prop:offset}). »
- Problème : cet énoncé est le Théorème 2(b) (étiquette [p]). La Proposition 4 (`prop:offset`, étiquette [f], formelle,
  entrées i.i.d.) ne fait qu'en donner la taille ; telle quelle, la cellule appuie un résultat prouvé sur une
  proposition formelle. Le bon théorème est cité dans la cellule voisine de la même ligne, d'où la gravité mineure.
- Preuve : l. 403 (Théorème 2(b)) et l. 449-458 (Proposition 4 : « size of the offset for minibatches; formal »).
- Correction proposée : `The limit gradient is minus a nuclear median of the noise (Theorem~\ref{thm:smd}), formally of order $1/b$ (Proposition~\ref{prop:offset}).`
- Confiance : moyenne (l'intention était peut-être de renvoyer à la taille du décalage).

### S4 (nit) — l. 443, unicité de la médiane géométrique / du minimiseur de `Lambda`

- Phrase : « For $d=1$ the limit in (d) is the geometric median of the $y_i/x_i$ ».
- Problème : le minimiseur de `Lambda` existe toujours mais n'est pas unique en général (points `y_i/x_i` alignés avec
  `n` pair ; plan de rang déficient). L'énoncé (d) dit correctement « a minimizer » ; la remarque et la légende de la
  Figure 3 (« the least-distances solution ») supposent l'unicité, vraie pour des données génériques.
- Preuve : `K = d = 1`, `x = (1,1)`, `y = (0,1)` : `Lambda(w) = 0.5` pour tout `w` de [0,1].
- Correction proposée : `a geometric median of the $y_i/x_i$`
- Confiance : élevée.

### S5 (nit) — l. 676-677 (conclusion), raccourci

- Phrase : « with decreasing lengths, polar steps converge to minus a nuclear median of the noise ».
- Problème : c'est le gradient qui converge, et sous bruit additif (rappelé l. 670, juste au-dessus) ; pour le bruit
  d'une somme finie, qui dépend de l'itérée, seul le cas de la Proposition 3(d) est prouvé. Raccourci acceptable, à
  aligner sur le résumé si la place le permet.
- Correction proposée : `with additive noise and decreasing lengths, the gradient of polar steps converges to minus a nuclear median of the noise`
- Confiance : moyenne.

## 5. Non vérifié

- Attributions et contenus des sources citées dans le périmètre (`duchi2012randomized`, `jiang2026adaptive`,
  `minsker2015geometric`, `maddison2021dual`, `balles2020geometry`, `chen2020distributed`, `cardot2013efficient`,
  `bernstein2019majority`, `karimireddy2019error`, `li2026implicit` Lemma H.1 et Theorem 4.10) : aucune source ouverte,
  dimension bibliographique.
- Mentions « was not found » (l. 420, 446) et « Searched only in an earlier form » (l. 272-273).
- Proposition 4 et les chiffres du paragraphe « Numerical checks » et de la Figure 3 (0.170, 2.6e-3, 2.1e-4, etc.).
- Convergence presque sûre au sens strict : les simulations montrent la convergence des moyennes de queue et la
  décroissance de la dispersion avec `eta_t`, pas une convergence trajectoire par trajectoire à horizon infini (la
  preuve, elle, a été vérifiée étape par étape).
- La convergence de SGD vers les moindres carrés (l. 447) n'a été contrôlée que par un run témoin.
