# Audit math-saga : enonces mathematiques de la partie SAGA

Papier : `paper/paper_saga_to_muon.tex` (numeros de ligne de l'outil Read).
Scripts de controle (tous lances depuis la racine du depot, un seul processus, BLAS mono-thread) :

- `paper/audit/checks/math-saga_check1_identity.py` (1 s) : identite (1) par enumeration exacte, I1-I6, (C3).
- `paper/audit/checks/math-saga_check2_identical.py` (2 min 20 de CPU) : SAGA sur n composantes identiques, recurrence
  exacte du second moment, borne de (C4).
- `paper/audit/checks/math-saga_check3_constants.py` (5 s) : meilleur (gamma, Delta) par force brute, (C5).
- Sorties : `paper/audit/checks/math-saga_check*.out`.

## Perimetre couvert

(a) l. 189-194 Proposition 5.9 telle que citee ; (b) l. 210-222 inegalites I1-I6, definitions, identite (1), valeurs pire
cas ; (c) (C2) l. 295-297 ; (d) (C3) l. 300-302 ; (e) (C4) l. 318-324 ; (f) (C5) l. 325-327 et l. 236-238.
Les chiffres mesures (3.2-220, 77-96 %, 2.45-2.62, 81, 4.5, etc.) ne sont pas dans ce perimetre ; seuls les chiffres qui
sont des consequences de formules (17 %, 8/3, 3, 4/3, "eight times") ont ete recalcules.

## Synthese

| Id | Severite | Lignes | Objet |
|----|----------|--------|-------|
| MS-1 | major | 321-323 | Borne inferieure de (C4) : le point initial n'est pas fixe ; l'enonce est faux pour d'autres theta_0 que (r, r) |
| MS-2 | minor | 295-296, 310-313 | "I1 and I2 together contribute at most a factor 8/3" n'est vrai que pour c_D ; dans c_M le facteur va jusqu'a 3, et I5 n'est pas borne |
| MS-3 | minor | 236-238, 325 | R_P^2 du papier est un majorant de la constante de regularite dans la metrique P, pas la meilleure constante (sans effet sur kappa_P >= kappa) |
| MS-4 | nit | 192-194, 210 | Prop. 5.9 paraphrasee par sa preuve (contraction de V) ; "six inequalities" omet deux majorations sans effet sur le taux |
| MS-5 | nit | 319-320 | "the largest step the proof admits" est une borne superieure non atteinte (gamma R^2 < 1/2) |

Tout le reste du perimetre est verifie correct (detail ci-dessous).

## (a) Proposition 5.9 telle que citee (l. 189-194) : OK

Enonce du papier : f_i convexes et R^2-lisses, F mu-fortement convexe, gamma = 1/(4R^2), z_i^(0) = f_i'(theta_0),
V = D + delta M, delta = 3 gamma^2, contraction 1 - min{1/(3n), 3 mu/(16 R^2)} par iteration.

- Concordance avec la transcription `theory/B3_prop59_tightness.tex` l. 8-12 (hypotheses), l. 29-40 (iteration, Lyapunov,
  Delta = 3 gamma^2 = 3/(16 R^4)), l. 42-47 (enonce : E||theta_t - eta*||^2 <= (1 - min{1/(3n), 3mu/(16R^2)})^t (1 + n/4)
  ||theta_0 - eta*||^2). Identique au papier.
- Preuve refaite a la main, independamment de la note : E_t D' = D - 2 gamma G + gamma^2 Omega ; Omega <= 2S/n + 2B (I1)
  <= 2 R^2 G + 2M/n (I2, I3) ; E_t M' = (1 - 1/n) M + S/n <= (1 - 1/n) M + R^2 G (I2). Donc
  E_t V' <= D - 2 gamma (1 - gamma R^2 - R^2 delta/(2 gamma)) G + (1 - 1/n + 2 gamma^2/(n delta)) delta M.
  Avec gamma = 1/(4R^2), delta = 3 gamma^2 : 1 - 1/4 - 3/8 = 3/8, 2 gamma * 3/8 = 3/(16 R^2), 2 gamma^2/(n delta) = 2/(3n) ;
  puis G >= mu D (I4) et max des deux coefficients (I5). Le resultat est celui du papier. Hypotheses utilisees :
  convexite + R^2-regularite de chaque f_i (cocoercivite), forte convexite de F, sum_i f_i'(eta*) = 0 : ce sont celles
  de la ligne 189.
- Texte du livre : absent du depot ; j'ai ouvert la version en ligne de l'auteur
  (https://www.di.ens.fr/~fbach/ltfp_book.pdf, datee du 26 mai 2025, p. 146-149). Section 5.4.4 : "each f_i is
  R^2-smooth ... F is mu-strongly convex ... kappa = R^2/mu". "Proposition 5.9 (Convergence of SAGA) If initializing with
  z_i^(0) = f_i'(theta_0) at the initial point theta_0, for all i, we have, for the choice of step size gamma = 1/(4R^2),
  E||theta_t - eta*||^2 <= (1 - min{1/(3n), 3mu/(16R^2)})^t (1 + n/4) ||theta_0 - eta*||^2 (5.30)". Preuve : "With
  Delta = 3 gamma^2 and gamma = 1/(4R^2), we get 1 - gamma R^2 - R^2 Delta/(2 gamma) = 3/8 and 2 gamma^2/(n Delta) = 2/(3n)",
  puis la contraction de ||theta_t - eta*||^2 + Delta sum_i ||f_i'(eta*) - z_i^(t)||^2 par 1 - min{1/(3n), 3mu/(16R^2)}.
  Le papier et la transcription B3 sont fideles (numero, pas, poids, initialisation, taux). La convexite de chaque f_i,
  que le papier ecrit, est utilisee par le livre via la cocoercivite (sa proposition 5.4).
- Controle numerique : sur 12 problemes (n = 20, d = 5, kappa de 5 a 5000) et 144 etats, E_t[V_{t+1}] calcule par
  enumeration des n indices est <= (1 - min{1/(3n), 3mu/(16R^2)}) V_t dans tous les cas (check1, colonne "OK").
- Remarque (nit MS-4) : la contraction en un pas de V vaut pour toute table ; l'initialisation z_i^(0) = f_i'(theta_0)
  ne sert qu'a la constante (I6, facteur 1 + n/4). La phrase du papier attribue a "Proposition 5.9" ce qui est le
  contenu de sa preuve (l'enonce du livre porte sur E||theta_t - eta*||^2 avec le facteur 1 + n/4). Pas une erreur.
- Reserve : c'est la version en ligne (2025) qui a ete ouverte, pas l'edition imprimee MIT Press 2024 citee.

## (b) I1-I6, definitions, identite (1), valeurs pire cas (l. 210-222) : OK

- Liste I1-I6 : conforme a la preuve refaite ci-dessus et a B3 l. 120-175. (B3 l. 118 signale deux autres majorations,
  E||theta_t - eta*||^2 <= E V_t et 1 + 3n/16 <= 1 + n/4, sans effet sur le taux : nit MS-4.)
- mu_eff = G/D et L_eff = S/(nG) rendent I4 (G >= mu D) et I2 (S/n <= R^2 G) exactes par construction. OK.
- r_sq = Omega/(2S/n + 2B), r_var = B/(M/n) sont dans [0, 1] (I1 et I3). OK.
- Identite (1), derivation refaite : S/n = L_eff mu_eff D, B = r_var M/n, Omega = 2 r_sq (L_eff mu_eff D + r_var M/n), d'ou
  E_t V' = D [1 - 2 gamma mu_eff + 2 gamma^2 r_sq L_eff mu_eff + delta L_eff mu_eff] + delta M [1 - 1/n + 2 gamma^2 r_sq r_var/(n delta)] ;
  avec gamma = 1/(4R^2), delta = 3 gamma^2 : c_D = 1 - (mu_eff/(2R^2)) (1 - (L_eff/R^2)(r_sq/4 + 3/8)),
  c_M = 1 - (1/n)(1 - (2/3) r_sq r_var). Identique a l'equation (1) du papier.
- Controle numerique attendu (1) : `math-saga_check1_identity.py`, regression logistique ridge n = 20, d = 5,
  kappa dans {5, 50, 500, 5000}, 3 graines, 11-12 etats par run (t = 0 a 39 passes ; 141 etats non encore converges
  a la precision machine). Membre de gauche : on execute
  reellement le pas SAGA pour chacun des n indices et on moyenne V ; membre de droite : c_D D + c_M delta M depuis les
  definitions. Erreur relative maximale 1.5e-13 (arrondi) ; I1-I6 et mu <= mu_eff <= L_eff <= R^2 verifiees partout.
- Valeurs pire cas : r_sq = r_var = 1, L_eff = R^2, mu_eff = mu donnent c_D = 1 - (mu/(2R^2))(1 - 5/8) = 1 - 3mu/(16R^2)
  et c_M = 1 - 1/(3n) (assertion dans check1). Ce sont bien des pires cas : c_D croit en L_eff et r_sq, decroit en mu_eff
  (le crochet est >= 3/8 > 0), c_M croit en r_sq r_var. Le taux certifie est le max des deux (I5). OK.
- Condition implicite de (1) : theta != eta* et M > 0 (sinon r_var, mu_eff non definis). Sans consequence.

## (c) (C2), l. 295-297

- "cocoercivity gives L_eff <= R^2" : OK (c'est I2).
- "mu_eff/mu is bounded only by a condition number" : OK. mu <= mu_eff <= L_eff <= R^2 (Cauchy-Schwarz et Jensen :
  S/n >= ||F'||^2 >= G^2/D), donc 1 <= mu_eff/mu <= kappa ; verifie numeriquement sur tous les etats de check1.
- "In (1) I1 and I2 together contribute at most a factor 8/3" : vrai pour le coefficient c_D seulement (constat MS-2).
  Dans c_D le crochet 1 - (L_eff/R^2)(r_sq/4 + 3/8) est dans [3/8, 1), donc le taux 1 - c_D vaut (mu_eff/mu) fois un
  facteur dans [1, 8/3) fois le taux certifie 3mu/(16R^2) : correct (max observe 2.58 dans check1).
  Dans c_M, le taux 1 - c_M = (1/n)(1 - (2/3) r_sq r_var) va de 1/(3n) a 1/n : I1 (avec I3) y contribue jusqu'a un
  facteur 3 > 8/3. Voir MS-2.

## (d) (C3), l. 300-302 : OK

- "The global strong-convexity constant of ridge logistic regression is exactly lambda" : vrai quelles que soient les
  donnees. F''(theta) = lambda I + (1/n) sum_i l''(y_i x_i.theta) x_i x_i^T >= lambda I, et pour theta_1 tel que
  x_i.theta_1 != 0 pour tout x_i non nul (presque tout theta_1), l''(t y_i x_i.theta_1) -> 0, donc F''(t theta_1) -> lambda I :
  inf_theta lambda_min(F''(theta)) = lambda (infimum, atteint seulement si les x_i n'engendrent pas R^d).
  Numerique (check1) : lambda_min(F''(t theta_1))/lambda - 1 = 15, 2.1, 4.5e-6, 0 pour t = 1, 10, 100, 1000.
  Concorde avec B2 Prop. "the global strong convexity constant is lambda" (l. 377-392).
- "near the optimum mu_eff is at least about the smallest eigenvalue of the Hessian at eta*" : vrai.
  mu_eff(theta) = u^T [int_0^1 F''(eta* + s(theta - eta*)) ds] u avec u unitaire, qui tend vers u^T F''(eta*) u >= lambda_min(F''(eta*))
  quand theta -> eta*. Borne non asymptotique de la note (B3 Prop. "local strong convexity", l. 481-499, preuve relue) :
  mu_eff >= lambda + psi(R_x ||theta - eta*||)(lambda_min(F''(eta*)) - lambda), psi(x) = (1 - e^{-x})/x.
  Numerique (check1) : dans la direction la plus plate, mu_eff/lambda_min(H*) = 0.9306, 0.9925, 0.99925, 0.99993 a distance
  1, 1e-1, 1e-2, 1e-3 ; sur 2000 directions aleatoires le minimum reste > 1.12. "at least about" est le bon niveau
  d'affirmation ; l'egalite est atteinte dans la direction propre la plus plate, comme le dit la suite de la phrase.

## (e) (C4), l. 318-324

### Premiere phrase : OK
- "Within the proof, the best step and Lyapunov weight gain at most a factor 3 (4/3 when kappa >= 9n/16 ...)".
  Recalcul : pour gamma, Delta > 0 la chaine I1-I5 certifie tau = min{2 gamma mu (1 - gamma R^2 - R^2 Delta/(2 gamma)),
  (1/n)(1 - 2 gamma^2/Delta)}. Avec Delta = c gamma^2 : max en gamma = 1/(R^2(2 + c)), tau = min{1/(kappa(2 + c)), (c - 2)/(cn)},
  croisement kappa c^2 - n c - 4 kappa = 0, tau* = 2/(4 kappa + n + sqrt(n^2 + 16 kappa^2)). Rapport au livre, s = n/kappa :
  Q = 32/(3(4 + s + sqrt(s^2 + 16))) <= 4/3 pour s <= 16/9 (kappa >= 9n/16), Q = 6s/(4 + s + sqrt(s^2 + 16)) < 3 sinon.
  Force brute (check3, grille 6000 x 3000 sur (gamma R^2, Delta/gamma^2)) : accord a 2e-3 pres avec tau* sur 10 couples
  (n, kappa) ; sup Q = 3.000 (limite n/kappa -> infini, non atteinte), sup sur kappa >= 9n/16 = 1.3333, min Q = 1.050.
- "at kappa = 10^3 they gain 17 %" (n = 1000) : Q(1) = 32/(3(5 + sqrt 17)) = 1.1692. OK.
- "eight times the largest step the proof admits" : alpha > 0 et Delta > 2 gamma^2 imposent gamma R^2 < 1 - R^2 Delta/(2 gamma)
  < 1 - gamma R^2, soit gamma R^2 < 1/2 ; 4/(1/2) = 8. OK (borne superieure non atteinte : nit MS-5).
- "the step of [hofmann2015variance]" : gamma* = a*(K)/(4L), a*(K) = 2K/(1 + K + sqrt(1 + K^2)), K = 4 kappa/n coincide
  algebriquement avec gamma* ci-dessus (check3, colonne "pas Hofmann"), et rho*(K) = (q/n) 2/(1 + K + sqrt(1 + K^2)) avec
  tau*. Source ouverte (arXiv 1506.03662, PDF) : "Theorem 1. Consider a uniform q-memorization algorithm. For any step
  size gamma = a/(4L) with a < 1 ... rho(gamma) = (q/n)(1-a)/(1-a/2) if gamma >= gamma*(K), otherwise rho(gamma) = mu gamma,
  where gamma*(K) := a*(K)/(4L), a*(K) := 2K/(1 + K + sqrt(1 + K^2)), K := 4qL/(n mu)" et "Corollary 1. In Theorem 1, rho is
  maximized for gamma = gamma*(K)". L'attribution "the step of [hofmann2015variance]" est exacte (SAGA : q = 1).

### Borne inferieure sur n composantes identiques : constat MS-1 (major)
Enonce du papier (l. 321-324) : "On n identical components (mu theta_1^2 + R^2 theta_2^2)/2 SAGA with any constant step
needs at least (kappa-1)/4 log(1/(2 eps)) iterations to bring E||theta_t||^2 below eps ||theta_0||^2, its mean iterate
being that of gradient descent".

Hypotheses exactes sous lesquelles la borne est vraie (preuve refaite, concorde avec B3 Prop. "lower bound", l. 406-416) :
1. E[theta_t] = (I - gamma Q)^t theta_0, Q = diag(mu, R^2), pour TOUTE table initiale deterministe (ou independante des
   indices) : seule l'absence de biais de la direction SAGA et la linearite de F' servent. Verifie par recurrence exacte
   (check2, partie A : ecart max 2.4e-15 pour les tables f_i'(theta_0), 0, et arbitraire).
2. Quantite bornee : E||theta_t - eta*||^2 >= ||E theta_t||^2 = (1 - gamma mu)^{2t} theta_{0,1}^2 + (1 - gamma R^2)^{2t} theta_{0,2}^2
   >= ((kappa-1)/(kappa+1))^{2t} min{theta_{0,1}^2, theta_{0,2}^2}, pour tout gamma > 0 (eta* = 0).
3. Point initial : la conclusion "au moins (kappa-1)/4 log(1/(2 eps)) iterations" demande min{theta_{0,1}^2, theta_{0,2}^2}
   = ||theta_0||^2/2, c'est-a-dire theta_0 = (r, +-r), r != 0 (et eps < 1/2 pour que la borne soit positive ; kappa > 1).
   Pour theta_0 general la borne correcte est (kappa-1)/4 log(m/eps) avec m = min_k theta_{0,k}^2/||theta_0||^2 <= 1/2.

Le papier contient : composantes, pas constant quelconque, quantite bornee, moyenne = descente de gradient. Il ne contient
PAS l'hypothese 3 sur theta_0 (ni la table, mais celle-ci est libre). Tel qu'ecrit, pour un theta_0 non precise, l'enonce
est faux :
- contre-exemple analytique : theta_0 = (0, 1), gamma = 1/R^2, table z_i = f_i'(theta_0) : le premier pas est un pas de
  gradient exact, theta_1 = 0, donc E||theta_1||^2 = 0 apres 1 iteration, pour tout kappa et tout eps ;
- contre-exemples non degeneres (check2, partie C, recurrence exacte, n = 5, eps = 1e-2) :
  theta_0 = (0.1, 1), gamma = 1/R^2, kappa = 100 : cible atteinte a t = 1, la formule du papier donne 96.8 ;
  theta_0 = (0.1, 1), pas du livre gamma = 1/(4R^2), kappa = 100 : t = 29 < 96.8 ;
  theta_0 = (0, 1), gamma = 1/(4R^2), kappa = 100 : t = 16 < 96.8.
- avec theta_0 = (1, 1) l'enonce est vrai : 864 cas (kappa dans {3, 10, 40}, n dans {1, 2, 5, 10}, 8 pas de 0.02/R^2 a
  2.5/R^2, 3 tables, eps dans {1e-1, 1e-2, 1e-4}), 0 violation, plus petit rapport t_atteint/borne = 1.06.

Lecture charitable : une borne inferieure "pire cas sur theta_0". Mais la phrase ne le dit pas, et la note B3 (l. 412) et le
rapport principal fixent theta_0 = (r, r). Correction : trois mots.

- "(the order of one-step stationary methods [Theorem 8, arjevani2016lower])" : OK. Source ouverte (JMLR 17 (2016) 1-51,
  https://jmlr.org/papers/volume17/15-106/15-106.pdf) : "Theorem 8 Let A be a consistent p-SCLI optimization algorithm for
  L-smooth mu-strongly convex functions over R^d. If the inversion matrix of A is diagonal, then there exists a quadratic
  function ... such that IC_A(eps, f) = Omega~((kappa^{1/p} - 1)/2 ln(1/eps))" ; p = 1 donne l'ordre kappa ln(1/eps)
  ("p-Stationary Canonical Linear Iterative", p = nombre de points precedents utilises). La condition "diagonal
  inversion matrix" n'est pas rappelee par le papier ; la parenthese ne pretend pas que SAGA releve du theoreme.

## (f) (C5), l. 325-327 et 236-238

- "Proposition 5.9 holds in any fixed metric P with R_P^2, mu_P" : OK. Le pas theta+ = theta - gamma P^{-1} omega est SAGA
  euclidien sur f~_i(u) = f_i(P^{-1/2} u) (u = P^{1/2} theta ; pour les matrices W -> W P^{1/2}), dont les constantes sont
  sup lambda_max(P^{-1/2} f_i'' P^{-1/2}) et inf lambda_min(P^{-1/2} F'' P^{-1/2}) = mu_P(F) pour la norme ||. P^{1/2}||_F.
  Le R_P^2 du papier, max_i(c_i x_i^T P^{-1} x_i) + lambda ||P^{-1}||_op, majore la premiere (lambda_max(A + B) <= lambda_max(A)
  + lambda_max(B)) : la Proposition 5.9 s'applique donc avec ce R_P^2.
- Formule de R_P^2 : c'est un majorant, pas la meilleure constante (constat MS-3, minor). Exemple : P = diag(1, 4),
  x = (0, 1), c = 1, lambda = 0.1 : formule 0.350, constante exacte lambda_max(c uu^T + lambda P^{-1}) = 0.275 (check3).
  Sur 4000 metriques aleatoires le rapport va de 1 a 1.077. Le code le dit (`vrlab/instrumentation.py` l. 96-99 :
  "This R_P^2 is the upper bound ... exact for P = I and otherwise above it (by at most 5.2% on the B1 problems)"),
  le papier non, alors que la l. 198-201 definit L_N comme "best constants".
- mu_P = lambda/lambda_max(P) pour la regression logistique ridge (meme suite t theta_1) : verifie (check3 : ecart relatif
  1.9, 0.2, 5e-3, 0 pour t = 1, 10, 100, 1000).
- "for ridge logistic regression every P has kappa_P >= kappa" : OK, demontre a la main :
  kappa_P = [max_i x_i^T P^{-1} x_i/4 + lambda/lambda_min(P)] lambda_max(P)/lambda >= [max_i ||x_i||^2/4 + lambda]/lambda = kappa,
  par x^T P^{-1} x >= ||x||^2/lambda_max(P) et lambda_min(P) <= lambda_max(P). Vrai aussi avec la constante exacte (quotient
  de Rayleigh en P^{1/2} x_i au point x_i.theta = 0 ou l'' = 1/4). Numerique (check3) : sur 4000 metriques, min
  kappa_P/kappa = 1.000009 avec les deux definitions. Concorde avec B1 Prop. "no free lunch" (l. 780-793) et avec le
  Theoreme 1 du papier pour N = ||. P^{1/2}||_F (c_N^2 = lambda_max(P), mu_N = lambda/c_N^2).

## Detail des constats

### MS-1 (major) l. 321-323 : hypothese manquante sur theta_0 dans la borne inferieure de (C4)
Voir (e). Preuve : contre-exemple analytique theta_0 = (0, 1), gamma = 1/R^2 ; check2 partie C.
Le rapport etendu (`archive/muon/report_spectral_muon_new.tex` l. 330-331) a deja la meme omission ; la note B3 (l. 412)
et le rapport principal (restate de prop:B3-lower) ont l'hypothese.
Correction proposee : `SAGA with any constant step, started at $\theta_0=(r,r)$, needs at least ...`
(et, si la place le permet, `for $\varepsilon<\frac12$`). Confiance : haute sur les faits ; la severite "major" applique
la grille (hypothese manquante qui rend l'enonce faux tel qu'ecrit), la correction tient en quelques mots.

### MS-2 (minor) l. 295-296 et 310-313 : le facteur 8/3 ne borne que c_D
Le papier : "In (1) I1 and I2 together contribute at most a factor 8/3 (cocoercivity gives L_eff <= R^2)" et, l. 310-313,
"as I1-I2 at most 8/3 ... beyond a bounded factor a run gains only through the curvature it meets".
- Dans c_M de la meme equation (1), I1 seule (r_sq -> 0, r_var = 1) multiplie le taux 1 - c_M par jusqu'a 3.
  Ce n'est pas un cas d'ecole : a t = 0 avec la table du livre, r_sq = 0.03-0.06 et 3n(1 - c_M) = 2.90-2.95, dont
  2.88-2.94 par I1 seule (check1, kappa >= 500) ; sur les runs du papier, les moyennes geometriques r_sq, r_var de
  `results/B3_summary.json` donnent 3n(1 - c_M) = 2.72 a 2.99 pour kappa >= 10^4.
- La contraction exacte est la moyenne ponderee (c_D D + c_M delta M)/V, le taux certifie le max (I5) : la repartition de
  la masse de Lyapunov est un troisieme canal, non borne a priori. Le rapport principal le dit
  (`archive/report/sections/overview/B3.tex` l. 51-56 : "... r_coco and r_sq change it by at most a factor 8/3, and the
  memory rate 1 - c_M lies between 1/(3n) and 1/n ... at most a factor 3; only mu_eff/mu ... and the step I5 ... can
  produce larger factors"). Le papier a perdu ces deux reserves en condensant.
- Portee : pour kappa >= 9n/16, ou c_D est le coefficient actif et D porte V (r_min ~ 1), l'enonce du papier est correct
  et c'est le regime auquel la suite de la phrase s'applique. C'est une imprecision de portee, pas une erreur de fond.
Correction proposee : `In $c_D$ of \eqref{eq:saga-repr} I1 and I2 together contribute at most a factor $8/3$
(cocoercivity gives $\LEff\le R^2$; in $c_M$, I1 and I3 at most $3$)` ; l. 311 : `as I1--I2 at most $8/3$ in $c_D$`.

### MS-3 (minor) l. 236-238 : R_P^2 est un majorant
Voir (f). Sans effet sur (C5) ni sur le Theoreme 1. L'etude Muon utilise la meme majoration pour ses constantes
"certifiees" (`muon/lab/vrmuon.py` l. 670 : `R2P = c_loss * quad.max() + lam * c_op`), donc les kappa_P de la figure 2(a)
sont des majorants hors de l'identite (5.2 % au plus sur les problemes B1 d'apres le commentaire du code ; l'ecart sur
les six instances de l'etude Muon n'a pas ete recalcule). Correction proposee : `the bound $R_P^2=\max_i(c_ix_i^\top P^{-1}x_i)+\lambda\nop{P^{-1}}$ on the
smoothness of the $f_i$`.

### MS-4 (nit) l. 192-194 et 210
Voir (a) et (b). Aucune correction necessaire.

### MS-5 (nit) l. 319-320
"the largest step the proof admits" : gamma R^2 < 1/2 strictement. Aucune correction necessaire
(eventuellement `the supremum of the steps the proof admits`).

## Non verifie

- Les chiffres mesures du perimetre (3.2-220, 77-96 %, 2.45-2.62, 81, 91-96 %, 59-96 %, 4.5, gamma R^2 = 4, 1.00 -> 1.11,
  etc.) : hors perimetre (autre auditeur). Seuls 17 %, 8/3, 3, 4/3 et "eight times" ont ete recalcules.
- L'edition imprimee MIT Press 2024 du livre de Bach : seule la version en ligne du 26 mai 2025 a ete ouverte
  (meme numero de proposition, meme enonce).
- Les attributions [dragomir2021fast] (l. 326), [yang2014data, gonen2016solving] (l. 327), [zhang2013gradient] (l. 214) :
  sources non ouvertes ici (dimension bibliographique). Le contenu mathematique associe (transfert de la Prop. 5.9 par
  changement de variable, definition de mu_eff) a ete verifie directement.
- "which SAGA reaches when its error aligns with the flattest direction, as it does at kappa = 10n for alpha in {0, 1}"
  (l. 302-303) : affirmation mesuree, non controlee ici.
- L'ecart entre le majorant R_P^2 (MS-3) et la constante exacte sur les six instances de la figure 2(a) : non recalcule.
- La proposition d'attribution de la note B3 (ordres de levee des inegalites, l. 248-304) n'a pas ete redemontree : elle
  ne porte que les parts mesurees.
