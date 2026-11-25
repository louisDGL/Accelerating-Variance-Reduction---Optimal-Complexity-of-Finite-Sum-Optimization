# Audit math-schedule : Proposition 5, remarques, recette (4), phrase F15

Statut : TERMINE (2026-11-20). Bilan : 0 blocker, 0 major, 7 minor, 1 nit. La Proposition 5 est exacte telle
qu'enoncee (items (a), (b), (c), bornes d'age) sous une hypothese d'initialisation implicite ; F15 est exact ; les
chiffres des remarques sont retrouves. Les constats portent sur la portee annoncee et sur des reserves manquantes.

## Perimetre couvert

- Proposition 5 (`paper/paper_saga_to_muon.tex`, l. 547-566) et son esquisse de preuve (l. 567-575).
- Remarques l. 577-581 ; recette (4) l. 508-515 ; phrase l. 648-650 (F15).
- Phrases qui resument la Proposition 5 : resume l. 105, introduction l. 131-133, tableau 1 l. 178-179, usage de
  l'item (c) dans (M5) l. 536-537.
- Sources lues : `muon/notes_theory.md` l. 483-792 (F12 a F15), `muon/lab/vrmuon.py` (recette, table, ordres),
  `muon/lab/polar.py` (ns5), `muon/experiments/families.py` (`recipe_family`), `muon/experiments/exp_M9_schedule.py`
  (ordre d'echantillonnage de M5), `muon/results/M1_summary.json`, `muon/results/M5b_summary.json`,
  `_archive/muon/tests/test_facts.py` (test archive d'ou viennent les chiffres des remarques).
- Scripts ecrits (sorties `.out` a cote) :
  - `paper/audit/checks/math-schedule_prop5_sim.py` : implementation independante du lab (numpy), moindres carres
    ridge matriciels K = 3, d = 4, n = 12 ; items (a), (b), (c), controles negatifs.
  - `paper/audit/checks/math-schedule_benchmark_numbers.py` : chiffres des remarques et de F15 sur soft-cond et ls-cond.
  - `paper/audit/checks/math-schedule_F15_lyapunov.py` : F15 par enumeration exacte des indices ; constante 10 de L_eff.

## 1. Item (a) : OK

Preuve refaite. grad F(W) = (1/n) sum_i grad g_i(W) + lam W, donc omega - grad F(W) = sum_i w_i c_i avec
c_i = grad g_i(W) - z_i et w_i = theta 1[i in B]/b - 1/n. c_i = (grad l_i(W x_i) - grad l_i(phi_i x_i)) x_i^T est de
rang <= 1, donc ||c_i||_* = ||grad l_i(W x_i) - grad l_i(phi_i x_i)|| ||x_i|| <= c_l ||(W - phi_i) x_i|| ||x_i||
<= L_i ||W - phi_i||_op. Enfin sum_i |w_i| L_i <= sum_{j in B} (theta/b + 1/n) L_j + (1/n) sum_{i hors B} L_i
<= theta L_max + L_bar. Aucun facteur de rang : exact.

Numerique (`math-schedule_prop5_sim.py a`) : 6000 etats aleatoires par perte (points phi_i quelconques, ecarts de rang
un alignes sur x_i, lot seul perime ; b dans {1,2,3,4,6,12} ; theta dans [0,1]) : max membre gauche / membre droit
= 0.812 (moindres carres, c_l = 1) et 0.100 (softmax, c_l = 1/2) ; 0 violation. Le long des runs de (b) : max 0.795.
Sur soft-cond et ls-cond (run de la recette a q = 0.8, 14 etats x 200 indices) : (a) verifiee a chaque indice.

Reserve de forme (constat S7) : l'esquisse ecrit "|w_i| <= theta/b + 1/n", ce qui, lu pour tout i, donne
(theta n/b + 1) L_bar et non (theta L_max + L_bar) ; il faut w_i = -1/n hors du lot.

## 2. Item (b) : OK sous l'hypothese implicite phi_i = W_0 au depart

Preuve refaite avec les constantes.

1. <grad F, polar(omega)> = <omega, polar(omega)> - <omega - grad F, polar(omega)> >= ||omega||_* - ||omega - grad F||_*
   >= ||grad F||_* - 2 ||omega - grad F||_* (||polar(omega)||_op <= 1). Cas omega = 0 : pas nul, D_{t+1} = D_t, et le
   membre droit de la recurrence vaut D_t + eta ||grad F||_* + (L_op/2) eta^2 >= D_t : trivial.
2. Lissite spectrale : F(W - eta O) <= F(W) - eta <grad F, O> + (L_op/2) eta^2 ||O||_op^2.
3. PL duale : minimiser en V les deux membres de F(V) >= F(W) + <grad F(W), V - W> + (mu_op/2)||V - W||_op^2 donne
   F_* >= F(W) - ||grad F(W)||_*^2 / (2 mu_op).
4. Peremption : ||W_t - phi_i||_op <= sum_{s = max(t-A,0)}^{t-1} eta_s = eta_t sum_{k=1}^{min(A,t)} rho^(-k/2)
   <= eta_t A rho^(-A/2). Pour t < A la somme a moins de A termes : la borne tient (si phi_i = W_0 au depart).
5. Donc D_{t+1} <= D_t - eta_t (2 mu_op D_t)^(1/2) + (1/2)[L_op + 4 (theta L_max + L_bar) sum_{k<=A} rho^(-k/2)] eta_t^2.
6. Role de mu_op <= L_max : L_eff >= 10 A L_max >= 10 A mu_op, donc rho >= 1 - 1/(10A) et
   rho^(-A/2) <= (1 - 1/(10A))^(-A/2) <= 0.9^(-1/2) = 1.0541 (decroissant en A) ; avec theta <= 1 et L_bar <= L_max,
   4 (theta L_max + L_bar) A rho^(-A/2) <= 8.44 A L_max <= 10 A L_max. Aussi L_eff >= 11 mu_op >= 2 mu_op.
   Numerique (`math-schedule_F15_lyapunov.py`, derniere section) : au pire cas (theta = 1, L_bar = L_max = mu_op = L_op),
   max sur A de 8 sum_{k<=A} rho^(-k/2) / (10 A) = 0.841 <= 1.
7. Convexite : phi(x) = x - eta (2 mu_op x)^(1/2) + (L_eff/2) eta^2 est convexe ; avec C = L_eff^2/(2 mu_op),
   phi(C eta^2) = (C - L_eff/2) eta^2 = C rho eta^2 (egalite), phi(0) = (L_eff/2) eta^2 <= C rho eta^2 ssi L_eff >= 2 mu_op,
   min phi = (L_eff - mu_op) eta^2 / 2 >= 0. Donc [0, C eta^2] -> [0, C rho eta^2]. C eta_0^2 = D_0 : D_t <= D_0 rho^t.

Bornes d'age refaites : cyclique, n/b (l'entree du lot B_t a exactement l'age n/b) ; permutation fraiche, 2n/b - 1
(premier lot de la passe p-1, dernier lot de la passe p) ; lots independants, P(age_i(t) > A) = (1 - b/n)^A
<= exp(-A b/n), borne d'union sur n entrees et T iterations : A = ceil((n/b) log(nT/delta)).

Numerique (`math-schedule_prop5_sim.py b adv iid`) : recette prescrite, facteur polaire exact, table en W_0.

| Controle | Resultat |
|---|---|
| 24 runs (2 problemes ; cyclique et permutation fraiche ; b = 1, 3 ; theta = 1, b/n ; departs aleatoire, PL-serre, W_0 = 0, proche de W_*), 20000 a 40000 iterations | max_t D_t/(D_0 rho^t) = 0.999966 ; 0 violation de la borne, de la recurrence, de (a), de l'inegalite 1 |
| ages observes | 12, 4 (cyclique b = 1, 3) et 23, 7 (permutation) : egaux aux bornes n/b et 2n/b - 1 |
| regle adverse gloutonne (b = 1, ages <= A, choisit l'indice qui maximise D_{t+1}), 8 runs de 4000 iterations | max ratio 0.999966 ; 0 violation |
| lots independants, delta = 0.05, T = 2000 (A = 157) et T = 20000 (A = 185) | 0/46 runs avec un age > A ; 0 violation |

Dependance en (T, delta) pour les lots independants : A, donc rho et eta_0, dependent de l'horizon ; la conclusion est
"pour t <= T avec probabilite 1 - delta", pas "pour tout t". Sur le petit probleme, 1 - rho passe de 5.2e-6 (T = 1e3) a
2.7e-6 (T = 1e8). Le papier ne donne pour ce cas que la borne d'age, ce qui est exact.

Controles negatifs (`math-schedule_prop5_sim.py neg`) :
- Table initialisee a zero (au lieu des gradients en W_0), ordre cyclique : 10/24 runs violent D_t <= D_0 rho^t
  (max ratio 1.0052), contre 0/24 pour les memes runs avec la table en W_0. L'hypothese phi_i = W_0 au depart est
  necessaire et n'est pas ecrite (constat S3).
- mu_op > L_max (lam = 1e4) : la condition suffisante de la preuve n'est plus satisfaite (L_eff = 3.11e4 contre 3.22e4
  requis en cyclique, 3.21e4 contre 4.81e4 en permutation) ; aucune violation de la conclusion n'a ete observee (max
  ratio 0.36). L'hypothese sert la preuve ; je n'ai pas de contre-exemple a la conclusion sans elle.

## 3. Item (c) : OK

Preuve refaite : pour les moindres carres, zbar + lam W = (1/n) sum_i e_i + lam E (car grad F(W_*) = 0) et
grad g_j(W) - z_j = E x_j x_j^T - e_j : omega est lineaire en u = (E, e) ; E_+ = E - eta o(omega), e_j <- E x_j x_j^T
sont homogenes de degre un en (u, eta) si o(sM) = o(M). D'ou u_{t+1} = eta_t Phi(u_t/eta_t; B_t) et
v_{t+1} = Phi(v_t; B_t)/r pour v_t = u_t/eta_t : chaine de Markov homogene pour des lots i.i.d.

"ns5 without epsilon" : `muon/lab/polar.py` l. 173, X = X / (||X||_F + eps) puis une iteration polynomiale fixe ; a
eps = 0 la carte est invariante d'echelle (M != 0).

Numerique (`math-schedule_prop5_sim.py c`) : identite a un pas Phi_{s eta}(s u; B) = s Phi_eta(u; B) sur 300 etats par
cas : ecart relatif <= 1.6e-15 pour le facteur polaire, ns5 a eps = 0, la normalisation de Frobenius et le signe ;
1.8e-4 a 0.18 pour ns5 a eps = 1e-7. Deux trajectoires aux echelles 1 et s (memes lots, eta_t = 0.3 x 0.995^t) :
ecart 8e-14 (s = 1e-3) et 3e-13 (s = 1e-7) apres 120 iterations pour le facteur polaire, 6e-9 pour ns5 a eps = 0
(amplification de l'arrondi), 0.16 a 0.49 pour ns5 a eps = 1e-7.

Reserve (constat S5) : l'etat de (c) ne contient pas le tampon de momentum, alors que (M5) l. 536-537 invoque (c) pour
la recette de Muon (ns5 avec momentum), sur softmax comme sur moindres carres.

## 4. Remarques l. 577-581

| Affirmation | Verdict | Preuve |
|---|---|---|
| "Part (a) has no rank factor ... a polar step has length exactly eta_t" | OK | section 1 ; ||S_t||_op = eta_t verifie a 1e-9 pres sur les runs du banc |
| "its cooldown is 3.3e4 to 8.9e4 times slower, in log(1/q), than q = 0.8" | OK pour les deux ordres a permutation ; reserve S1 | recalcul : cyclique 4.46e4 (soft-cond), 3.26e4 (ls-cond) ; permutation fraiche 8.90e4, 6.50e4 |
| "the error of the estimator is 2.3 to 3.5 times the gradient in nuclear norm" | OK | recalcul (q = 0.8, lr_0 = 0.1, graine 10, passes 2..26) : [2.69, 3.45] et [2.32, 3.42] |
| "a decrease that only the average over the index explains" | probleme mineur S4 | la moyenne a un pas E_j D_+/D vaut 0.990-0.999 (soft-cond) et 0.991-1.005 (ls-cond, > 1 en 2 etats sur 7) |
| "It gives no critical q and does not cover momentum" | OK, incomplet (S2) | (b) ne couvre pas non plus ns5 et prescrit eta_0 |

Constantes des instances (recalculees) : soft-cond lam = 5.0e-5, L_max = 0.5, L_bar = 0.0791, L_op = 0.0751,
mu_op = 5.0e-5 ; ls-cond lam = 1e-4, L_max = 1, L_bar = 0.158, L_op = 0.150, mu_op = 1.371e-4 ; mu_op <= L_max sur
les deux. Longueur initiale prescrite eta_0 = (2 mu_op D_0)^(1/2)/L_eff : 4.5e-6 a 1.6e-5, soit 6e3 a 6.6e4 fois
sous lr_0 = 0.1 a 0.3.

## 5. Recette (4) et portee de la Proposition 5

Recette conforme au code : lr_t = lr q^(t b/n) (`vrmuon.py` l. 478-480) ; longueur lr_t x max(1, K/d)^(1/2), facteur
egal a 1 sur les six instances (K <= d) ; ns5 a eps = 1e-7 (`polar.py` l. 157) ; Nesterov 0.9 avant la carte
(`families.py` l. 314-315) ; controles 'frob' et 'sign' ; theta = 1.

Correspondance : eta_t = eta_0 rho^(t/2) est la recette (4) avec q = rho^(n/(2b)) et lr_0 = eta_0. L'item (b) couvre
donc un seul point (q, lr_0) par borne d'age, pour le facteur polaire exact, sans momentum, de facon sure pour les
ordres cyclique et a permutation. Il ne couvre ni (q, lr_0) regles, ni ns5, ni le momentum, ni les cartes de controle,
ni les lots independants au-dela d'un horizon fixe. "at slower cooldowns" est exact (tout A' >= A est admissible, ce
qui donne une famille de refroidissements plus lents).

Phrases de synthese : "we prove this law for tables of bounded age, at slower cooldowns" (l. 105) et "certified for
tables of bounded age, at slower cooldowns [p]" (l. 178-179) sont vraies ; il leur manque "exact polar factor"
(constat S2). L'introduction l. 131-133 cite la Proposition 5 a l'appui d'une phrase experimentale (S2).

## 6. Phrase F15 (l. 648-650) : OK, reserve S6

Enonce de F15 (`muon/notes_theory.md` l. 740-770) : lot un, indice uniforme independant, facteur polaire exact,
theta dans [0,1], Psi = D + Theta/(8 n gamma), Theta = (1/n) sum_i ||W - phi_i||_op^2 ; si gamma <= 1/(14 n L_max) et
gamma L_op <= 1, E[Psi_+ | passe] <= (1 - min{(1 - gamma L_op) gamma mu_op/2, 1/(4n)}) Psi.

Preuve refaite : (I1) D_+ - D <= -gamma s^2 (1/2 + a/2) + gamma s x, puis s x <= s^2/4 + x^2 et
s^2 >= ||G||_*^2/2 - x^2 donnent D_+ - D <= -||S_j||_op^2/(4 gamma) - (gamma a/4)||G||_*^2 + gamma (1 + a/2) x^2, et
1 + a/2 <= 3/2 ; (I2) E_j x^2 <= 4 L_max^2 Theta (Jensen) ; (I3) avec e = 1/(2(n-1)),
E_j Theta_+ <= (1 - 1/(2n)) Theta + (2n - 3 + 2/n) E_j ||S_j||_op^2 ; les termes en ||S_j||^2 se compensent
(2n/(8 n gamma) = 1/(4 gamma)) ; 48 n gamma^2 L_max^2 <= 48/(196 n) = 0.245/n <= 1/(4n). Exact.

Numerique (`math-schedule_F15_lyapunov.py`, enumeration exacte des 12 indices) : a gamma = gamma_max et gamma_max/4,
1000 etats (table dispersee, ecarts de rang un, table fraiche ; theta = 1, 1/n, 0.5) : 0 echec de l'inegalite finale,
0 violation de (I1), (I2), F12 ; le long d'un run : 0 echec, decroissance exacte 14 a 108 fois la decroissance certifiee.
Hors hypothese l'inegalite tient encore a 4 et 16 gamma_max sur ces etats et echoue a 64 gamma_max (21 a 39 etats sur
250) et a 350 gamma_max (87 a 115 sur 250).

"350 to 1400 times below the tuned steps" : 1/(14 n L_max) = 3.574e-4/R2_op sur soft-cond et ls-cond ; pas regles de
`M1_summary.json`, ligne "SAGA-Muon (joint)" : 0.5/R2_op et 0.125/R2_op, rapports 1399 et 350. gamma L_op <= 1 est
satisfait sur les six instances (gamma_max L_op <= 5.4e-5 ; lam r <= (14n - 1) L_max). Le resume du papier est fidele.

## Constats

### S1 (minor) : les facteurs "3.3e4 a 8.9e4" et "1e4 a 1e5" valent pour les ordres a permutation ; (M5) tire des indices independants

- Lieu : l. 578-579 et l. 648-649.
- Citation : "its cooldown is $3.3\cdot10^4$ to $8.9\cdot10^4$ times slower, in $\log(1/q)$, than $q=0.8$" ;
  "Proposition~\ref{prop:schedule} at cooldowns $10^4$ to $10^5$ times slower than in \finding{M5}".
- Probleme : la fourchette melange deux instances et deux ordres (cyclique : 3.3e4, 4.5e4 ; permutation fraiche :
  6.5e4, 8.9e4) sans le dire. Les runs de (M5) (`exp_M9_schedule.py`, parties A a C : `order` par defaut 'iid' ;
  l'ordre n'est change qu'en partie D, l. 569-570) tirent des indices independants. Pour cet ordre, (b) ne donne
  qu'un enonce a horizon fini, de probabilite 1 - delta, avec A = ceil(n log(nT/delta)) : a T = 300 passes et
  delta = 0.05, A = 3860 et le refroidissement certifie est 6.3e5 (ls-cond) a 8.6e5 (soft-cond) fois plus lent que
  q = 0.8 (6.8e5 a 9.3e5 a delta = 0.01).
- Preuve : `math-schedule_benchmark_numbers.out`, section (i).
- Correction proposee (l. 578-579) : `its cooldown is $3.3\cdot10^4$ to $8.9\cdot10^4$ times slower, in $\log(1/q)$,
  than $q=0.8$ for the two permutation orders (and $6\cdot10^5$ to $9\cdot10^5$ times over the $300$ passes of
  \finding{M5}, whose indices are independent),` ; (l. 648-649) : `at cooldowns $10^4$ to $10^6$ times slower than in
  \finding{M5}`.
- Confiance : haute.

### S2 (minor) : portee de "we prove this law" et renvoi a la Proposition 5 pour une phrase experimentale

- Lieu : l. 105 ; l. 178-179 ; l. 131-133 ; l. 581.
- Citation : "we prove this law for tables of bounded age, at slower cooldowns" ; "fixed lengths on a geometric
  cooldown, fed by a SAGA table, converge to rounding error where other estimators freeze
  (Proposition~\ref{prop:schedule})" ; "It gives no critical $q$ and does not cover momentum."
- Probleme : (b) est prouve pour le facteur polaire exact, sans momentum, et pour un seul couple (eta_0, rho) par
  borne d'age ; eta_0 = (2 mu_op D_0)^(1/2)/L_eff demande D_0 et vaut 4.5e-6 a 1.6e-5 sur les deux instances, 6e3 a
  6.6e4 fois sous lr_0 = 0.1 a 0.3. La phrase du resume suit une phrase qui mentionne la carte de Newton-Schulz de
  Muon : "this law" peut se lire comme couvrant ns5. L'introduction cite la Proposition 5 comme les theoremes des
  points (1) et (2), alors que "converge to rounding error where other estimators freeze" est le constat experimental
  (M5) : la proposition ne dit rien des autres estimateurs ni des refroidissements utilises.
- Preuve : enonce l. 556-557 ; `math-schedule_benchmark_numbers.out` (eta_0 prescrit).
- Correction proposee : l. 105 `we prove this law for the exact polar factor and tables of bounded age, at slower
  cooldowns.` ; l. 132-133 `converge to rounding error where other estimators freeze \finding{M5}, a law certified at
  slower cooldowns (Proposition~\ref{prop:schedule});` ; l. 581 `It prescribes $\eta_0$, gives no critical $q$ and
  covers neither momentum nor ns5.`
- Confiance : moyenne (affaire de lecture), haute sur les faits.

### S3 (minor) : l'initialisation de la table n'est pas ecrite

- Lieu : l. 550 (et l. 556-559).
- Citation : "A SAGA table stores $z_i=\nabla g_i(\phi_i)$, $\phi_i$ the iterate at which entry $i$ was last replaced".
- Probleme : avant son premier remplacement, phi_i n'est pas defini ; (b) et les bornes d'age (n/b, 2n/b - 1) supposent
  phi_i = W_0 (table initialisee par les gradients en W_0, comme l. 192-193 pour SAGA et comme dans F12). Avec une
  table nulle au depart la conclusion est fausse.
- Preuve : `math-schedule_prop5_sim.out`, controles negatifs : 10/24 runs violent D_t <= D_0 rho^t (max 1.0052) avec une
  table nulle, 0/24 avec la table en W_0.
- Correction proposee : `$\phi_i$ the iterate at which entry $i$ was last replaced ($\phi_i=W_0$ initially);`
- Confiance : haute.

### S4 (minor) : "a decrease that only the average over the index explains" n'est pas etabli

- Lieu : l. 579-581.
- Citation : "along the fast runs the error of the estimator is $2.3$ to $3.5$ times the gradient in nuclear norm, a
  decrease that only the average over the index explains."
- Probleme : les rapports 2.3 a 3.5 sont exacts. Mais aux memes etats la moyenne exacte a un pas E_j D_+/D vaut
  0.9903 a 0.9993 sur soft-cond et 0.9906 a 1.0053 sur ls-cond, ou elle depasse 1 en deux etats sur sept (1.0052,
  1.0053) et vaut 1.0000 en un troisieme ; le pas augmente F pour 1 % a 73 % des indices. La moyenne sur l'indice ne fait donc pas
  decroitre F a chaque etat : la decroissance n'est observee que le long du run. La phrase affirme un mecanisme que
  la mesure ne montre pas ; ce qui est etabli est que la borne par trajectoire ne peut pas l'expliquer.
- Preuve : `math-schedule_benchmark_numbers.out`, section (ii) (un run par instance, graine 10, 7 etats).
- Correction proposee : `times the gradient in nuclear norm: the pathwise bound cannot explain their decrease (over
  the index, the one-step mean of $D_{t+1}/D_t$ is $0.990$--$1.005$).`
- Confiance : moyenne a haute (un seul run par instance).

### S5 (minor) : l'item (c) est invoque pour la recette de Muon, que son enonce ne couvre pas

- Lieu : l. 561-564 et l. 536-537.
- Citation : "the state $u_t$ made of $W_t-W_\ast$ and the errors $z_i-\nabla g_i(W_\ast)$" ; "a floor of ns5, whose
  constant $\epsilon$ breaks the scale invariance of Proposition~\ref{prop:schedule}(c)".
- Probleme : (c) est enonce sans momentum et pour les moindres carres. (M5) l'invoque pour "Muon's recipe" (ns5 avec
  momentum de Nesterov) et donne des planchers pour softmax et moindres carres. Avec momentum la covariance reste vraie
  si l'etat contient le tampon (F14 l'enonce ainsi) ; pour softmax elle n'est pas exacte (F14, point (2)).
- Preuve : enonce l. 561-564 ; `muon/notes_theory.md` l. 697-702 et 727-728.
- Correction proposee : l. 562 `the state $u_t$ made of $W_t-W_\ast$, the errors $z_i-\nabla g_i(W_\ast)$ and the
  momentum buffer, if any, satisfies` ; l. 537 `breaks the scale invariance of Proposition~\ref{prop:schedule}(c)
  (exact on least squares; with $\epsilon=10^{-30}$: $1.5\cdot10^{-29}$ and $4.1\cdot10^{-30}$)`.
- Confiance : moyenne.

### S6 (minor) : "350 to 1400 times below the tuned steps" sans instance ni grille

- Lieu : l. 649-650.
- Citation : "a Lyapunov function for $\gamma\le1/(14nL_{\max})$, $350$ to $1400$ times below the tuned steps".
- Probleme : 350 et 1400 sont les rapports sur ls-cond et soft-cond avec les pas de M1 (grille de facteur 2). Sur les
  grilles resolues que le papier utilise pour ce pas dans (M6) (`M5b_summary.json`, b = 1, 'fine' : 0.420/R2_op et
  0.149/R2_op) les rapports sont 1176 et 416 ; sur les six instances avec les pas de M1, 217 (ls-mem) a 3493 (soft-sq).
  F15 demande aussi gamma L_op <= 1 (vrai sur les six instances) et un indice uniforme independant.
- Preuve : `math-schedule_benchmark_numbers.out`, section (iii) ; `M5b_summary.json`.
- Correction proposee : `$350$ to $1400$ times below the steps tuned on the same two instances`.
- Confiance : haute.

### S7 (nit) : poids de l'esquisse de (a) et collisions de notation

- Lieu : l. 568-569.
- Citation : "$\omega-\nabla F(W)=\sum_iw_ic_i$ with $\abs{w_i}\le\vartheta/b+1/n$".
- Probleme : lue pour tout i, cette majoration donne sum_i |w_i| L_i <= (theta n/b + 1) L_bar et non
  theta L_max + L_bar ; il faut w_i = -1/n hors du lot. Par ailleurs c_i designe deja sup_z lambda_max(grad^2 l_i(z))
  (l. 202) et rho la longueur d'ecretage (l. 257), rho_star etant un facteur par passe de lr^2 (l. 597).
- Correction proposee : `with $w_i=\vartheta\mathbf{1}[i\in B]/b-1/n$, and`.
- Confiance : haute.

### S8 (minor) : les chiffres des remarques ne viennent pas des scripts cites

- Lieu : l. 578-580 et l. 686-688.
- Citation : "Numbers come from \nolinkurl{experiments/} and \nolinkurl{muon/experiments/} (...)".
- Probleme : "3.3e4 a 8.9e4" et "2.3 a 3.5" sont imprimes par `test_F13_limits_on_the_benchmark`, deplace le
  2026-11-20 dans `_archive/muon/tests/test_facts.py`, dossier liste dans `.gitignore` (non versionne) ; aucun script de
  `muon/experiments/` ne les calcule. Les valeurs sont exactes (recalculees ici).
- Preuve : `_archive/README.md` ; `.gitignore` l. 6 ; `muon/findings.md` l. 313-316.
- Correction proposee : aucune dans le texte si le test (ou `paper/audit/checks/math-schedule_benchmark_numbers.py`)
  est remis dans l'arbre versionne.
- Confiance : haute.

## Non verifie

- Existence d'un contre-exemple a la conclusion de (b) quand mu_op > L_max (la preuve tombe, aucune violation observee).
- "a decrease that only the average over the index explains" au-dela d'un run par instance (graine 10).
- F15 numeriquement sur softmax (la preuve ne depend pas de la perte) ; pas regles de M1 et M5b pris tels quels.
- F13 (e) (convergence presque sure), absent du papier.
- Affirmations bibliographiques ("not found", proximite de [zhu2026short]) et chiffres experimentaux de (M5) : autres
  dimensions de l'audit.
- Le test archive `test_F13_limits_on_the_benchmark` n'a pas ete relance ; ses chiffres ont ete recalcules a part.
