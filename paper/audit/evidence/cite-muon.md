# Audit cite-muon : bibliographie, groupe Muon 2024-2026

Statut : TERMINE (2026-11-20). Papier audite : `paper/paper_saga_to_muon.tex` (numeros de ligne de l'outil Read).

## 1. Perimetre couvert

Les 17 cles du troisieme groupe : bernstein2024old, chen2026when, do2026muon, hird2025quantifying, huang2026muon,
jawanpuria2026constant, jiang2026adaptive, jordan2024muon, lau2025polargrad, li2026implicit, liu2025marsm,
ma2026preconditioning, paquette2026phases, shulgin2026quadratic, vasudeva2025muon, xie2026muon, zhu2026short.
Pour chacune : (1) exactitude de l'entree .bib ; (2) chaque phrase du papier qui cite la cle ; (3) renvois precis
(numero d'equation, de theoreme, de lemme, constante) ouverts dans la source.
Point supplementaire demande : schedule de taux d'apprentissage des executions de reference de Muon et fidelite de
l'expression "Muon's own recipe" (resume l. 102-103, section 3.3 l. 506-515).

Methode. Metadonnees : API arXiv (`export.arxiv.org/api/query?id_list=...`, 16 identifiants, reponse du 2026-11-20) et
page JMLR. Contenu : PDF telecharges depuis arxiv.org avec curl (version indiquee ci-dessous), convertis par
`pdftotext`, lus directement (pas de resume par un modele). Billet Muon : HTML telecharge depuis
`kellerjordan.github.io/posts/muon/` ; code : fichiers bruts de GitHub aux commits lies par le billet.
Aucune de ces lectures ne passe par les notes du depot ; celles-ci (muon/literature_check.md,
archive/muon/notes_spectral_theory.tex) n'ont servi qu'a retrouver l'origine de deux phrases.
Deux controles numeriques : `paper/audit/checks/cite-muon_ray_factor4.py`, `paper/audit/checks/cite-muon_linear_cooldown.py`.

## 2. Entrees bibliographiques (existence, titre, auteurs, annee, identifiant)

| cle | identifiant | version consultee | verdict |
|---|---|---|---|
| bernstein2024old | arXiv:2409.20325 | v2, 6 dec. 2024 (OPT 2024 workshop) | OK : titre, 2 auteurs, annee |
| chen2026when | arXiv:2610.31728 | v1, 22 oct. 2026 | OK : titre, 4 auteurs |
| do2026muon | arXiv:2608.04607 | v1, 5 aout 2026 (82 p.) | OK : titre, 3 auteurs |
| hird2025quantifying | JMLR 26(119):1-51, 2025 ; arXiv:2312.04898 | JMLR + arXiv v2 | OK ; numero (119) absent du .bib (nit) |
| huang2026muon | arXiv:2611.18124 | v1, 13 nov. 2026 | OK : titre, 3 auteurs |
| jawanpuria2026constant | arXiv:2610.27747 | v1, 20 oct. 2026 | OK : titre, 3 auteurs |
| jiang2026adaptive | arXiv:2602.08232 | v1, 9 fev. 2026 | OK : titre, 4 auteurs |
| jordan2024muon | billet kellerjordan.github.io/posts/muon/ (8 dec. 2024) | page du 2026-11-20 | OK : le .bib reprend le BibTeX donne dans le billet (7 auteurs, titre, 2024) |
| lau2025polargrad | arXiv:2505.21799 | v4, 5 fev. 2026 | OK : titre, 3 auteurs, 2025 |
| li2026implicit | arXiv:2602.11557 | v2, 16 juin 2026 (et v1) | OK ; la numerotation citee est celle de v2 (voir F7) |
| liu2025marsm | arXiv:2510.21800 | v3, 29 janv. 2026 | OK : titre, 3 auteurs, 2025 |
| ma2026preconditioning | arXiv:2601.13474 | v1, 20 janv. 2026 | OK : titre, 4 auteurs |
| paquette2026phases | arXiv:2605.09552 | v2, 27 oct. 2026 (en-tete : NeurIPS 2026) | OK : titre, 6 auteurs |
| shulgin2026quadratic | arXiv:2610.30280 | v1, 21 oct. 2026 (47 p.) | OK : titre, 4 auteurs |
| vasudeva2025muon | arXiv:2510.22980 | v3, 12 dec. 2025 | OK : titre, 5 auteurs, 2025 |
| xie2026muon | arXiv:2609.32861 | v1, 26 sept. 2026 | OK : titre, auteur unique |
| zhu2026short | arXiv:2602.05304 | v2, 21 mai 2026 (ICML 2026 d'apres le commentaire arXiv) | OK : titre, 3 auteurs |

Aucune entree fantome, aucun titre ou auteur errone. Les 16 identifiants arXiv existent avec le titre et les auteurs du .bib.

## 3. Renvois precis : verdicts

| lignes | renvoi du papier | ce que dit la source | verdict |
|---|---|---|---|
| 231 | "Eq. (17) of shulgin2026quadratic" | Theoreme 3, Eq. (17) : f(X+)/f(X) = 1 - alpha(2-alpha) p(X), alpha = gamma tr(H), p(X) = \|HX\|_*^2/(tr(H) tr(HXX^T)), pour f = 1/2 tr(X^T H X), H > 0, rang plein en lignes. Avec gamma = 1/L : p = muPL/tr(H), J = tr(H), donc 1 - (muPL/L)(2 - J/L). | OK (cas unilateral, rang plein ; l'identite du papier pour un Hessien general en est l'extension immediate ; nit F4) |
| 231 | "Eq. (8) of jawanpuria2026constant" | Annexe C, Eq. (8) : f(X_{t+1}) = f(X_t) - alpha \|G_t\|_*^2 + (alpha^2/2) \|G_t\|_*^2 tr(U_t^T H U_t), "exact identity". | OK |
| 232 | "certified factor 1 - 1/kop [Theorem 3]" | Theoreme 3 : f(X_{t+1}) <= (1 - u(2-u) lambda_min/tr(H)) f(X_t), u = alpha tr(H) ; a u = 1 : 1 - lambda_min/tr(H) = 1 - muop/Lop (n >= d ; la preuve donne tr_n(H) = Lop sinon, Remark 2). | OK |
| 492 | "max(F(W_t),F(W_{t+1})) - F_* >= muop eta^2/8 [Lemma 4]" | Lemme 4 : max{\|E_t\|_F, \|E_{t+1}\|_F} >= alpha gamma/2 (gamma = 1 pour le facteur polaire), sur quadratiques, en distance de Frobenius ; pas d'enonce en valeur de fonction. | imprecis : corollaire du meme argument, pas l'enonce du lemme (F3, minor) |
| 313-314 | "the gap is about a factor 4 on the ray that attracts generic runs [shulgin2026quadratic]" | Corollaire 5 : sur la famille a deux amas (cas commutant) p = 4w(1-w), f contracte de (1-2w)^2 ; avec w = lambda_min/tr(H) (annexe E.2), exposant ~ 4w contre w certifie. Rien sur l'attraction des trajectoires generiques. | attribution partielle + enonce asymptotique presente comme general (F2, major) |
| 350-351 | "[Proposition 7]{hird2025quantifying}" | JMLR, Proposition 7 : "There exist distributions satisfying Assumption 6 with arbitrary condition number kappa >= 1 in arbitrary dimension d >= 1 for which any non-orthogonal linear preconditioner will cause the condition number to increase." (arXiv v2 : Proposition 1.) | OK (numerotation JMLR) |
| 439 | "[Lemma H.1]{li2026implicit}" | Lemme H.1 (v1 et v2) : pour un seul echantillon, gradient de rang un sigma_1 u v^T, directions spectrale et Frobenius identiques = u v^T. | OK |
| 445 | "[Theorem 4.10]{li2026implicit}" | v2, Theoreme 4.10 (v1 : Theoreme 4.9) : SignSGD et Normalized-SGD par echantillon, random reshuffling, eta_t = c t^-a, jeu "Orthogonal Scale-Skewed" : W_t/\|W_t\|_F -> Wbar/\|Wbar\|_F, direction differente du max-marge ; Lemme H.1 : Spectral-SGD = Normalized-SGD a batch un. | OK (v2 ; nit F7) |
| 168, 175, 654, 659 | "SVRG in polar steps", "SVRG anchors under reshuffling, without linear rate", "descent inequalities with SVRG anchors" | Algorithme 1 (reshuffling, snapshot par epoque, V_t = grad_B(W_t) - grad_B(W~) + grad L(W~)), normes de Schatten dont spectrale (phi_spec(G) = U V^T), Theoreme 4.7 et Corollaire 4.8 : taux polynomiaux avec eta_t = c t^-a ; Lemme C.13 (ecart de l'estimateur borne par la distance au snapshot), Lemme F.1 (inegalite de descente). | OK |
| 660-662 | "[Corollary 3.1, Theorem 3.2]{huang2026muon}" | Corollaire 3.1 : 1 - beta = Theta((L Delta_f/(N^2 D_0 T))^(1/3)), complexite O(eps^-3) ; Theoreme 3.2 : pour tout beta fixe dans [0,1), somme finie convexe lisse sur matrices 2x2 ou \|grad f(W_t)\|_* >= 1/N pour toute suite de permutations (pas constants par epoque, <= 1/(2N)). | OK |
| 162-163, 465-467 | do2026muon : dimension un, non-convergence sous donnees bornees asymetriques, toute limite est un zero du champ moyen | Theoreme 1.3 (et 6.16) : theta scalaire, L(theta,x) = \|theta - x\|^2, X bornes i.i.d., E[(X - EX)^3] != 0, Muon avec moment et NS generalise : limsup E[min{1,\|Theta_n - E X\|}] > 0 "for almost all M" ; Proposition 6.2 ("MUON-type limit theorem") : si Theta_n -> vartheta dans L^1 alors f(vartheta) = 0, f = champ de vecteurs de Muon. | OK (reserve : "pour presque toute taille de batch", nit F8) |
| 160, 461 | xie2026muon, chen2026when : "Noise smooths the polar map" ; derivee du champ moyen en 0 = sqrt(b) c_{K,d}/sigma | Xie, Prop. 3.1 : Psi_sigma(Y) = E P(Y + sigma Z) est C^inf, D Psi_sigma(0) = (c/sigma) Id, c = E\|Z\|_*/(mn) ; "Gaussian noise smooths the polar map". Chen et al., Prop. 3.1 : E[msgn(G + sigma Xi)] = (c_d/sigma) G + O(\|G\|_F^3/sigma^3), c_d = d^-2 E\|Xi\|_* (carre). Avec sigma/sqrt(b) : sqrt(b) c/sigma. | OK |
| 119 | paquette2026phases ("gradient noise") | Muon stochastique (EMA de gradients de minibatch) sur moindres carres matriciels en grande dimension, sensibilite au batch effectif. | OK |
| 119 | vasudeva2025muon, shulgin2026quadratic ("least squares") | Vasudeva et al. : SpecGD, modeles lineaires et bilineaires, perte quadratique en population (moments conjointement diagonalisables). Shulgin et al. : quadratiques matricielles. | OK |
| 120, 168, 496, 539 | liu2025marsm : MARS-M, reduction de variance a moment pour Muon | Resume : "MARS-M ... integrates MARS-style variance reduction with Muon" ; MARS derive de STORM (moment recursif). | OK |
| 206, 169 | lau2025polargrad : PolarGrad = pas a longueur nucleaire ; planchers des longueurs fixes | Section 3.3 : X_{k+1} = X_k - gamma_k tr(H_k) U_k, tr(H_k) = norme nucleaire ; Theoreme 3.4 : "if the learning rate is constant, convergence plateaus at a floor" pour matrix sign descent. | OK (le pas de Lau et al. est 1/(L r), pas 1/Lop : sans consequence) |
| 174-175, 657-658 | ma2026preconditioning : schedules geometriques pour Muon et la descente spectrale | Muon simplifie (sans moment) X - eta_t msign(grad f), eta_t = C lambda_max rho^t, 1/2 <= rho < 1, convergence lineaire (factorisation de matrices, apprentissage en contexte) ; "the length of each movement must decrease geometrically". | OK |
| 659-660 | zhu2026short : "a Euclidean staleness lemma for SAG and SAGA tables" | Lemme 3.1 (Bounded Staleness), Lemme 3.3 (Gradient Error) : \|e_k\|^2 <= 4 L^2 tau alpha^2 U_k pour SAG et SAGA. | OK |
| 247 | jiang2026adaptive : Phi_xi dans le sous-differentiel de psi_xi | Section 4.1, Eq. (6) : Psi(S;L) = E\|S + LZ\|_*, gradient = E[argmax_{\|X\|op<=1} <S+LZ, X>] = E polar(S + LZ) (bruit gaussien). | OK (le cas d'un bruit quelconque est l'extension du papier) |
| 117-118, 155 | bernstein2024old : Muon sans moment = descente la plus raide pour la norme spectrale | Proposition 5 : "Shampoo [sans accumulation] as steepest descent under the spectral norm", mise a jour U V^T. Le mot "Muon" n'apparait pas dans v2 ; le lien avec Muon est fait dans le billet jordan2024muon. | OK l. 117-118 ; nit l. 155 (F5) |
| 116, 513-514 | jordan2024muon : cinq iterations de Newton-Schulz, moment de Nesterov, constante 1e-7 | Billet : `def newtonschulz5(G, steps=5, eps=1e-7)`, `X /= (X.norm() + eps)` ; "using Nesterov-style momentum ... the default in the public Muon implementation". | OK |
| 102-103, 131-132, 506-515, 520, 678-679 | jordan2024muon : "Muon's own recipe" = pas de longueur fixe sur un refroidissement GEOMETRIQUE | aucun schedule dans le billet ; refroidissement LINEAIRE dans le code de reference (voir section 4). | probleme (F1, major) |

## 4. Point supplementaire : quel schedule utilisent les executions de reference de Muon ?

Sources ouvertes (toutes liees depuis le billet ou depuis le README du depot de reference).

1. Billet (https://kellerjordan.github.io/posts/muon/). La definition donne la mise a jour avec un taux eta constant ;
   le texte ne contient aucune occurrence de "learning rate", "schedule", "decay", "cooldown", "warmup" (recherche sur
   le HTML complet). Le billet ne prescrit donc AUCUN schedule.
2. Record NanoGPT lie par le billet ("An example usage in the current NanoGPT speedrun record") :
   `KellerJordan/modded-nanogpt`, commit 973030408364f8738b4ad9e8f912d8cbbf56e4d4, `train_gpt2.py`, lignes 370-372 et 458-471 :

       num_iterations : int = 1530 # number of iterations to run
       warmup_iters : int = 0
       cooldown_iters : int = 600 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
       ...
       # learning rate decay scheduler (linear warmup and cooldown)
       def get_lr(it):
           ...
           # 2) constant lr for a while
           elif it < args.num_iterations - args.cooldown_iters:
               return 1.0
           # 3) linear cooldown
           else:
               decay_ratio = (args.num_iterations - it) / args.cooldown_iters
               return decay_ratio

   Soit : constant pendant 930 iterations, puis LINEAIRE VERS ZERO sur les 600 dernieres (schedule trapezoidal).
   Meme fichier, l. 455 : `Muon(matrix_params, lr=0.05, momentum=0.95)` ; l. 82 : `nesterov=True` par defaut ;
   l. 545-547 : echauffement du moment de 0.85 a 0.95 sur 300 pas.
3. CIFAR-10 lie par le billet : `KellerJordan/cifar10-airbench`, commit 0e6f9614572d7e8e3c259905aebc7196f91d5d79,
   `research/clean_muon.py`, lignes 224-226 :

       def get_lr(step):
           total_train_steps = epochs * len(train_loader)
           return 1 - step / total_train_steps

   Soit : LINEAIRE VERS ZERO des le premier pas. (`Muon(filter_params, lr=0.24, momentum=0.6)`.)
4. Exemples lies par le README actuel du depot `KellerJordan/Muon` : record
   `records/track_1_short/2025-05-25_MuonWithAuxAdamExample`, lignes 482-490 : "learning rate schedule: stable then
   decay", constant sur 60 % puis lineaire de 1.0 a 0.1 (`cooldown_frac = 0.4`) ; `airbench94_muon.py` l. 401-403 :
   `group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)` (lineaire vers zero). README : "the default
   values of momentum (0.95), nesterov (True), and ns_steps (5) work well. Only the learning rate and weight decay
   must be tuned."

Conclusion. Les executions de reference de Muon utilisent un refroidissement LINEAIRE a horizon fini : constant puis
lineaire vers zero (NanoGPT, 2024), lineaire vers zero (CIFAR-10), constant puis lineaire vers 0.1 fois le taux
initial (records 2025). Aucune n'utilise un schedule geometrique. Sont fideles a Muon dans l'equation (4) du papier :
la longueur fixe du pas (lr fois une matrice orthogonalisee, independante de la norme du gradient), l'existence d'un
refroidissement du taux, les cinq pas de Newton-Schulz avec eps = 1e-7, le moment de Nesterov. Sont des choix du
papier : la FORME geometrique lr_t = lr_0 q^(tb/n) (motivee par la theorie des pas de longueur fixe : Goffin 1977,
Davis et al. 2018, Ma et al. 2026, que le papier cite), l'horizon infini, et le coefficient de moment 0.9 (defaut de
Muon : 0.95). L'expression "Muon's own recipe" n'est donc pas fidele pour la forme du schedule ; il faut le dire.

Le choix n'est pas neutre : controle `paper/audit/checks/cite-muon_linear_cooldown.py` (code du depot en lecture seule,
instance ls-cond de M9 : n = 200, 8 x 40, batch un, facteur polaire exact, table SAGA, 300 passes, graines 10-12),
distance carree relative finale, mediane [min, max] :

    geometrique q=0.8, lr0=0.3 (point du papier, Fig. 4a)        2.37e-29 [1.43e-29, 3.79e-29]
    lineaire vers zero, lr0=0.03                                  1.62e-06 [1.41e-06, 1.80e-06]
    lineaire vers zero, lr0=0.1                                   1.67e-05 [1.44e-05, 2.29e-05]
    lineaire vers zero, lr0=0.3                                   1.49e-04 [1.44e-04, 1.58e-04]
    lineaire vers zero, lr0=1                                     1.52e-03 [1.50e-03, 1.56e-03]
    constant puis lineaire vers zero (39 % finaux), lr0=0.03      9.70e-06 [9.08e-06, 1.21e-05]
    constant puis lineaire vers zero (39 % finaux), lr0=0.1       1.20e-04 [1.15e-04, 1.22e-04]
    constant puis lineaire vers zero (39 % finaux), lr0=0.3       1.07e-03 [9.83e-04, 1.22e-03]

Le point geometrique reproduit le plancher d'arrondi du papier (2.3e-29, l. 535). Avec le schedule lineaire des
executions de reference, la meme table SAGA s'arrete entre 1e-6 et 1e-3 (proportionnel a lr0^2) : pres de la fin la
decroissance relative du taux par passe, n/(T - t), depasse toute valeur critique et l'iteration gele, comme le papier
le decrit pour un refroidissement trop rapide. Le resultat "down to rounding error" tient donc a la forme geometrique,
qui est le choix du papier, et non a la recette de reference de Muon. (Petite reproduction : 3 graines, une instance,
lr0 non regle finement ; elle tranche l'ordre de grandeur, pas une valeur.)

## 5. Constats

### F1 = CM-1-recipe-schedule (major, a la limite du blocker) : "Muon's own recipe" attribue a Muon un refroidissement geometrique
- Lignes : 102-103 (resume), 131-132, 173 (Table 1), 506-515 (titre de 3.3, Eq. (4)), 520 (legende Fig. 4), 678-679
  (conclusion) ; aussi l. 4 (commentaire) et paper/README.md l. 11.
- Phrases : "\emph{Muon's own recipe.} Steps of fixed length on a geometric cooldown of the learning rate, fed by a
  SAGA table, follow the schedule down to rounding error" ; "Muon does not shrink its steps with the estimator: it
  keeps steps of fixed length and cools the learning rate down. We feed that recipe with the SAGA estimator, ...
  lr_t = lr_0 q^{tb/n}" ; "with or without Nesterov momentum 0.9 (\emph{Muon's recipe}: ns5 with momentum)" ;
  "Muon's own recipe, fixed lengths on a geometric cooldown, converges to rounding error".
- Probleme : la source citee (billet et code de reference) ne contient aucun schedule geometrique ; les executions de
  reference refroidissent LINEAIREMENT (section 4). La forme geometrique et le coefficient 0.9 sont des choix du
  papier, non signales. Et le resultat annonce depend de ce choix : avec le refroidissement lineaire de reference la
  table SAGA s'arrete a 1e-6 - 1e-3 en 300 passes (controle ci-dessus), pas a l'arrondi.
- Correction proposee : voir cite-muon.json (resume, section 3.3, conclusion). Confiance : haute.

### F2 = CM-2-ray-factor4 (major) : "factor 4 on the ray that attracts generic runs [shulgin2026quadratic]"
- Lignes 313-315 : "On least squares the curvature is constant and the gap is about a factor $4$ on the ray that
  attracts generic runs \cite{shulgin2026quadratic}; a large gap needs a global constant far below the curvature met,
  as for logistic and softmax losses (not measured for the polar step)."
- Probleme (a), attribution : Shulgin et al. donnent le taux (1-2w)^2 sur la famille a deux amas du cas commutant
  (Corollaire 5, annexe E.2), d'ou le facteur 4 sur ce rayon. Ils ne disent pas que les trajectoires generiques y
  sont attirees ; ils observent seulement que sous recherche lineaire "p(X_k) decays to 0.01 within 150 iterations".
  Jawanpuria et al. (annexe C) ecrivent : "we do not know its rate in closed form ... the measured asymptotic factor
  ... varies with the initialization". L'attraction est une observation numerique de l'etude elle-meme, que ses notes
  classent [N], "not proved", et posent en question ouverte (archive/muon/notes_spectral_theory.tex l. 800-801, 829,
  1210-1214 : "Is the flattest ray attracting for the non-commuting iteration, and why?" ; les rayons y sont
  "linearly unstable for the commuting map").
- Probleme (b), portee : le facteur 4 ne vaut que dans la queue. Controle `cite-muon_ray_factor4.py` (K x d = 20x20,
  40x20, 100x100, spectres k^-1 et k^-2, departs gaussiens, pas nucleaire avec L = Lop) : exposant de queue
  4.02 a 4.23 fois le certifie (rayon : 4.01 a 4.06), ce qui confirme l'observation ; mais l'exposant moyen jusqu'a
  F/F_0 = 1e-2, 1e-4, 1e-6, 1e-8 vaut 55-420, 13-60, 8-13 et 7-9 fois le certifie, et 107 a 1750 fois au premier pas.
  Sur moindres carres, a courbure constante, l'ecart est donc grand tant que l'erreur est repartie sur plusieurs
  directions : la source citee le dit ("Row-isotropic iterates give p(X) = 1", convergence en un pas, Theoreme 3 ;
  le pas regle "keeps p(X_k) near 0.6"). La phrase "a large gap needs a global constant far below the curvature met"
  est contredite par ce cas : muPL/muop depend de la forme de l'erreur, pas seulement de la courbure.
- Correction proposee : voir JSON. Confiance : haute pour (a), moyenne-haute pour (b).

### F3 = CM-3-lemma4-floor (minor) : plancher muop eta^2/8 attribue au Lemme 4 de Jawanpuria et al.
- Ligne 491-492. Le Lemme 4 enonce max{\|E_t\|_F, \|E_{t+1}\|_F} >= alpha/2 pour la quadratique f = 1/2 tr(X^T H X) ;
  le plancher en valeur de fonction, pour F quelconque et un estimateur quelconque, est le corollaire du papier
  (inegalite triangulaire en norme spectrale + forte convexite spectrale). Exact comme calcul, inexact comme renvoi.

### F4 = CM-4-eq17-scope (nit) : l. 231, les deux equations citees sont etablies pour la quadratique unilaterale f = 1/2 tr(X^T H X)
(Eq. (17) de Shulgin et al. en rang plein, ou J = tr(H)). L'identite (2) du papier, pour un Hessien general, en est
l'extension immediate ; un "for one-sided quadratics" suffirait.

### F5 = CM-5-bernstein-muon (nit) : l. 155, "Muon as spectral steepest descent [bernstein2024old]"
L'anthologie (v2) ne nomme pas Muon : sa Proposition 5 concerne Shampoo sans accumulation. Le lien avec Muon est dans
jordan2024muon. Citer les deux.

### F6 = CM-6-hird-bib (nit) : .bib, hird2025quantifying : ajouter `number = {119}` ; la "Proposition 7" est celle de la version JMLR
(Proposition 1 dans arXiv v2, que la note du .bib donne aussi).

### F7 = CM-7-li-version (nit) : l. 445, "Theorem 4.10" est la numerotation de arXiv v2 (16 juin 2026) ; c'est le Theoreme 4.9 dans v1.
Le theoreme porte sur SignSGD et Normalized-SGD par echantillon, sur un jeu de donnees particulier ("Orthogonal
Scale-Skewed") ; le cas spectral vient du Lemme H.1. Preciser la version dans le .bib.

### F8 = CM-8-do-almost-every-batch (nit) : l. 466, do2026muon : la non-convergence est prouvee "for almost all M" (presque toute taille de batch).

## 6. Non verifie
- Lieux de publication definitifs des preprints 2025-2026 (le .bib les donne comme arXiv, ce qui est exact ; zhu2026short
  est annonce a ICML 2026 et paquette2026phases porte l'en-tete NeurIPS 2026 : non verifie sur les sites des conferences).
- Versions arXiv anterieures a celle consultee pour shulgin, jawanpuria, huang, do, xie, chen, ma, jiang (une seule
  version existe a ce jour : v1) ; pour lau (v4), liu (v3), vasudeva (v3), paquette (v2), zhu (v2), seule la derniere
  version a ete lue, le papier n'y renvoyant a aucun numero.
- La figure de definition de Muon dans le billet est une image (non lue) ; le code du billet et des depots a ete lu.
- Les affirmations "not found" (nouveaute) relevent de la dimension novelty, pas de celle-ci.
