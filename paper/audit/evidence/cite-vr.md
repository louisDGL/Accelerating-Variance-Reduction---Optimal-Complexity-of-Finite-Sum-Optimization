# Audit cite-vr : bibliographie, premier groupe (variance reduite, sommes finies, theorie classique)

Statut : TERMINE (blocs 1 et 2). 21 cles controlees ; 6 constats (0 blocker, 0 major, 4 minor, 2 nit).

## Perimetre

21 cles : allenzhu2018katyusha arjevani2016lower bach2024learning defazio2014finito defazio2014saga dragomir2021fast
driggs2019biased gonen2016solving hofmann2015variance johnson2013accelerating kovalev2020dont lan2018optimal
mairal2015incremental mokhtari2018surpassing na2023hessian park2020linear qin2023convergence xiao2014proximal yang2014data
ying2017variance zhang2013gradient.

Pour chaque cle : (1) exactitude de l'entree .bib (auteurs, titre, annee, lieu, arXiv/DOI) ; (2) chaque phrase du papier
qui la cite (grep de la cle dans paper/paper_saga_to_muon.tex).

Occurrences (lignes du .tex, outil Read) :
- allenzhu2018katyusha 115, 655 ; arjevani2016lower 324 ; bach2024learning 114, 147, 192 (+ "Proposition 5.9" 210, 303, 306, 325)
- defazio2014finito 607 ; defazio2014saga 113, 147 ; dragomir2021fast 153, 326 ; driggs2019biased 259
- gonen2016solving 154, 327 ; hofmann2015variance 319 ; johnson2013accelerating 113 ; kovalev2020dont 115, 253, 305
- lan2018optimal 115 ; mairal2015incremental 170, 256 ; mokhtari2018surpassing 177, 609 ; na2023hessian 298
- park2020linear 608 ; qin2023convergence 175, 659 ; xiao2014proximal 259 ; yang2014data 154, 240, 327, 655
- ying2017variance 177, 608 ; zhang2013gradient 148, 214

Sources ouvertes (texte telecharge et lu, pdftotext, dans le scratchpad de la session, hors depot) : PDF arXiv de
1611.00347, 1506.03662, 1303.4645, 2204.09266, 2104.09813, 1408.3115, 1602.02350, 2310.15976, 1906.01133, 1402.4419,
1603.05953, 1708.01383, 1810.11167, 1403.4699, 1901.08689, 1507.02000 ; PDF JMLR 15-106 (Arjevani et al.) ; PDF PMLR
v32 defazio14 (Finito) ; PDF en ligne du livre de Bach (https://www.di.ens.fr/~fbach/ltfp_book.pdf, version du
26 mai 2025). Metadonnees : API arXiv (export.arxiv.org, 19 identifiants), API Crossref (8 DOI), balises citation_* des
pages PMLR (v117 kovalev20a, v32 defazio14, v48 gonen16, v139 dragomir21a), JMLR (v17 15-106, v18 16-410) et NeurIPS
(Hofmann 2015, Johnson-Zhang 2013).

## 1. Entrees .bib (metadonnees)

| cle | verdict | preuve |
|---|---|---|
| allenzhu2018katyusha | OK | JMLR v18 16-410 : titre, auteur, vol 18, no 221, p. 1-51, 2018 ; Crossref 10.1145/3055399.3055448 : STOC 2017, p. 1200-1205 ; arXiv 1603.05953 (v6 : "Appeared in STOC 2017 and JMLR 2018"). |
| arjevani2016lower | OK | JMLR v17 15-106 : "On Lower and Upper Bounds in Smooth and Strongly Convex Optimization", 3 auteurs, vol 17, no 126, p. 1-51, 2016. (Le titre arXiv 1503.06833 differe : "... for Smooth and Strongly Convex Optimization Problems" ; le .bib suit JMLR, correct.) |
| bach2024learning | OK (sous reserve) | Livre MIT Press ; version en ligne de l'auteur (26 mai 2025) : "Copyright ... licensed exclusively to The MIT Press ... releasing the final version to the public in 2024". Edition imprimee non ouverte. |
| defazio2014finito | OK | PDF PMLR v32 p. 1125-1133 : auteurs "Aaron J. Defazio, Tiberio S. Caetano, Justin Domke" (ordre du .bib ; la balise PMLR liste Domke en 2e, le PDF fait foi) ; arXiv 1407.2710. |
| defazio2014saga | OK | arXiv 1407.0202 v3 (commentaire : NIPS 2014) ; titre et auteurs identiques. Pages (1646-1654) non donnees : pas une erreur. |
| dragomir2021fast | OK | PMLR v139 dragomir21a : titre, 3 auteurs, p. 2815-2825, ICML 2021 ; arXiv 2104.09813. |
| driggs2019biased | constat CV-3 | arXiv 1906.01133 v2 (27 fev. 2020, commentaire "journal version") : titre et auteurs exacts. Mais l'article est publie : https://jmlr.org/papers/v23/20-316.html, balises citation_* : "Journal of Machine Learning Research", volume 23, issue 24, p. 1-43, 2022. Le .bib ne donne que le preprint. |
| gonen2016solving | OK | PMLR v48 gonen16 : titre, 3 auteurs, p. 1397-1405 ; arXiv 1602.02350. |
| hofmann2015variance | OK | NeurIPS 2015 (vol. 28) : titre, 4 auteurs ; Metadata.json NeurIPS : page_first 2305, page_last 2313 ; arXiv 1506.03662 v4. |
| johnson2013accelerating | OK | NeurIPS 2013 (vol. 26) : titre, 2 auteurs ; Metadata.json NeurIPS : page_first 315, page_last 323. |
| kovalev2020dont | OK | PMLR v117 kovalev20a : titre, 3 auteurs, p. 451-467, ALT 2020 ; arXiv 1901.08689. |
| lan2018optimal | OK | Crossref 10.1007/s10107-017-1173-0 : Lan, Zhou, Math. Program. 171(1-2):167-215 (impression sept. 2018). |
| mairal2015incremental | OK | Crossref 10.1137/140957639 : SIAM J. Optim. 25(2):829-855, 2015 ; arXiv 1402.4419. |
| mokhtari2018surpassing | OK | Crossref 10.1137/16M1101702 : SIAM J. Optim. 28(2):1420-1447, 2018 ; arXiv 1611.00347. |
| na2023hessian | OK | Crossref 10.1007/s10107-022-01913-5 : Math. Program. 201(1-2):473-520 (impression sept. 2023) ; arXiv 2204.09266. |
| park2020linear | OK | Crossref 10.1007/s11590-019-01520-y : Optim. Lett. 14(6):1583-1598, 2020 ; arXiv 1810.11167. |
| qin2023convergence | OK (preprint) | arXiv 2310.15976 : titre et 3 auteurs exacts ; v1 du 24 oct. 2023 (annee du .bib), v4 du 11 aout 2026 ; aucune version publiee trouvee (Crossref, recherche bibliographique). |
| xiao2014proximal | OK | Crossref 10.1137/140961791 : SIAM J. Optim. 24(4):2057-2075, 2014 ; arXiv 1403.4699. (DOI absent du .bib : nit.) |
| yang2014data | constat CV-3 | arXiv 1408.3115 v4 (25 sept. 2015) : titre et 4 auteurs exacts. Mais l'article est publie : Crossref 10.1007/s10994-015-5536-6, "Machine Learning" 103(1):57-79 (en ligne 20 oct. 2015, numero de 2016). Le .bib ne donne que le preprint. |
| ying2017variance | OK | Crossref 10.1109/TSP.2020.2968280 : IEEE TSP 68:1390-1408, 2020 ; arXiv 1708.01383. |
| zhang2013gradient | OK | arXiv 1303.4645 v2 : titre et auteurs exacts. https://ww3.math.ucla.edu/camreport/cam13-17.pdf est bien ce rapport (page de titre : meme titre, Hui Zhang, Wotao Yin, 20 mars 2013) ; aussi Rice CAAM TR13-04 (Optimization Online 2013/03/3805). Aucune version en revue trouvee. |

## 2. Renvois precis

### arjevani2016lower, "Theorem 8" (l. 321-324) : OK
JMLR 17 (2016), p. 23 : "Theorem 8 Let A be a consistent p-SCLI optimization algorithm for L-smooth mu-strongly convex
functions over R^d. If the inversion matrix of A is diagonal, then there exists a quadratic function f_{A,b}(x) in
Q_d([mu, L]) such that IC_A(eps, f_{A,b}(x)) = Omega~((kappa^{1/p} - 1)/2 ln(1/eps)), where kappa = L/mu." Definition 1
(p. 9) : p-SCLI = "p-stationary canonical linear iterative" ; p = "lifting factor" (nombre de points precedents). Pour
p = 1 : ordre (kappa - 1)/2 ln(1/eps), atteint par la descente de gradient (section 4.2 : "for p = 1 ... the lower bound
stated in Theorem 8 is simply attained by FGD"). "the order of one-step stationary methods" = 1-SCLI : fidele. La
moyenne de SAGA est la descente de gradient (matrice d'inversion scalaire) : le renvoi est correct comme comparaison
d'ordre. (La validite de la borne (kappa-1)/4 log(1/(2 eps)) elle-meme releve de math-saga.)

### mokhtari2018surpassing, "kappa < 4.08" (l. 607-609) et Table 1 (l. 177) : constats CV-1, CV-2
- Corollaire 3 (arXiv v2, p. 10) : "the error of the proposed DIAG method after m > 1 passes ... is bounded above by
  ||x^{n(m-1)+1} - x*|| <= rho^m [1 - (1-rho)/n x min{1, (n-1)/2}] ||x0 - x*||", rho = (kappa-1)/(kappa+1), kappa = L/mu,
  pas 2/(mu+L), "cyclic scheme". Hypothese 1 : chaque f_i mu-fortement convexe, gradient L-lipschitzien.
- Au carre, rho^2 < e^{-1} ssi kappa < (sqrt(e)+1)/(sqrt(e)-1) = 4.083 : arithmetique exacte (deja verifiee par
  math-order). Mais le texte de la source ne contient ni e^{-1}, ni 4.08, ni comparaison avec l'echantillonnage i.i.d. :
  grep de "e^{-1}", "exp(-1)", "4.08", "nat" dans le texte : rien. La note B4 du depot le dit (theory/B4_sampling_order.tex
  l. 515 : "a comparison that these papers do not make"). Le papier, lui, attribue le seuil a la source (constat CV-1).
- DIAG est une methode cyclique (ordre fixe), pas remaniee : Table 1 (l. 177) "reshuffled SAGA-type methods
  \cite{ying2017variance,mokhtari2018surpassing}" range a tort la source parmi les methodes remaniees (constat CV-2).

### hofmann2015variance, "the step of" (C4), l. 319 : OK
arXiv 1506.03662 v4, Theoreme 1 : "For any step size gamma = a/(4L) with a < 1, the algorithm converges at a geometric
rate of at least (1 - rho(gamma)) with rho(gamma) = (q/n)(1-a)/(1-a/2) ... if gamma >= gamma*(K), otherwise rho(gamma) =
mu gamma, where gamma*(K) := a*(K)/(4L), a*(K) := 2K/(1 + K + sqrt(1 + K^2)), K := 4qL/(n mu) = (4q/n) kappa" ;
Corollaire 1 : "In Theorem 1, rho is maximized for gamma = gamma*(K)". SAGA : q = 1, K = 4 kappa/n. L'egalite de ce pas
avec le meilleur pas de la preuve de la Proposition 5.9 a ete verifiee algebriquement et par force brute par math-saga
(evidence/math-saga.md, l. 115-127) ; je n'ai controle que le texte de la source : conforme.

### bach2024learning, Proposition 5.9 et section 5.4.4 : OK (version en ligne)
PDF en ligne du 26 mai 2025, p. 146-149. Section "5.4.4 Variance Reduction" : "each f_i is R^2-smooth ... F is
mu-strongly convex ... kappa = R^2/mu". "Proposition 5.9 (Convergence of SAGA) If initializing with z_i = f_i'(theta_0)
... for the choice of step size gamma = 1/(4R^2), E||theta_t - eta*||^2 <= (1 - min{1/(3n), 3mu/(16R^2)})^t (1 + n/4)
||theta_0 - eta*||^2 (5.30)". Preuve : Lyapunov ||theta - eta*||^2 + Delta sum_i ||f_i'(eta*) - z_i||^2, "With Delta =
3 gamma^2 and gamma = 1/(4R^2)", contraction par 1 - min{1/(3n), 3mu/(16R^2)} ; inegalites utilisees dans l'ordre :
"||a + b||^2 <= 2||a||^2 + 2||b||^2" (I1), "co-coercivity of all functions f_i (see proposition 5.4)" (I2), "the
inequality between variance and second moment" (I3), "(theta - eta*)^T F'(theta) >= mu ||theta - eta*||^2 as a
consequence of strong convexity" (I4), le min des deux taux (I5), l'initialisation "(1 + 3n/16) <= (1 + n/4)" (I6). Les
l. 113-114, 147, 192-194, 210-212 sont fideles (la contraction de V est le contenu de la preuve, l'enonce porte sur
E||theta_t - eta*||^2 : deja note par math-saga, MS-4, pas une erreur). Reserve : l'edition imprimee MIT Press 2024 n'a
pas ete ouverte ; le numero 5.9 est celui de la version en ligne.

### zhang2013gradient, "restricted-secant constant" (l. 213-214) et "one-point curvature conditions" (l. 148) : OK, nit CV-5
arXiv v2, Definition 2 : "Restricted secant inequality - RSI(nu). A function f satisfies the restricted secant
inequality (RSI) with constant nu > 0 if ... <grad f(x), x - x_prj> >= nu ||x - x_prj||^2 ... where x_prj = Proj_{X*}(x)".
Suite : "Constant nu can be viewed as a lower bound of the average curvature". Le nom et l'objet sont les bons. Nuance :
mu_eff(theta) du papier est le rapport ponctuel, dont l'infimum est la meilleure constante RSI (nit CV-5).

### na2023hessian, "the designs of" (l. 297-298) : OK
arXiv 2204.09266 v2, Section 5 : "For low coherence, we generate a n x d random matrix with each entry ... N(0,1), and
let U be the left singular vectors of that matrix. For high coherence, we divide each row of U by sqrt(z_i), where z_i
is independently sampled from Gamma distribution with shape 0.5 and scale 2 ... we vary kappa_A = {d^0.5, d, d^1.5}.
For each kappa_A, we let singular values in Sigma vary from 1 to kappa_A with equal spacing ... A = U Sigma ... x ~
N(0, 1/d I), and the response b_i with P(b_i = 1) = 1/(1 + exp(-a_i^T x))". Le portage du depot
(vrlab/problems.py, make_logistic_ndm, l. 258-298) reproduit ces deux familles avec kappa_A in {d^0.5, d} (n = 1000,
d = 50, ridge choisi pour fixer kappa) : "designs of" est exact. (Deux des huit jeux ont des etiquettes modifiees,
variante "labels" de experiments/exp_B3_families.py : le plan des variables est celui de Na et al. ; le decompte "eight"
releve de num-f1.)

### dragomir2021fast, "SAGA in a fixed metric" (l. 153) et "Proposition 5.9 holds in any fixed metric" (l. 325-326) : OK, nit CV-6
arXiv 2104.09813, Section 4 : "Algorithm 1, which is a Bregman adaptation of the SAGA algorithm" ; Assumption 3 : "For
all i, f_i is L_{f/h} relatively smooth w.r.t. h, and f is mu_{f/h} relatively strongly convex w.r.t. h" ; "Corollary 1.
If nabla^2 h is constant (h is quadratic), then Assumption 3 is satisfied with G = 1, so that E[psi_t] <= (1 - min(1/(8
kappa_{f/h}), 1/(2n)))^t psi_0" ; "the rate recovers that of Euclidean SAGA in case h is a quadratic form". Pour h =
1/2 ||.||_P^2, L_{f/h} et mu_{f/h} sont les constantes de la metrique P : "SAGA in a fixed metric" est exact. La source
certifie SAGA en metrique fixe avec ses propres constantes (1/8, 1/2), pas la Proposition 5.9 elle-meme ; le transfert
de la Proposition 5.9 est un changement de variable (verifie par math-saga). Nit CV-6.

### yang2014data et gonen2016solving, "preconditioning where the loss curvature is bounded below" (l. 154, 327) : OK ; metrique de covariance (l. 240) : constat CV-4
- Yang et al., arXiv v4, Assumption 1 : "The scalar loss function l(z, y) w.r.t z satisfies l''(z, y) >= beta for
  |z| <= r and beta > 0" ; preconditionneur : "H = rho I + C, where rho = lambda/beta", "P^{-1} = H^{-1/2}" ;
  experiences : SAG et SVRG sur les donnees preconditionnees (preconditionneur calcule une fois, eventuellement par
  sous-echantillonnage) : l. 655 "with a preconditioner computed once \cite{yang2014data}" exact.
- Gonen et al. : "We develop a novel preconditioning method for ridge regression ... By equipping SVRG with this
  preconditioning process" (perte quadratique, courbure 1) : exact.
- l. 240 : la metrique omega -> omega(c_bar Sigma_hat + lambda I)^{-1} attribuee a Yang et al. emploie c_bar, non defini
  dans le papier ; dans le code (muon/lab/vrmuon.py l. 641-649) c'est la constante de regularite c_l de la perte
  ("model Hessian c_l Sigma_hat + lam I"), une borne SUPERIEURE de la courbure, alors que Yang et al. prennent la borne
  INFERIEURE beta (H proportionnel a beta C + lambda I). Constat CV-4 (minor).


### Bloc 2 : autres attributions

- **qin2023convergence**, Table 1 l. 175 "SVRG anchors under reshuffling, without linear rate" et l. 658-660 "descent
  inequalities with SVRG anchors in sign ... steps under reshuffling" : OK. arXiv v4, Algorithm 2 "Sign-based random
  reshuffling with variance reduction (SignRVR)" : "v_t^i := grad f_{pi_i^t}(x_t^i) - grad f_{pi_i^t}(y_t) + grad f(y_t),
  y_t := x_t" (ancre SVRG en debut d'epoque, permutation uniforme a chaque epoque) ; Theoreme 4.1 : inegalite de
  descente pathwise, borne de stationnarite min E||grad f||_1 (cadre non convexe, pas de taux lineaire). SignRVR est deja
  dans la v1 du 24 oct. 2023 (resume v1 via l'API arXiv : "We then present SignRVR and SignRVM"), annee du .bib coherente.
- **ying2017variance**, l. 177 et 608 "linear rates are proved for reshuffled ... orders" : OK. Resume : "provides the
  first theoretical guarantee of linear convergence under random reshuffling for SAGA" ; "Theorem 1 (Linear convergence
  of SAGA)".
- **park2020linear**, l. 608 "cyclic orders" : OK. Resume : "We prove C-SAGA converges linearly under the standard
  assumptions of strong convexity" ; C-SAGA = "cyclic variant of SAGA".
- **defazio2014finito**, l. 606-607 "A permutation order was seen to help SAGA-type methods with Finito" : OK. PDF PMLR,
  section 3 : "Sampling without replacement is much faster ... the permuted case ... this approach does not give any
  speedup with SAG, however it works spectacularly well with Finito. We see speedups of up to a factor of two ... we
  have no theory to support this case". Recoupe par le papier SAGA (arXiv 1407.0202) : "The big advantage of
  Finito/MISOmu is that when using a per-pass re-permuted access ordering, empirical speed-ups of up-to a factor of 2x
  has been observed."
- **driggs2019biased**, l. 259 "damped SAGA (B-SAGA)" : OK. arXiv v2 : "Biased SAGA (B-SAGA) ... For theta > 0 ...
  grad~_k^{B-SAGA} = (1/theta)(grad f_{j_k}(x_k) - grad f_{j_k}(phi_{j_k}^k)) + (1/n) sum_i grad f_i(phi_i^k) ... When
  theta = n, grad~^{B-SAGA} recovers the SAG gradient estimator" : innovation amortie par vartheta = 1/theta, comme
  l. 254-255.
- **xiao2014proximal**, l. 259 "importance sampling" : OK. arXiv 1403.4699 : "When the sampling probabilities for f_i
  are proportional to their Lipschitz constants ... O((n + L_avg/mu) log(1/eps))" ; conclusion : "a weighted sampling
  scheme".
- **mairal2015incremental**, l. 170 "MISO" et l. 255-256 "gamma(m + vartheta c) with vartheta = 1/n, gamma = 1/lambda is
  MISO" : OK. arXiv 1402.4419 v3, eq. (3.8) : surrogates "g_n^t : theta -> f^t(kappa) + grad f^t(kappa)^T(theta - kappa)
  + (mu/2)||theta - kappa||^2, which are lower bounds of the functions f^t ... performing the update (3.6) when replacing
  L by mu", Proposition 3.4 (convergence si T >= 2L/mu, "may diverge if T is small"). Calcul : avec f_i = g_i +
  (lambda/2)||.||^2, mu = lambda, le minimiseur de la moyenne des surrogates est theta = -(1/lambda) zbar (zbar moyenne de
  la table des grad g_i) ; apres rafraichissement de l'indice j : theta+ = -(1/lambda)(zbar + (grad g_j(theta) - z_j)/n),
  et theta - (1/lambda)(m + c/n) avec m = zbar + lambda theta donne la meme expression. L'identification est exacte ; la
  condition T >= 2L/mu de la source s'accorde avec "fails where MISO fails (kappa >> n)" (l. 503).
  Controle numerique : paper/audit/checks/cite-vr_miso_and_diag.py (5 runs x 200 pas, logistique ridge, n = 20,
  d = 5) : ecart max entre la mise a jour MISO et la forme du papier 4.4e-16 ; seuil DIAG (sqrt e + 1)/(sqrt e - 1) =
  4.08299.
- **allenzhu2018katyusha**, l. 115 "(n + sqrt(n kappa)) log(1/eps)" : OK (arXiv v6, Theorem 2.1 : "O((n + sqrt(n kappa))
  log(1/eps)) stochastic gradient iterations") ; l. 655 "finite-sum estimators ... in general norms" : OK, section 6
  "Katyusha in the Non-Euclidean Norm Setting" : "smoothness (and strongly convexity) with respect to an arbitrary norm
  ... Bregman divergence".
- **kovalev2020dont**, l. 115 : OK ("L-Katyusha ... E Psi^k <= eps Psi^0 as long as k = O((n + sqrt(nL/mu)) log 1/eps)") ;
  l. 253 "loopless SVRG" et l. 305 "loopless Katyusha" : OK (resume : "we design loopless variants of both of these
  methods").
- **lan2018optimal**, l. 115 : OK (RPDG, "O((m + sqrt(mL/mu)) log 1/eps)" et borne inferieure correspondante (1.11)).
- **johnson2013accelerating** (SVRG, l. 113) et **defazio2014saga** (l. 113, 147 "Lyapunov proof") : OK. SAGA arXiv
  1407.0202, "Theorem 1. With x* the optimal solution, define the Lyapunov function T as ...".

## Constats

### CV-1 (minor, confiance haute) l. 607-609 : seuil kappa < 4.08 attribue a la source
Phrase : "linear rates are proved for reshuffled and cyclic orders \cite{ying2017variance,park2020linear}, some beating
$e^{-1}$ per pass for $\kappa<4.08$ \cite{mokhtari2018surpassing}."
Probleme : Mokhtari et al. prouvent ||x^{n(m-1)+1} - x*|| <= rho^m (...) ||x0 - x*||, rho = (kappa-1)/(kappa+1)
(Corollaire 3, cyclique). Le seuil 4.08 et la comparaison au facteur memoire e^{-1} sont un calcul des auteurs
(rho^2 < e^{-1} ssi kappa < (sqrt e + 1)/(sqrt e - 1) = 4.083), que la source ne fait pas (aucune occurrence de e^{-1},
4.08 ou d'une comparaison avec l'echantillonnage i.i.d. dans le texte ; la note B4 du depot, l. 515, le dit
explicitement : "a comparison that these papers do not make"). "some" suggere en outre que plusieurs des taux prouves le
font ; parmi les sources citees, seule celle-ci (Ying et al. et Park-Ryu ont des taux 1 - O(1/kappa^2) par passe).
Correction : "linear rates are proved for reshuffled and cyclic orders \cite{ying2017variance,park2020linear}, and that
of cyclic DIAG, $\frac{\kappa-1}{\kappa+1}$ per pass on the distance \cite[Corollary~3]{mokhtari2018surpassing}, beats
$e^{-1}$ for $\kappa<4.08$ (our comparison)."

### CV-2 (minor, confiance haute) l. 177 : source cyclique classee "reshuffled"
Phrase (Table 1) : "reshuffled SAGA-type methods \cite{ying2017variance,mokhtari2018surpassing}".
Probleme : DIAG est une methode a ordre cyclique fixe ("The cyclic scheme is critical in proving the result in (21)" ;
"If we pick functions uniformly at random, as in MISO, it is not possible to write the expression in (21)"). Seul Ying et
al. traite le remaniement.
Correction : "reshuffled and cyclic SAGA-type methods \cite{ying2017variance,mokhtari2018surpassing}".

### CV-3 (minor, confiance haute) .bib : deux articles publies cites comme preprints
driggs2019biased : publie dans JMLR 23(24):1-43, 2022 (https://jmlr.org/papers/v23/20-316.html) ; yang2014data : publie
dans Machine Learning 103(1):57-79, 2016, doi 10.1007/s10994-015-5536-6 (Crossref). Le .bib ne donne que arXiv
1906.01133 et 1408.3115. Les autres entrees du perimetre citent la version publiee quand elle existe.
Correction (.bib) : driggs2019biased -> @article, journal = {Journal of Machine Learning Research}, volume = {23},
number = {24}, pages = {1--43}, year = {2022}, url = {https://jmlr.org/papers/v23/20-316.html}, eprint = {1906.01133} ;
yang2014data -> @article, journal = {Machine Learning}, volume = {103}, number = {1}, pages = {57--79}, year = {2016},
doi = {10.1007/s10994-015-5536-6}, eprint = {1408.3115}.

### CV-4 (minor, confiance haute) l. 240 : c_bar non defini ; differe du preconditionneur de la source
Phrase : "the feature-covariance metric $\omega\mapsto\omega(\bar c\hat\Sigma+\lambda I)^{-1}$ \cite{yang2014data}".
Probleme : c_bar n'est defini nulle part dans le papier (grep : une seule occurrence). Dans le code
(muon/lab/vrmuon.py l. 641-649, muon/lab/problems.py l. 319 et 415) c'est c_loss = 0.5 (softmax), 1.0 (moindres carres),
la borne superieure de la courbure (= c_i de la l. 202). Yang et al. (Assumption 1 et section 4) prennent H = C +
(lambda/beta) I, beta borne INFERIEURE de l'' : la metrique du papier est une variante de celle de la source (meme
famille, autre scalaire). Pour les moindres carres (beta = c = 1) les deux coincident.
Correction : "the feature-covariance metric $\omega\mapsto\omega(\bar c\hat\Sigma+\lambda I)^{-1}$, $\bar c=\max_ic_i$
(\cite{yang2014data} use a lower bound of the loss curvature in its place)".

### CV-5 (nit, confiance moyenne) l. 213-214 : "constant" pour un rapport ponctuel
Phrase : "$\muEff(\theta)=\inner{\theta-\etas}{F'(\theta)}/\norm{\theta-\etas}^2$ (the restricted-secant constant of
\cite{zhang2013gradient})". Zhang-Yin, Definition 2 : RSI(nu) avec une constante nu valable pour tout x ; mu_eff(theta)
est le rapport en un point, dont l'infimum est la meilleure constante RSI. Correction : "(the pointwise ratio of the
restricted secant inequality of \cite{zhang2013gradient})".

### CV-6 (nit, confiance moyenne) l. 325-326 : la source certifie une version de SAGA, pas la Proposition 5.9
Phrase : "Proposition~5.9 holds in any fixed metric $P$ with $R_P^2,\mu_P$ \cite{dragomir2021fast}". Dragomir et al.,
Corollaire 1 : Bregman-SAGA avec h quadratique, taux 1 - min(1/(8 kappa_{f/h}), 1/(2n)), constantes relatives =
celles de la metrique. La Proposition 5.9 en metrique P est un changement de variable (verifie par math-saga). Correction :
"Proposition~5.9 holds in any fixed metric $P$ with $R_P^2,\mu_P$ (a change of variables; cf.\
\cite[Corollary~1]{dragomir2021fast})".

## Non verifie

- Edition imprimee MIT Press (24 dec. 2024, ISBN 978-0-262-04944-3) du livre de Bach : seule la version en ligne du
  26 mai 2025 a ete ouverte ; le numero "Proposition 5.9" et "section 5.4.4" y sont conformes, l'imprime peut differer.
- Egalite algebrique entre le meilleur pas de la preuve de la Proposition 5.9 et gamma*(K) de Hofmann et al. : non
  refaite ici (faite par math-saga, check3) ; seul le texte de la source a ete controle.
- DBLP inaccessible (page anti-robot) : metadonnees recoupees par Crossref, PMLR, JMLR, NeurIPS et l'API arXiv a la place.
