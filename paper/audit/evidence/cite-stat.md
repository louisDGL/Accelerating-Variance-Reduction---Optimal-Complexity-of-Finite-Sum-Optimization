# Audit cite-stat : bibliographie, deuxieme groupe (signe, medianes, pas normalises, miroir, outils)

Statut : TERMINE (blocs 1 a 3). Synthese des constats en fin de fichier.

## Perimetre
Cles : balles2020geometry bernstein2019majority cardot2013efficient carlson2015stochastic chen2020distributed
chzhen2023signsvrg cornish1938moments cutkosky2019momentum davis2018subgradient duchi2012randomized goffin1977convergence
hu2017unified karimireddy2019error maddison2021dual minsker2015geometric sidford2018coordinate taylor2017smooth.
Pour chaque cle : (1) exactitude de l'entree .bib ; (2) exactitude de l'attribution dans chaque phrase qui la cite.
Sources ouvertes : PDF arXiv telecharges par WebFetch puis texte extrait par pdftotext (copies dans le scratchpad de la
session, non conservees dans le depot) ; pages arXiv abs ; pages editeur quand indique.

## Bloc 1 - renvois precis

### sidford2018coordinate (l. 354) - PROBLEME
Phrase : "the spectral geometry has its own cost, $\kop\ge\min(K,d)$ \cite{sidford2018coordinate}".
Source (arXiv:1808.01278v2, PDF) : la seule affirmation pertinente est l'Appendice A.1 "Folklore bound on size of
ell_inf-strongly-convex functions", Lemma A.1 : "Suppose psi is 1-strongly convex with respect to the ell_inf norm on
[-1,1]^n. Then, max psi(x) - min psi(x) >= n/2" (p. 57), annonce p. 4 : "it is a folklore result that any function
strongly-convex over [-1,1]^n in the ell_inf norm has range at least n/2". La source ne parle ni de norme spectrale, ni de
min(K,d), ni meme de nombre de conditionnement : c'est une borne sur l'amplitude (range) d'une fonction fortement convexe
en ell_inf. Un nombre de conditionnement en ell_inf >= n/4 s'en deduit (lissite en ell_inf sur la boite : amplitude
<= 2L), pas l'enonce matriciel. L'enonce du papier est vrai (controle checks/cite-stat_kop_minKd.py, voir bloc 3) mais
il n'est pas dans la source : attribution inexacte. L'ancien texte (archive/muon/paper_spectral_muon.tex:230) disait
correctement "the spectral form of a folklore fact on ell_inf \cite{sidford2018coordinate}".

### balles2020geometry (l. 415) - OK
Phrase : "(c): $2\muop(F(W)-F_\ast)\le\nnuc{\nabla F(W)}^2\le2\Lop(F(W)-F_\ast)$ \cite{balles2020geometry}".
Source (arXiv:2002.08056v1, PDF) : Definition 1 (Appendice A) "A function f satisfies the PL condition with constant mu
w.r.t. a norm if ||grad f(x)||_*^2 >= 2 mu (f(x) - f*)" suivie de "Note that strong convexity implies the PL condition"
(inegalite de gauche) ; Lemma 2 (Sect. 3.1.3), Eq. (12) : "f(x_{t+1}) <= f(x_t) - (1/2L)||grad f(x_t)||_*^2", d'ou
||grad f||_*^2 <= 2L(f - f*) en minorant f(x_{t+1}) par f* (inegalite de droite ; c'est l'Eq. (94) de la preuve de la
Proposition 8). Les deux inegalites y sont (la droite comme consequence immediate). Dual de la norme spectrale =
norme nucleaire : correct.

### maddison2021dual (l. 410) - OK
Phrase : "mirror descent in the space of gradients with mirror map $F^\ast$ (as in \cite{maddison2021dual})".
Source (arXiv:1902.02257v4, PDF), Section 5 : "The second interpretation is as a Bregman gradient method in the dual
space, which minimizes the designed k while the conjugate f* plays the role of the 'reference function'". L'identite
$D_{F^\ast}(G',G)=D_F(W,W')$ utilisee ensuite est le Lemma 3.2 (egalite pour f Legendre). Reserve (non bloquante, le
papier dit "as in") : l'analyse de Maddison et al. suppose k differentiable et la source precise "Our analysis does not
apply in the case of other norms or normalized steepest descent" ; ici k est la norme nucleaire (pas normalise). Le
papier ne pretend pas que l'analyse s'applique : OK.

### minsker2015geometric (l. 248-249) - OK
Phrase : "$\Med(\xi)=\arg\min_C\E[\nnuc{\xi-C}-\nnuc\xi]$ is the set of nuclear medians of $\xi$ \cite{minsker2015geometric}".
Source (arXiv:1308.1334, PDF), Section 2 : "Let X be a Banach space ... and let mu be a probability measure ... Define the
geometric median ... of mu as x* = argmin_y int (||y - x|| - ||x||) mu(dx)". Definition identique (avec la norme
nucleaire). OK.

### chen2020distributed (l. 160-161, 419) - OK
Phrases : "sign methods follow a median \cite{chen2020distributed,bernstein2019majority}" ; "Its scalar and coordinatewise
analogues are published (sign methods \cite{chen2020distributed}, ...)".
Source (arXiv:1906.01736v2, PDF) : Proposition 1 "sign(sum_i sign(g_{t,i})) = sign(median({g_t}))" et "signSGD actually
goes along the signed median direction" ; Sect. 1 "median-based algorithms like signSGD, medianSGD ... drive the median of
gradients to 0" ; Theorem 3 (contre-exemple : signSGD converge vers un point de gradient non nul fixe par l'ecart
mediane-moyenne). Analogue coordonnee par coordonnee de "la limite est moins une mediane" : OK (mediane sur les
travailleurs, analogue au bruit).

### bernstein2019majority (l. 160-161, 442) - OK
Phrases : "sign methods follow a median" ; "Item (a) parallels the noise condition of \cite{bernstein2019majority}".
Source (arXiv:1810.05291v3, PDF, en-tete "Published as a conference paper at ICLR 2019") : "Assumption 4 (Unimodal,
symmetric gradient noise). At any given point x, each component of the stochastic gradient vector has a unimodal
distribution that is also symmetric about the mean" ; contre-exemple bimodal (p. 6) : signSGD va dans le mauvais sens
quand la mediane et la moyenne ont des signes opposes. L'item (a) (loi symetrique => 0 dans Med) est bien le parallele
de la symetrie de l'Assumption 4. Le vote majoritaire est le signe de la mediane des signes (Prop. 1 de Chen et al.) :
"follow a median" acceptable.

### karimireddy2019error (l. 444) - OK
Phrase : "batch-one sign descent need not converge \cite{karimireddy2019error}".
Source (arXiv:1901.09847v2, PDF), Theorem I : "... Then SIGNSGD with batch-size 1 and stochastic gradients ... for i
chosen uniformly at random does not converge to x* a.s. for any adaptive sequence of step-sizes" (sous hypothese de
motifs de signes communs). Exact. (Precision possible : \cite[Theorem~I]{...}.)

### duchi2012randomized (l. 246-248) - reserve mineure
Phrase : "No moment is required: $\psi_\xi$ is convex and $1$-Lipschitz for $\nnuc\cdot$, and
$\Phi_\xi(G)\in\partial\psi_\xi(G)$ \cite{duchi2012randomized,jiang2026adaptive}, with equality to $\nabla\psi_\xi(G)$ when
$G+\xi$ has full rank almost surely."
Source (arXiv:1103.4296v2, PDF), Appendice E : "since mu is a density with respect to Lebesgue measure, the function f_mu
is in fact differentiable [4]" ; Lemma 6(iv) "E[grad F(x+Z)] = grad f_mu(x)" pour mu uniforme sur une boule (Lemmas 6-9 :
boules ell_inf, ell_2, gaussienne). La source traite le lissage par une densite ; le papier enonce le fait pour une loi
quelconque (sans densite, ex. la loi a deux points de la l. 393 ou les moyennes de lots d'une somme finie), et la
condition "full rank a.s." n'y figure pas. Le fait general est elementaire (esperance de l'inegalite de sous-gradient
de la norme nucleaire en G+xi ; differentiabilite par convergence dominee quand la norme est differentiable en G+xi
p.s.) et vrai, donc l'attribution est seulement imprecise.

### carlson2015stochastic (l. 118, 206) - OK, nit
Source (PMLR v38, p. 111-119, PDF) : Eq. (28) "W* = W - (||lambda||_1 / MJ) sum_k a_k b_k^T", minimisant la majoration
(20) en ||W - W^k||_{S_inf}^2 (Theorem 3) : pas a longueur nucleaire avec gamma = 1/L, L = MJ constante de lissite (borne)
pour la norme S_inf, sur gradients estimes par PCD. "steepest descent for the spectral norm" : OK. Nit : l. 205-206
"with exact gradients and $\gamma=1/\Lop$, stochastic spectral descent" : la source applique ce pas a des gradients
stochastiques et avec une borne MJ de L ; "stochastic" pour gradients exacts est un abus mineur (forme).


## Bloc 2 - autres attributions

### cardot2013efficient (l. 162, 419-420, 443-444) - OK, une imprecision
Source (arXiv:1101.4316, version HTML ar5iv) : Eq. (1) "m := argmin_u E[||X - u|| - ||X||]" ; Eq. (8)
"Z_{n+1} = Z_n + gamma_n (X_{n+1} - Z_n)/||X_{n+1} - Z_n||", Theorem 3.1 sous sum gamma_n = inf, sum gamma_n^2 < inf.
- l. 162 "normalized steps and the geometric median" : exact (pas normalises, limite = mediane geometrique).
- l. 443-444 "For $d=1$ the limit in (d) is the geometric median of the $y_i/x_i$" : exact. Pour d = 1,
  Lambda(W) = (1/n) sum ||W x_i - y_i||/|x_i| = (1/n) sum ||W - y_i/x_i|| et le pas polaire a lot un est
  W - eta (W - y_j/x_j)/||W - y_j/x_j||, soit exactement l'Eq. (8) sur la mesure empirique.
- l. 419-420 "Its scalar and coordinatewise analogues are published (sign methods \cite{chen2020distributed},
  normalized steps and the geometric median \cite{cardot2013efficient})" : Cardot et al. traitent la mediane
  geometrique dans un espace de Hilbert, c'est-a-dire l'analogue vectoriel (euclidien) - exactement le cas K = 1 du
  Theoreme 2 (polar(g) = g/||g||, norme nucleaire = norme euclidienne) - et non un analogue "scalaire" : la phrase
  sous-decrit l'anteriorite (constat cs-cardot-scalar, minor).

### cornish1938moments (l. 456-457) - OK, nit de signe
Phrase : "in every entry, the scalar offset $\kappa_3/(6\sigma^2b)$ of the median of a mean \cite{cornish1938moments} times
$a_{K,d}/c_{K,d}$". Developpement de Cornish-Fisher (formule verifiee sur la page Wikipedia "Cornish-Fisher expansion",
qui cite Cornish et Fisher 1938, Rev. Inst. Int. Stat. 5(4):307-320, doi:10.2307/1400905 ; l'article original (JSTOR)
n'a pas ete ouvert) : w_p = x + gamma_1 (x^2 - 1)/6 + ... ; en p = 1/2 la mediane de la moyenne de b copies centrees vaut
-kappa_3/(6 sigma^2 b) + o(1/b). Controle checks/cite-stat_cornish_median.py : moyennes de lois Gamma(a) centrees,
b * mediane = -0.3327 (a = 1, b = 32), -0.3333 (b = 512), limite -1/3 = -kappa_3/(6 sigma^2) ; lognormale centree
(Monte Carlo, b = 32) -0.00543 +- 0.00021 contre -0.00550 predit. Le decalage de la mediane est donc
$-\kappa_3/(6\sigma^2b)$ ; $+\kappa_3/(6\sigma^2b)$ est $G_b^\dagger=-\Med$ (coherent avec (eq:offset) :
$a_{1,1}=c_{1,1}=\sqrt{2/\pi}$). Ecart de signe purement verbal (constat cs-cornish-sign, nit).

### cutkosky2019momentum (l. 539) - imprecision
Phrase : "as for MARS-M \cite{liu2025marsm} and STORM-Muon \cite{cutkosky2019momentum}". Cutkosky et Orabona (arXiv:1905.10018,
NeurIPS 2019) introduisent l'estimateur STORM (2019, avant Muon) ; "STORM-Muon" est, dans le depot, Muon nourri par un tampon
STORM (muon/lab/mvr.py, docstring : "g = 1, same batch  STORM / MVR2 (Cutkosky and Orabona; Chang et al. arXiv:2509.15816;
Qian et al. arXiv:2512.16598)"). Le nom n'est defini nulle part dans le papier et la citation suggere que la source
propose une methode Muon (constat cs-storm-muon, minor).

### chzhen2023signsvrg (l. 169) - OK
Phrase : "SVRG in polar \cite{li2026implicit} and sign steps \cite{chzhen2023signsvrg}". Source (arXiv:2305.13187v1, PDF) :
"x_{t+1} = x_t - gamma sign(grad f_{i_t}(x_t) - grad f_{i_t}(x~) + grad f(x~) + G_t U_t)" (SignSVRG), Algorithm 1 : estimateur
SVRG dans un pas de signe (avec un bruit uniforme). Exact.

### davis2018subgradient, goffin1977convergence (l. 174) - OK (Goffin via source secondaire)
Phrase : "Geometric schedules of fixed lengths with exact gradients \cite{goffin1977convergence,davis2018subgradient}".
Davis et al. (arXiv:1803.02461, PDF), Section 5, Algorithm 3 : "Set stepsize alpha_k = lambda q^k. Update iterate
x_{k+1} = proj_X(x_k - alpha_k zeta_k/||zeta_k||)", zeta_k sous-gradient exact ; Theorem 5.1 : convergence lineaire ;
"The analogous strategy in the convex setting goes back to [19]" ([19] = Goffin 1977). Pas de longueur fixe, programme
geometrique, sous-gradients exacts : exact. Goffin 1977 lui-meme n'a pas ete ouvert (Springer, acces authentifie) ;
le contenu est atteste par Davis et al. Section 5.

### taylor2017smooth, hu2017unified (l. 147-148) - imprecision pour hu2017unified
Phrase : "worst cases by performance estimation \cite{taylor2017smooth,hu2017unified}".
- Taylor et al. (arXiv:1502.05666, abstract ; Math. Program. 161(1-2):307-345) : probleme d'estimation de performance,
  pire cas exacts de methodes du premier ordre a pas fixe (gradient, gradient rapide) : exact.
- Hu, Seiler, Rantzer (arXiv:1706.08141, PDF ; PMLR 65:1157-1189) : "jump system theory with several simple quadratic
  inequalities to derive sufficient conditions for convergence" (abstract, l. 30 du texte extrait) ; ces conditions sont
  des LMI ; le texte ne contient ni "performance estimation" ni "worst case" (seule occurrence : le titre de Taylor et
  al. en bibliographie). Ce sont des certificats de taux (conditions suffisantes), pas des pires cas (constat
  cs-hu-pep, minor). La source a l'avantage de traiter SAGA, Finito et SDCA, ce qui justifie sa presence dans la ligne.

## Bloc 3 - entrees .bib (paper_saga_to_muon.bib) et controles numeriques

| cle | verification | verdict |
|---|---|---|
| balles2020geometry | arXiv:2002.08056 : titre, 3 auteurs, v1 19 fev. 2020, pas de journal-ref | OK (statut ICLR 2020 indique par mlanthology.org mais non confirme : OpenReview bloque) |
| bernstein2019majority | arXiv:1810.05291 v1 11 oct. 2018 ; en-tete du PDF "Published as a conference paper at ICLR 2019" | OK |
| cardot2013efficient | arXiv:1101.4316 ; Bernoulli 19(1):18-43, 2013, doi:10.3150/11-BEJ390 (projecteuclid) | OK |
| carlson2015stochastic | PDF PMLR v38 (premiere page 111, en-tete AISTATS) ; PMLR 38:111-119 (deja ouvert dans muon/literature_check.md:1229) | OK |
| chen2020distributed | arXiv:1906.01736 ; NeurIPS 2020 (papers.neurips.cc, neurips.cc/virtual/2020/poster/18179) | OK |
| chzhen2023signsvrg | arXiv:2305.13187 v1 22 mai 2023, pas de version publiee trouvee | OK |
| cornish1938moments | Rev. Inst. Int. Stat. 5(4):307-320, 1938, JSTOR 1400905 (scilit, Wikipedia) | OK |
| cutkosky2019momentum | arXiv:1905.10018 ; NeurIPS 2019, NeurIPS 32 (papers.nips.cc) | OK (pas de pages, nit non retenu) |
| davis2018subgradient | Crossref 10.1007/s10957-018-1372-8 : JOTA 179(3):962-982, 2018 | OK |
| duchi2012randomized | arXiv:1103.4296 ; Crossref 10.1137/110831659 : SIAM J. Optim. 22(2):674-701, 2012 | OK |
| goffin1977convergence | Crossref 10.1007/BF01584346 : Math. Program. 13, numero 1, 329-347, dec. 1977 (Davis et al. citent 13(3)) | OK selon Crossref |
| hu2017unified | PMLR 65:1157-1189 (page PMLR) ; arXiv:1706.08141 | OK |
| karimireddy2019error | arXiv:1901.09847, commentaire "ICML 2019 (long talk)" | OK |
| maddison2021dual | arXiv:1902.02257 ; SIAM J. Optim. 31(1):991-1016, 2021, doi:10.1137/19M130858X (Edinburgh research portal) | OK |
| minsker2015geometric | arXiv:1308.1334, journal-ref "Bernoulli 2015, Vol. 21, No. 4, 2308-2335", doi:10.3150/14-BEJ645 | OK |
| sidford2018coordinate | arXiv:1808.01278, commentaire "A preliminary version appeared in FOCS 2018, with an error in the accelerated coordinate descent proof" | OK |
| taylor2017smooth | Crossref 10.1007/s10107-016-1009-3 : Math. Program. 161(1-2):307-345 (numero de janv. 2017) | OK |

Rendu verifie dans paper/paper_saga_to_muon.pdf (pdftotext de la bibliographie) : entrees [4], [6]-[9], [11]-[14], [20],
[21], [25], [31], [38], [40], [47], [48] conformes au .bib.

Controle checks/cite-stat_kop_minKd.py (enonce attribue a Sidford-Tian, vrai) : pour un hessien PSD generique sur
R^{KxD}, moyenne de <D,HD> sur les directions de rang un unitaires = tr(H)/(Kd) (ex. K=4, d=6 : 1.1195 contre 1.1185
predit) et sur les isometries partielles de Haar = min(K,d) tr(H)/(Kd) (4.4678 contre 4.4740), d'ou
mu_op <= tr(H)/(Kd) <= L_op/min(K,d), donc kappa_op >= min(K,d), avec egalite pour H = I.

## Synthese des constats
- cs-sidford-kop (major, l. 354) : $\kop\ge\min(K,d)$ attribue a Sidford-Tian, qui ne prouvent qu'une borne
  d'amplitude en $\ell_\infty$ (Lemma A.1).
- cs-hu-pep (minor, l. 147-148) : Hu-Seiler-Rantzer donnent des conditions suffisantes (LMI), pas des pires cas par
  estimation de performance.
- cs-cardot-scalar (minor, l. 419-420) : Cardot et al. sont l'analogue euclidien (K = 1) du Theoreme 2, pas un analogue
  "scalaire".
- cs-storm-muon (minor, l. 539) : "STORM-Muon \cite{cutkosky2019momentum}" attribue a la source de STORM une variante
  de Muon qu'elle ne contient pas.
- cs-duchi-density (minor, l. 246-248) : Duchi et al. traitent le lissage par une densite ; le papier enonce
  l'inclusion pour toute loi (vraie, argument elementaire).
- cs-cornish-sign (nit, l. 456-457) : le decalage de la mediane est $-\kappa_3/(6\sigma^2b)$.
- cs-carlson-exact (nit, l. 205-206) : "with exact gradients ... stochastic spectral descent".

## Non verifie
- Texte de Goffin 1977 (Springer, acces authentifie) : contenu atteste seulement par Davis et al. Section 5.
- Texte original de Cornish et Fisher 1938 (JSTOR) : formule verifiee par source secondaire (Wikipedia, qui cite
  l'article) et par calcul.
- Statut de publication de Balles et al. (ICLR 2020 selon mlanthology.org ; OpenReview et DBLP refusent l'acces
  automatise) : l'entree arXiv est exacte de toute facon.
- jiang2026adaptive (cite avec duchi2012randomized l. 247) : hors perimetre, non ouvert ; il peut couvrir le cas
  general.

## Corrections proposees (texte LaTeX, minimal)
- cs-sidford-kop (major, l. 354, confiance high) :
  remplacer `and the spectral geometry has its own cost, $\kop\ge\min(K,d)$ \cite{sidford2018coordinate}.`
  par `and the spectral geometry has its own cost, $\kop\ge\min(K,d)$ (average $D^2F$ over unit rank-one and over orthogonal directions), the spectral form of a folklore $\ell_\infty$ bound \cite[Lemma~A.1]{sidford2018coordinate}.`
- cs-hu-pep (minor, l. 147-148, confiance high) :
  remplacer `worst cases by performance estimation \cite{taylor2017smooth,hu2017unified}`
  par `worst cases by performance estimation \cite{taylor2017smooth} and rate certificates by quadratic constraints \cite{hu2017unified}`
- cs-cardot-scalar (minor, l. 418-420, confiance high) :
  remplacer `Its scalar and coordinatewise analogues are published (sign methods \cite{chen2020distributed}, normalized steps and the geometric median \cite{cardot2013efficient}); the statement for polar steps was not found.`
  par `Its coordinatewise and Euclidean ($K=1$) analogues are published (sign methods \cite{chen2020distributed}, normalized steps and the geometric median \cite{cardot2013efficient}); the statement for polar steps was not found.`
- cs-storm-muon (minor, l. 539, confiance high) :
  remplacer `as for MARS-M \cite{liu2025marsm} and STORM-Muon \cite{cutkosky2019momentum}`
  par `as for MARS-M \cite{liu2025marsm} and Muon on a STORM buffer \cite{cutkosky2019momentum}`
- cs-duchi-density (minor, l. 247, confiance medium) :
  remplacer `and $\Phi_\xi(G)\in\partial\psi_\xi(G)$ \cite{duchi2012randomized,jiang2026adaptive}`
  par `and $\Phi_\xi(G)\in\partial\psi_\xi(G)$ (take expectations in the subgradient inequality of $\nnuc\cdot$ at $G+\xi$; for smoothing densities, \cite{duchi2012randomized,jiang2026adaptive})`
- cs-cornish-sign (nit, l. 456-457, confiance high) :
  remplacer `in every entry, the scalar offset $\kappa_3/(6\sigma^2b)$ of the median of a mean \cite{cornish1938moments} times $a_{K,d}/c_{K,d}$.`
  par `in every entry, minus the scalar offset $-\kappa_3/(6\sigma^2b)$ of the median of a mean \cite{cornish1938moments}, times $a_{K,d}/c_{K,d}$.`
- cs-carlson-exact (nit, l. 205-206, confiance medium) :
  remplacer `(with exact gradients and $\gamma=1/\Lop$, stochastic spectral descent \cite{carlson2015stochastic} or PolarGrad \cite{lau2025polargrad})`
  par `(with $\gamma=1/\Lop$, the step of stochastic spectral descent \cite{carlson2015stochastic} and of PolarGrad \cite{lau2025polargrad})`
