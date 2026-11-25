# Audit math-offset : Proposition 4 (taille de l'offset) et debut de "Numerical checks"

Version finale (2026-11-20). Auditeur : dimension `math-offset`.

Bilan : la Proposition 4, l'equation (3), ses constantes, son signe et les chiffres des lignes 463-465 et 482-484 sont
**corrects** ; le statut "formal" est honnete et plutot prudent. Aucun constat blocker ni major. Quatre constats :
3 minor (MO-1, MO-3, MO-4) et 1 nit (MO-2).

## Perimetre couvert

- `paper/paper_saga_to_muon.tex` lignes 449-467 (Proposition 4, equation (3), commentaire) et lignes 482-484.
- Coherence des mentions liees : resume (l. 101), tableau 1 (l. 165-166), section 2.3 (l. 249-251), tableau 2 (l. 632),
  limitations (l. 671), et, pour la question (e), la phrase voisine l. 486-488.
- Sources lues : `muon/theory_checks/check_offset_first_order.py`, `check_minibatch_offset_exact.py`,
  `paper_figures_data.json`, `archive/muon/notes_spectral_theory.tex` (l. 958-1040, 1055-1090).
- Scripts de l'audit (tous sous `paper/audit/checks/`, lances depuis la racine, un processus, BLAS mono-fil) :
  - `math-offset_constants.py` (1 min 44 s) : constantes a, c, a/c ; quadrature exacte K = 2 ; K = 1 ; jacobienne ; cas scalaire.
  - `math-offset_zero_meanfield.py A` (51 s) : rejeu de `check_minibatch_offset_exact.py` (meme graine, meme flux).
  - `math-offset_zero_meanfield.py B` (68 s) : estimation independante du zero du champ moyen, 2 x 3, b = 1..1024.
  - `math-offset_other_laws.py` (1 min 49 s) : autres lois (autre variance, autre asymetrie), tailles 2 x 2 et 2 x 5.
  - `math-offset_3x5.py` (47 s) : taille 3 x 5 (K = 3), b = 32.

## (a) Developpement d'Edgeworth a un terme : refait independamment -> OK

Notations : S_b = sqrt(b) xi_b / sigma (entrees i.i.d. centrees reduites, asymetrie lambda_3/sqrt(b),
lambda_3 = kappa_3/sigma^3), Z gaussienne standard K x d.

1. Homogeneite : polar(tM) = polar(M) pour t > 0, donc Phi_{xi_b}(0) = E polar(xi_b) = E polar(sqrt(b) xi_b). OK (l. 460).
2. Densite d'Edgeworth a un terme, entrees independantes :
   p_b(S) = phi(S) [1 + lambda_3/(6 sqrt(b)) sum_{kl} He_3(S_kl) + O(1/b)], d'ou
   E polar(S_b)_{ij} = lambda_3/(6 sqrt(b)) sum_{kl} E[polar(Z)_{ij} He_3(Z_kl)] + ... (E polar(Z) = 0 par imparite).
   Les termes croises (k,l) != (i,j) sont nuls : si k != i, changer le signe de la ligne i de Z change polar(Z)_{ij} en
   son oppose (polar(DZ) = D polar(Z)) sans changer Z_kl ; idem par colonne si l != j. Il reste le terme diagonal,
   identique pour tout (i,j) par permutation des lignes et des colonnes :
   Phi_{xi_b}(0) = -(kappa_3/(6 sigma^3 sqrt(b))) a_{K,d} 1 1^T, a_{K,d} = -E[polar(Z)_11 (Z_11^3 - 3 Z_11)].
   Le terme suivant (ordre 1/b, polynomes pairs) est nul par imparite de polar : reste formel O(b^{-3/2}).
   Controle numerique : sqrt(b) ||Phi_b(0)||_F -> (2/6) a sqrt(6) = 0.2339 pour 2 x 3, Exp(1)-1 ; mesure (partie B)
   0.2293, 0.2321, 0.2333, 0.2331, 0.2347, 0.2349, 0.2349, 0.2355, 0.2314, 0.2333 pour b = 1, 2, 4, ..., 256, 1024.
   Pour 3 x 5, b = 32 : entree moyenne de sqrt(b) Phi_b(0) = -0.05338, predit -0.05353.
3. Jacobienne : Phi_{xi_b}(G) = E polar(sqrt(b) G/sigma + S_b) ; a l'ordre dominant S_b ~ Z et
   D Phi_Z(0)[Delta] = E[polar(Z) <Z, Delta>] (Stein, polar bornee). Element de matrice E[polar(Z)_{ij} Z_kl] : nul si
   (k,l) != (i,j) (memes changements de signe), constant sur la diagonale (permutations), donc D Phi_Z(0) = c Id avec
   c = E<polar(Z), Z>/(Kd) = E||Z||_*/(Kd) = c_{K,d}. Donc D Phi_{xi_b}(0) = sqrt(b) c_{K,d}/sigma au premier ordre :
   **c_{K,d} est bien la bonne constante** (jacobienne de polar moyennee sous bruit gaussien).
   Controles numeriques : difference finie centree (h = 0.1, 2e6 tirages, 2 x 3) : D Phi_Z(0)[E_11]_11 = 0.5037 +- 0.0004
   contre c_{2,3} = 0.5043, autres entrees nulles a 4e-4 pres ; pente du champ moyen non gaussien au zero divisee par
   6 sqrt(b) c : 1.137, 1.065, 1.033, 1.016, 1.007, 1.002, 1.002, 1.001, 1.000 pour b = 1, ..., 256 (correction O(1/b)).
4. Resolution : G = -D Phi(0)^{-1} Phi(0) = (sigma/(sqrt(b) c)) (kappa_3 a/(6 sigma^3 sqrt(b))) 1 1^T
   = kappa_3/(6 sigma^2 b) (a/c) 1 1^T : c'est l'equation (3), facteur kappa_3/(6 sigma^2 b), constantes et **signe** compris.
5. "i.i.d. entries" suffit pour la forme 1 1^T : le tenseur des troisiemes cumulants est diagonal et la covariance vaut
   sigma^2 Id (il suffirait d'entrees independantes de memes variance et troisieme cumulant). De plus la loi de xi_b est
   invariante par permutation des lignes et des colonnes, donc l'ensemble des minimiseurs de psi aussi : quand le
   minimiseur est unique (c'est le cas des moyennes Gamma des controles) il est **exactement** proportionnel a 1 1^T
   pour tout b. Les cosinus 0.998 et
   0.996 des notes a b = 128, 256 sont du bruit de Monte-Carlo (rejeu : 0.99997 a b = 2, 0.99561 a b = 256).

## (b) Cas scalaire -> OK

- Esseen : F_b(x) = Phi(x) - lambda_3 (x^2 - 1) phi(x)/(6 sqrt(b)) + o(b^{-1/2}) uniformement pour une loi non
  arithmetique de troisieme moment fini ; mediane de S_b = -lambda_3/(6 sqrt(b)), donc mediane de xi_b
  = -kappa_3/(6 sigma^2 b) + o(1/b). Cornish-Fisher en p = 1/2 donne la meme valeur. Dans le cas scalaire l'enonce est
  donc un theoreme, et "non-lattice" + "o(1/b)" est exactement la bonne paire hypothese/reste.
- a_{1,1} = -E[sign(Z)(Z^3 - 3Z)] = -(E|Z|^3 - 3E|Z|) = sqrt(2/pi) = c_{1,1} : (3) redonne kappa_3/(6 sigma^2 b).
- Signe : G^dagger = -Med(xi_b) = +kappa_3/(6 sigma^2 b) ; asymetrie positive => mediane negative => offset positif. Coherent.
- Numerique : b * mediane(Gamma(b,1)/b - 1) = -0.30685, -0.32165, -0.32794, -0.33075, -0.33207, -0.33271 (b = 1..32),
  -0.33326 (b = 256), -0.33333 (b = 4096) -> -1/3 = -kappa_3/(6 sigma^2). Medianes -0.3069, -0.1608, -0.0820, -0.0413,
  -0.0208, -0.0104 : identiques a celles des notes.

## (c) Constantes -> OK (un nit d'arrondi, MO-2)

- K = 1 : polar(z) = z/||z||, z = rho u ; a = -E[rho^3]E[u_1^4] + 3E[rho]E[u_1^2] = 3E[rho]/(d(d+2)) avec
  E[rho^3] = (d+1)E[rho] ; c = E[rho]/d ; **a/c = 3/(d+2)** exactement. Monte-Carlo : 0.59996 +- 0.0023 (d = 3, exact 0.6),
  0.29725 +- 0.0015 (d = 8, exact 0.3), 0.05788 +- 0.0006 (d = 50, exact 0.05769).
- Formule exacte obtenue pendant l'audit (moyenne de Haar sur U, V de Z = U diag(s) V^T, puis independance de ||Z||_F et
  de Z/||Z||_F) :
  a_{K,d} = 3[(2K+2d+3) E sum s_k - 2 E sum s_k^3] / (K(K+2)d(d+2)),
  **a/c = 3[(2K+2d+3) - 2 E[sum s^3]/E[sum s]] / ((K+2)(d+2))**.
  Elle redonne 3/(d+2) pour K = 1, coincide avec le Monte-Carlo direct de a = -E[polar(Z)_11 He_3(Z_11)] aux 11
  tailles (ecarts <= 1.6 erreur standard), et ne demande que des valeurs singulieres (a nombre de tirages egal, erreur
  environ 2 fois plus petite que l'estimateur direct aux petites tailles et 15 fois a 32 x 32). Limites (calcul de
  l'audit, appuye par les valeurs numeriques ci-dessous) : max(K,d) a/c -> 12/5 pour les matrices carrees (loi du quart
  de cercle) et -> 3 pour K fixe, d -> infini (a/c ~ 3/(d+2) quand d >> K).
- K = 2, quadrature exacte (200 x 200 noeuds, stable a 6 chiffres) : a_{2,3} = 0.286449, c_{2,3} = 0.504295,
  a/c = 0.568019 ; a_{2,2} = 0.391084, c_{2,2} = 0.572898, a/c = 0.682641. Identiques aux valeurs des notes
  (0.28645, 0.50430, 0.39108, 0.57290).
- Dix tailles du papier, max(K,d) a/c (formule en valeurs singulieres, 1e5 a 4e5 tirages, erreur standard entre parentheses) :

  | K x d | 2x2 | 2x3 | 3x5 | 4x4 | 8x8 | 16x16 | 32x32 | 4x16 | 8x32 | 16x64 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | max(K,d) a/c | 1.364 (0.004) | 1.705 (0.004) | 2.005 (0.004) | 1.712 (0.003) | 1.985 (0.002) | 2.167 (0.002) | 2.276 (0.001) | 2.599 (0.005) | 2.735 (0.004) | 2.814 (0.004) |
  | notes (table) | 1.37 | 1.70 | 2.01 | 1.71 | 2.00 | 2.17 | 2.29 | 2.59 | 2.75 | 2.80 |

  Etendue 1.364 (exact 1.36528 a 2 x 2) a 2.814 : "between 1.4 and 2.8" est exact a l'arrondi pres (MO-2).
- "per entry the offset is a factor of order max(K,d) below that of sign descent" : OK. Le signe coordonnee par
  coordonnee a l'offset scalaire dans chaque entree (entrees independantes) ; le rapport vaut max(K,d)/(1.4 a 2.8) sur
  les dix tailles, et les limites 12/5 et 3 ci-dessus montrent que l'ordre max(K,d) tient au-dela
  (64 x 64 : 2.334 ; 128 x 128 : 2.367 ; 2 x 64 : 2.88 +- 0.03 ; 4 x 256 : 3.01 +- 0.06).

## (d) Prediction 0.464 et valeurs mesurees 0.450-0.498 -> OK, avec une reserve (MO-1)

- Prediction : (kappa_3/(6 sigma^2)) (a/c) sqrt(Kd) = (2/6)(0.568019) sqrt(6) = **0.46379** -> "0.464" OK
  (Exp(1) - 1 : sigma^2 = 1, kappa_3 = 2 ; ||1 1^T||_* = sqrt(6)).
- Les valeurs 0.450-0.498 ne sont stockees nulle part dans le depot (`paper_figures_data.json` ne contient que le
  probleme de somme finie de la figure 3) : elles viennent de la sortie de `check_minibatch_offset_exact.py`.
  Rejeu avec la meme graine et le meme flux (facteur polaire en forme close au lieu de la SVD, meme appel `hybr`) :

  | b | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|---|---|
  | b \|\|G_b\|\|_* (rejeu, 1.5e6 tirages) | 0.4503 | 0.4605 | 0.4610 | 0.4620 | 0.4703 | 0.4811 | 0.4853 | 0.4976 |
  | erreur standard (composante 1 1^T) | 0.0013 | 0.0019 | 0.0026 | 0.0037 | 0.0053 | 0.0075 | 0.0106 | 0.0150 |
  | cosinus avec 1 1^T | 0.99997 | 0.99990 | 0.99985 | 0.99989 | 0.99975 | 0.99934 | 0.99793 | 0.99561 |
  | audit, independant (3e6 tirages, variables de controle) | 0.4500 (4) | 0.4571 (6) | 0.4600 (8) | 0.4640 (12) | 0.4655 (17) | 0.4657 (24) | 0.4671 (33) | 0.4591 (47) |

  (b = 1 : 0.4334 (3) ; b = 1024 : 0.4625 (94).) Le rejeu redonne exactement l'etendue du papier (0.4503 a 0.4976),
  "1.5e6 exactly sampled batch means" est exact (Gamma(b,1)/b - 1 est la loi exacte de la moyenne), "b = 2, ..., 256" aussi.
- Estimation independante (autre graine, racine en dimension 1 le long de 1 1^T, deux variables de controle de moyenne
  connue) : la prediction est confirmee, 0.4640 +- 0.0012 a b = 16 et tous les b >= 16 a une erreur standard au plus
  de 0.4638 ; pour b <= 8 l'ecart vaut -0.027/b a -0.030/b (correction relative O(1/b), compatible avec le o(1/b) de (3)).
- Reserve (MO-1) : le haut de l'etendue du papier (0.47 a 0.498 pour b = 32..256) est de l'erreur de Monte-Carlo
  (erreur standard 0.005 a 0.015, plus un biais positif de la norme nucleaire du a la composante de bruit orthogonale a
  1 1^T : ||G||_* >= ||G||_F = composante / cosinus, soit +0.4 % a +9 % a b = 256), pas un ecart reel ; sans barre
  d'erreur, la suite du script s'eloigne de 0.464 quand b croit.
- "Parameter-free" teste hors des cas du depot : entrees Gamma(alpha, theta) centrees, rapport mesure/prediction
  (b = 16, 64, 256) : 2 x 2, sigma^2 = 9, kappa_3 = 54 : 0.998 (3), 0.996 (7), 1.003 (13) ; 2 x 2, sigma^2 = 1,
  kappa_3 = 4 : 0.983 (2), 0.990 (4), 0.995 (7) ; 2 x 5, kappa_3 = 4 : 0.987 (1), 0.998 (2), 1.000 (4) ; 3 x 5,
  Exp(1) - 1, b = 32 : 0.996 (5). La dependance en kappa_3/sigma^2, en taille (cas carre compris) et la convergence en
  1/b sont confirmees.

## (e) Qualifications et coherence des statuts

- "formal" : juste. Le developpement n'est pas demontre dans le cas matriciel (ni dans les notes, qui le marquent [F]+[N]
  et le listent comme question ouverte). Le qualificatif est repris dans le titre de la proposition, l. 462, le tableau 1
  (etiquette [f]), le resume ("formally") et les limitations (l. 671) : coherent partout.
- "non-lattice" : juste et necessaire. Dans le cas scalaire c'est l'hypothese du theoreme d'Esseen ; pour une loi
  arithmetique b fois la mediane oscille (exemple des notes, X dans {-1, 2}). Des entrees i.i.d. non arithmetiques
  donnent un vecteur fortement non arithmetique (|fonction caracteristique| < 1 hors de 0), l'hypothese naturelle du
  developpement a un terme en dimension > 1. Le troisieme moment fini est implicite dans "third cumulant kappa_3".
- "o(1/b)" : coherent avec l'hypothese (Esseen donne o(b^{-1/2}) sur la fonction de repartition, donc o(1/b) sur le
  zero) ; numeriquement le reste est O(1/b^2) pour les lois Gamma.
- "against a discontinuous map" : juste (polar est discontinue sur les matrices de rang deficient ; c'est la raison pour
  laquelle le developpement d'Edgeworth n'est pas automatique).
- "parameter-free" : juste (a et c ne dependent que de (K, d) ; aucun ajustement ; verifie sur 5 couples loi/taille).
- "the minimizer" : l'unicite n'est pas garantie par la seule hypothese "non-lattice" (lois a atomes ou a trou autour de
  la mediane : arg min psi peut etre un ensemble) ; dans le cas scalaire l'enonce vaut alors pour tout minimiseur
  (Esseen est uniforme). Simplification sans consequence pour un enonce formel ; non retenu comme constat.
- Tableau 1, "a limit gradient of order 1/b [f]" et resume, "an offset formally of order 1/b" : coherents avec ce qui est
  etabli (developpement formel + zeros du champ moyen). Deux precisions utiles (MO-3, MO-4) : le renvoi du tableau 2
  (l. 632) et le fait que le probleme de somme finie de la figure 3(c) sort des hypotheses de (3).
- Piste (non verifiee, hors perimetre) : le statut "formal" parait prudent. Un developpement a un terme uniforme sur les
  translatees d'une fonction bornee continue presque partout (Bhattacharya-Rao) et la monotonie du sous-gradient de psi
  pourraient donner une preuve, au moins pour K != d ; le cas carre demande un controle de l'oscillation moyenne de
  polar pres de {det = 0} (en eps log(1/eps)).

## Releve au passage (hors perimetre strict, donnees de `paper_figures_data.json`)

- l. 485 : ||W_LAD - W_LS||_F = 0.16963 -> "0.170" OK.
- l. 486-487 : b fois le zero du champ moyen = 0.168, 0.240, 0.276, 0.285, 0.303, 0.308 (b = 1..32) -> "rising from 0.17
  to a plateau near 0.3" OK ; ecart runs / zero : 1.4 %, 2.1 %, 1.4 %, 1.2 %, 3.8 %, 4.6 % -> "within 5%" OK.
  (Runs non relances.)

## Constats

### MO-1 (minor) — l. 482-484 : l'etendue "0.450-0.498" melange un ecart reel et l'erreur de Monte-Carlo
- Citation : "the zero of the mean field computed from $1.5\cdot10^6$ exactly sampled batch means gives $0.450$--$0.498$
  for $b=2,\dots,256$."
- Probleme : la phrase est une transcription exacte de la sortie du script, mais elle ne donne pas l'erreur de
  Monte-Carlo, qui croit comme sqrt(b). La borne basse 0.450 (b = 2) est un ecart reel d'ordre 1/b ; la borne haute
  0.498 (b = 256) est du bruit : erreur standard 0.015 sur la composante 1 1^T, plus un biais positif de la norme
  nucleaire (cosinus 0.9956 avec 1 1^T). Telle que calculee, la suite croit avec b (0.450, 0.461, 0.461, 0.462, 0.470,
  0.481, 0.485, 0.498) et depasse la limite annoncee 0.464 en s'en eloignant : un jury qui relance le script peut y
  lire une non-convergence.
- Preuve : tableau de la section (d) ; estimation independante a variance reduite : 0.4640 +- 0.0012 (b = 16),
  0.4655 +- 0.0017 (32), 0.4657 +- 0.0024 (64), 0.4671 +- 0.0033 (128), 0.4591 +- 0.0047 (256), 0.4625 +- 0.0094 (1024).
  L'accord reel avec (3) est donc bien meilleur que ce que l'etendue publiee laisse voir.
- Correction minimale : "the zero of the mean field computed from $1.5\cdot10^6$ exactly sampled batch means gives
  $0.450$ at $b=2$ and $0.46$--$0.50$ for $b=4,\dots,256$ (Monte Carlo standard error growing to $0.015$ at $b=256$)."
  Variante plus forte, si le script de l'audit est repris dans le depot : "... gives $0.450$ at $b=2$, $0.460$ at $b=8$
  and $0.464\pm0.005$ for $b=16,\dots,256$ (control variates)."
- Confiance : haute.

### MO-2 (nit) — l. 464-465 : "between 1.4 and 2.8"
- Citation : "$\max(K,d)\,a_{K,d}/c_{K,d}$ lies between $1.4$ and $2.8$ for ten sizes from $2\times2$ to $32\times32$ and $16\times64$"
- Probleme : les extremes valent 1.365 (2 x 2, exact 1.36528) et 2.814 +- 0.004 (16 x 64 ; la valeur 2.80 des notes
  avait 3 % d'erreur) : les deux sortent de [1.4, 2.8] pris a la lettre. Arrondi seulement.
- Correction : "lies between $1.37$ and $2.81$".
- Confiance : haute.

### MO-3 (minor) — l. 632 (tableau 2) : renvoi a la Proposition 4 pour un enonce du Theoreme 2
- Citation : "The limit gradient is minus a nuclear median of the noise (Proposition~\ref{prop:offset})."
- Probleme : cet enonce est le Theoreme 2(b), prouve ; la Proposition 4 n'en donne que la taille, et formellement. Le
  renvoi fait reposer un resultat [p] sur un enonce [f].
- Correction : "The limit gradient is minus a nuclear median of the noise (Theorem~\ref{thm:smd}), formally of order
  $1/b$ (Proposition~\ref{prop:offset})."
- Confiance : haute (hors perimetre strict : signale parce que le renvoi vise la proposition auditee).

### MO-4 (minor) — l. 486-488 et figure 3(c) : la somme finie sort des hypotheses de (3), sans que ce soit dit
- Citation : "for $b\le32$ the runs end within $5\%$ of the zero of the mean field, $b$ times the offset rising from
  $0.17$ to a plateau near $0.3$"
- Probleme : dans ce probleme les gradients par echantillon sont de rang un, (W x_j - y_j) x_j^T : les entrees du bruit
  ne sont pas independantes et (3) ne s'applique pas. Le paragraphe "Numerical checks" enchaine la verification de (3)
  (0.464) et ce plateau (0.3) sans le dire ; le lecteur peut prendre 0.3 pour un test de la constante de (3). Les notes
  le disent explicitement (`archive/muon/notes_spectral_theory.tex` l. 1038 : "so (3) does not apply as such: what is
  observed is the 1/b regime"). De meme l'ordre 1/b du resume et du tableau 1 n'est derive dans le papier que pour des
  entrees i.i.d. (la version generale, G = -(1/b) J^{-1} T, n'est que dans les notes).
- Correction : "... rising from $0.17$ to a plateau near $0.3$ (per-sample gradients have rank one: \eqref{eq:offset}
  does not apply as such, the $1/b$ regime does), and ..."
- Confiance : moyenne (constat de presentation ; les chiffres sont exacts).

## Non verifie

- Contenu des sources citees : \cite{xie2026muon,chen2026when} (constante de reponse lineaire c_{K,d}), \cite{do2026muon}
  (l. 465-467), \cite{cornish1938moments} (l'entree bibliographique est la reference classique et la formule de la
  mediane est recalculee ici, mais l'article n'a pas ete ouvert) : dimension bibliographique.
- Validite rigoureuse du developpement d'Edgeworth dans le cas matriciel : non demontree (le papier ne le pretend pas).
- "an intercept suffices to create one" (l. 463) : argument seulement (avec une colonne constante, la colonne
  correspondante du bruit est le residu lui-meme, asymetrique) ; pas de controle numerique.
- Zero du champ moyen contre (3) pour K >= 4 ou des entrees non Gamma : seules les constantes sont verifiees aux grandes
  tailles ; les zeros le sont pour 2 x 2, 2 x 3, 2 x 5, 3 x 5.
- Runs de la somme finie (l. 484-488) : non relances, seulement compares a `paper_figures_data.json`.
