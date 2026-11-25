export const meta = {
  name: 'audit-paper-saga-to-muon',
  description: 'Audit scientifique du papier paper_saga_to_muon : preuves, chiffres, code, citations, nouveaute, coherence',
  phases: [
    { title: 'Audit', detail: 'un auditeur par dimension (preuves, chiffres, code, citations, nouveaute, coherence)' },
    { title: 'Contre-verification', detail: 'un verificateur adverse par dimension ayant des constats bloquants ou majeurs' },
  ],
}

const ROOT = '/users/eleves-b/2023/louis.darrigol/projetSTATS'
const EVID = ROOT + '/paper/audit/evidence'
const CHECKS = ROOT + '/paper/audit/checks'

const COMMON = `Tu es un auditeur scientifique independant. Objet de l'audit : le papier ${ROOT}/paper/paper_saga_to_muon.tex (12 pages, PDF a cote, bibliographie paper_saga_to_muon.bib, figures dans paper/figures/, scripts des figures dans paper/scripts/). Titre : "From SAGA to Muon-type steps: no norm repairs the logistic worst case, and variance reduction acts on the fixed point". L'auteur veut un audit de niveau soutenance de these : chaque affirmation doit etre exacte, demontree ou mesuree comme annonce, et correctement attribuee. Tu traites UNE dimension de l'audit, decrite sous PERIMETRE : reste dedans et va au fond. D'autres auditeurs traitent les autres dimensions en parallele.

CARTE DU DEPOT (${ROOT})
- Papier : paper/paper_saga_to_muon.tex (700 lignes ; les numeros de ligne cites ci-dessous sont ceux de l'outil Read), paper/README.md.
- Etude principale (SAGA) : notes de theorie avec preuves completes theory/B3_prop59_tightness.tex, B2_effective_curvature.tex, B1_preconditioned_saga.tex, A1_lower_bound.tex, A3_lkatyusha_lyapunov.tex, B4_sampling_order.tex ; resultats results/*_summary.json, results/digest.md (tous les chiffres), results/findings.md (analyse) ; rapport final archive/report/main.pdf et archive/report/sections/*.tex ; code vrlab/ et experiments/.
- Etude Muon : muon/findings.md (analyse, sections 4 et 5.1 a 5.13), muon/results/digest.md et muon/results/*_summary.json (chiffres), muon/notes_theory.md (faits F1 a F16 avec preuves), archive/muon/notes_spectral_theory.tex (preuves des theoremes et propositions), archive/muon/report_spectral_muon_new.tex (rapport etendu d'ou viennent les etiquettes (C1)-(C5) et (M1)-(M4)), muon/theory_checks/ (controles numeriques de la Proposition 4), code muon/lab/ et muon/experiments/, muon/literature_check.md (verifications bibliographiques anterieures, 500 Ko : utilise grep, ne le lis pas en entier), bibliography/literature_review.md.
- Etiquettes de statut du papier : [p] prouve, [m] mesure exactement le long des trajectoires, [e] experience, [f] developpement formel verifie par simulation.

METHODE
1. Lis d'abord le papier en entier, puis les sources de ton perimetre.
2. Pour chaque affirmation de ton perimetre, etablis de facon independante si elle est vraie : refais la demonstration ou le calcul, retrouve le chiffre dans les donnees brutes (JSON, pickles) plutot que dans la prose des notes, ouvre la source citee. Les notes du depot ont ete ecrites par les memes auteurs : elles aident a comprendre mais ne prouvent rien par elles-memes.
3. Quand un controle numerique bon marche tranche, fais-le et garde le script dans ${CHECKS}/ (nom prefixe par la cle de ta dimension).
4. Un constat doit etre etaye : contre-exemple, calcul, fichier:ligne, valeur recalculee, ou URL avec le passage. Un constat non etaye nuit a l'audit autant qu'une erreur manquee. Ne rapporte pas des preferences de style comme des erreurs. Si le papier est correct, dis-le dans checked_ok avec la preuve du controle : c'est un resultat utile.
5. Si tu ne peux pas trancher, mets l'element dans not_checked en disant ce qui manque.

REGLES
- Lecture seule sur tout le depot. Tu n'ecris QUE dans ${EVID}/ (ton rapport) et ${CHECKS}/ (tes scripts). Ne modifie jamais paper/*.tex, muon/, results/, theory/, vrlab/, experiments/. Pas de commit git.
- Python : python3 du systeme (3.9, numpy 1.22, scipy 1.9, mpmath ; pas de venv), toujours avec OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, un seul processus a la fois, chaque calcul sous 3 minutes de CPU, rien en arriere-plan. La machine est partagee et deja chargee. Ne relance pas les experiences completes du depot (elles durent des heures) : petites reproductions seulement. Lance les scripts depuis ${ROOT}.
- Budget : vise 30 a 40 minutes au total. Traite d'abord ce qui porte les affirmations de tete (titre, resume, enonces des theoremes, intitules des constats), ensuite le detail.
- Ecris ton rapport ${EVID}/<cle>.md AU FUR ET A MESURE (apres chaque bloc de verifications), pas seulement a la fin : si tu es interrompu, ce qui est ecrit doit rester exploitable. Langue du rapport : francais ; citations du papier telles quelles (anglais).

SEVERITES
- blocker : un enonce de tete (titre, resume, theoreme, proposition, intitule d'un constat) est faux ou non etaye, ou une preuve annoncee est fausse.
- major : chiffre faux, renvoi ou attribution fausse, hypothese manquante qui rend un enonce faux tel qu'ecrit, sur-affirmation qu'un jury releverait.
- minor : imprecision, ambiguite, reserve manquante, petit ecart d'arrondi, incoherence locale.
- nit : forme.

SORTIE
Ton rapport ${EVID}/<cle>.md contient : le perimetre couvert ; pour chaque affirmation controlee, verdict (OK / probleme / non verifie) et preuve ; le detail des constats. Ta reponse structuree reprend : scope_covered ; checked_ok (affirmations verifiees correctes, avec la preuve du controle, une ligne chacune) ; findings (id court, severity, location = lignes du .tex, claim = phrase du papier citee, problem, evidence, proposed_fix = texte LaTeX de remplacement en anglais dans le style du papier, minimal car le papier est a la limite de 12 pages, ou chaine vide si aucune correction de texte n'est necessaire, confidence) ; not_checked ; report_path.`

const CITE_METHOD = `METHODE BIBLIOGRAPHIQUE. Pour chaque cle : (1) l'entree de paper/paper_saga_to_muon.bib (auteurs, titre, annee, lieu de publication, identifiant arXiv ou DOI) est-elle exacte ? Verifie par recherche web (arXiv, editeur, DBLP, Semantic Scholar, OpenReview). (2) Chaque phrase du papier qui cite la cle (grep de la cle dans le .tex) attribue-t-elle a la source quelque chose qu'elle contient reellement ? Pour les renvois precis (numero de theoreme, de lemme, d'equation, constante), ouvre la source (page arXiv abs, version HTML, PDF) et cite le passage. Le depot contient des verifications anterieures : muon/literature_check.md (section "Citation check" vers la ligne 1098, addendum vers la ligne 1277), bibliography/literature_review.md, theory/theory_citations.json. Sers-t'en pour gagner du temps (grep par cle, auteur ou titre) mais reverifie toi-meme sur le web les renvois precis et tout ce qui date de 2025-2026. Si une source est inaccessible, dis-le (not_checked) plutot que de deviner. N'invente jamais le contenu d'une source. Outils web : WebSearch et WebFetch (charge-les avec ToolSearch si besoin). N'envoie aucune information personnelle sur l'auteur du depot dans les requetes.`

const DIMENSIONS = [
  {
    key: 'math-saga',
    scope: `PERIMETRE : les enonces mathematiques de la partie SAGA (pas les chiffres mesures, traites par un autre auditeur).
(a) Lignes 189-194 : la Proposition 5.9 de Bach telle que citee (pas gamma = 1/(4R^2), delta = 3 gamma^2, V = D + delta M, table initialisee en theta_0, contraction 1 - min{1/(3n), 3mu/(16R^2)}). Le livre n'est pas dans le depot : compare avec la transcription utilisee dans theory/B3_prop59_tightness.tex et refais la preuve si necessaire.
(b) Lignes 210-222 : la liste des six inegalites I1-I6, les definitions de mu_eff, L_eff, r_sq, r_var, l'identite (1) avec c_D et c_M, et l'affirmation que les valeurs pire cas redonnent le taux certifie.
(c) (C2) lignes 295-297 : "I1 and I2 together contribute at most a factor 8/3 (cocoercivity gives L_eff <= R^2)" et "mu_eff/mu is bounded only by a condition number".
(d) (C3) lignes 300-302 : "The global strong-convexity constant of ridge logistic regression is exactly lambda" ; "near the optimum mu_eff is at least about the smallest eigenvalue of the Hessian at eta*".
(e) (C4) lignes 318-324 en entier : "the best step and Lyapunov weight gain at most a factor 3 (4/3 when kappa >= 9n/16, the step of Hofmann et al.)" ; la borne inferieure sur n composantes identiques ((kappa-1)/4 * log(1/(2 eps)) iterations, "with any constant step", "its mean iterate being that of gradient descent"). Etablis les hypotheses exactes sous lesquelles cette borne est vraie (point initial, initialisation de la table, quantite bornee) et dis si l'enonce du papier les contient.
(f) (C5) lignes 325-327 et 236-238 : "Proposition 5.9 holds in any fixed metric P with R_P^2, mu_P", les formules de R_P^2 et mu_P, "for ridge logistic regression every P has kappa_P >= kappa".
SOURCES : theory/B3_prop59_tightness.tex, theory/B2_effective_curvature.tex, theory/B1_preconditioned_saga.tex, archive/report/sections/, results/findings.md sections 1 a 4, vrlab/instrumentation.py, vrlab/algorithms.py.
CONTROLES NUMERIQUES ATTENDUS : (1) verifier l'identite (1) par enumeration exacte des n indices sur une petite regression logistique ridge (n = 20, d = 5), en calculant independamment l'esperance conditionnelle de V_{t+1} et le membre de droite ; (2) calculer E||theta_t||^2 pour SAGA sur n composantes identiques (petit n ; recurrence exacte ou Monte-Carlo large) pour plusieurs pas et plusieurs points initiaux, et comparer a la borne de (C4).`,
  },
  {
    key: 'math-nfl',
    scope: `PERIMETRE : le Theoreme 1 et la geometrie des pas polaires.
(a) Lignes 196-206 : definitions (normes, polar, mu_N, L_N, kappa_N, modele (GLM), c_i, pas polaire fixe ou a longueur nucleaire) ; "without momentum this is steepest descent for the spectral norm" (ligne 117) ; "variance-reduced methods are governed by the per-sample ratios L_N(f_i)/mu_N(F)".
(b) Theoreme 1 (lignes 334-347) : enonce, hypotheses, demonstration ligne a ligne ; l'exemple "logistic and softmax losses with nonzero x_i: W_t = t W_1 for almost every W_1" (la courbure de TOUTES les pertes tend-elle vers 0 ?) ; mu_N(F) = lambda/c_N^2 ; L_N(g) >= L_F(g)/c_N^2 ; les deux consequences ; L_F(f_i) = c_i ||x_i||^2 + lambda.
(c) Remarques 349-354 : couverture annoncee (norme spectrale et ses versions preconditionnees, norme l_infini de la descente de signe, metriques a facteurs de Kronecker) ; "the hypothesis is needed (for ridge least squares ||Delta Sigma^{1/2}||_F, Sigma the Hessian, has condition number one)" ; "kappa_op >= min(K,d)" : demontre-le ou refute-le pour toute F, et dis si la reference citee (sidford2018coordinate) contient bien cet enonce ou un analogue.
(d) Identite (2) du pas polaire a longueur nucleaire sur une quadratique (lignes 226-232) : l'identite, J dans [mu_op, L_op], mu_PL dans [mu_op, J], le facteur certifie 1 - 1/kappa_op.
(e) Paragraphe "The same anatomy for polar steps" (309-315) : "(mu_PL/mu_op)(2 - J/L_op) times the certified one", "gains at most a factor 2", "only the sharpness slack is unbounded", "On least squares the curvature is constant and the gap is about a factor 4 on the ray that attracts generic runs", et le resume (ligne 96-97) "the nuclear-scaled polar step has the same anatomy" avec l'etiquette [p] du tableau 1 : qu'est-ce qui est reellement prouve ?
(f) (M4) ligne 367-369 : "Computed once, the polar metric makes the method SAGA in a fixed metric, certified with R_P^2, mu_P" ; formule de la metrique polaire (lignes 240-242).
SOURCES : archive/muon/notes_spectral_theory.tex, muon/notes_theory.md (F10, F11), archive/muon/report_spectral_muon_new.tex.
CONTROLES NUMERIQUES ATTENDUS : identite (2) sur une quadratique matricielle aleatoire ; mu_N numerique sur un petit probleme softmax ou logistique le long de t W_1 pour deux ou trois normes (spectrale, l_infini, Frobenius ponderee) ; borne kappa_op >= min(K,d) sur des quadratiques aleatoires.`,
  },
  {
    key: 'math-smd',
    scope: `PERIMETRE : la theorie des pas polaires stochastiques.
(a) Lignes 244-249 : psi_xi convexe et 1-lipschitzienne pour la norme nucleaire sans hypothese de moment ; Phi_xi(G) dans le sous-differentiel de psi_xi en G, avec egalite au gradient quand G + xi est de plein rang presque surement ; definition des medianes nucleaires.
(b) Lignes 389-393 : argmin psi_xi = -Med(xi) non vide, convexe, compact ; il contient tout zero de Phi_xi ; il est forme de zeros de Phi_xi sous plein rang presque sur ; le contre-exemple scalaire P(xi = 1) = 0.4, P(xi = -1) = 0.6.
(c) Theoreme 2 (lignes 395-416) : enonces (a), (b), (c) et esquisse de preuve, etape par etape : bijection du gradient, lecture miroir dans l'espace des gradients, identite des trois points (signes et ordre des arguments des divergences de Bregman), borne D_F(W_{t+1}, W_t) <= L_op eta^2 / 2, inegalite de sous-gradient, moyenne ponderee, lemme de Robbins-Siegmund, coercivite de psi_xi SANS moment, argument de l'ensemble denombrable dense de minimiseurs, les deux inegalites de (c). Dis pour chaque etape si elle est correcte et si une hypothese manque (mesurabilite, integrabilite, existence de W^dagger, cas polar(0) = 0, independance des xi_t, bruit additif independant de l'iteree).
(d) Lignes 418-420 : "replaces E xi = 0 by Med(xi) = {0}" ; "an offset that no schedule of this class removes".
(e) Proposition 3 (lignes 422-440), items (a) a (d) et leurs preuves ; remarques 442-447 ("For d = 1 the limit in (d) is the geometric median of the y_i/x_i" ; "With lengths proportional to the sampled gradient the step is plain stochastic gradient descent") ; existence et unicite du minimiseur de Lambda ; l'item (c) vaut-il aussi pour SVRG comme ecrit ?
SOURCES : archive/muon/notes_spectral_theory.tex (preuves completes), muon/notes_theory.md, archive/muon/report_spectral_muon_new.tex.
CONTROLES NUMERIQUES ATTENDUS : une simulation (scalaire puis 2x2) verifiant que la limite du Theoreme 2 est un zero du champ moyen pour un bruit asymetrique centre ; un controle de l'item (d) de la Proposition 3 sur un petit probleme (la limite minimise-t-elle Lambda et differe-t-elle des moindres carres ?) ; un test Monte-Carlo de l'inegalite (a) du Theoreme 2.`,
  },
  {
    key: 'math-offset',
    scope: `PERIMETRE : la Proposition 4 et son commentaire (lignes 449-467), et le debut du paragraphe "Numerical checks" (lignes 482-484).
(a) Refais le developpement d'Edgeworth a un terme : correction d'ordre b^(-1/2) de Phi(0) = E polar(sqrt(b) xi_b), derivee du champ moyen en 0 egale a sqrt(b) c_{K,d} / sigma au premier ordre, resolution de Phi(G) = 0. Retrouve la constante de l'equation (3) : facteur kappa_3 / (6 sigma^2 b), a_{K,d} = -E[polar(Z)_11 (Z_11^3 - 3 Z_11)], c_{K,d} = E||Z||_* / (K d), direction 1 1^T, signe. c_{K,d} est-il bien la bonne constante (jacobienne de polar moyennee sous bruit gaussien) ? L'hypothese "i.i.d. entries" suffit-elle pour la forme 1 1^T ?
(b) Cas scalaire : mediane d'une moyenne, kappa_3 / (6 sigma^2 b) ; coherence du signe avec G^dagger = -Med.
(c) "a_{K,d}/c_{K,d} equals 3/(d+2) for K = 1" (calcul exact) ; "max(K,d) a_{K,d}/c_{K,d} lies between 1.4 and 2.8 for ten sizes from 2x2 to 32x32 and 16x64" (Monte-Carlo independant avec erreur statistique) ; "per entry the offset is a factor of order max(K,d) below that of sign descent".
(d) "For 2x3 matrices with entries Exp(1)-1, (3) predicts b ||G_b^dagger||_* -> 0.464" et "the zero of the mean field computed from 1.5e6 exactly sampled batch means gives 0.450-0.498 for b = 2,...,256" : recalcule la prediction ; retrouve les valeurs mesurees dans muon/theory_checks/ (scripts et paper_figures_data.json) et, si le cout le permet, refais un point independamment.
(e) Qualifications : "formal", "non-lattice", "parameter-free", "o(1/b)", "against a discontinuous map" : justes et suffisantes ? L'etiquette [f] du tableau 1 ("a limit gradient of order 1/b") et le resume ("an offset formally of order 1/b") sont-ils coherents avec ce qui est etabli ?
SOURCES : muon/theory_checks/check_offset_first_order.py, check_minibatch_offset_exact.py, paper_figures_data.json ; archive/muon/notes_spectral_theory.tex ; muon/findings.md.`,
  },
  {
    key: 'math-schedule',
    scope: `PERIMETRE : la Proposition 5 (lignes 547-575), les remarques 577-581, la recette (4) (lignes 508-515) et la phrase des lignes 648-650 sur la fonction de Lyapunov du pas a longueur nucleaire (gamma <= 1/(14 n L_max), fait F15).
(a) Item (a) : decomposition de omega - grad F(W), rang un, constante (theta L_max + L_bar), "no rank factor".
(b) Item (b) : refais toute la preuve avec les constantes : inegalite <grad F, polar(omega)> >= ||grad F||_* - 2 ||omega - grad F||_*, lissite spectrale, inegalite de Polyak-Lojasiewicz duale, majoration de la peremption par les A dernieres longueurs avec eta_t = eta_0 rho^(t/2), la constante 10 A L_max, le role de l'hypothese mu_op <= L_max, la recurrence D_{t+1} <= D_t - eta_t sqrt(2 mu_op D_t) + L_eff eta_t^2 / 2, l'argument de convexite ([0, C eta^2] envoye dans [0, C rho eta^2]), l'initialisation (table initiale, t < A), le cas omega_t = 0. Bornes d'age : n/b (ordre cyclique), 2n/b - 1 (permutation fraiche), ceil((n/b) log(nT/delta)) (lots independants ; dependance de A, donc de rho et du schedule, en T et delta).
(c) Item (c) : covariance d'echelle exacte sur moindres carres, chaine de Markov homogene ; "ns5 without epsilon" satisfait-il o(sM) = o(M) ?
(d) Remarques : "The proposition certifies the mechanism, not the speed" ; "It gives no critical q and does not cover momentum" ; la proposition couvre-t-elle la recette (4) telle qu'ecrite (lr_t = lr_0 q^(tb/n) contre eta_t = eta_0 rho^(t/2), eta_0 prescrit) ? Les phrases du resume ("we prove this law for tables of bounded age, at slower cooldowns") et du tableau 1 ("certified for tables of bounded age, at slower cooldowns [p]") disent-elles exactement ce qui est prouve ?
(e) F15 : enonce et preuve dans muon/notes_theory.md ; la phrase du papier le resume-t-elle fidelement ?
SOURCES : muon/notes_theory.md (F12 a F15, a partir de la ligne 483), muon/lab/vrmuon.py pour la recette.
CONTROLES NUMERIQUES ATTENDUS : simuler la recette de l'item (b) (facteur polaire exact, table SAGA, ordre cyclique puis permutation fraiche, eta_t prescrit) sur un petit probleme de moindres carres ridge matriciel (K = 3, d = 4, n = 12) et verifier D_t <= D_0 rho^t a chaque iteration ; verifier la borne (a) sur des etats aleatoires ; verifier (c) en comparant deux trajectoires a deux echelles.`,
  },
  {
    key: 'math-order',
    scope: `PERIMETRE : les enonces sur l'ordre d'echantillonnage et le terme memoire de SAGA que le papier reprend de la note B4.
- Lignes 382-384 : "near the ceiling of SAGA under independent sampling, e^{-1} per pass in expectation, since stale entries keep their error".
- (M7) lignes 597-606 : "memory factor (1-b/n)^(n/b) ~ e^{-1} of the table" ; "Linear SAGA explains part of it (note B4 of the main study; n quadratics with a common Hessian, c = gamma n lambda): as n -> infinity its expected squared error loses min{2c,1} nats per pass under independent sampling, the ceiling of (M3), and up to two under a fresh permutation, min{2, log((2-g)/g)} at a fixed step g = gamma lambda (proved)" ; "single runs under independent sampling lose about 1.6 (measured, n = 1000)" ; "the memory factor bounds the mean error, not a single run".
- Tableau 2, ligne Memory (lignes 637-640) : "At most one nat per pass under independent sampling, two under a permutation (note B4)".
- Conclusion (lignes 679-681) : "a fresh permutation per pass, which lifts the memory rate of SAGA from e^{-1} to nearly e^{-2}".
Pour chaque enonce : retrouve l'enonce exact et sa preuve dans theory/B4_sampling_order.tex, verifie la preuve, et verifie que le papier le restitue fidelement : regime (pas fixe par passe c = gamma n lambda contre pas fixe par iteration g = gamma lambda), limite n -> infini, esperance contre trajectoire typique, unite (nats par passe de l'erreur quadratique), classe de problemes (hessiennes identiques). Signale toute phrase du papier plus forte que ce qui est prouve ou mesure (par exemple une affirmation generale sur "SAGA" qui n'est etablie que sur le modele lineaire).
SOURCES : theory/B4_sampling_order.tex, results/findings.md section 9, vrlab/order_model.py, results/B4_summary.json, results/B4b_summary.json, experiments/exp_B4_sampling_order.py, exp_B4b_refinements.py.
CONTROLES NUMERIQUES ATTENDUS : avec vrlab/order_model.py ou un script autonome, calculer le taux exact par passe du second moment pour des indices i.i.d. et pour une permutation fraiche (n = 100 et 1000) a plusieurs c et g, comparer a min{2c,1} et min{2, log((2-g)/g)} ; retrouver le chiffre 1.6.`,
  },
  {
    key: 'num-f1',
    scope: `PERIMETRE : tous les chiffres et descriptions de protocole de la partie SAGA : resume (lignes 94-96), tableau 1 ligne 1 (lignes 150-151), protocole (lignes 222-226 : n = 1000, d = 50, kappa de 10 a 1e5, spectres j^(-alpha), cinq graines, passes 2 a 20), legende de la figure 1 (283-286), (C1) a (C5) (lignes 290-331).
Liste chaque chiffre (3.2-3.7 ; 19-220 ; 3.2-220 ; 77-96 % ; 39-44 % ; 32-37 % ; 0.6 % ; 2.45-2.62 ; 81 ; 91-96 % ; huit jeux de donnees, 64-98 %, 13 a 198 ; 59-96 % ; 2.8 et 4.1 contre 5.2 et 15.7 ; facteur 3, 4/3 ; 17 % ; 4.5 ; gamma R^2 = 4 et "eight times" ; 1.00 -> 1.11 ; 0.61 -> 1.26 ; 0.13 -> 0.33 ; 0.42 -> 0.017 ; kappa = 10 n pour alpha dans {0,1}) et retrouve-le dans les donnees : results/B3_summary.json, B3_families_summary.json, B2_summary.json, B2_checks_summary.json, B1_summary.json, B1_coherence_summary.json, A3_summary.json, GammaDelta_summary.json, results/digest.md ; a defaut results/findings.md et archive/report/sections/.
Recalcule les plages (minimum et maximum sur quelles configurations ? lesquelles sont exclues ?), les arrondis, les pourcentages. Verifie que le texte decrit bien la quantite calculee ("exact contraction exponent ... times the certified one" : quel rapport, quelle fenetre, quelle agregation sur les graines ; "share of the logarithmic gap" ; "tightness of I2 / I4" ; "observed exponent per pass").
Regarde paper/figures/figR1_classical.png (outil Read) et lis paper/scripts/make_figR1_classical.py : la legende et les phrases qui renvoient a la figure decrivent-elles ce qui est trace ?
Verifie l'ordre de retrait des inegalites annonce (I5, I4, I2, I3, I1) contre le code (vrlab/analysis.py, experiments/exp_B3_tightness.py) et la reserve de la section Limitations (lignes 667-668) sur la dependance des parts a cet ordre.`,
  },
  {
    key: 'num-geometry',
    scope: `PERIMETRE : (M4) et (M3) (lignes 367-384), la legende de la figure 2 (359-363), le paragraphe "Geometries" (236-242), le paragraphe "Experiments" (253-265 : estimateurs, methodes, six instances, n, K x d, cibles 1e-6 et 1e-9, "Each family is tuned on five seeds and run on ten fresh ones, ten families also on two further draws of the data; differences below about 10% are not resolved"), tableau 1 ligne 2 colonne "Added here" (lignes 156-158).
Chiffres : 17.5 (551 contre 9659) ; 1.6 a 103 ; 7.6 a 163 ; 42 contre 4364 ; 13.3 / 27.1 / 13.1 passes ; 69.5 / 98.7 ; 22.1-29.0 / 24.4-43.5 ; 15.1-33.8 / 16.1-31.9 ; 27.1 / 31.9 ; 0.20-0.37 ; 0.78-1.08 ; 16-18 et 17-21 passes ; "best ratio to the Euclidean one exactly 1" ; "one seed of the joint method misses the target" ; "the covariance metric is the best method on least squares for every draw, and on softmax for one draw of three" ; R^2/mu "of the order of n" et "from 9n to 50n".
Retrouve chacun dans muon/results/M7_summary.json, M1_summary.json (et M1_pass*_summary.json), M8_summary.json, M5b_summary.json, muon/results/digest.md ; a defaut muon/findings.md. Verifie les definitions (quelle famille est "the best method recomputing the polar factor", "the best Euclidean method without preconditioner", quelles instances sont "memory-limited" et "conditioning-limited"), la coherence des comparaisons avec la resolution annoncee de 10 % et avec le nombre de graines, et que la figure paper/figures/figP2_geometry.png (outil Read ; script paper/scripts/make_figP2_geometry.py) montre ce que disent la legende et le texte.
Verifie enfin les intitules de (M3) "orthogonalization buys a trajectory effect that Euclidean devices reproduce" et de (M4) "A frozen polar metric certifies a gain only where Theorem 1 does not apply" contre les chiffres : la conclusion suit-elle des donnees, et sur combien d'instances ?`,
  },
  {
    key: 'num-noise',
    scope: `PERIMETRE : (M1) et (M2) (lignes 491-503), le paragraphe "Numerical checks" a partir de "On the finite sum of Figure 3" (lignes 484-488), la legende de la figure 3 (472-479), la derniere phrase du paragraphe "Stochastic polar steps" (249-251), tableau 1 lignes 3 et 4 colonne "Added here".
Chiffres : 0.170 ; 2.6e-3 ; 2.1e-4 ; "within 5 %" pour b <= 32 ; 0.17 vers un plateau pres de 0.3 ; "ends at rounding level" ; 3.8 eta^2 (softmax) et 6.0 eta^2 (moindres carres) ; "the floor of minibatch Muon is linear in its learning rate" ; 1.2e-1 et 4.8e-2 (b = 32) ; "a momentum buffer as in MARS-M keeps a floor" ; rang nucleaire 1.07 ; 11.8 contre 11.9 ; SAGA 20.2 ; "fails where MISO fails (kappa >> n)" ; les parametres de la legende (eta = 1e-2, lr = 1e-2, b = 32, n = 2000, K = 2, d = 3, eta_t = 0.3 (1 + t/200)^(-3/4), moyenne sur (t/2, t] et 32 essais, 64 essais).
Retrouve chacun dans muon/results/M3_summary.json, M6_summary.json, M2_summary.json, M4_summary.json, muon/theory_checks/paper_figures_data.json, muon/results/digest.md ; a defaut muon/findings.md sections 5.1, 5.2, 5.7.
Verifie aussi par le calcul : la borne citee "max(F(W_t), F(W_{t+1})) - F_* >= mu_op eta^2 / 8 whenever the step is nonzero" (redemontre-la ou refute-la) ; "gamma (m + theta c) with theta = 1/n, gamma = 1/lambda is MISO" (lignes 255-256) et "the predictable part m is the gradient of an isotropic quadratic" (lignes 499-500).
Verifie que la figure paper/figures/figP3_noise.png (outil Read ; script paper/scripts/make_figP3_noise.py) montre ce que disent la legende et le texte, et que les conclusions "Only a finite-sum estimator reaches rounding precision" et "it is not a spectral method" suivent des donnees (combien d'instances, quelles alternatives testees).`,
  },
  {
    key: 'num-m5',
    scope: `PERIMETRE : la recette (4) et (M5) (lignes 508-544), la legende de la figure 4 partie (a) (lignes 520-523), les chiffres des remarques sur la Proposition 5 (lignes 578-581 : "3.3e4 to 8.9e4 times slower, in log(1/q), than q = 0.8" ; "the error of the estimator is 2.3 to 3.5 times the gradient in nuclear norm"), les lignes 648-650 ("1e4 to 1e5 times slower" ; "gamma <= 1/(14 n L_max), 350 to 1400 times below the tuned steps"), le resume (lignes 102-105) et le tableau 1 ligne 5.
Chiffres de (M5) : rapport a C lr_t^2 variant d'un facteur 3.2 a 5.7 ; planchers 2.8e-29 (softmax) et 2.3e-29 (moindres carres) ; "near 1e-18" pour la recette de Muon a q = 0.8, lr_0 = 0.1 ; 1.5e-29 et 4.1e-30 avec epsilon = 1e-30 ; gel a 8 % a 93 % de la distance initiale ; b = 1, 32 ; MARS-M et STORM-Muon ; 13.4-25.5 passes ; 12.9-24.4 ; 13.5-16.1 "but for a tie" ; 7/10, 9/10, 9/10 ; 21.3-26.4 ; 300 passes, cinq graines ; "the two non-square conditioning-limited instances" contre "the four conditioning-limited instances".
Retrouve chacun dans muon/results/M9_summary.json, M5b_summary.json, M1_summary.json, muon/results/digest.md (partie M9), le cache muon/results/M9_law.pkl s'il existe ; a defaut muon/findings.md section 5.11. Recalcule 3.3e4-8.9e4 a partir des constantes des instances (mu_op, L_op, L_max, n, b, A) : quelle valeur de A, quelle definition de "slower in log(1/q)" ? Recalcule 350-1400.
Verifie dans le code (muon/lab/polar.py, vrmuon.py, muon/experiments/exp_M9_schedule.py) que ns5 a cinq iterations et epsilon = 1e-7 par defaut, que "Muon's recipe" = ns5 avec moment de Nesterov 0.9, que l'explication "a floor of ns5, whose constant epsilon breaks the scale invariance" est etayee par l'essai epsilon = 1e-30, et que "the rounding floor" pour 2.8e-29 est plausible (distance au carre relative ou absolue ? precision machine).
Verifie la figure paper/figures/figP4_recipe.png partie (a) (outil Read ; script paper/scripts/make_figP4_recipe.py) contre la legende et le texte.
Dis si "With a table the cooldown converges; without one it freezes the iterate" est etabli avec l'equite voulue : les concurrents sans table ont-ils ete regles sur une grille (q, lr_0) ou seulement evalues au schedule choisi pour une autre methode ? Le resume ("where minibatch and momentum-based estimators freeze") est-il couvert ?`,
  },
  {
    key: 'num-m6m7',
    scope: `PERIMETRE : (M6) et (M7) pour leurs chiffres experimentaux (lignes 584-609), la legende de la figure 4 parties (b) et (c) (lignes 524-528), le resume (lignes 105-107 : "0.58-0.93", "36-49 %", "batches of 20 and 50"), le tableau 2 ligne Memory ("36-49 % fewer passes where memory limits (b = 20, 50)"), la section Limitations (lignes 668-669 : "(M3)-(M7) rest on two to four instances and one to three draws of the data").
Chiffres de (M6) : 0.92-1.06 ; 1.16-1.36 ; 0.58-0.93 ; 0.41-0.89 ; 0.81-1.04 ; 0.74 et 0.75 (b = 4 et 16, softmax) ; 0.74 et 0.55 (moindres carres) ; 0.91 et 1.04 a b = 1 ; 1.30 et 1.11 a b = 1 et 4 sur l'instance carree ; "1.7 to 2.7 times slower at b = 16" sur grilles de facteur 2 ; "the spectral maps need no momentum while the controls do".
Chiffres de (M7) : rho_star = 0.27-0.43 pour b = 5 a 50 ; 0.10-0.18 a b = 20 et 50 ; passes 16.7-20.0 vers 9.1-12.2 ; 1.2-1.3 et 1.7-1.9 nats ; "about 1.6" ; 1.7 a 2.7 fois et 1.0 a 1.3 fois les passes du modele ; "four of five seeds" (lignes 596-597) contre "every seed" dans la legende (ligne 527) : coherent ? "36-49 %" se deduit-il de 16.7-20.0 vers 9.1-12.2, instance par instance et lot par lot ?
Retrouve chaque chiffre dans muon/results/M9_summary.json, M9b_summary.json, M5b_summary.json, muon/results/digest.md ; a defaut muon/findings.md sections 5.11 a 5.13. Pour chaque rapport, identifie numerateur, denominateur, instances, tailles de lot, graines, et recalcule. Confronte chaque comparaison a la resolution annoncee (10 %, grilles de raison 2^(1/4)) et au nombre de graines : quelles affirmations sont dans le bruit ?
Verifie la figure paper/figures/figP4_recipe.png parties (b) et (c) (outil Read ; script paper/scripts/make_figP4_recipe.py).
Dis si l'intitule "Minibatches: where the spectral map pays" et la phrase du resume suivent des donnees (combien d'instances, quels controles, cout des factorisations exclu) et si "a fresh permutation per pass lifts the memory ceiling" est etabli au-dela du modele a hessiennes identiques.`,
  },
  {
    key: 'code-fidelity',
    scope: `PERIMETRE : la fidelite du code aux descriptions du papier et l'equite des comparaisons.
(a) Algorithmes de muon/lab/ (algorithms.py, vrmuon.py, polar.py, mvr.py, katyusha.py, maps.py) contre les descriptions des lignes 196-206, 253-262, 508-515 : estimateur SAGA omega = m + theta c, mise a jour de la table, pas polaire fixe / a longueur nucleaire / ecrete, carte "soft", metrique polaire figee, B-SAGA, echantillonnage d'importance, MISO, ns5 (coefficients de l'implementation de Muon de Jordan et al., normalisation, epsilon = 1e-7, cinq iterations), moment de Nesterov 0.9, MARS-M, STORM-Muon, controles (normalisation de Frobenius, signe coordonnee par coordonnee). Signale tout ecart entre ce que le papier dit et ce que le code fait, et tout bug qui changerait un resultat.
(b) Protocole : grilles de pas, graines de reglage et de confirmation, critere d'arret (distance au carre relative 1e-6 / 1e-9), comptage des passes (cout initial de la table compris ?), tirage des lots (indices distincts dans un lot ? "fresh permutation per pass"), dans muon/experiments/common.py, families.py, exp_M1_headtohead.py, exp_M5b_batch_fine.py, exp_M9_schedule.py, exp_M9b_order_model.py. Les concurrents (SAGA, B-SAGA, controles, MARS-M, STORM-Muon, minibatch) sont-ils regles avec le meme soin que les methodes mises en avant ?
(c) Reproduction : rejoue de facon independante 2 ou 3 resultats de tete a petite echelle avec le code du depot, dans les limites de calcul (par exemple : plancher pres de 1e-29 du facteur polaire exact avec table contre gel sans table sur une instance, lignes 532-539 ; MISO contre SAGA-Muon joint, lignes 501-503 ; un rapport de (M6)). Compare aux valeurs du papier.
(d) Partie SAGA : vrlab/algorithms.py (saga et ses ordres d'echantillonnage), vrlab/instrumentation.py : l'enumeration "exacte" des esperances conditionnelles est-elle exacte ? Le pas "textbook" est-il gamma = 1/(4 R^2) avec R^2 = max_i (||x_i||^2/4 + lambda) ?`,
  },
  {
    key: 'cite-vr',
    effort: 'medium',
    scope: `PERIMETRE : bibliographie, premier groupe (methodes a variance reduite, sommes finies, theorie classique). Cles : allenzhu2018katyusha arjevani2016lower bach2024learning defazio2014finito defazio2014saga dragomir2021fast driggs2019biased gonen2016solving hofmann2015variance johnson2013accelerating kovalev2020dont lan2018optimal mairal2015incremental mokhtari2018surpassing na2023hessian park2020linear qin2023convergence xiao2014proximal yang2014data ying2017variance zhang2013gradient.
Renvois precis a ouvrir en priorite : "Theorem 8" d'arjevani2016lower et "the order of one-step stationary methods" (ligne 323-324) ; "kappa < 4.08" pour mokhtari2018surpassing (ligne 608-609) ; "the step of hofmann2015variance" dans (C4) (ligne 319) ; "the designs of na2023hessian" (ligne 297-298) ; Proposition 5.9 et section 5.4.4 de bach2024learning (enonce exact du taux) ; "restricted-secant constant of zhang2013gradient" (ligne 213-214) ; "(n + sqrt(n kappa)) log(1/eps)" pour allenzhu2018katyusha, kovalev2020dont, lan2018optimal (ligne 115) ; dragomir2021fast = "SAGA in a fixed metric" ; yang2014data et gonen2016solving = "preconditioning where the loss curvature is bounded below" ; qin2023convergence = "SVRG anchors under reshuffling, without linear rate" (sign steps ?) ; ying2017variance et park2020linear = taux lineaires sous ordre remanie ou cyclique ; defazio2014finito = "a permutation order was seen to help" ; driggs2019biased = B-SAGA (SAGA amorti) ; xiao2014proximal = echantillonnage d'importance ; mairal2015incremental = MISO.
${CITE_METHOD}`,
  },
  {
    key: 'cite-stat',
    effort: 'medium',
    scope: `PERIMETRE : bibliographie, deuxieme groupe (signe, medianes, pas normalises, miroir, outils). Cles : balles2020geometry bernstein2019majority cardot2013efficient carlson2015stochastic chen2020distributed chzhen2023signsvrg cornish1938moments cutkosky2019momentum davis2018subgradient duchi2012randomized goffin1977convergence hu2017unified karimireddy2019error maddison2021dual minsker2015geometric sidford2018coordinate taylor2017smooth.
Renvois precis a ouvrir en priorite : sidford2018coordinate cite pour "kappa_op >= min(K,d)" (ligne 354) : que dit exactement la source (norme l_infini ? enonce ? numero) ; balles2020geometry cite pour les deux inegalites 2 mu_op (F - F_*) <= ||grad F||_*^2 <= 2 L_op (F - F_*) (ligne 415) ; maddison2021dual cite pour la lecture miroir dans l'espace des gradients (ligne 410) ; duchi2012randomized (lissage aleatoire, sous-gradient du lisse) (ligne 247) ; minsker2015geometric (medianes dans un espace norme) (ligne 248-249) ; chen2020distributed et bernstein2019majority ("sign methods follow a median", condition de bruit) (lignes 160-161, 419, 442) ; cardot2013efficient (mediane geometrique, pas normalises) ; karimireddy2019error ("batch-one sign descent need not converge") (ligne 444) ; cornish1938moments ("the scalar offset kappa_3/(6 sigma^2 b) of the median of a mean") ; cutkosky2019momentum (STORM) ; chzhen2023signsvrg ("SVRG in sign steps") ; davis2018subgradient et goffin1977convergence ("Geometric schedules of fixed lengths with exact gradients") ; taylor2017smooth et hu2017unified ("worst cases by performance estimation") ; carlson2015stochastic ("stochastic spectral descent", pas a longueur nucleaire avec gamma = 1/L_op).
${CITE_METHOD}`,
  },
  {
    key: 'cite-muon',
    scope: `PERIMETRE : bibliographie, troisieme groupe (litterature Muon 2024-2026, la plus fragile). Cles : bernstein2024old chen2026when do2026muon hird2025quantifying huang2026muon jawanpuria2026constant jiang2026adaptive jordan2024muon lau2025polargrad li2026implicit liu2025marsm ma2026preconditioning paquette2026phases shulgin2026quadratic vasudeva2025muon xie2026muon zhu2026short.
Renvois precis a ouvrir en priorite : "Eq. (17)" de shulgin2026quadratic et "the gap is about a factor 4 on the ray that attracts generic runs" (lignes 231, 313-314) ; "Eq. (8)", "Theorem 3" (facteur 1 - 1/kappa_op) et "Lemma 4" (plancher mu_op eta^2 / 8) de jawanpuria2026constant (lignes 231-232, 492) ; "Proposition 7" de hird2025quantifying (ligne 350-351) ; "Lemma H.1" et "Theorem 4.10" de li2026implicit (lignes 439, 445) et "SVRG in polar steps" ; "Corollary 3.1, Theorem 3.2" de huang2026muon (ligne 660-662) ; ce que le papier attribue a do2026muon (lignes 162-163 et 465-467 : dimension un, non-convergence sous donnees bornees asymetriques, toute limite est un zero du champ moyen) ; xie2026muon et chen2026when ("Noise smooths the polar map", derivee du champ moyen en 0) ; paquette2026phases ; vasudeva2025muon et shulgin2026quadratic (moindres carres) ; liu2025marsm (MARS-M : reduction de variance a moment pour Muon) ; lau2025polargrad (PolarGrad = pas a longueur nucleaire ; planchers des longueurs fixes) ; ma2026preconditioning (schedules geometriques pour Muon et la descente spectrale) ; zhu2026short ("a Euclidean staleness lemma for SAG and SAGA tables") ; jiang2026adaptive ; bernstein2024old (Muon sans moment = descente la plus raide pour la norme spectrale) ; jordan2024muon (cinq iterations de Newton-Schulz, moment de Nesterov, constante de normalisation 1e-7).
Pour chaque cle, verifie aussi que l'article existe bien avec ce titre, ces auteurs et cette annee (identifiant arXiv), et que la version consultee numerote les resultats comme le papier les cite (les numerotations changent entre versions arXiv : indique la version).
${CITE_METHOD}`,
  },
  {
    key: 'novelty',
    scope: `PERIMETRE : les affirmations de nouveaute et de positionnement. La section 2.4 (lignes 267-274) dit ce qui n'a "pas ete trouve" ; voir aussi le tableau 1 (colonne "Published" contre "Added here", lignes 137-183), les lignes 418-420, 442-447, 607-609 et la section 4.2 (lignes 652-663).
Contributions revendiquees : (1) Theoreme 1 : aucune norme n'ameliore le conditionnement pire cas quand la courbure de la perte s'annule le long d'une suite ; (2) Theoreme 2 : les pas polaires stochastiques sont une descente miroir stochastique sur la norme nucleaire lissee par le bruit et convergent vers (moins) une mediane nucleaire du bruit ; (3) Proposition 3(d) : au lot un sur moindres carres, limite = estimateur des moindres distances ; (4) Proposition 4 : decalage d'ordre 1/b et sa constante ; (5) Proposition 3(c), table SAGA dans un pas polaire a longueur nucleaire, lecture MISO (M2), metrique figee a partir d'une decomposition polaire ; (6) Proposition 5 : longueurs fixes sur refroidissement geometrique alimentees par une table SAGA, taux lineaire ; (7) (M7) : permutation fraiche et plafond memoire de la table ; (8) (C1)-(C3) : instrumentation exacte de la preuve de SAGA inegalite par inegalite.
Pour chacune, cherche sur le web (arXiv 2023-2026 surtout ; mots-cles : Muon, spectral descent, polar factor, orthogonalized update, steepest descent spectral norm, sign descent median, normalized SGD bias, nuclear norm smoothing, SAGA Muon, variance reduction Muon, MARS-M, finite-sum Muon, reshuffling SAGA, incremental aggregated gradient staleness, preconditioning lower bound condition number norm) un travail anterieur qui contient deja le resultat, un resultat plus fort, ou qui devrait etre cite. Pour chaque contribution : verdict (deja publie / proche et a citer / rien trouve), references precises avec URL et passage, et jugement sur la formulation du papier ("was not found", "not found", portee de la recherche annoncee : assez prudente et honnete ?).
Le depot contient muon/literature_check.md (recherches des 12-18 novembre 2026) : lis ses verdicts (lignes 9-36 et 1277-1400) pour ne pas refaire ce qui est fait, puis concentre-toi sur ce qui a pu etre manque et sur les travaux les plus recents. Ne rapporte une anteriorite que si tu as ouvert la source et verifie le passage. Outils web : WebSearch et WebFetch (charge-les avec ToolSearch si besoin). N'envoie aucune information personnelle sur l'auteur du depot dans les requetes.`,
  },
  {
    key: 'consistency',
    scope: `PERIMETRE : coherence interne et rigueur de la redaction, a l'echelle du papier entier.
(a) Chaque chiffre ou affirmation repete (titre, resume, introduction, tableaux 1 et 2, corps, discussion, conclusion) est-il identique partout, avec les memes qualifications ? Dresse la table des repetitions et de leurs ecarts.
(b) Etiquettes de statut [p] [m] [e] [f] du tableau 1 et section 2.4 : chaque element etiquete [p] a-t-il un enonce et une preuve, dans le papier ou dans une note du depot effectivement citee ? Des elements presentes comme prouves ne sont-ils que des "proof sketch" renvoyant a des notes ? Les renvois ("Details: theory notes of the Muon study, F12-F14", "the main report proves") pointent-ils vers des preuves qui existent (verifie l'existence, pas le detail) ?
(c) Sur-affirmations : titre ("no norm repairs the logistic worst case", "variance reduction acts on the fixed point"), resume, intitules des constats, conclusion : chaque formulation est-elle couverte par ce qui est demontre ou mesure, avec ses hypotheses (fortement convexe, donnees synthetiques, deux a quatre instances, pas de moment dans les theoremes, bruit additif) ? Repere les generalisations ("Muon", "the spectral map pays", "SAGA") que les experiences ou les preuves ne portent pas.
(d) La section Limitations (lignes 667-672) couvre-t-elle les limites reelles (cout des factorisations, taille des problemes, reglage, nombre de graines, ordre de retrait des inegalites, hypotheses des theoremes) ?
(e) Definitions et notations : chaque symbole est-il defini avant usage et employe de facon coherente (mu_eff et L_eff de SAGA contre L_eff de la Proposition 5 ; rho de la Proposition 5, rho du pas ecrete, rho_star ; kappa_3 contre kappa ; theta et vartheta ; eta, gamma, lr ; delta ; D ; c_i (constante de courbure, ligne 202) contre c_i (matrice, ligne 569) ; g et g_j) ?
(f) Compilation : copie paper/ dans un repertoire temporaire (hors du depot ou sous paper/audit/checks/build_consistency/), compile (pdflatex, bibtex, pdflatex deux fois), releve avertissements, references ou citations non resolues, depassements de boite importants, nombre de pages (12 attendu), entrees .bib non citees et cles citees absentes.
(g) Lis le PDF paper/paper_saga_to_muon.pdf page par page (outil Read avec pages) pour reperer les problemes de rendu (tableaux, figures, legendes, formules tronquees) et verifier que chaque figure et chaque tableau est appele dans le texte.
(h) Paragraphe "Reproducibility" (lignes 683-690) et commentaires d'en-tete : les chemins et scripts cites existent-ils dans le depot (archive/report/main.pdf, muon/notes_theory.md, archive/muon/..., theory/main.pdf, exp_M1 a exp_M9b, exp_B3, exp_B4, paper/scripts/) ? Le paper/README.md dit-il la meme chose que le papier (55 entrees, 12 pages, sources des chiffres) ?`,
  },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    scope_covered: { type: 'string' },
    checked_ok: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          location: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['claim', 'evidence'],
      },
    },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          location: { type: 'string' },
          claim: { type: 'string' },
          problem: { type: 'string' },
          evidence: { type: 'string' },
          proposed_fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['id', 'severity', 'location', 'claim', 'problem', 'evidence', 'proposed_fix', 'confidence'],
      },
    },
    not_checked: { type: 'array', items: { type: 'string' } },
    report_path: { type: 'string' },
  },
  required: ['scope_covered', 'checked_ok', 'findings', 'not_checked', 'report_path'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'downgraded', 'refuted', 'uncertain'] },
          severity_after: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit', 'none'] },
          reasoning: { type: 'string' },
          independent_evidence: { type: 'string' },
          final_fix: { type: 'string' },
        },
        required: ['id', 'verdict', 'severity_after', 'reasoning', 'independent_evidence', 'final_fix'],
      },
    },
  },
  required: ['verdicts'],
}

function finderPrompt(d) {
  return COMMON + '\n\nCLE DE TA DIMENSION : ' + d.key + '\nTon rapport : ' + EVID + '/' + d.key + '.md\n\n' + d.scope
}

function verifierPrompt(d, serious) {
  return `Tu es un contre-verificateur independant dans l'audit du papier ${ROOT}/paper/paper_saga_to_muon.tex (niveau soutenance de these). Un auditeur de la dimension "${d.key}" a rendu les constats bloquants ou majeurs ci-dessous. Son rapport detaille est ${EVID}/${d.key}.md.

Ta tache : pour CHAQUE constat, travaille comme un rapporteur qui chercherait a le refuter. Relis le passage vise du papier (les numeros de ligne sont ceux de l'outil Read), relis la source ou refais le calcul toi-meme, sans te fier au raisonnement de l'auditeur ; verifie aussi que l'auditeur n'a pas mal lu le papier (reserve presente ailleurs dans le texte, definition donnee plus haut, arrondi legitime). Verdicts :
- confirmed : tu as reproduit le probleme de facon independante ;
- downgraded : le probleme existe mais la severite est surestimee (donne la bonne dans severity_after) ;
- refuted : le papier est correct ou le constat repose sur une erreur (explique laquelle) ;
- uncertain : tu n'as pas pu trancher (dis ce qui manque).
Echelle de severite : blocker = enonce de tete faux ou preuve fausse ; major = chiffre, renvoi ou attribution faux, hypothese manquante qui rend un enonce faux tel qu'ecrit, sur-affirmation qu'un jury releverait ; minor = imprecision ou reserve manquante ; nit = forme ; none = pas de probleme.
Pour chaque constat confirme ou retrograde, donne dans final_fix la correction minimale exacte : texte LaTeX de remplacement, en anglais, dans le style du papier, sans allonger le papier de plus de quelques mots (il est a la limite de 12 pages) ; chaine vide si aucune correction de texte n'est necessaire.

Regles : lecture seule sur le depot ; tu peux ajouter une section "Contre-verification" a la fin de ${EVID}/${d.key}.md et des scripts dans ${CHECKS}/ (prefixe verif-${d.key}-). Python : python3 du systeme avec OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, un seul processus, chaque calcul sous 3 minutes de CPU. Outils web disponibles (WebSearch, WebFetch via ToolSearch) si un constat porte sur une source externe ; n'envoie aucune information personnelle sur l'auteur du depot. Budget : 20 a 25 minutes.

CONSTATS A CONTRE-VERIFIER (JSON) :
${JSON.stringify(serious, null, 1)}`
}

phase('Audit')
log('Audit du papier : ' + DIMENSIONS.length + ' auditeurs, puis contre-verification des constats bloquants ou majeurs')

const results = await pipeline(
  DIMENSIONS,
  d => agent(finderPrompt(d), {
    label: 'audit:' + d.key,
    phase: 'Audit',
    schema: FINDINGS_SCHEMA,
    effort: d.effort || 'high',
  }),
  (res, d) => {
    if (!res) {
      log('auditeur ' + d.key + ' : pas de resultat (voir ' + EVID + '/' + d.key + '.md pour le partiel)')
      return { key: d.key, audit: null, verification: null }
    }
    const serious = (res.findings || []).filter(f => f.severity === 'blocker' || f.severity === 'major')
    log('auditeur ' + d.key + ' : ' + (res.findings || []).length + ' constats dont ' + serious.length + ' bloquants ou majeurs')
    if (!serious.length) return { key: d.key, audit: res, verification: { verdicts: [] } }
    return agent(verifierPrompt(d, serious), {
      label: 'verif:' + d.key,
      phase: 'Contre-verification',
      schema: VERDICT_SCHEMA,
      effort: 'high',
    }).then(v => ({ key: d.key, audit: res, verification: v }))
  }
)

const out = results.map((r, i) => r || { key: DIMENSIONS[i].key, audit: null, verification: null })
const missing = out.filter(r => !r.audit).map(r => r.key)
const unverified = out.filter(r => r.audit && !r.verification).map(r => r.key)
if (missing.length) log('Dimensions sans resultat d auditeur : ' + missing.join(', '))
if (unverified.length) log('Dimensions dont la contre-verification a echoue : ' + unverified.join(', '))
return { results: out, missing, unverified }
