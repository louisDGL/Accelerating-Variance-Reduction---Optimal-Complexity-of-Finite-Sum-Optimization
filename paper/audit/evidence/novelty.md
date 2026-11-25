# Audit `novelty` : affirmations de nouveauté et de positionnement

Papier : `paper/paper_saga_to_muon.tex` (version du 20 novembre 2026, 700 lignes ; numéros de ligne = outil Read).
Auditeur : dimension `novelty`. **Statut : terminé** (20 novembre 2026).

**Bilan : 0 bloquant, 0 majeur, 8 mineurs, 3 nits.** Aucune antériorité trouvée qui contienne déjà un des résultats
revendiqués. Les formulations « not found » du papier sont étroites et la section 2.4 annonce honnêtement ce qui n'a
été cherché que sous une forme antérieure. Les constats portent sur des travaux proches non cités (dont cinq figurent
déjà dans les registres du dépôt) et sur une clause trop large (l. 656).

## 1. Périmètre couvert et méthode

- Texte : section 2.4 (l. 267-274), tableau 1 (l. 137-183), l. 418-420, l. 442-447, l. 465-467, l. 606-609,
  section 4.2 (l. 652-663) ; les six occurrences de « not found » (l. 270, 274, 420, 446, 656, 663).
- Huit contributions : (1) Théorème 1 ; (2) Théorème 2 ; (3) Prop. 3(d) ; (4) Prop. 4 ; (5) Prop. 3(c), table SAGA
  dans le pas polaire à longueur nucléaire, lecture MISO (M2), métrique figée ; (6) Prop. 5 ; (7) (M7) ; (8) (C1)-(C3).
- Registres internes lus pour ne pas refaire le travail : `muon/literature_check.md` (verdicts l. 9-36 ; addendum du
  18 novembre l. 1277-1412 ; section « Frozen polar metric » l. 39-59), `bibliography/literature_review.md`
  (sections 6, 8, 10), `muon/findings.md` (l. 107-225), `archive/muon/notes_spectral_theory.tex` (remarques
  d'attribution l. 289-302, 1076-1135), `archive/muon/report_spectral_muon_new.tex` (l. 284-291).
- Recherches propres (20 novembre 2026) : 11 requêtes WebSearch ; 6 requêtes au niveau des résumés de l'API arXiv
  (`export.arxiv.org/api/query`, triées par date, jusqu'aux dépôts du 18 novembre 2026) ; 18 PDF arXiv téléchargés et
  lus en texte (`pdftotext`) aux passages cités ci-dessous ; 10 pages de résumé arXiv lues par l'outil de fetch, dont
  4 sans lecture du PDF (Auger-Hansen, Poon et al., Davis-Drusvyatskiy, Négiar et al. : signalé « résumé seul »).
- Script : `paper/audit/checks/novelty_close_works.py` (seuil 4.08 ; présence des travaux proches dans le `.tex`, le
  `.bib` et les registres du dépôt ; lignes « not found »).

Limites de mes recherches : moteur web généraliste et recherche arXiv sur les résumés seulement (pas de plein texte,
pas d'OpenReview, pas d'ateliers) ; « rien trouvé » veut dire « rien trouvé par ces requêtes ».

## 2. Verdict par contribution

| # | Contribution | Verdict | Travaux les plus proches (ouverts) | Formulation du papier |
|---|---|---|---|---|
| 1 | Théorème 1 (aucune norme n'améliore le pire cas) | rien trouvé | Diakonikolas-Guzmán arXiv:2101.11041 (« basic fact » : pas de conditionnement indépendant de la dimension en norme l_p, p≠2) ; Hird-Livingstone arXiv:2312.04898 (pas d'énoncé pour la logistique ridge : `grep logistic` ne donne que le g-prior) ; Davis-Drusvyatskiy arXiv:2512.04299 (critère par pas, **à citer**, constat N5) | prudente : l. 273 dit « for right-preconditioned spectral norms only » |
| 2 | Théorème 2 (descente miroir stochastique, médiane nucléaire) | rien trouvé pour les pas polaires | Xie 2609.32861, Chen et al. 2610.31728 (bruit gaussien, réponse moyenne ; ni médiane ni norme lissée) ; Beneventano et al. 2606.08388 ; Do-Dereich-Jentzen 2608.04607 (dimension un + borne matricielle, constat N8) ; Li-Tang-Zou 2602.11557 (constat N6) | honnête : l. 272 « searched only in an earlier form » ; analogues scalaires crédités l. 419-420 |
| 3 | Prop. 3(d) (lot un, moindres distances) | rien trouvé pour K,d≥2 ; **proche et à citer** pour K=1 | Song-Steinke-Thakkar-Thakurta arXiv:2006.06783, lemme 5.1 (constat N4) ; Li-Tang-Zou th. 4.10 et lemme H.1 (cités) | exacte mais muette sur K=1, d≥2 |
| 4 | Prop. 4 (décalage 1/b et constante) | rien trouvé | Do-Dereich-Jentzen th. 1.2 : E‖Θ_n−ϑ‖² ≤ C(M⁻¹+γ_n) en dimension matricielle (constat N8) | honnête (« formal », constante « searched only in an earlier form ») |
| 5 | Prop. 3(c), table SAGA dans le pas à longueur nucléaire, (M2), métrique figée | rien trouvé | CacheMuon arXiv:2606.16371 (transformée polaire en cache, constat N2) ; Xie et al. arXiv:2205.02273 (SAGA en normes non euclidiennes, nit N11) | clause l. 656 trop large (N2) |
| 6 | Prop. 5 (longueurs fixes, refroidissement géométrique, table SAGA) | proche (les deux moitiés sont publiées séparément, comme le dit le registre du 18 novembre) ; rien de plus proche trouvé | Singh-Mishra-Raut arXiv:2607.12360 (constat N1) ; Auger-Hansen arXiv:1310.7697 (constat N7) ; Beretti et al. arXiv:2611.15825 (11 nov. 2026, taux locaux déterministes, pas de recouvrement) | honnête pour (a) et (b) ; (c) sans travail proche nommé (N7) |
| 7 | (M7) permutation fraîche et plafond mémoire | **proche et à citer mieux** | Ying-Yuan-Sayed arXiv:1708.01383 : observation et mécanisme pour SAGA (constat N3) | attribution incomplète l. 606-609 |
| 8 | (C1)-(C3) instrumentation de la preuve | rien trouvé (recherche légère : 1 requête) | cohérent avec `literature_review.md` §6.3 (« no paper computes per-inequality slack along trajectories ») | pas de « not found » explicite ; tableau 1 correct |

## 3. Affirmations contrôlées et correctes

1. **l. 418-420, « the statement for polar steps was not found »** : OK. Requête API arXiv « abs:muon AND (median OR
   "mirror descent" OR "nuclear norm" OR biased OR skewed) » : 28 entrées sur Muon, aucune ne mentionne médiane,
   bruit asymétrique ou descente miroir sur une norme lissée. PDF ouverts : Xie 2609.32861 (résumé : « Under a
   Gaussian noise model, the expected Muon update becomes a scaled gradient step » ; l. 611 du texte : le bruit réel
   « anisotropic, non-Gaussian » est renvoyé aux limites), Chen et al. 2610.31728 (Prop. 3.1, réponse moyenne
   gaussienne isotrope), Beneventano-Abdelmoneum-Poggio 2606.08388 (pas de bruit dans le biais étudié),
   Jagannath-Sarangian 2611.19059 (dynamiques effectives en grande dimension), Wang-Klabjan 2410.14158 (descente
   miroir pour la descente de signe lissée, biais implicite déterministe) : aucun n'énonce le théorème 2.
2. **l. 272-274, Théorème 1 cherché « for right-preconditioned spectral norms only »** : OK et honnête. Trois
   requêtes (normes quelconques, préconditionnement de la logistique ridge, équivalence à une norme euclidienne) :
   aucun énoncé du type « aucune norme n'améliore κ quand la courbure de la perte s'annule ». Diakonikolas-Guzmán
   (2101.11041, introduction) : « No function can be both smooth and strongly convex with respect to an ℓp norm and
   have a dimension-independent condition number, unless p = 2. This is a basic fact from convex analysis » : c'est
   le coût κ_op ≥ min(K,d) que le papier cite déjà (l. 354), pas le théorème 1.
3. **l. 662-663, « The SAGA-table version with a linear rate was not found »** : OK. Requête API arXiv
   « (Muon OR "spectral descent" OR orthogonalized) AND (SAGA OR SVRG OR finite-sum OR reshuffling OR "variance
   reduction") » : 52 résultats, 15 pertinents, aucun avec table SAGA/SVRG, taux linéaire ou refroidissement
   géométrique. Parus depuis la vérification du 18 novembre ou non vus par elle : 2611.11387 et 2611.10212
   (réduction de variance par momentum), 2611.15825 (taux locaux déterministes) : pas de recouvrement.
4. **l. 119-120, « its published variance-reduced versions are mostly momentum-based »** : OK. Même listing :
   MARS-M 2510.21800, 2509.15816, LiMuon 2509.14562, 2512.16598, 2605.11850, 2608.12710, 2611.10212 sont de type
   momentum/STORM ; exception SVRG : Li-Tang-Zou (cité). « mostly » est exact.
5. **l. 606-607, permutation et Finito** : OK. Papier SAGA (arXiv:1407.0202, section 3) : « The big advantage of
   Finito/MISOµ is that when using a per-pass re-permuted access ordering, empirical speed-ups of up-to a factor of
   2x has been observed. This access order can also be used with the other methods discussed, but with smaller
   empirical speed-ups. »
6. **l. 608-609, seuil κ < 4.08** : OK comme calcul. DIAG (arXiv:1611.00347, corollaire 3) : erreur après m passes
   ≤ ρ^m(...)‖x⁰−x*‖, ρ=(κ−1)/(κ+1) ; ρ² < e⁻¹ ⟺ κ < (√e+1)/(√e−1) = 4.0830 (script). Le seuil ne figure pas dans
   l'article cité (voir nit N9).
7. **l. 444-445, Li-Tang-Zou th. 4.10** : OK. « Theorem 4.10 (Implicit Bias of Per-sample SignSGD and Per-sample
   Normalized-SGD) [...] with random reshuffling » ; lemme H.1 : directions spectrale et de Frobenius identiques pour
   un échantillon (gradient de rang un).
8. **l. 162-163 et 465-467, Do-Dereich-Jentzen en dimension un** : OK. Th. 1.3 : L(θ,x)=|θ−x|², θ∈R, données
   bornées, E[(X−EX)³]≠0, non-convergence « for almost all M » ; Prop. 6.2 : toute limite est un zéro du champ
   (« very specific one-dimensional quadratic example class »).
9. **Prop. 4, ordre 1/b et constante** : OK (rien trouvé). Requête API arXiv « (signSGD OR normalized SGD OR
   steepest descent OR Muon) AND (mirror descent OR dual space OR Edgeworth OR skewness OR non-convergence) » :
   21 entrées, aucune ne donne un développement du biais en la taille du lot.
10. **l. 654-656, (M2) et table SAGA dans le pas à longueur nucléaire** : rien trouvé de plus proche que ce que le
    registre du 12 novembre donne (recherche légère de ma part).
11. **Tableau 1, ligne (C1)-(C3)** : OK. Une requête ; rien qui mesure les inégalités de la preuve le long des
    trajectoires ; cohérent avec `bibliography/literature_review.md` §6.3.
12. **Portée annoncée de la recherche (l. 270-274)** : globalement prudente et honnête. Le papier distingue ce qui a
    été cherché sous une forme antérieure (Th. 2, constante de (offset), Prop. 3(c)), ce qui n'a été cherché que
    pour des normes spectrales préconditionnées (Th. 1) et ce qui n'a pas été trouvé (Prop. 5 avec table, (a), (c)) ;
    cela concorde avec `report_spectral_muon_new.tex` l. 288-291 et `literature_check.md` l. 1285-1326. Réserves :
    constats N7 et nit N10.

## 4. Constats

### N1 (mineur) : plancher en lr² et nécessité du refroidissement sous bruit qui s'annule, déjà publiés et non cités
- **Lieu** : l. 491-498 (M1), l. 532-540 (M5), tableau 1 l. 167-180, section 4.2 l. 657-663.
- **Texte** : « at a constant length variance reduction does not remove the floor: with the SAGA estimator the iterates
  settle at a squared distance $3.8\,\eta^2$ [...] whereas the floor of minibatch Muon is linear in its learning rate » ;
  « With a table the cooldown converges; without one it freezes the iterate ».
- **Problème** : la dichotomie (plancher quadratique en la longueur quand le bruit s'annule à l'optimum, plancher qui
  subsiste sous bruit additif, minimiseur atteint seulement si le taux tend vers zéro) est publiée pour les pas de
  signe et normalisés par Singh, Mishra et Raut (arXiv:2607.12360, 14 juillet 2026). Le papier ne les cite pas, alors
  que la vérification du dépôt les classe parmi les travaux les plus proches. La nouveauté réelle (table SAGA, pas
  polaire, taux linéaire) reste intacte.
- **Preuve** : résumé (PDF) : « sign-based and normalized methods [...] settle on a noise floor of order η² and reach
  the minimizer only as the learning rate is driven to zero; any additive noise then reinstates a floor for every
  method [...] extend the floor to normalized SGD in dimension d > 1 by a scale-invariance argument » ; introduction :
  « Cooldown is necessary ». Registres : `muon/literature_check.md` l. 1314 et 1345, `muon/findings.md` l. 174-175.
  Script : absent du `.tex` et du `.bib`.
- **Correction proposée** (l. 658, après `\cite{ma2026preconditioning};`) :
  `sign and normalized steps under noise that vanishes at the optimum settle at a floor of order $\mathrm{lr}^2$ and reach the minimizer only when the learning rate goes to zero \cite{singh2026same};`
- **Confiance** : haute (faits) ; la sévérité est une appréciation.

### N2 (mineur) : « a metric frozen from one polar decomposition were not found » est trop large
- **Lieu** : l. 655-656.
- **Problème** : lue seule, la clause affirme qu'aucune métrique figée issue d'une décomposition polaire n'a été
  trouvée. CacheMuon réutilise sur plusieurs pas une transformée d'orthogonalisation calculée auparavant, appliquée
  linéairement. Ce qui n'a pas été trouvé, d'après le registre du dépôt lui-même, c'est SAGA (ou SVRG) dans une telle
  métrique.
- **Preuve** : Dev, Bohara, Takáč, Horváth, arXiv:2606.16371 (15 juin 2026), introduction : « we propose CacheMuon, a
  Muon variant that reuses a previously computed orthogonalization transform across optimization steps » ;
  `muon/literature_check.md` l. 45 : « CacheMuon reuses a polar transform but has no finite-sum estimator ».
- **Correction proposée** : `and SAGA in a metric frozen from one polar decomposition were not found (Muon with a cached polar transform: \cite{dev2026cachemuon}).`
  Variante sans nouvelle référence : `and SAGA in a metric frozen from one polar decomposition were not found.`
- **Confiance** : moyenne à haute (la lecture stricte dépend du contexte de la phrase).

### N3 (mineur) : l'observation et le mécanisme de (M7) pour SAGA sont dans Ying-Yuan-Sayed, cité pour autre chose
- **Lieu** : l. 598 et l. 606-609 ; tableau 1 l. 176-177.
- **Texte** : « A permutation order was seen to help SAGA-type methods with Finito \cite{defazio2014finito}, and linear
  rates are proved for reshuffled and cyclic orders \cite{ying2017variance,park2020linear} ».
- **Problème** : Ying, Yuan et Sayed observent que SAGA sous permutation converge plus vite que sous tirage uniforme
  et donnent le mécanisme que (M7) reprend (« refreshes every entry once per pass »). Le papier ne leur attribue que
  la preuve du taux linéaire et rattache l'observation à Finito seul.
- **Preuve** : arXiv:1708.01383, section sur les variables d'historique : « This is because random reshuffling updates
  every history variable during each run, while uniform sampling may leave some variables [...] untouched. As we are
  going to illustrate in later experiments, this difference helps explain why SAGA under random reshuffling tends to
  have faster convergence rate. » Le gain chiffré (ρ⋆, de e⁻¹ vers e⁻²) n'y figure pas.
- **Correction proposée** (l. 606-608) :
  `A permutation order was seen to help Finito \cite{defazio2014finito} and SAGA, where it refreshes every table entry in each pass \cite{ying2017variance}; linear rates are proved for reshuffled and cyclic orders \cite{ying2017variance,park2020linear},`
- **Confiance** : haute.

### N4 (mineur) : Prop. 3(d), le cas K=1 est la limite d'un lemme publié
- **Lieu** : l. 442-447.
- **Texte** : « for $K,d\ge2$ (d) was not found ».
- **Problème** : l'énoncé est exact mais ne dit rien de K=1, d≥2. Pour une sortie scalaire, le gradient par
  échantillon tronqué d'un modèle linéaire généralisé convexe est le gradient d'une perte « Huberisée » de pente
  B/‖x‖ ; la normalisation (B→0, pas renormalisé) donne |⟨θ,x⟩−y|/‖x‖, soit Λ pour K=1.
- **Preuve** : Song, Steinke, Thakkar, Thakurta, arXiv:2006.06783v2, section 5.1 « Characterizing Clipping in Convex
  GLMs », lemme 5.1 : g_x affine de pente ±B/‖x‖₂ hors de [y₁,y₂] et « ∂_θ ℓ_g(θ;x) = { min{1, B/‖u‖₂}·u : u ∈
  ∂_θ ℓ_f(θ;x) } ». Le passage à la limite B→0 est ma lecture, pas un énoncé de l'article.
- **Correction proposée** (l. 446) :
  `for $K=1$ it is the zero-threshold limit of per-sample clipping, a Huberized loss \cite[Lemma~5.1]{song2020evading}; for $K,d\ge2$ (d) was not found.`
- **Confiance** : moyenne à haute.

### N5 (mineur) : « a property of the trajectory » sans le critère publié par pas
- **Lieu** : l. 351-353, l. 331, l. 500-501, tableau 2 l. 627.
- **Texte** : « any advantage of another norm is a property of the trajectory » ; « nuclear rank
  $\nnuc\omega^2/\nF\omega^2$ ».
- **Problème** : le complément positif du théorème 1 est publié : Davis et Drusvyatskiy donnent une condition par
  couche et par pas, qui compare le rapport nucléaire/Frobenius au carré du gradient au rang stable des activations.
  Le papier utilise cette quantité (l. 500-501) sans attribution ; les notes du dépôt citent ce travail, pas le papier.
- **Preuve** : arXiv:2512.04299, résumé (lu par l'outil de fetch, résumé seul) : « propose a simple layerwise
  condition that predicts when a spectral update yields a larger decrease in the loss », en comparant « the squared
  nuclear-to-Frobenius ratio of the gradient » à « the stable rank of the incoming activations ».
  `archive/muon/notes_spectral_theory.tex` l. 290-291.
- **Correction proposée** (l. 352-353) :
  `and any advantage of another norm is a property of the trajectory (a per-step criterion is in \cite{davis2025when}).`
- **Confiance** : moyenne (résumé seul).

### N6 (mineur) : Li-Tang-Zou montrent déjà que la réduction de variance rétablit la limite
- **Lieu** : titre du constat 2 (l. 387), tableau 1 l. 167-172, section 4.2 l. 654.
- **Problème** : le message de tête « variance reduction acts on the fixed point » a une forme publiée sur données
  séparables : pour la descente la plus raide en lots (norme spectrale comprise), la réduction de variance supprime
  la dépendance de la limite à la taille du lot. Le papier cite cet article trois fois (algorithme SVRG, lemme H.1,
  th. 4.10) sans jamais dire ce résultat.
- **Preuve** : arXiv:2602.11557v2, résumé : « we prove that variance reduction can recover the exact full-batch
  implicit bias for any batch size, albeit at a slower convergence rate » ; après le th. 4.7 : « variance reduction
  removes the dependence of the implicit bias on algorithmic parameters such as the batch size and momentum ».
- **Correction proposée** (l. 654) :
  `finite-sum estimators in fixed-length polar steps, where they restore the full-batch limit direction on separable data \cite[Theorem~4.7]{li2026implicit},`
- **Confiance** : haute.

### N7 (mineur) : Prop. 5(c) déclarée « not found » sans le travail proche que le registre du dépôt nomme
- **Lieu** : l. 274 et l. 657-663.
- **Problème** : pour (a) le papier nomme les travaux proches ; pour (c) aucun, alors que la vérification du
  18 novembre classe (c) « CLOSE » : la normalisation de l'état par le pas, qui donne une chaîne de Markov homogène
  pour un algorithme invariant d'échelle, est la technique d'Auger et Hansen ; Singh et al. ont un lemme
  d'invariance d'échelle.
- **Preuve** : `muon/literature_check.md` l. 1312-1317 et 1351 ; Auger-Hansen, arXiv:1310.7697 (SIAM J. Optim. 2016),
  résumé (outil de fetch, résumé seul) : « On scaling-invariant functions, we show the existence of a homogeneous
  Markov chain », conditions de convergence linéaire globale « expressed in terms of different stability conditions
  of the normalised homogeneous Markov chain ».
- **Correction proposée** (l. 660, après « are close to Proposition~\ref{prop:schedule}(a) ») :
  `, and the normalization of (c) is that of scale-invariant randomized search \cite{auger2016linear}`
- **Confiance** : moyenne à haute.

### N8 (mineur) : Do-Dereich-Jentzen réduits à la dimension un
- **Lieu** : tableau 1 l. 162-163 ; l. 465-467.
- **Texte** : « in dimension one Muon need not converge under skewed noise \cite{do2026muon} ».
- **Problème** : leur théorème 1.2 porte sur des paramètres matriciels (blocs R^{d_i×𝔡_i}), avec momentum et
  Newton-Schulz généralisé, et borne l'erreur quadratique moyenne par C(M⁻¹+γ_n). C'est le pendant publié de la
  Prop. 3(b) (écart de F d'ordre 1/b) et la borne que la Prop. 4 affine (décalage 1/b, donc 1/b² en erreur
  quadratique). Le papier ne le mentionne pas.
- **Preuve** : arXiv:2608.04607v1, th. 1.2 : « A = ×_{i=1}^δ R^{d_i×𝔡_i} [...] E‖Θ_n − ϑ‖² ≤ C(M⁻¹ + γ_n) » (1.5) ;
  th. 1.3 et Prop. 6.2 seulement en dimension un.
- **Correction proposée** (l. 465-467) :
  `For Muon, \cite{do2026muon} bound the mean-square error by $C(1/b+\mathrm{lr}_t)$ for matrix parameters and, in dimension one, prove non-convergence under skewed bounded data and show that every limit is a zero of the mean field.`
- **Confiance** : haute.

### N9 (nit) : DIAG est cyclique ; le seuil 4.08 est un calcul du papier
- **Lieu** : tableau 1 l. 176-177 ; l. 608-609.
- **Problème** : « reshuffled SAGA-type methods \cite{ying2017variance,mokhtari2018surpassing} » : la méthode de
  Mokhtari et al. choisit les fonctions « based on a cyclic scheme » (résumé), pas par permutation aléatoire. Le
  seuil κ<4.08 n'est pas dans l'article : il vient de ρ²<e⁻¹ avec le corollaire 3 (le registre du dépôt le dit :
  « a comparison not made in the papers », `literature_review.md` §10.2).
- **Correction proposée** : tableau 1 : `reshuffled and cyclic SAGA-type methods` ; l. 608-609 :
  `and the cyclic rate of \cite[Corollary~3]{mokhtari2018surpassing} beats $e^{-1}$ per pass for $\kappa<4.08$.`
- **Confiance** : haute.

### N10 (nit) : « (M5)-(M7) are new » et « opened the closest sources »
- **Lieu** : l. 270-272.
- **Problème** : dans un paragraphe sur le statut de nouveauté, « are new » se lit comme une revendication alors
  qu'il s'agit d'étiquettes absentes du rapport étendu. « opened the closest sources (about sixty papers) » est un peu
  fort : le registre indique des lectures par résumé ou balayage de mots-clés pour une partie des sources, et Goffin
  1977 (cité au tableau 1) comme non lu.
- **Preuve** : `muon/literature_check.md` l. 7 et l. 1392-1397 (« Not read: Goffin 1977 (publisher page requires
  authentication) »).
- **Correction proposée** : `\finding{M5}--\finding{M7} are added here.` et
  `that opened the closest sources (about sixty papers, some by abstract only).`
- **Confiance** : moyenne (je n'ai pas vérifié si Goffin a été ouvert depuis).

### N11 (nit) : SAGA en normes non euclidiennes existe (pas de Bregman)
- **Lieu** : l. 654-656.
- **Problème** : pour « in general norms » le papier cite Katyusha (estimateur SVRG). Une table SAGA avec normes non
  euclidiennes est publiée, avec des pas de Bregman et non des pas de plus forte pente : plus proche côté « table ».
  La revendication (table SAGA dans le pas polaire à longueur nucléaire) reste vraie.
- **Preuve** : Xie, Jin, Zhou, Meng, Cheng, arXiv:2205.02273v2, résumé : « several novel adaptive variants of the
  popular SAGA algorithm [...] convergence guarantees under general settings that allow non-Euclidean norms in the
  definition of smoothness and the composite objectives » ; section 3 : divergence de Bregman 1-fortement convexe.
  `muon/literature_check.md` l. 1322 et 1350.
- **Correction proposée** : aucune nécessaire ; facultatif : `in general norms \cite{allenzhu2018katyusha} (with a SAGA table and Bregman steps: arXiv:2205.02273)`.
- **Confiance** : haute.

Entrées bibliographiques pour les corrections (toutes vérifiées sur la première page du PDF ou la page arXiv) :
`singh2026same` = S. Singh, A. Mishra, S. Raut, « Same Loss, Same Noise, Opposite Schedules: Noise Structure and
Optimizer Normalization Jointly Determine Whether Learning-Rate Cooldown Helps », arXiv:2607.12360 (2026) ;
`dev2026cachemuon` = B. Dev, S. Bohara, M. Takáč, S. Horváth, « CacheMuon: Using Temporal Preconditioning To
Approximate Polar Factor », arXiv:2606.16371 (2026) ; `song2020evading` = S. Song, T. Steinke, O. Thakkar,
A. Thakurta, « Evading the Curse of Dimensionality in Unconstrained Private GLMs via Private Gradient Descent »,
arXiv:2006.06783 ; `davis2025when` = D. Davis, D. Drusvyatskiy, « When do spectral gradient updates help in deep
learning? », arXiv:2512.04299 ; `auger2016linear` = A. Auger, N. Hansen, « Linear Convergence of Comparison-based
Step-size Adaptive Randomized Search via Stability of Markov Chains », SIAM J. Optim. 2016, arXiv:1310.7697.
Chaque référence ajoutée coûte environ deux lignes de bibliographie sur deux colonnes : N3, N6, N8, N9, N10 et la
variante de N2 n'en ajoutent aucune ; N1 est la seule que je juge nécessaire.

## 5. Travail récent à connaître (pas un constat)

Beretti, Do, Dereich, Jentzen, « Optimal higher order convergence rates for the MUON optimizer », arXiv:2611.15825
(11 novembre 2026), absent des registres du dépôt. Taux locaux optimaux, déterministes ; introduction : « due to the positive
constant b_{0,1} in the normalization step, the function NS is, to first order, the multiplication by [a constant]
close to zero [...] close to a local minimizer, MUON behaves like the momentum GD optimizer with a rescaled learning
rate ». C'est le mécanisme publié derrière la remarque de (M5) sur le plancher de ns5 (l. 535-537 : la constante ε
« breaks the scale invariance »). Pas de recouvrement avec les énoncés du papier.

## 6. Non vérifié

- Section 6 de Katyusha (normes générales), citée l. 654-655 : non rouverte (registre du dépôt : « not re-read »).
- Yang et al. (arXiv:1408.3115) : la remarque sur la régression logistique rapportée par un résumé de moteur de
  recherche n'a pas été vérifiée dans le texte ; aucun constat n'en est tiré.
- Négiar et al. (arXiv:2002.11860, table de type SAG dans un pas LMO, d'après le registre) : résumé seul, qui ne
  confirme pas la table ; Kereta et al. (arXiv:2106.03172) et Freund-Grigas-Mazumder (arXiv:1810.08727) non ouverts.
- Goffin 1977 : non accessible ; je n'ai pas vérifié s'il a été lu depuis le 18 novembre.
- Davis-Drusvyatskiy et Auger-Hansen : résumés seulement (N5, N7).
- Le décompte « about sixty papers » et le détail des recherches du 16 et du 17 novembre (pas de registre séparé
  retrouvé pour le 17).
- Plein texte, OpenReview, ateliers, thèses : non couverts. Les lois à pas fixe de la note B4 citées dans (M7)
  (l. 600-603) n'ont fait l'objet d'aucune recherche dédiée (ni par le dépôt, `literature_review.md` l. 233, ni par
  moi au-delà d'une requête).
- L'exactitude mathématique des énoncés (autres auditeurs).
