# Lecture critique « jury » (session VSCode, 20 novembre 2026)

Périmètre : pertinence scientifique des affirmations de tête (titre, résumé, tableaux 1 et 2, conclusion) au regard de
ce que le corps du papier démontre ou mesure. Lecture faite sur le texte seul (`paper/paper_saga_to_muon.tex`, numéros
de ligne de l'outil Read), sans rouvrir les données ni les sources externes : chaque constat dit ce qui reste à
vérifier. Les preuves relues à la main (Théorèmes 1 et 2, Propositions 3 à 5, identité (2)) sont résumées dans
`paper/audit/process/consigne_session_distante.md`.

## Constats

### jury-1 (majeur, confiance haute) : « the same anatomy » n'est établi que sur les quadratiques

- Résumé, lignes 96-97 : « the nuclear-scaled polar step has the same anatomy. » Tableau 1, ligne 151 : « the same
  anatomy for the polar step [p] ». Tableau 2, ligne 626 : « The same exact anatomy (2) ».
- L'identité (2) est énoncée « For a quadratic with Hessian H » (ligne 226) ; le corps dit lui-même (lignes 313-315)
  que sur moindres carrés l'écart est d'environ 4 et que le cas à grand écart, logistique et softmax, n'est « not
  measured for the polar step ». Or la phrase du résumé suit immédiatement des chiffres mesurés sur la régression
  logistique (3,2 à 220 ; 77-96 %) : un lecteur comprend que la même mesure vaut pour le pas polaire.
- Ce qui est prouvé : une identité exacte à un pas sur les quadratiques et les plages des deux rapports
  (2 − J/L_op ∈ [1, 2), μ_PL/μ_op ∈ [1, κ_op]). C'est une analogie de structure, pas une mesure.
- Correction : ajouter « on quadratics » aux trois endroits. Résumé : « on quadratics, the exact one-step formula of
  the nuclear-scaled polar step has the same anatomy. »

### jury-3 (majeur si les concurrents n'ont pas été réglés, confiance moyenne) : « freeze » sans « on the same schedule »

- Résumé, ligne 104 : « where minibatch and momentum-based estimators freeze » ; introduction, lignes 132-133 :
  « converge to rounding error where other estimators freeze » ; tableau 1, ligne 178 : « without one it freezes [e] ».
- Le corps est plus précis : (M5), ligne 538, « Without a table the same schedule freezes the iterate at 8% to 93% of
  the initial squared distance » ; légende de la figure 4, ligne 523 : « the schedule selected for Muon's recipe
  (q = 0.75, lr_0 = 0.3) on the minibatch gradient and on MARS-M » ; conclusion, ligne 679 : « where the same
  schedule freezes without one ».
- Un gel à 8-93 % de la distance initiale signifie que les concurrents n'ont presque pas avancé : c'est le signe d'un
  refroidissement réglé pour une autre méthode, pas une limite de ces estimateurs. Le mécanisme de fond est juste et
  peut se dire : une suite de longueurs sommable ne laisse qu'un déplacement total fini, et un estimateur dont le
  bruit ne s'annule pas en W_* ne converge alors pas vers W_* ; mais la hauteur du gel dépend du réglage.
- À vérifier dans les données (auditeur `num-m5`) : les méthodes sans table ont-elles été réglées sur une grille
  (q, lr_0), et jusqu'où descendent-elles avec leur meilleur réglage en 300 passes ? Si oui, donner ce chiffre ; si
  non, restreindre les trois phrases : « where, on the same schedule, minibatch and momentum-based estimators
  freeze ».

### jury-4 (mineur, confiance haute) : le plafond mémoire est une borne en espérance, sur le modèle linéaire

- Tableau 2, ligne 638 : « At most one nat per pass under independent sampling (M3), two under a permutation (note
  B4). » Le papier mesure lui-même 1,6 nat par passe pour une trajectoire sous tirage indépendant (ligne 603) et écrit
  « the memory factor bounds the mean error, not a single run » (lignes 604-605) : la cellule du tableau contredit
  (M7) si l'on ne lit pas « in expectation ».
- Conclusion, lignes 680-681 : « a fresh permutation per pass, which lifts the memory rate of SAGA from e^{-1} to
  nearly e^{-2} » : énoncé général sur SAGA, alors que le résultat prouvé porte sur n quadratiques de même hessienne,
  en espérance, pour n → ∞ (lignes 599-603) ; la section Limitations le dit (« the model of (M7) has a single
  curvature »), pas la conclusion.
- Corrections : « In expectation, at most one nat per pass under independent sampling (M3), two under a permutation
  (common Hessian, note B4). » ; « which lifts the expected memory rate of linear SAGA from e^{-1} to nearly e^{-2} ».

### jury-5 (mineur, confiance haute) : « certify nothing » est trop fort

- Lignes 238-239 : « local ones, at the minimizer [...], describe the end of a run but certify nothing. »
- Pour une méthode monotone (descente de gradient, pas polaire à longueur nucléaire avec gradient exact), des
  constantes prises sur l'ensemble de sous-niveau du point initial certifient un taux ; ce sont les constantes au
  minimiseur qui ne certifient pas un taux global. C'est aussi la première objection qu'un jury fera au Théorème 1 :
  le pire cas est pris sur tout l'espace, là où aucune itérée ne va. Le papier y répond déjà (lignes 325-327,
  351-353, 669-670) ; la phrase de la ligne 239 gagne à dire exactement cela.
- Correction : « describe the end of a run but certify no global rate ».

### jury-6 (renforcement proposé, vérifié numériquement) : le Théorème 1 vaut aussi pour la constante de Polyak-Łojasiewicz

- Objection prévisible : le taux certifié de la descente la plus raide en norme N s'écrit avec la constante de
  Polyak-Łojasiewicz en norme duale, μ_PL,N = inf_W ‖∇F(W)‖²_{N*} / (2(F(W) − F_*)) ≥ μ_N, pas avec μ_N ; un
  certificat de ce type échappe-t-il au théorème ?
- Réponse (calcul fait ici) : non, pour des pertes lipschitziennes (logistique, softmax). Soit Δ_0 un
  maximiseur de N/‖·‖_F, ‖Δ_0‖_F = 1 ; alors ‖Δ_0‖_{N*} = 1/c_N (la boule de Frobenius de rayon 1/c_N est inscrite dans
  la boule de N et la touche en Δ_0/c_N). Le long de W = tΔ_0, ∇F(W) = λtΔ_0 + O(1) et F(W) − F_* = λt²/2 + O(t),
  donc le rapport tend vers λ/c_N². Avec μ_PL,N ≥ μ_N = λ/c_N², on a μ_PL,N = λ/c_N² exactement, et
  L_N/μ_PL,N ≥ L_F/λ : même conclusion que le théorème. L'hypothèse devient « pertes à gradient borné », sans
  condition sur la direction.
- Contrôle numérique : `paper/audit/checks/jury-6_pl_constant.py` (softmax ridge, K = 3, d = 4, n = 30, λ = 0,1 ;
  normes spectrale, max des entrées, Frobenius pondérée). Le rapport le long de tΔ_0 tend vers λ/c_N² (0,10001 ;
  0,10003 ; 0,02500 à t = 10⁵) et reste au-dessus sur 20 000 points aléatoires.
- Ajout possible après la ligne 354, si la place le permet : « For Lipschitz losses the Polyak--Łojasiewicz constant
  in the dual norm is also λ/c_N² (take W = tΔ_0, t → ∞), so certificates of steepest descent in N do not escape. »
  C'est un choix d'auteur : recommandation, pas correction.

### jury-2 (mineur, à vérifier sur la source) : « Muon's own recipe » et le refroidissement géométrique

- Résumé, lignes 102-103 : « Muon's own recipe. Steps of fixed length on a geometric cooldown of the learning rate » ;
  section 3.3, ligne 508 : « it keeps steps of fixed length and cools the learning rate down ».
- Ce qui est propre à Muon : la mise à jour orthogonalisée de longueur spectrale fixe, cinq itérations de
  Newton-Schulz, le moment de Nesterov. La forme géométrique du refroidissement est un choix du papier (celle qui
  donne un taux linéaire avec gradients exacts) ; de mémoire, les exécutions de référence de Muon font décroître le
  taux d'apprentissage linéairement. À vérifier sur la source de Jordan et al. (2024) avant d'écrire quoi que ce soit.
- Si confirmé, précision en quelques mots ligne 508 : « ... and cools the learning rate down (linearly in its
  reference runs; we use the geometric cooldown that gives a linear rate with exact gradients). »

### jury-7 (mineur, à vérifier dans les données) : « a normalized step » dans le résumé

- Résumé, lignes 105-106 : « the spectral map needs 0.58--0.93 of the passes of a normalized step ». (M6), lignes
  588-589 : « of the fastest control », les contrôles étant la normalisation de Frobenius et le signe coordonnée par
  coordonnée (lignes 514-515). Si le contrôle le plus rapide est parfois le signe, écrire « of the fastest
  non-spectral control ». Par ailleurs 0,93 est dans la bande « about 10% not resolved » (ligne 264) : sur l'instance
  correspondante l'avantage n'est pas résolu (auditeur `num-m6m7`).

## Points contrôlés sans problème à la lecture

- (C5), « every P has κ_P ≥ κ » : μ_P = λ/λ_max(P), R_P² ≥ (max_i c_i‖x_i‖² + λ)/λ_max(P), donc κ_P ≥ κ. Correct.
- « γ(m + ϑc) with ϑ = 1/n, γ = 1/λ is MISO » (lignes 255-256) : θ⁺ = −(1/λ)(z̄ + (∇g_i(θ) − z_i)/n), l'itérée de MISO
  avec le terme ridge comme partie quadratique. Correct. « m is the gradient of an isotropic quadratic » : m est le
  gradient de (λ/2)‖W + z̄/λ‖_F². Correct.
- Plancher « near 1e-18 » de ns5 attribué à ε (lignes 535-537) : pour ‖ω‖_F ≪ ε la carte de Newton-Schulz devient
  linéaire (gain d'environ 3,4445⁵ ≈ 485 sur ω/ε), le pas n'est plus de longueur fixe et l'itérée se maintient là où
  ‖ω‖_F est de l'ordre de ε : mécanisme cohérent avec l'essai ε = 1e-30.
- « 16-18 passes to 1e-9 [...] near the ceiling [...] e^{-1} per pass in expectation » (lignes 382-384) : e^{-1} par
  passe donnerait 20,7 passes ; les trajectoires vont plus vite que la borne en espérance, ce que (M7) explique
  (1,6 nat par passe mesuré). Cohérent à condition de garder « in expectation » partout (voir jury-4).
- Honnêteté d'ensemble : le papier dit ce qu'il ne fait pas (« We propose no method that beats the state of the
  art », « certifies the mechanism, not the speed », cooldowns certifiés 3,3·10⁴ à 8,9·10⁴ fois plus lents, section
  Limitations). Les sur-affirmations relevées ci-dessus sont dans les phrases courtes (résumé, tableaux, conclusion),
  pas dans le corps.
