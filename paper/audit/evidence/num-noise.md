# Audit `num-noise` : (M1), (M2), contrôles numériques du bruit, figure 3

Papier : `paper/paper_saga_to_muon.tex` (numéros de ligne de l'outil Read). Audit du 2026-11-20.

## 1. Périmètre couvert

- (M1) et (M2), l. 491-503 ; paragraphe « Numerical checks » à partir de « On the finite sum of Figure 3 », l. 484-488 ;
  légende de la figure 3, l. 472-479, et l'image `paper/figures/figP3_noise.png` ; dernière phrase de « Stochastic polar
  steps », l. 249-251 ; tableau 1, lignes 3 et 4, colonne « Added here » (l. 164-166 et 171-172) ; phrase « γ(m+ϑc) …
  is MISO », l. 255-256.
- Sources lues : `muon/results/M3_summary.json`, `M6_summary.json`, `M2_summary.json`, `M4_summary.json`,
  `M1_summary.json` (pour MISO et L-SVRG-Muon), `muon/theory_checks/paper_figures_data.json`,
  `muon/theory_checks/make_paper_figures.py`, `check_minibatch_offset_exact.py`, `check_offset_first_order.py`,
  `muon/experiments/exp_M2_mechanism.py`, `exp_M3_floors.py`, `exp_M6_miso.py`, `common.py`, `muon/lab/mvr.py`,
  `paper/scripts/make_figP3_noise.py`, `make_figR2_muon_vr.py`, `muon/results/digest.md` (sections M1, M2, M3, M4, M6),
  `muon/findings.md` (sections 4, 5.1, 5.2, 5.7).
- Scripts de contrôle (tous relancés, un processus, BLAS mono-fil, moins de 3 minutes chacun), dans `paper/audit/checks/` :
  - `num-noise_floor_constants.py` : ‖W⋆‖²_F des deux instances de M3, constantes 3.8 η² / 6.0 η², rapport des planchers
    de Muon, réexécution courte du pas polaire de longueur fixe ;
  - `num-noise_bound_and_miso.py` : borne μ_op η²/8 (24 000 paires d'itérés, cas d'égalité) ; identité
    γ(m+ϑc), ϑ = 1/n, γ = 1/λ = MISO (implémentation indépendante) ;
  - `num-noise_miso_regime.py` : portée de « fails where MISO fails » (parties a, b) et « rounding precision » sur
    ls-mem au-delà de 60 passes (partie c) ;
  - `num-noise_finite_sum_ls.py` : problème de moindres carrés à somme finie de la figure 3(b)-(c), autres graines
    (parties a, b1, b2, c, d) ;
  - `num-noise_lsvrg_joint.py` : pas polaire à longueur nucléaire avec l'estimateur L-SVRG (lignes stockées de M1 et
    réexécution à 150 passes).

## 2. Affirmations contrôlées : verdicts

| # | Affirmation (papier) | Lignes | Verdict | Preuve |
|---|---|---|---|---|
| 1 | ‖W_LAD − W_LS‖_F = 0.170 | 485 | OK | cache `noise_trace.dist_lad_ls` = 0.16963 ; recalculé (Weiszfeld) 0.16963 ; champ moyen exact du pas polaire à b = 1 en W_LAD : 8.7e-16 (énumération des n = 2000 indices), 0.194 en W_LS |
| 2 | « averaged batch-one polar steps end 2.6·10⁻³ from W_LAD » | 485 | OK | cache `mb_to_lad[-1]` = 2.609e-3 ; autre graine (1010) : 2.10e-3 (5 derniers points 2.1e-3 à 4.8e-3) |
| 3 | « the SAGA estimator 2.1·10⁻⁴ from W_LS » | 486 | OK | cache `saga_to_ls[-1]` = 2.147e-4 (trois derniers points 1.4e-4, 1.6e-4, 2.1e-4, non monotones) ; autre graine (1011) : 7.4e-5 |
| 4 | « for b ≤ 32 the runs end within 5 % of the zero of the mean field » | 486-487 | OK (remarque) | écarts runs/zéro du cache : −1.4, −2.1, −1.4, −1.2, −3.8, −4.6 % pour b = 1, 2, 4, 8, 16, 32. Remarque : les zéros (b ≥ 2) sont des estimations de Monte-Carlo (2 × 200 000 lots) ; sur 6 répliques à b = 32 j'obtiens b×‖∇F‖⋆ = 0.286 à 0.309 (écart-type ≈ 3 %), sur 3 répliques à b = 16 : 0.297 à 0.310 (2.1 %). Les écarts de −3.8 et −4.6 % ne sont donc pas résolus du bruit de Monte-Carlo ; l'énoncé « within 5 % » tient |
| 5 | « b times the offset rising from 0.17 to a plateau near 0.3 » | 487 | OK | b×zéro = 0.168, 0.240, 0.276, 0.285, 0.303, 0.308 ; runs 0.166, 0.235, 0.272, 0.281, 0.292, 0.294 ; mes répliques : 0.2395 (b = 2), moyenne 0.303 (b = 16), 0.302 (b = 32) |
| 6 | « the SAGA estimator with the nuclear-scaled length ends at rounding level » | 487-488 | OK | cache `noise_offsets.saga` = 1.8e-15 à 2.4e-15 (norme nucléaire du gradient) ; réexécution, autres graines : b = 1 : 1.1e-4, 1.3e-8, 2.0e-15 à T/4, T/2, T ; b = 32 : 1.3e-7, 1.7e-11, 2.3e-15 |
| 7 | Légende fig. 3 : η = 10⁻², lr = 10⁻², b = 32 ; n = 2000, K = 2, d = 3 ; η_t = 0.3 (1+t/200)^(−3/4) ; moyenne sur (t/2, t] et 32 runs ; 64 runs ; non régularisé, intercept, résidus asymétriques | 472-479 | OK | `make_figR2_muon_vr.py` l. 61-66 (clés M3 tracées) ; `make_paper_figures.py` l. 202-211 (n, K, d, `X[:,0]=1`, résidus 0.5(Exp(1)−1), asymétrie mesurée −2.1 et −2.0), l. 229 (longueur), l. 251 (moyenne sur (c/2, c]), l. 319 (32 chaînes), l. 352 (64 chaînes, 20 000 pas), pas de terme ridge (‖∇F(W_LS)‖ = 1.7e-16) |
| 8 | La figure montre ce que disent la légende et le texte | fig. 3 | OK (réserves : constats F3, F7) | (a) plateau bleu 4.6e-7, courbe orange sous 1e-24 vers la passe 48, tirets roses ≈ 1e-2 encore décroissants, pointillés rouges ≈ 0.12 ; (b) « minibatch, to LS » plat à 0.17, « to LAD » décroissant jusqu'à 2.6e-3, SAGA de 13 à 2e-4 ; (c) ligne de 0.17 à 0.31, points à moins de 5 %, carrés à 0 |
| 9 | « max(F(W_t),F(W_{t+1})) − F⋆ ≥ μ_op η²/8 whenever the step is nonzero » | 491-492 | OK (attribution : F8) | Redémontrée : ‖W_{t+1} − W_t‖_op = η si ω ≠ 0, donc max(‖W_t − W⋆‖_op, ‖W_{t+1} − W⋆‖_op) ≥ η/2 ; forte convexité spectrale et ∇F(W⋆) = 0 : F(W) − F⋆ ≥ (μ_op/2)‖W − W⋆‖²_op. Vraie pour tout estimateur. Numérique : 24 000 paires sur 60 quadratiques (gradient exact, bruité, direction arbitraire), min du rapport = 1.0445 ; égalité atteinte (2-cycle ±η/2, rapport 1.000000) |
| 10 | « with the SAGA estimator the iterates settle at a squared distance 3.8 η² (softmax) or 6.0 η² (least squares) » | 493-494 | OK pour la valeur (unité : F3) | M3 : niveaux relatifs 4.64e-5, 4.61e-7, 4.58e-9 (softmax) et 5.26e-5, 5.25e-7, 5.23e-9 (moindres carrés) pour η = 1e-1, 1e-2, 1e-3 ; ‖W⋆‖²_F recalculé = 826.97 et 1146.99 ; d'où 3.79 à 3.84 η² et 6.00 à 6.06 η² (distance ABSOLUE). Réexécution (32 passes, η = 1e-2, graine 0) : 3.78 η² et 6.00 η² |
| 11 | « the floor of minibatch Muon is linear in its learning rate » | 494-495 | OK (deux valeurs de lr ; F7) | M3, niveau stationnaire (départ à 1e-14 de W⋆) : 5.74e-2 / 5.71e-3 = 10.05 (softmax), 1.59e-2 / 1.66e-3 = 9.58 (moindres carrés) pour lr = 1e-1 / 1e-2, 3 graines ; traces stationnaires plates sur les passes 19-30 |
| 12 | « 1.2·10⁻¹ and 4.8·10⁻² with the minibatch gradient, b = 32 » | 495-496 | OK (unité : F3) | M3 « minibatch b=32, nuclear-scaled », niveau stationnaire 1.241e-1 [1.24e-1, 1.27e-1] et 4.847e-2 [4.82e-2, 4.91e-2] ; depuis W_0 = 0 : 1.26e-1 et 4.89e-2 (distances RELATIVES) |
| 13 | « a momentum buffer as in MARS-M keeps a floor » | 496 | Problème mineur (F4) | le tampon testé avec la longueur nucléaire est STORM (g = 1), pas MARS-M |
| 14 | « Only a finite-sum estimator reaches rounding precision, with a length that vanishes with it (nuclear-scaled or clipped; on least squares also the soft map) » | 497-498 | OK pour l'estimateur SAGA ; problèmes F5, F6, F9 | M3 : softmax 2.0e-27 (nucléaire), 5.3e-27 et 2.1e-27 (écrêté) ; moindres carrés 3.3e-20, 1.1e-20, 7.2e-21 à 60 passes, encore décroissant ; ma réexécution de ls-mem à 110 passes s'arrête sous 1e-30 à la passe 87 (graines 0 et 1), soft-mem plafonne à 1.1e-29 dès la passe 60 |
| 15 | « γ(m+ϑc) with ϑ = 1/n, γ = 1/λ is MISO » | 255-256 | OK | algèbre : W⁺ = W − (1/λ)(z̄ + λW) − (1/(λn))(∇g_j(W) − z_j) = −z̄⁺/λ = argmin de la moyenne des minorants d_i(W) = g_i(φ_i) + ⟨∇g_i(φ_i), W − φ_i⟩ + (λ/2)‖W‖² ; implémentation indépendante (softmax et moindres carrés, 399 pas) : écart relatif max 1.6e-15 |
| 16 | « The predictable part m is the gradient of an isotropic quadratic … the step drives m to zero, which MISO does exactly » | 499-502 | OK (m quadratique isotrope ; MISO annule m) ; « drives m to zero » dépend du régime (F1) | m = z̄ + λW = ∇[(λ/2)‖W + z̄/λ‖²_F], hessienne λI (vérifié : 3e-15) ; après un pas de MISO ‖m‖_F ≤ 1.1e-16 ; M6 : ‖m‖⋆/‖∇F‖⋆ = 0.02 à 0.04 (softmax κ = n), 0.011 à 0.014 (moindres carrés κ = n) ; mais 0.35 à 0.47 aux pas réglés sur soft-cond et ls-cond (M2) |
| 17 | « nuclear rank … of 1.07, median over the recorded states » | 500-501 | Problème mineur (F2) | M2, soft-mem : 1.095, 1.058, 1.066, 1.077, 1.064, médiane 1.066 ; cinq autres instances : 1.36, 1.76, 1.81, 1.65, 1.36 |
| 18 | « 11.8 against 11.9, softmax with κ = n; SAGA: 20.2 » | 502-503 | OK | M6, softmax κ/n = 1, trois graines : joint c = 1/4 : 11.8, 13.2, 11.7 ; MISO : 12.0, 11.9, 11.5 ; SAGA c = 1/4 : 20.1, 20.2, 20.2. Autres cas où MISO converge : 8.8 / 8.7 (softmax κ = n/10), 10.5 / 10.4 et 18.1 / 18.9 (moindres carrés ; là SAGA est plus rapide : 12.9) |
| 19 | « fails where MISO fails (κ ≫ n): it is not a spectral method » | 503 | Problème majeur (F1) | vrai aux constantes c = 1/4 et 1 de M6 seulement ; contre-exemples par réexécution |
| 20 | « Predictions are checked with zeros of the mean field from exactly sampled batch means, quadrature or Monte Carlo, and runs on a finite-sum least-squares problem with an intercept and skewed residuals » | 249-251 | OK | `check_minibatch_offset_exact.py` (moyennes de lots Gamma(b,1)/b − 1 tirées exactement, 1.5·10⁶ matrices) ; `check_offset_first_order.py` (quadrature 2D et Monte-Carlo des constantes) ; `make_paper_figures.py` `ls_problem` (intercept, résidus Exp(1) − 1) |
| 21 | Tableau 1, ligne 3 : « at batch one on least squares, to a least-distances estimator [p] ; a limit gradient of order 1/b [f] » | 165-166 | OK (formulation : F10) | Prop. 3(d) (`prop:comp`) ; W_LAD minimise Λ (Λ(W_LAD) = 0.49570 < Λ(W_LS) = 0.51236, 200 perturbations toutes défavorables) ; b×décalage entre 0.17 et 0.31 pour b = 1 à 32 |
| 22 | Tableau 1, ligne 4 : « rounding precision with a length that goes to zero, with the estimator or on a cooldown [e] ; at batch one the nuclear-scaled polar step of SAGA behaves as MISO [e] » | 171-172 | Problèmes F1, F6 | voir constats |

## 3. Constats

### F1 (major) : « fails where MISO fails (κ ≫ n): it is not a spectral method » n'est vrai qu'aux deux constantes de pas de M6

- Lignes 499-503 ; même portée manquante l. 172 (tableau 1, ligne 4) et l. 636 (tableau 2).
- Citation : « With the same indices the joint SAGA-Muon needs the passes of MISO where MISO converges ($11.8$ against
  $11.9$, softmax with $\kappa=n$; SAGA: $20.2$) and fails where MISO fails ($\kappa\gg n$): it is not a spectral method. »
- Problème : M6 compare à pas fixes non réglés γ = c/R²_op, c ∈ {1/4, 1} (trois graines, 40 passes, instances n = 1000,
  κ/n ∈ {0.1, 1, 10}). Le papier ne le dit pas. Avec un pas 4 à 16 fois plus petit, la même méthode converge sur les
  instances mêmes où MISO échoue ; et le constat (M3) du papier (l. 375-377) rapporte que le joint SAGA-Muon réglé atteint
  10⁻⁶ en 29.0 et 43.5 passes sur des instances κ ≈ 48n et 36n. L'énoncé tel qu'écrit est donc faux hors de son
  hypothèse de pas, et (M2) contredit (M3) pour un lecteur attentif. L'intitulé « is MISO in disguise » et la conclusion
  « it is not a spectral method » héritent de la même limite : aux pas réglés des instances limitées par le
  conditionnement, m n'est pas petit (‖m‖⋆/‖∇F‖⋆ médian 0.35 et 0.47, jusqu'à 0.99) et le rang nucléaire est 1.76 à 1.81.
- Preuves :
  - `num-noise_miso_regime.py a` (instances κ/n = 10 de M6, graine 0, 40 passes). Softmax : MISO 0.88 (stagne) ;
    joint c = 1 : 0.84 ; c = 1/4 : 0.20 ; **c = 1/16 : 10⁻⁶ en 25.1 passes, 5.3e-10 à 40 passes** ; SAGA c = 1/4 :
    2.3e-3. Moindres carrés : MISO diverge (4.8e27) ; joint c = 1, 1/4, 1/16 divergent ; **c = 1/64 : 10⁻⁶ en 25.6
    passes, 2.3e-10 à 40 passes** ; SAGA 19.0 passes.
  - `num-noise_miso_regime.py b` (soft-cond, ls-cond, graine 10, 100 passes) : MISO (δ = 1) stagne à 9.5 (soft-cond)
    ou diverge (ls-cond, 3e29) ; joint SAGA-Muon au pas de M1 : 30.6 et 31.0 passes.
  - `M1_summary.json` / digest l. 326 et 335 : « MISO » (table relâchée, δ réglé) 42.7, 120.7, non atteint, 29.1 passes
    sur soft-cond, ls-cond, ls-orth, soft-sq ; « SAGA-Muon (joint) » 29.0, 43.5 (9/10), 28.1, 20.3 : 0.36 à 0.70 des
    passes de MISO là où les deux convergent.
  - `M2_summary.json` : base_ratio du joint aux pas réglés, médianes 0.062 (soft-mem), 0.23, 0.35, 0.47, 0.17, 0.19.
- Ce qui reste vrai : aux constantes 1/4 et 1, là où MISO converge (κ ≤ n), les passes coïncident (ligne 18 du tableau
  ci-dessus) et à κ = 10n les deux stagnent ou divergent.
- Correction proposée (l. 499, 502-503, 172) :
  - l. 499 : `\item[(M2)] \emph{At batch one and at textbook steps the nuclear-scaled polar step of SAGA is MISO in disguise.}`
  - l. 502-503 : `exactly. With the same indices and untuned step constants ($1/4$ and $1$, three seeds) the joint SAGA-Muon needs the passes of MISO where MISO converges ($11.8$ against $11.9$, softmax with $\kappa=n$; SAGA: $20.2$) and stalls or diverges with it at $\kappa=10n$: at these steps it is not a spectral method. Steps $4$ to $16$ times smaller converge there, as do the tuned steps of \finding{M3}, where MISO does not.`
  - l. 172 : `at batch one and at textbook steps the nuclear-scaled polar step of SAGA behaves as MISO \nat{e}.`
- Confiance : élevée.

### F6 (major) : le résultat négatif pour l'estimateur L-SVRG n'est pas rapporté ; « a finite-sum estimator » n'est établi que pour la table SAGA

- Lignes 497-498 ; l. 171-172 (tableau 1) ; l. 636 (tableau 2).
- Citation : « Only a finite-sum estimator reaches rounding precision, with a length that vanishes with it
  (nuclear-scaled or clipped; on least squares also the soft map) or with fixed lengths cooled geometrically ».
- Problème : la proposition 3(c) (`prop:comp`) couvre « a SAGA or SVRG estimator », et le paragraphe « Experiments » présente les
  deux estimateurs. Or le pas polaire à longueur nucléaire de l'estimateur L-SVRG (« L-SVRG-Muon (joint) », pas réglé
  sur une grille de 2⁻¹⁰ à 1 fois 1/R²_op) manque la cible sur les six instances de M1 : 0/10 graines sur cinq
  instances, 1/10 sur soft-cond, distances relatives finales 5.5e-5 à 5.9e-2. Le papier ne le mentionne pas. La
  phrase est littéralement une condition nécessaire, mais les tableaux 1 et 2 la présentent comme une propriété de
  « a finite-sum estimator » ; un jury qui demande « et avec SVRG ? » reçoit une réponse négative absente du texte.
  C'est aussi l'illustration directe de la réserve du papier « (c) identifies a fixed point, not its attraction ».
- Preuves : `M1_summary.json` (digest l. 55, 111, 153, 211, 257, 304, 336) ; `num-noise_lsvrg_joint.py` :
  réexécution à 150 passes au pas retenu par M1, graine 10 : soft-mem erre entre 3e-4 et 2e-3 (7.3e-2, 7.5e-3, 3.2e-4,
  7.9e-4, 4.1e-4, 1.7e-3 aux passes 10, 20, 40, 80, 120, 150) ; ls-mem converge lentement (6.1e-8 à 150 passes, contre
  10⁻⁹ en 23 passes avec la table SAGA).
- Correction proposée (l. 497-498) :
  `Among the estimators tested at constant parameters, only the SAGA estimator reaches rounding precision, with a length that vanishes with it (nuclear-scaled or clipped; on least squares also the soft map) or with fixed lengths cooled geometrically \finding{M5}; with the loopless-SVRG estimator the nuclear-scaled step misses the targets of Section~\ref{sec:experiments} on all six instances.`
- Confiance : moyenne (faits certains ; la sévérité est un jugement, la phrase n'étant pas littéralement fausse).

### F2 (minor) : le rang nucléaire 1.07 est celui d'une seule instance

- Lignes 500-501. Citation : « (nuclear rank $\nnuc\omega^2/\nF\omega^2$ of $1.07$, median over the recorded states) ».
- Problème : 1.07 est la médiane de 5 états d'une trajectoire (graine 0, passes 1, 2, 5, 10, 20) sur soft-mem au pas
  réglé γ = 0.125/R²_op (expérience M2, pas celle de M6). Sur les cinq autres instances la même médiane vaut 1.36
  (ls-mem), 1.76 (soft-cond), 1.81 (ls-cond), 1.65 (ls-orth), 1.36 (soft-sq). La phrase donne la valeur la plus
  favorable sans nommer l'instance ; (M6) du papier s'appuie dessus (« the estimator being close to rank one (M2) »)
  pour des instances où le rang nucléaire est 1.76 à 1.81.
- Preuve : `M2_summary.json`, clé `nrank_omega` de `SAGA-Muon (joint)` → `own` ; digest l. 452, 471, 490, 509, 528, 547.
- Correction proposée : `(nuclear rank $\nnuc\omega^2/\nF\omega^2$ of $1.07$ on the softmax instance with $\kappa=n$, $1.36$--$1.81$ on the five others; medians over the recorded states)`.
- Confiance : élevée.

### F3 (minor) : deux unités dans (M1) ; 3.8 η² n'est pas dans l'unité de la figure 3(a)

- Lignes 493-496. Citation : « the iterates settle at a squared distance $3.8\,\eta^2$ (softmax) or $6.0\,\eta^2$ (least
  squares) … ($1.2\cdot10^{-1}$ and $4.8\cdot10^{-2}$ with the minibatch gradient, $b=32$ ».
- Problème : 3.8 η² et 6.0 η² sont des distances carrées absolues ‖W − W⋆‖²_F ; 1.2·10⁻¹ et 4.8·10⁻² sont des
  distances carrées relatives à ‖W⋆‖²_F (826.97 et 1146.99), comme l'axe de la figure 3(a). Le plateau bleu de la
  figure (η = 10⁻²) est à 4.6·10⁻⁷, soit 4.6·10⁻³ η² : un lecteur qui confronte texte et figure trouve un facteur 830.
  L'ancienne rédaction disait « an absolute squared distance » (`muon/findings.md` l. 891) ; le mot a disparu à la
  condensation.
- Preuve : `num-noise_floor_constants.py`.
- Correction proposée : `the iterates settle at an absolute squared distance $3.8\,\eta^2$ (softmax) or $6.0\,\eta^2$ (least squares)` et `(relative squared distances $1.2\cdot10^{-1}$ and $4.8\cdot10^{-2}$ with the minibatch gradient, $b=32$;`.
- Confiance : élevée.

### F4 (minor) : le tampon à quantité de mouvement testé avec la longueur nucléaire est STORM, pas MARS-M

- Ligne 496. Citation : « a momentum buffer as in MARS-M \cite{liu2025marsm} keeps a floor ».
- Problème : dans M3, la configuration à longueur nucléaire est « STORM buffer, nuclear-scaled » (g = 1, même lot, SVD
  exacte : `exp_M3_floors.py` l. 70-75), niveaux 4.1e-2 et 9.1e-3 (b = 32), 0.35 et 0.14 (b = 1). MARS-M (g = 0.025,
  correction écrêtée, ns5) n'a été lancé qu'à longueur fixe lr = 10⁻² (plancher 4.8e-3 et 1.7e-3). La phrase, placée
  dans la parenthèse sur la longueur nucléaire, attribue à MARS-M un essai fait avec STORM.
- Correction proposée : `a STORM buffer \cite{cutkosky2019momentum} with this length keeps a floor, $4.1\cdot10^{-2}$ and $9.1\cdot10^{-3}$)`.
- Confiance : élevée (la valeur par défaut de MARS-M est lue dans la docstring de `muon/lab/mvr.py`, pas dans l'article).

### F5 (minor) : portée de « Only … reaches rounding precision »

- Lignes 497-498.
- Problème : la conclusion repose sur M3 : deux instances limitées par la mémoire (soft-mem, ls-mem ; n = 1000, même
  graine de données), trois graines d'algorithme, 60 passes, alternatives toutes à paramètres constants (gradient de
  mini-lot b = 32 avec deux règles de longueur ; Muon b = 32 à lr 10⁻¹, 10⁻² et cosinus ; MARS-M ; STORM-Muon constant
  et cosinus ; tampon STORM à longueur nucléaire, b = 32 et 1). Non testés : estimateurs de mini-lot ou à quantité de
  mouvement avec paramètres décroissants (convergence sous-linéaire selon `findings.md` l. 366-368). Par ailleurs, dans
  les données stockées, ls-mem finit à 3.3e-20 à 60 passes, encore en décroissance linéaire (1.8e-21 au dernier point) :
  la précision d'arrondi n'y est pas atteinte dans le budget. Ma réexécution à 110 passes la confirme (arrêt sous
  1e-30 à la passe 87). Sur ls-cond au pas réglé, le même pas n'est qu'à 3.5e-10 à 3.9e-8 après 300 passes (phase
  lente, graines 10 à 12) : la démonstration d'arrondi vaut pour les deux instances limitées par la mémoire.
- Correction proposée : préfixe `Among the estimators tested at constant parameters,` (inclus dans la correction de F6).
- Confiance : élevée.

### F7 (minor) : la courbe Muon de la figure 3(a) n'est pas à son plancher ; « linear » vient de deux valeurs de lr

- Lignes 491, 494-495 ; figure 3(a).
- Problème : dans la figure 3(a) (titre « two floors »), la courbe Muon (lr = 10⁻², b = 32) vaut 1.3e-2 à la passe 60 et
  décroît encore ; son niveau stationnaire (run démarré à 10⁻¹⁴ de W⋆) est 5.7e-3. Le plancher de bruit de la figure
  est porté par la courbe « minibatch, nuclear-scaled ». La linéarité en lr est lue sur les niveaux stationnaires à
  deux valeurs de lr (10⁻¹ et 10⁻²) ; depuis W_0 = 0 le rapport des niveaux atteints à 60 passes est 3.5 (softmax) et
  1.2 (moindres carrés), pas 10.
- Preuve : `M3_summary.json`, traces `Muon b=32, lr 1e-2` (`trace` et `trace_stat`) ; `num-noise_floor_constants.py`.
- Correction proposée : `whereas the floor of minibatch Muon is linear in its learning rate ($5.7\cdot10^{-2}$ and $5.7\cdot10^{-3}$ at $\mathrm{lr}=10^{-1}$ and $10^{-2}$ on softmax, runs started at $W_\ast$; the run of Figure~\ref{fig:noise}(a) has not reached it).`
- Confiance : élevée.

### F8 (minor) : la borne μ_op η²/8 est juste, mais le lemme cité est sa forme en distance, pour des gradients exacts

- Lignes 491-492. Citation : « $\max(F(W_t),F(W_{t+1}))-F_\ast\ge\muop\eta^2/8$ whenever the step is nonzero
  \cite[Lemma~4]{jawanpuria2026constant} ».
- Problème : d'après la page HTML arXiv:2610.27747 (lue par l'outil de récupération, 100 000 premiers caractères), le
  lemme 4 (« Nonconvergence ») énonce max{‖E_t‖_F, ‖E_{t+1}‖_F} ≥ αγ/2 pour l'itération à gradient exact sur une
  quadratique fortement convexe (γ = 1 pour le facteur polaire). C'est une borne en distance de Frobenius, pour des
  gradients exacts. La forme en écart d'objectif avec μ_op, valable pour tout estimateur, est le corollaire des
  auteurs (notes de théorie, F2 et F13(b)). Elle est correcte (ligne 9 du tableau), mais elle ne se lit pas dans le
  lemme cité (la version Frobenius ne donne que μ_F η²/8 ≤ μ_op η²/8).
- Correction proposée : `whenever the step is nonzero, for any estimator (in distance and for exact gradients: \cite[Lemma~4]{jawanpuria2026constant})`.
- Confiance : moyenne (source lue à travers le résumé d'un outil, pas le PDF).

### F9 (nit) : « on least squares also the soft map » sous-affirme

- Ligne 497-498. Sur softmax la carte « soft » à longueur fixe n'a pas de plancher non plus : le run démarré à 10⁻¹⁴
  de W⋆ descend sans palier jusqu'à 5.7e-19 à la passe 30, et depuis W_0 = 0 il est à 1.4e-9 à la passe 60, en
  décroissance linéaire (`M3_summary.json`, « SAGA est., bounded gain »). La restriction aux moindres carrés reflète le
  budget de 60 passes.
- Correction proposée : aucune nécessaire (chaîne vide) ; au besoin `(nuclear-scaled or clipped; within $60$ passes on least squares also the soft map)`.

### F10 (nit) : raccourci du tableau 1, ligne 3

- Lignes 164-165. « Polar steps … converge to minus a nuclear median of the noise » : c'est le gradient qui converge
  vers l'opposé d'une médiane nucléaire (théorème 2(b), `thm:smd`), pas les itérés.
- Correction proposée : `Polar steps are stochastic mirror descent on a smoothed nuclear norm and their gradient converges to minus a nuclear median of the noise \nat{p};`.

## 4. Non vérifié

- Les expériences M2, M3, M6 complètes n'ont pas été relancées (heures de calcul) : réexécutions courtes seulement
  (graines 0, 1 ou 10 à 12 ; 32 à 300 passes ; instances soft-mem, ls-mem, soft-cond, ls-cond, κ/n = 10 de M6).
- L'identité des indices tirés entre les runs MISO et SAGA-Muon de M6 (« With the same indices ») n'a pas été
  contrôlée dans le code ; les passes par graine concordent à 11 % près.
- Les règles « clipped » et « soft map » de (M1) sont lues dans `M3_summary.json` sans réexécution.
- Les zéros du champ moyen pour b = 4 et 8 n'ont pas été répliqués (b = 2, 16, 32 l'ont été).
- Le lemme 4 de Jawanpuria et al. n'a été lu qu'à travers le résumé de l'outil de récupération (HTML arXiv) ; les
  valeurs par défaut de MARS-M (g = 0.025, écrêtage) ne sont lues que dans la docstring de `muon/lab/mvr.py`.
- Le début du paragraphe « Numerical checks » (0.464 ; 0.450-0.498, entrées Exp(1) − 1) est hors périmètre.
- La partie « or with fixed lengths cooled geometrically (M5) » de (M1) relève d'une autre dimension.
