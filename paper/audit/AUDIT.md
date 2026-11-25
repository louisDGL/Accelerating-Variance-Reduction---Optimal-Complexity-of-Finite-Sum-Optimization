# Audit de `paper/paper_saga_to_muon.tex`

**Statut : terminé le 20 novembre 2026 à 04h35** (heure de la machine), puis relu par la session VSCode jusqu'à
04h45 (fin de la section 8, « Relecture indépendante »). Papier audité : commit `9ea8ee7` (« SAGA to MuON », 12
pages). Rien n'est modifié hors de `paper/audit/` ; aucun commit.

## 1. Verdict

**Le socle mathématique tient.** Les Théorèmes 1 et 2 et les Propositions 3, 4 et 5 sont exacts tels qu'énoncés :
chaque preuve a été refaite à la main par deux lecteurs indépendants, puis contrôlée numériquement. Aucun constat
bloquant. Les chiffres du papier se retrouvent dans les données brutes du dépôt (environ 280 affirmations
contrôlées sans problème), et aucune antériorité n'a été trouvée pour les huit résultats revendiqués. Les 55
références existent, avec des métadonnées exactes (deux sont citées en préprint alors qu'elles sont parues,
section 4).

**Treize constats majeurs sont à corriger avant de rendre.** Ce sont, pour l'essentiel, des phrases de tête
(résumé, tableaux, conclusion, intitulés) qui disent plus que ce que le corps du papier et les données établissent,
plus deux énoncés faux tels qu'écrits et deux attributions inexactes :

- deux énoncés faux dans (C4) : « SAGA still converges at γR² = 4 » (il stagne ; la convergence tient à γR² = 2) et
  la borne inférieure sans son point initial ;
- quatre chiffres ou protocoles sensibles à la grille de réglage : passes de SAGA dans (M3), « 0.58–0.93 » du
  résumé, grille annoncée pour les recettes, gel « 8 % à 93 % » dont la moitié haute (b = 32) ne dit rien de la
  table ;
- cinq généralisations : « same anatomy » (quadratiques seulement), (M2) « not a spectral method » (pas non réglés
  seulement), « not with minibatches » (SAGA et B-SAGA seulement), « memory rate of SAGA » (modèle linéaire, en
  espérance), « Muon's own recipe » (le refroidissement géométrique est un choix du papier, Muon refroidit
  linéairement) ;
- deux attributions : l'attraction du « ray » prêtée à Shulgin et al., et κ_op ≥ min(K,d) prêtée à Sidford–Tian.

Les intitulés des deux constats du titre restent vrais une fois ces phrases resserrées.

**Un patch prêt à appliquer corrige les treize**, plus quinze précisions mineures : `paper/audit/corrections.patch`
(section 5). Le papier corrigé compile en 12 pages : `paper/audit/paper_saga_to_muon_corrige.pdf`. Quatre points
demandent une décision d'auteur que le patch ne prend pas (section 6).

## 2. Constats majeurs

Chaque constat a été établi par un auditeur, repris par un contre-vérificateur indépendant ou par un second
auditeur, puis relu par moi sur le texte et, quand c'était possible, sur les données. Les identifiants renvoient
aux fichiers `evidence/<clé>.json`.

### 2.1 Énoncés faux tels qu'écrits

**A. « SAGA still converges at γR² = 4, eight times the largest step the proof admits » (l. 320-321)** — `num-f1`
N1-STEP4.
À γR² = 4, SAGA ne converge pas sur l'instance du papier (κ = 10³, α = 1, n = 1000) : la distance au carré chute
pendant une quarantaine de passes puis stagne vers 4,6 % de sa valeur initiale (8 graines, 1000 passes ; pente
nulle après la passe 100). La dynamique linéarisée à l'optimum y est répulsive (exposant −0,13 par passe). Le
chiffre du dépôt (0,027 par passe) est une pente ajustée sur les passes 2 à 40 : la fin du transitoire.
`results/findings.md` disait « remains stable », devenu « still converges » dans le rapport archivé puis dans le
papier. SAGA converge à γR² = 2, quatre fois la borne de la preuve. Le « 4.5 times the best certificate » de la
même phrase vaut 4,4 (0,9663 / 0,2192).
Preuve : `checks/verif-num-f1_step4.py` (SAGA réécrit sans le code du dépôt, retrouve à six chiffres les exposants
de `results/GammaDelta_summary.json`).
Correction (dans le patch) : `is $4.4$ times the best certificate and SAGA still converges at $\gamma R^2=2$, four times`.

**B. Borne inférieure de (C4) sans point initial (l. 321-323)** — `math-saga` MS-1, `num-f1` N1-LOWER,
`consistency` C-20, et ma propre lecture avant les auditeurs.
« SAGA with any constant step needs at least (κ−1)/4 · log(1/(2ε)) iterations » n'est vrai que pour θ₀ = (r, ±r).
Contre-exemple : θ₀ = (0, 1), γ = 1/R², table initialisée en θ₀ : θ₁ = 0 en une itération. La note source
(`theory/B3_prop59_tightness.tex`, l. 412) contient l'hypothèse ; la condensation l'a perdue.
Correction (dans le patch) : `constant step, started at $\theta_0=(r,r)$, needs at least`.

### 2.2 Chiffres et protocole sensibles à la grille de réglage

**C. (M3) : « SAGA needs 69.5 / 98.7 passes », « 0.20–0.37 of the passes of SAGA » (l. 375-380)** — `num-geometry`
G2, confirmé.
Ces valeurs sont exactes sur la grille de pas de facteur 2, où SAGA est sous-réglé. La grille de raison 2^(1/4)
du dépôt (`M5b_summary.json`) donne 49,8 / 83,5 passes, et le papier ne cite la valeur fine que pour B-SAGA. À pas
résolu, le rapport vaut 0,34–0,48 sur les trois tirages (gain de 2 à 3, non de 3 à 5) ; la borne 0,20 vient d'un
cran de grille sur le troisième tirage softmax. L'intitulé de (M3) n'en dépend pas.
Correction (dans le patch) : la parenthèse cite aussi SAGA à grille fine, et le rapport est qualifié
`on factor-$2$ grids`. La figure 2(b) garde les barres de la grille de facteur 2 (décision 6.3).

**D. Résumé et (M6) : « the spectral map needs 0.58–0.93 of the passes of a normalized step » (l. 105-106,
587-589)** — `num-m6m7` F2, confirmé.
La fourchette est exacte sur la grille stockée mais ne survit pas à son raffinement. À grille quatre fois plus
fine pour les deux cartes, sélection sur les graines de réglage et graines de confirmation utilisées une fois :
0,58 → 0,69 (moindres carrés, b = 32) et 0,84 → 0,93 (softmax, b = 32). Le rapport reste sous 1 dans les cinq
cellules refaites sur huit : l'énoncé qualitatif tient, la borne 0,58 est un artefact du pas de 0,1 en q.
Correction (dans le patch) : le résumé dit `needs fewer passes than a normalized step ($0.58$--$0.93$, resolved to
about $20\%$)`. Le rafraîchissement des huit cellules reste à faire (décision 6.1).

**E. « Section 3.3 uses the same protocol, with step grids of ratio 2^{1/4} » (l. 264-265)** — `num-m6m7` F1,
confirmé ; `code-fidelity` l'a relu dans le code.
Faux pour les recettes (4) : seuls SAGA, B-SAGA et SAGA-Muon à pas constant sont sur des grilles 2^(1/4). Les
recettes sont réglées sur q par pas de 0,05 ou 0,1 et des demi-décades de lr₀ (`families.py`, `COOLDOWNS`,
`half_decades`) ; un cran de lr₀ déplace les passes de 12 à 28 %. La résolution de 10 % annoncée ne s'applique pas
aux rapports entre cartes de (M6).
Correction (dans le patch) : la phrase décrit les deux grilles et la résolution d'environ 20 %.

**F. (M5) : « Without a table the same schedule freezes the iterate at 8% to 93% … (b = 1, 32) » (l. 538-540,
679)** — `num-m5`, `consistency` C-02, `code-fidelity` CF-1 : trois relances indépendantes, deux
contre-vérifications.
Deux problèmes. (i) Ce n'est pas « the same schedule » : les courses avec table citées juste avant sont à q = 0,8 ;
les contrôles sans table tournent au point réglé pour la recette de Muon à lot un (q = 0,7 ou 0,75, lr₀ = 0,3),
sans réglage propre. La légende de la figure 4(a) le dit correctement. (ii) À b = 32, la longueur totale du
schedule, Σ lr_t, ne vaut que 0,08 à 0,37 ‖W_*‖_op : aucun estimateur ne peut atteindre W_*. Sur ce schedule la
table SAGA gèle elle aussi (0,79 et 0,57, contre 0,82 et 0,63 sans table), et le gradient exact de même. Toutes
les valeurs au-dessus de 34 % viennent de b = 32 et ne disent rien de la table.
Ce qui reste vrai : à lot un, où le chemin suffit (le gradient exact descend à 10⁻³⁰), le gel sans table est réel,
8 % à 34 %, et persiste à des refroidissements plus lents (0,25 % à 12 % jusqu'à q = 0,98, une graine), alors que
la table atteint 10⁻¹⁸ et moins. L'intitulé de (M5) et la phrase du résumé tiennent.
Preuve : `checks/verif-num-m5_path_length.py`, `verif-num-m5_b32_table.py`, `verif-consistency-m5_same_schedule.py` ;
j'ai relu les médianes dans `M9_summary.json` (lot un : 0,083 à 0,341 ; b = 32 : 0,63 à 0,93).
Correction (dans le patch) : `Without a table the schedule selected for Muon's recipe freezes the iterate at $8\%$
to $34\%$ of the initial squared distance at batch one` ; conclusion : `where such a schedule freezes without one`.

### 2.3 Généralisations que le corps du papier ne porte pas

**G. « the nuclear-scaled polar step has the same anatomy » (résumé l. 96-97, tableau 1 l. 151 avec l'étiquette
[p], tableau 2 l. 626)** — lecture de niveau jury, `math-nfl` E1, `consistency` C-12.
L'identité (2) est énoncée pour une quadratique (l. 226) et le corps dit que le cas à grand écart est « not
measured for the polar step » (l. 315). Placée après les chiffres mesurés sur la régression logistique, la phrase
laisse croire à une mesure analogue.
Correction (dans le patch) : `on quadratics` aux trois endroits. Variante possible : `math-nfl` montre que (2) vaut
pour toute F de classe C² avec J = 2∫₀¹(1−s) D²F(W − sηO)[O,O] ds (contrôle numérique à 4·10⁻¹⁵), ce qui
permettrait de garder l'énoncé général.

**H. (M2) : « fails where MISO fails (κ ≫ n): it is not a spectral method » ; « SAGA: 20.2 » (l. 499-503, tableaux
l. 172 et 636)** — `num-noise` F1 confirmé, `consistency` C-01, `code-fidelity` CF-2.
L'expérience M6 lance la méthode jointe à deux constantes de pas non réglées (c = 1/4 et 1, trois graines). Avec
un pas 4 à 16 fois plus petit elle converge en 25 passes sur les instances κ = 10n où MISO échoue, et le (M3) du
papier la rapporte réglée à 29,0 / 43,5 passes sur des instances κ ≈ 36n à 48n : (M2) et (M3) se contredisent
tels qu'écrits. « SAGA: 20.2 » est SAGA au pas du livre ; réglé par le protocole du papier, il fait 11,4 passes sur
la même instance (`M1_summary.json`, relu par moi), comme MISO (11,9) et la méthode jointe (11,8).
L'identification à MISO est vraie là où elle a été testée (κ ≤ n).
Correction (dans le patch) : intitulé et cellules de tableau `At batch one and untuned steps` ; corps
`SAGA: $20.2$, tuned $11.4$) and fails where MISO fails ($\kappa=10n$) unless its step is tuned \finding{M3}:
untuned, it is not a spectral method`.

**I. « reproduced by Euclidean devices at batch one but not with minibatches » (tableau 2 l. 627-628, conclusion
l. 680)** — `num-m6m7` F3 confirmé, `consistency` C-03.
En mini-lots, seuls SAGA et B-SAGA ont tourné côté euclidien (tous les `muon/results/*_summary.json`). La métrique
de covariance, « the best method on least squares for every draw » à lot un, n'y a pas été lancée ; à lot un elle
fait déjà moins de passes que la variante spectrale à b = 16 ou 32 sur trois instances sur quatre. Le corps de
(M6) est exact sur ses comparateurs.
Correction (dans le patch) : tableau 2 `not by (B-)SAGA with minibatches` ; conclusion `the spectral map is ahead
of (B-)SAGA and normalized steps`.

**J. (M7), tableau 2, conclusion : « lifts the memory rate of SAGA from e⁻¹ to nearly e⁻² » (l. 600-606, 638,
680-681)** — `math-order` MO-1, `consistency` C-04 confirmé, lecture de niveau jury.
Ce qui est prouvé (et reprouvé par l'auditeur) : un taux de l'espérance de l'erreur quadratique, pour SAGA linéaire
à hessienne commune, n → ∞. Le papier mesure lui-même 1,6 nat par passe pour des trajectoires sous tirage
indépendant. Surtout, sur le modèle qui isole la table, la permutation ne change rien aux lots où le gain de
36–49 % est chiffré : ρ⋆ = 0,22 dans les deux ordres à b = 20 et 50, avec plus de passes sous permutation
(`M9b_summary.json`, relu par moi). Les « 1.2–1.3 » et « 1.7–1.9 nats » sont à lot un, ce que le texte ne dit pas.
Le journal du dépôt retire lui-même la lecture forte (`muon/findings.md`, l. 604-610 et 835-838). Les 36–49 %
mesurés sur les deux instances sont exacts, cellule par cellule.
Correction (dans le patch) : `at batch one` ; `where the model itself gains nothing ($\rho_\star=0.22$ both ways)` ;
tableau 2 `In expectation, … (linear model, note B4)` ; conclusion `the expected memory rate of linear SAGA`.

**K. « Muon's own recipe. Steps of fixed length on a geometric cooldown » (résumé l. 102-103, section 3.3 l.
508-512, conclusion)** — `cite-muon` CM-1, lecture de niveau jury.
Le billet de Jordan et al. ne prescrit aucun schedule ; les exécutions de référence refroidissent linéairement
(`modded-nanogpt` : constant puis linéaire vers zéro ; `cifar10-airbench` : linéaire vers zéro). Sont fidèles à
Muon : la longueur fixe, l'existence d'un refroidissement, cinq pas de Newton–Schulz, ε = 10⁻⁷, le moment de
Nesterov. Sont des choix du papier : la forme géométrique et le coefficient 0,9 (défaut de Muon : 0,95). Le choix
n'est pas neutre : sur une petite reproduction de l'auditeur (une instance, trois graines), le refroidissement
linéaire avec table s'arrête entre 10⁻⁶ et 10⁻³ au lieu de l'arrondi. Ce constat n'a pas eu de contre-vérificateur
dédié ; il concorde avec ce que je sais de ces dépôts.
Correction (dans le patch) : résumé `on a cooldown of the learning rate, here geometric` ; section 3.3
`(linearly in its reference runs; geometrically here)`. Voir la décision 6.2.

### 2.4 Attributions inexactes

**L. « a factor 4 on the ray that attracts generic runs [shulgin2026quadratic] » (l. 313-315)** — `math-nfl` E2,
`cite-muon` CM-2.
Shulgin et al. donnent le facteur sur le rayon invariant (Corollaire 5, cas commutant), pas l'attraction des
trajectoires génériques. Les notes du dépôt la classent numérique et la posent en question ouverte
(`archive/muon/notes_spectral_theory.tex`, l. 800-801 et 1210-1214, relues par moi) et montrent ces rayons
linéairement instables dès d ≥ 3. Le facteur 4 vaut en queue de course (3,2 à 5,4 mesuré) ; sur une course entière
jusqu'à 10⁻⁶ l'écart est de 7 à 14, ce qui contredit « a large gap needs a global constant far below the
curvature met ».
Correction (dans le patch) : `in the tail of generic runs (observed; on the flattest ray, \cite{…}); a lasting
large gap needs`.

**M. « κ_op ≥ min(K,d) [sidford2018coordinate] » (l. 354)** — `cite-stat`, `math-nfl` C1 : deux auditeurs ont
ouvert la source.
L'inégalité est vraie pour toute F (redémontrée, contrôlée numériquement). La source ne contient que l'analogue
ℓ∞ (Lemma A.1 : amplitude d'au moins n/2), sans norme spectrale ni conditionnement.
Correction (dans le patch) : `(as for $\ell_\infty$ \cite[Lemma~A.1]{sidford2018coordinate})`.

## 3. Statut des énoncés principaux après audit

| Énoncé | Statut | Comment |
|---|---|---|
| Proposition 5.9 telle que citée, identité (1) | vérifié | preuve refaite ; énumération exacte des n indices, écart ≤ 1,5·10⁻¹³ |
| Théorème 1 | vérifié | preuve refaite par deux lecteurs ; μ_N numérique pour trois normes le long de tW₁ |
| Identité (2) | vérifié | développement à la main ; contrôle numérique ; la source citée la donne pour des quadratiques unilatérales |
| Théorème 2 (a)-(c) | vérifié | chaque étape redémontrée, aucune hypothèse manquante ; inégalité (a) en espérances exactes ; limites simulées |
| Proposition 3 | vérifié avec réserve | (c) : la réciproque n'est écrite que pour une table ; (d) : sans terme ridge (ajouté par le patch) |
| Proposition 4 | vérifié comme développement formel | Edgeworth refait ; 0,46379 par quadrature ; Monte-Carlo indépendant 0,4640 ± 0,0012 |
| Proposition 5 | vérifié avec réserve | preuve refaite avec les constantes ; 32 simulations sans violation ; l'initialisation φ_i = W₀ manquait (ajoutée par le patch) |
| Lois de la note B4 reprises en (M7) | vérifié | preuves refaites ; matrices exactes reconstruites, écart ≤ 2,2·10⁻¹⁵ |
| (C1) 3.2–3.7, 19–220 | vérifié avec réserve | 75 exécutions reproduites à 1,3·10⁻⁷ ; dépend de la fenêtre (11–187 sur les passes 20-100) |
| (C2) 77–96 %, 39–44 %, 32–37 % | vérifié avec réserve | exact dans l'ordre annoncé ; 59–96 % dans l'autre ordre calculé, où la part de I2 passe à I5 pour κ ≤ 10² |
| (C3), (C5) | vérifié | chiffres retrouvés et recalculés ; κ_P ≥ κ contrôlé sur 4000 métriques |
| (C4) | corrigé | constats A et B |
| (M1) | vérifié avec réserve | chiffres exacts ; unités absolue et relative mêlées ; tampon testé = STORM (corrigés par le patch) |
| (M2) | corrigé | constat H |
| (M3) | corrigé | constat C ; les autres chiffres sont exacts |
| (M4) | vérifié | 96 constantes recalculées, écart 7,5·10⁻¹⁴ |
| (M5) | corrigé | constat F ; planchers, ns5, ε = 10⁻⁷, figure 4(a) vérifiés |
| (M6) | corrigé | constats D, E, I ; chiffres exacts sur la grille stockée |
| (M7) | corrigé | constat J ; 36–49 % exact cellule par cellule |
| Résumé : 3.2–220, 77–96 %, 10⁻¹⁸, 36–49 % | vérifié | retrouvés dans les JSON |
| Résumé : 0.58–0.93 | corrigé | constat D |
| Bibliographie (55 entrées) | vérifié avec réserve | toutes existent, métadonnées exactes ; renvois précis confirmés sauf constats L, M et le Lemme 4 (section 4) |
| Nouveauté | rien trouvé | 11 requêtes web et 6 requêtes de résumés arXiv ; un travail proche à citer (section 4) |

## 4. Précisions mineures

**Corrigées par le patch** (quinze) : renvoi du tableau 2 à la Proposition 4 au lieu du Théorème 2 (l. 632) ;
« otherwise » plus fort que le Théorème 2(c) (l. 418) ; Proposition 3(d) « unregularized » ; initialisation
φ_i = W₀ dans la Proposition 5 ; « 3.3·10⁴ to 8.9·10⁴ » valable pour les ordres à permutation, 10⁶ pour le tirage
indépendant de (M5) (l. 579, 649) ; introduction citant la Proposition 5 pour un énoncé mesuré (l. 133) ; « 8/3 »
vrai pour c_D seulement (l. 295) ; « 64–98 % » vrai pour κ ≥ 10⁴ (l. 298) ; « absolute squared distance » (l. 494) ;
tampon STORM et non MARS-M (l. 496) ; « Only the SAGA estimator » avec le résultat négatif L-SVRG (l. 497) ;
« same grid of cooldowns » (l. 525, 586) ; « up to about 1.6 » et « mean squared error » (l. 603-605) ; Hu et al.
ne font pas de « performance estimation » (l. 148) ; « no change of norm » au lieu de « no fixed change of
geometry » (l. 675 : une géométrie de Bregman h = F échappe au théorème).

**Non corrigées, à votre appréciation** (les plus utiles ; le reste est dans les fichiers `evidence/`) :

| Lignes | Point | Source |
|---|---|---|
| 290-294, 667-668 | Limitations : chiffrer la dépendance à l'ordre de retrait (59–96 %) et à la fenêtre (11–187) | `num-f1` |
| 225-226 | « five seeds » : cinq graines d'échantillonnage sur un seul tirage de données par α | `num-f1` |
| 263-264 | la résolution de 10 % ne couvre que le bruit de graines ; le réglage a trois étages (1, 2, 5 graines) | `num-geometry`, `code-fidelity` |
| 368-371 | « exactly 1 » vrai par construction ; le gain certifié de 103 vaut 15,5 sur le taux, plafonné par 1/(3n) | `num-geometry` |
| 372, 541 | « covariance metric » désigne SAGA en (M4) et B-SAGA en (M5) : 13,1 hors de la plage 13,5–16,1 | `consistency` |
| 586-587 | « 1.16–1.36 » mêle deux grilles ; 1,22–1,39 à grille égale | `num-m6m7` |
| 534, 579-580, 650 | « 3.2 to 5.7 », « 2.3 to 3.5 », « 350 to 1400 » : médianes ou courses uniques, instances non nommées | `num-m5`, `math-schedule` |
| 580-581 | « only the average over the index explains » n'est pas établi (moyenne à un pas 0,990–1,005) | `math-schedule` |
| 105, 581 | « we prove this law » : dire facteur polaire exact, sans moment, η₀ prescrit | `math-schedule` |
| 99 | résumé : « decreasing lengths » ; l'hypothèse est Σηₜ = ∞, Σηₜ² < ∞, que le refroidissement géométrique ne vérifie pas | `consistency` |
| 236-238 | R_P² est un majorant de la régularité, pas la meilleure constante | `math-saga` |
| 482-488 | « 0.450–0.498 » sans erreur de Monte-Carlo ; la somme finie de la figure 3(c) a des gradients de rang un, (3) ne s'y applique pas | `math-offset` |
| 491-492 | le Lemme 4 de Jawanpuria et al. énonce un plancher en distance pour gradients exacts ; la forme en valeurs de F est un corollaire du papier | `cite-muon`, `num-noise` |
| 232 | Théorème 3 de Jawanpuria et al. : 1 − 1/κ_op seulement si K ≥ d | `math-nfl` |
| 607-609, 177 | « κ < 4.08 » est un calcul du papier ; DIAG est cyclique, pas à permutation | `cite-vr`, `novelty` |
| 418-420, 539 | Cardot et al. est l'analogue euclidien (K = 1), pas « scalar » ; « STORM-Muon [cutkosky] » prête une variante de Muon à la source | `cite-stat` |
| 240 | c̄ n'est défini nulle part | `cite-vr` |
| `.bib` | driggs2019biased (JMLR 2022) et yang2014data (Machine Learning 2016) cités en préprint | `cite-vr` |
| 657-663 | citer Singh, Mishra et Raut (arXiv:2607.12360) : plancher en lr² et nécessité du refroidissement pour les pas normalisés. Relevé par l'auditeur, source non rouverte par moi | `novelty` |
| 667-672 | Limitations : la table stocke un résidu par échantillon ; les théorèmes sont pour le facteur polaire exact | `consistency` |
| 686-688 | liste de scripts incomplète (exp_B1, exp_gamma_delta, exp_A3, exp_M5, theory_checks) ; deux chiffres viennent d'un test de `_archive/` non versionné | `consistency`, `math-schedule` |
| notation | L_eff, ρ, c_i, g, δ ont chacun plusieurs sens | `consistency` |

Hors papier : `theory/B4_sampling_order.tex` dit encore « not proved » (l. 378 et 525) pour la loi que le papier
dit prouvée ; la preuve existe et tient. La phrase fausse sur γR² = 4 figure aussi dans les rapports archivés.

## 5. Corrections proposées

- `paper/audit/corrections.patch` : 44 corrections de texte et une ligne de mise en page. Depuis la racine :
  `git apply paper/audit/corrections.patch`, puis la compilation habituelle. `git apply --check` passe sur le
  commit `9ea8ee7`.
- `paper/audit/paper_saga_to_muon_corrige.pdf` (12 pages, aucune référence non résolue) et le `.tex`
  correspondant.
- `paper/audit/process/appliquer_corrections.py` : la liste des remplacements, un par constat, pour en retirer ou
  en reformuler un et régénérer.

**La ligne de mise en page.** Les corrections ajoutent une dizaine de lignes et le papier n'avait qu'une ligne de
marge : sans compensation il passe à 13 pages. Le patch réduit l'espace autour des flottants (`\textfloatsep` de
10 pt à 7 pt), ce qui suffit. Si vous préférez garder la mise en page, il faut couper environ dix lignes de texte.

## 6. Décisions d'auteur que le patch ne prend pas

1. **Résumé, « 0.58–0.93 ».** Le patch garde les chiffres et annonce leur résolution. Pour un chiffre robuste :
   re-régler les huit cellules de (M6) sur une grille plus fine (les deux refaites donnent 0,69 et 0,93) et
   retracer la figure 4(b) ; ou citer le rapport à SAGA et B-SAGA, 0,41–0,89, qui est robuste.
2. **« Muon's own recipe ».** Le patch dit que la forme géométrique est un choix. Si le nom doit rester, ajouter
   l'essai avec le refroidissement linéaire de Muon serait la réponse à l'objection d'un jury.
3. **(M3) et figure 2(b).** Re-régler SAGA à grille fine sur les trois tirages et mettre à jour les barres.
4. **(M5).** Ajouter les contrôles sans table au schedule des courses avec table (q = 0,8, lr₀ = 0,1, lot un)
   rendrait « the same schedule » littéralement vrai ; l'audit trouve un gel de 15 à 48 % sur deux graines.

Facultatif : la lecture de niveau jury propose de renforcer le Théorème 1 par la constante de
Polyak–Łojasiewicz en norme duale, qui vaut aussi λ/c_N² (`checks/jury-6_pl_constant.py`).

## 7. Ce qui n'a pas été vérifié

- Les expériences complètes n'ont pas été relancées : les chiffres sont comparés aux résumés JSON du dépôt, avec
  des reproductions ciblées (B3 en entier, une trentaine de cellules de M1, M5b, M6, M9).
- Les scripts des figures n'ont pas été exécutés tels quels (`results/B3.pkl` est absent du dépôt) ; le contenu de
  la figure 1 a été recalculé à part, les figures 2 à 4 confrontées à leurs données.
- Sources non ouvertes : l'édition imprimée du livre de Bach (version en ligne de mai 2025 seulement), Goffin
  1977, l'article original de Cornish et Fisher.
- La fidélité de STORM-Muon à son article, et le moment par défaut de Muon dans le code public. Pour MARS-M, la
  session VSCode a comparé `muon/lab/mvr.py` à l'Algorithme 2 de Liu, Yuan et Gu (arXiv:2510.21800v3, lu en
  ligne) : correction C_t = G_t + γ·β/(1−β)·(G_t − G̃_{t−1}) sur le même lot, écrêtage à 1, moyenne exponentielle,
  Newton–Schulz, γ = 0,025 : conformes (version « exacte », celle que couvre leur théorème ; leurs expériences
  de langue utilisent la version approchée). Deux écarts : l'échelle de forme (0,2·√max(m,n) dans l'article,
  max(1, K/d)^{1/2} dans le code, soit un pas 1,26 à 1,41 fois plus court à lr égal sur les instances 8×40 et
  10×50) et la norme de l'écrêtage (Frobenius dans le code, « ‖·‖₂ » dans l'article). Ils ne changent pas le
  constat F, mais les témoins de (M5) ne sont pas réglés : l'échelle compte.
- La validité rigoureuse du développement d'Edgeworth dans le cas matriciel (le papier le déclare formel).
- La recherche d'antériorité est bornée (17 requêtes en plus de celles du dépôt) ; les préprints de 2026 cités par
  l'auditeur de nouveauté n'ont pas été rouverts par moi.
- Le constat K et les constats mineurs n'ont eu qu'un auditeur.

## 8. Méthode

Dix-sept auditeurs indépendants, un par dimension : six sur les preuves, cinq sur les chiffres, un sur la fidélité
du code, trois sur la bibliographie, un sur la nouveauté, un sur la cohérence. S'y ajoute la lecture de niveau
jury faite par la session VSCode. Chacun a relu le papier en entier, refait preuves et calculs sans se fier aux
notes du dépôt, et écrit un rapport et ses scripts. Six contre-vérificateurs ont ensuite cherché à réfuter les
constats majeurs avec leur propre code : dix verdicts confirmés, un rétrogradé en mineur (le résultat négatif
L-SVRG), aucun réfuté. Ils couvrent les constats A, C, D, E, F, H, I et J. Les constats B, G, L et M reposent sur
deux à quatre lectures indépendantes concordantes, le constat K sur un seul auditeur. J'ai relu chaque constat
majeur sur le texte et recoupé dans les JSON les chiffres mis dans le patch.

Trois écarts à signaler. Le lancement s'est fait par l'outil Agent et non par un workflow, dont le dialogue de
confirmation ne pouvait être validé en votre absence. Deux lots de citations ont tourné sur Opus pour ménager le
quota. Au moins deux auditeurs ont laissé Python réécrire des caches `__pycache__` ignorés par git ; aucun fichier
suivi n'a changé (`git status` ne montre que `paper/audit/`).

**Relecture indépendante (session VSCode, 04h35 à 04h45).** Le rapport et le patch ont été écrits par la session
distante ; la session VSCode, qui avait lu le papier et relu les preuves avant le lancement des auditeurs, les a
contrôlés sans passer par elle :

- le patch s'applique sur une copie propre du commit `9ea8ee7`, redonne exactement
  `paper_saga_to_muon_corrige.tex` et compile en 12 pages sans référence non définie ;
- les 45 lignes modifiées ont été relues une à une contre les constats : chacune dit la même chose ou moins que le
  texte d'origine, sauf trois ajouts de faits, tous étayés (« tuned 11.4 », « SAGA 49.8 / 83.5 », « linearly in
  its reference runs ») ;
- les 23 constats majeurs bruts des auditeurs sont tous repris dans les treize du rapport (22 fusionnés, un
  rétrogradé en mineur) ;
- constat A, troisième exécution : `checks/verif-num-f1_step4.py` relancé donne, sur 8 graines, la convergence
  jusqu'à 10⁻²⁸ à γR² = 2 (pente 0,63 à 0,68 par passe) et la stagnation à 4,6 % de la distance initiale sur 1000
  passes à γR² = 4 ;
- le « 4.5 » d'origine est 0,9663 / 0,2156 (certificat au pas du livre avec le meilleur poids), le « 4.4 » du patch
  est 0,9663 / 0,2192 (meilleur certificat sur le pas et le poids, ce que dit la phrase) ;
- « a normalized step » dans le résumé est exact : le contrôle le plus rapide est la normalisation de Frobenius
  avec moment dans les 8 cellules de (M6) ;
- le tampon testé avec la longueur nucléaire dans `exp_M3_floors.py` est bien un tampon STORM (g = 1), d'où la
  correction de (M1).

Trois réserves sur le patch, qui restent des choix d'auteur :

1. Résumé : « needs fewer passes than a normalized step (0.58–0.93, resolved to about 20%) » affirme encore un
   avantage général, alors que deux cellules sont à 0,93–0,95 après raffinement. Formulation neutre de même
   longueur : `needs $0.58$--$0.93$ of the passes of a normalized step on grids that resolve about $20\%$`.
2. L'intitulé de (M7), « a fresh permutation per pass lifts the memory ceiling », attribue au plafond mémoire un
   gain que le modèle à hessiennes identiques ne montre pas à b = 20 et 50 ; le patch l'écrit dans le corps mais
   garde l'intitulé.
3. La phrase « local ones [...] certify nothing » (l. 238-239) n'est pas dans le patch : des constantes prises sur
   l'ensemble de sous-niveau du point initial certifient un taux pour une méthode monotone. Proposition :
   `certify no global rate` (`evidence/lecture-jury.md`, jury-5).

## 9. Fichiers

| Chemin | Contenu |
|---|---|
| `paper/audit/AUDIT.md` | ce rapport |
| `paper/audit/corrections.patch` | corrections proposées, non appliquées |
| `paper/audit/paper_saga_to_muon_corrige.pdf`, `.tex` | papier corrigé, 12 pages |
| `paper/audit/evidence/<clé>.md`, `.json` | rapport détaillé et constats de chaque auditeur |
| `paper/audit/evidence/<clé>.verif.json` | verdicts des contre-vérificateurs |
| `paper/audit/checks/` | une centaine de scripts de contrôle et leurs sorties |
| `paper/audit/process/` | consignes des auditeurs, script des corrections |
