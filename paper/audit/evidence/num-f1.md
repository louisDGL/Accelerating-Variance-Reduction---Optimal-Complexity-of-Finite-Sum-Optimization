# Audit num-f1 : chiffres et protocole de la partie SAGA (Finding 1)

Etat : TERMINE (2026-11-20). Papier audite : paper/paper_saga_to_muon.tex (numeros de ligne de l'outil Read).

## 1. Perimetre couvert

Resume (l. 94-96), tableau 1 ligne 1 (l. 150-151), protocole (l. 222-226), legende de la figure 1 (l. 283-286) et
figure paper/figures/figR1_classical.png, constats (C1) a (C5) (l. 290-331), ordre de retrait des inegalites
(vrlab/analysis.py, experiments/exp_B3_tightness.py) et reserve des Limitations (l. 667-668).

Sources de donnees : results/B3_summary.json, B3_families_summary.json, B2 (results/digest.md), B1_summary.json,
A3_summary.json et A3.pkl, GammaDelta_summary.json et GammaDelta.pkl. Les caches results/B3.pkl, B1.pkl,
B3_families.pkl ne sont pas dans le depot (git-ignores) : les chiffres correspondants ont ete **recalcules par de
petites reproductions** (memes fonctions vrlab, memes graines), pas seulement relus dans les JSON.

## 2. Scripts de controle (paper/audit/checks/, tous lances depuis la racine, un seul processus, BLAS mono-thread)

| Script | Objet | Resultat |
|---|---|---|
| num-f1_b3_summary.py | plages de (C1)/(C2) depuis B3_summary.json, ordre A | toutes les plages du papier retrouvees |
| num-f1_b3_rerun.py | 75 executions SAGA (15 problemes x graines 0-4, 20100 iterations), attribution passes 2-20 | moyennes identiques au JSON (pire ecart 1.3e-7) ; plages par graine |
| num-f1_b3_families.py | phrase "eight data sets" depuis B3_families_summary.json | 8 jeux, 64-98 % (kappa >= 1e4 seulement), 12.6-198.4 |
| num-f1_families_rerun.py | reproduction ponctuelle de 3 problemes des familles a kappa = 1e5 | identique au JSON (198.42 ; 12.64 ; 16.77) |
| num-f1_gamma_delta.py | (C4) : optimum (gamma, Delta) dans la preuve, rapports 4.5 / 4.4, pas gamma R^2 = 4 (400 passes) | gain 1.169 ; 4.48 ou 4.41 ; **pas de convergence a gamma R^2 = 4** |
| num-f1_gamma_delta_long.py | gamma R^2 = 2 (60 passes) et 4 (1000-2000 passes) | converge a 2 ; stagne a 4 |
| num-f1_b1_rerun.py | (C5) : B1 (kappa = 1e3, alpha = 1, 2 ; P = I et Sigma_hat + lam I), 5 puis 10 graines | identique au JSON |
| num-f1_b2_alignment.py | (C3) : alignement a kappa = 10n, graines 0-4 | confirme pour alpha = 0, 1 ; pas pour alpha = 2 |
| num-f1_a3_from_pkl.py | (C3) : L-Katyusha 2.8 / 4.1 depuis A3.pkl | 2.775 / 4.117 (2.789 / 4.134 sur passes 2-20) |
| num-f1_fig1a.py | figure 1(a) : les trois courbes recalculees | coherent avec la figure et le digest |
| num-f1_repr_and_lower.py | identite (4) du papier ; contre-exemple a la borne inferieure de (C4) sans condition sur theta_0 | identite exacte a 4e-16 ; contre-exemple en 1 iteration |

## 3. Ce que calcule le code (definition des quantites citees)

- "Exact contraction exponent" : pour chaque etat instrumente (toutes les n/10 = 100 iterations), rho_exact =
  E_t[V_{t+1}]/V_t calcule exactement (enumeration des n indices, vrlab/instrumentation.py:126-148) ; exposant
  c = -log rho. "Certified" : -log max(rho_D, rho_M) = -log(1 - min{1/(3n), 3 mu/(16 R^2)}).
- Speed-up (C1, figure 1(b)) : (moyenne de fenetre de c_exact) / (moyenne de fenetre de c_bound), fenetre = passes 2 a 20
  (181 enregistrements par execution, aucun ecarte par le seuil V >= 1e-18), puis **moyenne sur 5 graines
  d'echantillonnage** (vrlab/analysis.py:139-148, experiments/exp_B3_tightness.py:150-161). Un seul jeu de donnees
  (graine 0) par alpha ; kappa ne change que lambda.
- Parts (C2, figure 1(c)) : differences des exposants moyens de fenetre le long de la chaine
  exact <= I1 <= I3 <= I2 <= I4 <= I5 = borne (vrlab/analysis.py:22-35), divisees par c_exact - c_bound : c'est bien
  "retirer, depuis la borne, I5, I4, I2, I3, I1" (ordre B). Le "logarithmic gap" est donc la difference des exposants
  -log rho (= log(rho_bound/rho_exact)), pas le logarithme du speed-up. Moyenne des parts sur les 5 graines.
- Ordre A (stocke dans le meme JSON, cle shares_orderA) : retirer I4, I2, I3, I1 en gardant le max, puis I5.
  La part I4+I2 de l'ordre A est exactement "ce que recupere la Prop. 5.9 reecrite avec mu_eff et L_eff" (59-96 % de C3).

## 4. Affirmations controlees

### Resume, tableau 1, (C1), figure 1(b)

| Affirmation (lignes) | Recalcul | Verdict |
|---|---|---|
| "3.2-3.7 times ... for kappa <= 10^2" (290) | moyennes 5 graines, 6 configurations : 3.172 a 3.710 | OK |
| "19-220 times at kappa = 10^5" (291) | 18.95 (alpha=2), 100.3 (alpha=1), 220.4 (alpha=0) | OK |
| "3.2 to 220" (95, 150) | min 3.172 (kappa=1e2, alpha=0), max 220.4 sur les 15 configurations | OK |
| realite de l'exposant : pente realisee de -log D_t sur passes 2-20 / certifie | 3.3-4.3 (kappa<=1e2) ; 210-220, 99-100, 18.8-18.9 (kappa=1e5) | OK, coherent avec l'exposant conditionnel |

Reserves (constats N1-SEEDS, N1-WINDOW) : par graine isolee les facteurs vont de 2.90 a 4.73 pour kappa <= 1e2 et de
18.9 a 223.9 a kappa = 1e5 ; sur la fenetre tardive passes 20-100 (meme JSON) les facteurs a kappa = 1e5 sont
11.3 / 71.6 / 186.6 au lieu de 18.9 / 100.3 / 220.4.

### (C2)

| Affirmation (lignes) | Recalcul | Verdict |
|---|---|---|
| "I2 and I4 ... carry 77-96 %" pour kappa >= 1e4 (292-293, 95, 151) | I2+I4, ordre B, moyennes : 0.7691 a 0.9606 (par graine 0.748-0.961) | OK (ordre B) |
| "I1 (39-44 %) and I2 (32-37 %)" pour kappa <= 1e2 (293-294) | I1 0.3923-0.4428 ; I2 0.3170-0.3651 | OK (ordre B) ; ordre A : I2 = 0, I5 = 56-61 % |
| "where the memory term is active" (293) | 1/(3n) < 3 mu/(16 R^2) ssi kappa < 9n/16 = 562.5 | OK |
| "I3 ... at most 0.6 %" (294) | max des moyennes 0.54 % (kappa=10, alpha=1) ; 0.75 % pour une graine ; 6.3 % en ordre A | OK (moyennes, ordre B) |
| "at most a factor 8/3" (295) | vrai pour le coefficient c_D ; dans c_M le facteur du a I1 (et I3) va jusqu'a 3 (mesure 1.97-3.00) | imprecision (N1-83) |
| "first factor is 2.45-2.62" (296) | moyennes geometriques de fenetre 2.4509-2.6244 ; par enregistrement 2.36-2.63 | OK |
| "the second reaches 81" (297) | mu_eff/mu = 81.34 (kappa=1e5, alpha=0, moyenne geometrique ; jusqu'a 147 par enregistrement) | OK |
| "product reproduces 91-96 %" (297) | 0.9060 a 0.9635 | OK |
| "eight data sets ... 64-98 %" (297-298) | 8 jeux (P x4, L x2, R x2) ; I2+I4 0.6400-0.9834 **pour kappa >= 1e4** ; 0.37-0.98 sur tous les kappa | OK si kappa >= 1e4 (N1-FAM) |
| "13 to 198 at kappa = 10^5" (298-299) | 12.64 a 198.42 | OK |

### (C3)

| Affirmation (lignes) | Recalcul | Verdict |
|---|---|---|
| alignement "at kappa = 10n for alpha in {0,1}" (302-303) | graines 0-4, 60 passes : part de l'erreur sur le vecteur propre le plus plat 0.997-0.9997, mu_eff/lmin(H*) = 1.001-1.011 ; alpha=2 : 0.002, 1.285 | OK (le JSON ne contient qu'une execution ; confirme sur 5) |
| "recovers 59-96 % ... for kappa >= 10^4" (303-304) | part I4+I2 en ordre A : 0.5898 a 0.9636 ; figure 1(a) (kappa=1e4, alpha=1) : 0.91 | OK |
| "2.8 and 4.1 ... against 5.2 and 15.7" (305-306) | A3.pkl : 2.775, 4.117 (3 graines, enregistrements Psi >= 1e-10) ; sur passes 2-20 : 2.789, 4.134 ; SAGA alpha=1 : 5.155, 15.73 | OK |

### (C4)

| Affirmation (lignes) | Recalcul | Verdict |
|---|---|---|
| "at most a factor 3 (4/3 when kappa >= 9n/16)" (318-319) | optimum par balayage : gain -> 2.994 quand kappa -> 1 (n = 1000), 1.05 a 1.333 pour kappa >= 9n/16 | OK |
| "at kappa = 10^3 they gain 17 %" (319) | 0.2192/0.1875 = 1.169 | OK |
| "observed exponent is 4.5 times the best certificate" (320) | 0.9663/0.2156 = 4.48 (certificat au pas du livre, Delta optimise) ; 0.9663/0.2192 = 4.41 (meilleur certificat) | ecart (N1-45) |
| "still converges at gamma R^2 = 4, eight times the largest step" (320-321) | pente 0.020-0.036/passe sur passes 2-40 (transitoire) puis **stagnation** : D mediane 22 (4.5 % de D_0 = 486) sur passes 100-2000 ; converge a gamma R^2 = 2 (D = 1e-15 a 60 passes) ; pas admissibles gamma R^2 < 1/2 | **FAUX** (N1-STEP4) |
| borne inferieure (kappa-1)/4 log(1/(2 eps)) (321-324) | vraie pour theta_0 = (r, r) (note B3, Prop. B3-lower) ; fausse sans condition : theta_0 = (0,1), gamma = 1/R^2 donne theta_1 = 0 | hypothese manquante (N1-LOWER, hors perimetre strict) |

### (C5)

| Affirmation (lignes) | Recalcul (reproduction 5 graines) | Verdict |
|---|---|---|
| "from 1.00 to 1.11 (alpha = 1)" (328) | 1.0003 -> 1.1137 (par graine I : 0.98-1.02 ; P : 0.94-1.20 ; 10 graines : 1.000 -> 1.098, 9/10) | OK |
| "from 0.61 to 1.26 (alpha = 2)" (329) | 0.6104 -> 1.2569 | OK |
| "certified bound ranks it below the identity" (329) | exposant certifie par passe 0.1875 (I) contre 0.0369 (alpha=1) et 0.0515 (alpha=2) ; kappa_P = 5083, 3639 > 1000 | OK |
| "I2 from 0.13 to 0.33, I4 from 0.42 to 0.017" (330-331) | alpha=1 : r_coco 0.1289 -> 0.3288, r_sc 0.4233 -> 0.01692 (alpha=2 : 0.019 -> 0.19 ; 0.79 -> 0.0094) | OK (valeurs de alpha = 1, non precise) |

### Protocole (l. 222-226) et figure 1 (l. 283-286)

- n = 1000, d = 50, kappa dans {10, 1e2, 1e3, 1e4, 1e5}, spectres j^(-alpha), alpha dans {0,1,2}, 5 graines, fenetre
  passes 2-20 : conforme a experiments/exp_B3_tightness.py:20-25, 133, 151-152. Precision manquante : les 5 graines sont
  des graines d'echantillonnage sur un seul tirage de donnees par alpha (N1-SEEDS).
- Ordre "I5, I4, I2, I3, I1" (l. 224) : conforme a vrlab/analysis.py (cumulative_chain, attribution, ordre B).
- Identite (4) du papier : verifiee numeriquement sur les etats instrumentes (ecart relatif <= 4e-16).
- Figure 1(a) : borne de la Prop. 5.9 (constante 1 + 3n/16), borne a posteriori avec mu_eff, L_eff le long de la graine 0
  (depart V_0), moyenne de D sur 5 graines : valeurs recalculees a 0/20/50/100 passes 3.68e5/2.53e5/1.44e5/5.65e4 ;
  2.08e3/8.2/7.4e-3/6.8e-8 ; 1.95e3/3.28/1.78e-3/7.11e-9 (digest : 7.11e-09), conformes a l'image. La borne a posteriori
  reste au-dessus de la moyenne (rapport 1.06 a 9.5).
- Figure 1(b) : moyennes sur 5 graines du speed-up contre kappa, trois alpha : conforme au JSON.
- Figure 1(c) : alpha = 1, moyenne et min-max sur 5 graines des cinq parts : conforme au JSON et a la reproduction.
- Limitations (l. 667-668) : la reserve "the shares of (C2) depend on the order" est exacte et necessaire ; son ampleur
  n'est pas donnee (N1-ORDER).

## 5. Constats

### N1-STEP4 (major) - l. 320-321
Citation : "SAGA still converges at $\gamma R^2=4$, eight times the largest step the proof admits."
Probleme : a gamma R^2 = 4 SAGA ne converge pas sur ce probleme (kappa = 1e3, alpha = 1, n = 1000). Le chiffre source
(GammaDelta_summary.json : exposant 0.027 par passe) est une pente de log D ajustee sur les passes 2 a 40 de runs de
40 passes (experiments/exp_gamma_delta.py:23, 38-44), qui ne capte que la fin du transitoire initial.
Preuve : num-f1_gamma_delta.py et num-f1_gamma_delta_long.py. Memes graines 0-2 : pentes passes 2-40 = 0.0204,
0.0357, 0.0260 (identiques au pickle), puis pente passes 100-400 = -0.0008, -0.0003, +0.0002 par passe ;
D aux passes 100/200/400/1000/2000 (graine 0) = 19.6/11.0/26.9/18.2/32.2, mediane 22 (min 5.3, max 142) pour
D_0 = 486. A gamma R^2 = 2 : D = 4.2e-15, 5.3e-16, 1.3e-15 a 60 passes (converge). Le journal results/findings.md:230
disait d'ailleurs "remains stable", pas "converges".
Correction : `and SAGA still converges at $\gamma R^2=2$, four times the largest step the proof admits (at
$\gamma R^2=4$ it stalls).`

### N1-LOWER (major, en bordure de perimetre : enonce prouve, pas un chiffre) - l. 321-324
Citation : "On $n$ identical components ... SAGA with any constant step needs at least
$\frac{\kappa-1}4\log\frac1{2\varepsilon}$ iterations to bring $\E\norm{\theta_t}^2$ below $\varepsilon\norm{\theta_0}^2$".
Probleme : faux sans condition sur theta_0. La Prop. B3-lower de theory/B3_prop59_tightness.tex (l. 406-412) le prouve
pour theta_0 = (r, r), r != 0 (et eps < 1/2) : la borne est ((kappa-1)/(kappa+1))^{2t} min{theta_{0,1}^2, theta_{0,2}^2}.
Preuve : num-f1_repr_and_lower.py : theta_0 = (0, 1), gamma = 1/R^2, z_i^(0) = f_i'(theta_0) : theta_1 = 0 exactement.
Correction : inserer `from $\theta_0=(r,r)$` apres "SAGA with any constant step".

### N1-45 (minor) - l. 320
Citation : "the observed exponent is $4.5$ times the best certificate".
Probleme : 4.5 est le rapport au certificat au pas du livre avec Delta optimise (0.9663/0.2156 = 4.48) ; le "best
certificate" de la phrase (pas et poids optimaux, celui qui "gagne 17 %") vaut 0.2192 par passe et le rapport est 4.41.
Coherence interne : 5.155/1.169 = 4.41.
Correction : `$4.4$ times the best certificate`.

### N1-ORDER (minor) - l. 292-294, 667-668 (et 95-96, 150-151)
Probleme : la dependance a l'ordre est signalee mais pas chiffree, alors qu'elle renverse une des deux phrases de (C2) :
dans l'ordre A (calcule par les auteurs, results/B3_summary.json cle shares_orderA) I2 porte exactement 0 % pour
kappa <= 1e2 et I5 56-61 % ; pour kappa >= 1e4, I2+I4 portent 59-96 % (0.59 a kappa = 1e4, alpha = 0) au lieu de
77-96 % ; I3 jusqu'a 6.3 %. Le chiffre du resume est le plus favorable des deux ; (C3) cite lui-meme 59-96 % pour la
meme quantite dans l'autre ordre sans faire le lien.
Correction (Limitations) : `the shares of \finding{C2} depend on the order in which the inequalities are undone (in
the other order tested, I2 and I4 carry $59$--$96\%$ for $\kappa\ge10^4$, and I5, not I2, $56$--$61\%$ for
$\kappa\le10^2$)`.

### N1-FAM (minor) - l. 297-298
Citation : "On eight data sets of the designs of \cite{na2023hessian}, I2 and I4 carry $64$--$98\%$ of the gap".
Probleme : vrai pour kappa >= 1e4 seulement (16 problemes : 0.6400-0.9834) ; sur les cinq kappa la plage est 0.37-0.98
(ordre B). La phrase ne reprend pas le qualificatif.
Correction : `carry $64$--$98\%$ of the gap for $\kappa\ge10^4$`.

### N1-83 (minor) - l. 294-296 (repris l. 311)
Citation : "In \eqref{eq:saga-repr} I1 and I2 together contribute at most a factor $8/3$".
Probleme : vrai pour c_D ; dans c_M de la meme equation, I1 (avec I3) contribue un facteur (1 - 2 r_sq r_var/3)/(1/3)
qui va jusqu'a 3 (mesure 1.97 a 3.00). L'enonce n'est exact que pour le terme de parametre.
Correction : `In the coefficient $c_D$ of \eqref{eq:saga-repr} I1 and I2 together contribute at most a factor $8/3$`.

### N1-WINDOW (minor) - l. 290-291 (et 95, 150)
Probleme : les facteurs de (C1) dependent de la fenetre, ce que ni (C1) ni les Limitations ne disent. A kappa = 1e5 les
runs sont loin d'avoir converge (pour alpha = 2, E D_t = 878 apres 100 passes, soit 8 % de D_0 = 1.09e4) et mu_eff
decroit le long du run ; sur les passes 20-100 (B3_summary.json, speedup_late_20_100) les facteurs sont
11.3 / 71.6 / 186.6 au lieu de 18.9 / 100.3 / 220.4, soit "11-187" au lieu de "19-220". Pour kappa <= 1e2 la fenetre ne
change rien (3.2-3.7), et la part I2+I4 pour kappa >= 1e4 reste du meme ordre (83-97 % sur passes 20-100).
Correction (Limitations, apres la reserve sur l'ordre) : `and the factors of \finding{C1} on the window ($11$--$187$
at $\kappa=10^5$ over passes 20--100)`.

### N1-SEEDS (minor) - l. 225-226, 285-286
Citation : "five seeds, passes 2--20" ; "(passes 2--20, five seeds)".
Probleme : (i) les cinq graines sont des graines d'echantillonnage sur un seul tirage de donnees par alpha
(experiments/exp_B3_tightness.py:36) ; (ii) les plages de (C1) sont des plages, sur les configurations, de moyennes
sur 5 graines : par execution, 2.90-4.73 pour kappa <= 1e2 (cite 3.2-3.7), 18.9-223.9 a kappa = 1e5 ; sur deux autres
tirages du meme generateur, 14.0-185 a kappa = 1e5 (B3_families_summary.json, famille C).
Correction : l. 225-226 `five sampling seeds on one data set per $\alpha$, passes 2--20` ; l. 285 `(passes 2--20,
mean of five seeds)`.

### N1-C5ALPHA (nit) - l. 330-331
"tightness of I2 from $0.13$ to $0.33$, of I4 from $0.42$ to $0.017$" : valeurs de alpha = 1 (alpha = 2 : 0.019 -> 0.19
et 0.79 -> 0.0094). Correction : ajouter `, $\alpha=1$` dans la parenthese.

## 6. Non verifie

- Le script paper/scripts/make_figR1_classical.py n'a pas ete execute tel quel (results/B3.pkl absent, reconstruction
  documentee dans paper/README.md:27) ; son contenu (courbes de (a), points de (b), (c)) a ete recalcule par mes scripts.
- Famille L (etiquettes remises a l'echelle) de B3_families : non relancee ; 3 problemes P/R reproduits, le reste lu
  dans le JSON. La valeur 98 % (L, forte coherence, kappa = 1e5 : 0.9834) n'est donc verifiee que dans le JSON.
- Enonces de theorie des lignes 290-331 hors chiffres : "global strong-convexity constant ... exactly lambda",
  "every P has kappa_P >= kappa", attributions (hofmann2015variance, arjevani2016lower, na2023hessian,
  dragomir2021fast), et le paragraphe polaire (l. 309-315, dont "about a factor 4" de shulgin2026quadratic).
- L-Katyusha : non relance (instrumentation en O(n^2 d)) ; recalcul depuis A3.pkl seulement.
