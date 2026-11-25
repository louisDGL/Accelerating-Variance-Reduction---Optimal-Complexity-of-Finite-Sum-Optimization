# Audit math-order : ordre d'echantillonnage et terme memoire de SAGA (note B4)

Statut : TERMINE (2026-11-20). Scripts : `paper/audit/checks/math-order_exact_rates.py`, `paper/audit/checks/math-order_typical_runs.py` (36 s et 5 s de CPU, un seul processus, BLAS mono-fil).

## Bilan

Les enonces mathematiques que le papier reprend de la note B4 sont exacts : les deux lois limites ($\min\{2c,1\}$ sous tirage independant, $\min\{2,\log\frac{2-g}g\}$ sous permutation fraiche) sont prouvees dans la note, les preuves ont ete refaites, et les taux exacts recalcules independamment a n = 100 et 1000 les confirment. Le chiffre 1.6 est dans les donnees brutes et se reproduit. Aucun constat bloquant. Les problemes sont des problemes de restitution : 1 constat majeur (la conclusion enonce pour 'SAGA' un doublement du taux memoire qui n'est etabli que pour l'esperance, sur le modele lineaire, et que les sources du depot disent explicitement ne pas valoir 2 pour des trajectoires) et 7 constats mineurs (reserves manquantes : esperance, modele lineaire, regime c fixe contre g fixe, pas auquel vaut 1.6, quantite bornee par le facteur memoire, controle sur le modele a b = 20 et 50, incoherence de statut dans la note citee).

## Perimetre couvert

Enonces du papier paper/paper_saga_to_muon.tex sur l'ordre d'echantillonnage et le terme memoire de SAGA repris de theory/B4_sampling_order.tex : (M3) l. 381-384, (M7) l. 595-606, Table 2 ligne Memory l. 637-640, conclusion l. 679-681. Papier et note B4 lus en entier ; preuves des propositions B4 utilisees par le papier refaites a la main (reduction scalaire, recurrence des moments i.i.d., limites a c fixe et a g fixe, matrice de passe sous permutation, loi min{2, log((2-g)/g)}) ; matrices exactes reconstruites independamment et validees par enumeration exhaustive ; taux exacts a n = 100, 1000 (1e4, 1e5) compares aux lois ; chiffre 1.6 retrouve dans results/B4b_summary.json et reproduit ; coherence avec muon/results/M9b_summary.json (recette sur le modele) et muon/notes_theory.md (F3).

## Preuves refaites (note B4)

| Enonce de la note | Controle | Verdict |
|---|---|---|
| Lemme B4-reduction (SAGA sur hessienne commune = recurrence scalaire par mode, les $b_i$ disparaissent) | refait : $\omega_t=H(e_{t-1}-\psi_{i(t)}+\bar\psi)$ | OK |
| Prop. B4-iid (recurrence exacte de $(A,B,C,P)$) | les 4 lignes refaites a la main ; enumeration de toutes les suites d'indices (n=2..5), ecart 1.4e-15 | OK |
| Prop. B4-iid-limit (a) $\lim r^{iid}_n(c)=\min\{2c,1\}$ | $G$ et $G'$ recalculees coefficient par coefficient ; triangularite dans l'ordre $(A,B,P,C)$ ; excitation du mode dominant par la memoire uniforme verifiee numeriquement | OK |
| Prop. B4-iid-limit (b) $r_n=1-\frac1n(\frac{c^2}{2c-1}-\frac12)$ | $\nu'(0)=wG'v/wv=c^2/(2c-1)$ refait ; a n=1000 la formule donne 0.9995, 0.9992, 0.9982, 0.9962, 0.9922 contre l'exact 0.9995, 0.9992, 0.9982, 0.9962, 0.9922 (c=1..16) | OK |
| Prop. B4-iid-fixed $\frac{2(1-g)}{2-g}$ a $g$ fixe | $T^{(0)}$, $T^{(1)}$, vecteurs propres et matrice reduite refaits ; exact n=1e5 : 0.9744, 0.9474, 0.8889, 0.8571, 0.8095, 0.6667, 0.1818, -0.2222 | OK |
| Prop. B4-rr (premiers moments sur une passe permutee) et B4-fluid | recurrence refaite ; limites de $M_n(g)$ refaites ; matrice fluide contre RK4 (6e-14) | OK |
| Prop. B4-rr2 (seconds moments exacts sur une passe) | $R_n(g)$ reconstruite par une autre voie (application lineaire d'une passe + covariance sans remise) ; enumeration de toutes les permutations (n=3..7), ecart 2.2e-15 ; Monte Carlo n=100 | OK |
| Prop. B4-rr-ms (i) limite a $c$ fixe | borne des poids $u_n\le\frac{2c}ne^{c/2}$ refaite ; exact n=1000 contre loi fluide : ecart <= 1.1e-3 | OK |
| Prop. B4-rr-ms (ii) $\min\{2,\log\frac{2-g}g\}$ a $g$ fixe | recurrence de $\tau_j$, $\alpha=O(1/n)$, $w_j=g_nr_-^j+O(n^{-2})$, $(\Sigma_k)_{11}\to\varphi(1-(1-g)^{2k})$, matrice limite, polynome $x^2(x-e^{-2})(x-\varphi)$ : refaits ; vecteurs propres a gauche : la memoire uniforme excite les deux modes ; exact n=1e5 a 1e-4 de la loi | OK (prouve) |
| Section typical (c) : trajectoire au moins aussi rapide que l'esperance | Markov + Borel-Cantelli : correct | OK |

## Taux exacts recalcules (nats par passe, erreur quadratique)

| i.i.d., c fixe | 0.1 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| n=100 | 0.2001 | 0.5005 | 0.9574 | 0.9950 | 0.9915 | 0.9815 | 0.9602 | 0.9145 | 0.8099 |
| n=1000 | 0.2000 | 0.5000 | 0.9850 | 0.9995 | 0.9992 | 0.9982 | 0.9962 | 0.9922 | 0.9840 |
| n=1e5 | 0.2000 | 0.5000 | 0.9984 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9998 |
| loi min{2c,1} | 0.2 | 0.5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

| i.i.d., g fixe | 0.05 | 0.1 | 0.2 | 0.25 | 0.32 | 0.5 | 0.9 | 1.1 |
|---|---|---|---|---|---|---|---|---|
| n=100 | 0.9763 | 0.9491 | 0.8901 | 0.8580 | 0.8099 | 0.6659 | 0.1806 | -0.2192 |
| n=1000 | 0.9746 | 0.9476 | 0.8890 | 0.8572 | 0.8096 | 0.6666 | 0.1817 | -0.2219 |
| loi 2(1-g)/(2-g) | 0.9744 | 0.9474 | 0.8889 | 0.8571 | 0.8095 | 0.6667 | 0.1818 | -0.2222 |

| permutation, c fixe | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| n=100 | 0.4455 | 0.7801 | 1.1937 | 1.5464 | 1.7629 | 1.8727 | 1.9329 | 1.6030 |
| n=1000 | 0.4446 | 0.7775 | 1.1880 | 1.5380 | 1.7534 | 1.8636 | 1.9240 | 1.9596 |
| loi fluide -2 log nu(c) | 0.4445 | 0.7772 | 1.1874 | 1.5370 | 1.7523 | 1.8625 | 1.9230 | 1.9586 |

| permutation, g fixe | 0.05 | 0.1 | 0.2 | 0.2384 | 0.25 | 0.32 | 0.5 | 0.9 | 1.1 |
|---|---|---|---|---|---|---|---|---|---|
| n=100 | 1.8069 | 1.8956 | 1.9465 | 1.8971 | 1.8524 | 1.6030 | 1.0796 | 0.1987 | -0.1986 |
| n=1000 | 1.9738 | 1.9870 | 1.9939 | 1.9879 | 1.9350 | 1.6521 | 1.0966 | 0.2005 | -0.2005 |
| n=1e5 (vrlab) | 1.9997 | 1.9999 | 1.9999 | 1.9999 | 1.9458 | 1.6582 | 1.0986 | 0.2007 | -0.2007 |
| loi min{2, log((2-g)/g)} | 2 | 2 | 2 | 2 | 1.9459 | 1.6582 | 1.0986 | 0.2007 | -0.2007 |

- Plus grand exposant exact sous permutation sur une grille de pas : 1.949 (n=100, g=0.21), 1.995 (n=1000, g=0.23). Au meme pas l'esperance i.i.d. perd 0.88 et 0.87 nat.
- Sup sur le pas de l'exposant exact i.i.d. : 0.99503 (n=100), 0.99950 (n=1000), sous la borne $-n\log(1-1/n)$ = 1.00503, 1.00050.
- Accord avec `vrlab.order_model` (iid_exponent, rr_ms_exponent, rr_exponent) : 4.7e-14.

Trajectoires i.i.d. (n=1000), exposant median :

| c | 0.25 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 125 | 250 (g=1/4) |
|---|---|---|---|---|---|---|---|---|---|---|
| results/B4b_summary.json (12 graines) | - | 1.614 | 1.635 | 1.556 | 1.484 | 1.395 | 1.352 | 1.235 | 1.128 | 0.956 |
| reproduction (8 trajectoires) | 0.500 | 1.616 | 1.639 | 1.559 | 1.516 | - | 1.342 | - | 1.111 | 0.958 |
| esperance exacte | 0.5000 | 0.9995 | 0.9992 | 0.9982 | 0.9962 | 0.9922 | 0.9840 | 0.9672 | 0.9335 | 0.8572 |
| permutation, exact | 0.4446 | 1.1880 | 1.5380 | 1.7534 | 1.8636 | 1.9240 | 1.9596 | 1.9795 | 1.9898 | 1.9350 |

## Affirmations controlees

| Affirmation (papier) | Lignes | Verdict | Preuve du controle |
|---|---|---|---|
| as $n\to\infty$ its expected squared error loses $\min\{2c,1\}$ nats per pass under independent sampling | l. 600-601 | OK | Prop. B4-iid + B4-iid-limit refaites a la main (4 lignes de la recurrence, matrices G et G', triangularite, valeurs propres -2c,-1,-2,-(c+1)). Matrice T_n(g) validee par enumeration de toutes les suites d'indices (n=2..5, ecart 1.4e-15). Exposant exact -n log rho(T_n(c/n)) : n=1000 : 0.2000, 0.5000, 0.9850, 0.9995, 0.9992, 0.9982, 0.9962, 0.9922, 0.9840 pour c=0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32 ; n=1e5 : 0.2, 0.5, 0.9984, 1.0000... ; loi min{2c,1}. La memoire uniforme excite le mode dominant (exposant de E e^2 entre passes 150 et 300 identique). Script math-order_exact_rates.py sections 1, 3. |
| up to two under a fresh permutation, $\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$ (proved) | l. 602-603 | OK | Preuve de Prop. B4-rr-ms(ii) (theory/B4_sampling_order.tex l. 313-336) refaite : recurrence des poids w_j, racines r_+-, alpha=O(1/n), variance d'un tirage sans remise, a_n,b_n -> g/(2-g), c_n,p_n -> 0, matrice limite de polynome caracteristique x^2(x-e^-2)(x-phi) ; le mode dominant est excite par la memoire uniforme (vecteurs propres a gauche calcules). Matrice de passe R_n(g) reconstruite sans la recurrence de covariance de la note, validee par enumeration de toutes les permutations (n=3..7, ecart 2.2e-15) et Monte Carlo (n=100, z entre -1.1 et -0.4). Exposants exacts : n=1000 : 1.9738, 1.9870, 1.9939, 1.9350, 1.6521, 1.0966, 0.2005, -0.2005 pour g=0.05, 0.1, 0.2, 0.25, 0.32, 0.5, 0.9, 1.1 ; n=1e5 : 1.9997, 1.9999, 1.9999, 1.9458, 1.6582, 1.0986, 0.2007, -0.2007 ; loi : 2, 2, 2, 1.9459, 1.6582, 1.0986, 0.2007, -0.2007. Maximum sur le pas : 1.949 (n=100), 1.995 (n=1000), pres de g=0.21-0.23 : 'up to two' exact. Script sections 1, 2, 6, 7. |
| fixed-c limit under a permutation (used implicitly by 'up to two'): $r^{rr}(c)=-2\log\nu(c)\to2$ | l. 602 ; note B4 Prop. B4-fluid, B4-rr-ms(i) | OK | Matrice fluide re-derivee et comparee a une integration RK4 de l'EDO (ecart 6e-14) ; exposants exacts n=1000 : 0.4446, 0.7775, 1.1880, 1.5380, 1.7534, 1.8636, 1.9240, 1.9596 (c=0.25..32) contre la loi 0.4445, 0.7772, 1.1874, 1.5370, 1.7523, 1.8625, 1.9230, 1.9586 ; c(2-r) = 1.398, 1.429, 1.433 a c=100, 500, 1000 (limite annoncee 2(e-2)=1.4366). Script section 5. |
| the memory factor $(1-b/n)^{n/b}\approx e^{-1}$ of the table | l. 597-598 | OK | Calcul : n=1000 : 0.3670 (b=5), 0.3642 (b=20), 0.3585 (b=50), e^-1=0.3679 ; identiques a memory_factor de muon/results/M9b_summary.json (0.36696, 0.36417, 0.35849). La borne E[M_{t+1}\|passe] >= (1-b/n) M_t (F3, muon/notes_theory.md l. 104-112) est exacte pour un lot uniforme de b indices distincts, quelle que soit la regle de pas. |
| $e^{-1}$ per pass in expectation, since stale entries keep their error (ceiling under independent sampling) ; 'At most one nat per pass under independent sampling' | l. 382-383, l. 638 | OK | Mecanisme exact : E_i[M_t] >= (1-1/n) M_{t-1} (theory/B3_prop59_tightness.tex l. 420, eq. B3-EM ; F3) ; pour l'erreur quadratique elle-meme, modele a hessienne commune : le sup sur le pas de l'exposant exact de E e^2 vaut 0.99503 (n=100) et 0.99950 (n=1000), sous la borne -n log(1-1/n) (script section 9). Reserve sur la quantite bornee : voir constat MO-6. |
| single runs under independent sampling lose about $1.6$ (measured, $n=1000$) : le chiffre existe et se reproduit | l. 603 | OK | results/B4b_summary.json typical.n=1000 : medianes 1.614 (c=1), 1.635 (c=2), 1.556 (c=4) sur 12 graines ; results/B4_summary.json : 1.611 (c=1), 1.579 (c=2). Reproduction independante (8 trajectoires, 260 passes, ajustement 60-260) : 1.616, 1.639, 1.559 ; meme exposant pour e^2 seul que pour la taille de l'etat. Le chiffre ne vaut que pour 1 <= c <= 4 : constat MO-3. Script math-order_typical_runs.py. |
| the memory factor bounds the mean [...], not a single run (inegalite 'trajectoire typique >= esperance' et ecart mesure) | l. 604-605 | OK | Note B4 sec. typical (c) : Markov + Borel-Cantelli, correct. Reproduction : n=1000, c=2, apres 20 passes E e^2 exact = 3.72e-12 alors que les 8 trajectoires simulees ont e^2 de mediane 1.1e-16 et de maximum 3.1e-16 (la note : 3.7e-12 contre mediane 7.3e-17). Formulation 'mean error' : constat MO-5. |
| On such a sum the recipe follows $1.2$--$1.3$ nats of $\mathrm{lr}^2$ per pass with independent indices and $1.7$--$1.9$ with a permutation (lecture du JSON seulement) | l. 603-604 | OK | muon/results/M9b_summary.json L.cells.n=<n>.orders.<order>.level_end.x_star : iid 1.2, 1.3, 1.3 ; reshuffle 1.9, 1.8, 1.7 pour n=100, 1e3, 1e4 (lot un, 80 passes). Non relance. Avec le critere strict x_star_clean : 1.2, 1.2, 1.2 et 1.5, 1.7, 1.4. |
| $36$--$49\%$ fewer passes where memory limits ($b=20$, $50$) ; passes $16.7$--$20.0$ to $9.1$--$12.2$ ; $\rho_\star$ 0.27-0.43 puis 0.10-0.18 ; instances/model 1.7-2.7 et 1.0-1.3 | l. 596-599, 605-606, 640 | OK | muon/results/M9b_summary.json D_vs_M9 : soft-mem 17.9->9.1 (b=20), 16.7->10.4 (b=50) ; ls-mem 20.0->11.3, 19.1->12.2 : economies 49.2, 37.7, 43.5, 36.1 % ; rho_star iid 0.368/0.32/0.27 et 0.43/0.368/0.32 (b=5/20/50), permutation b=20,50 : 0.10, 0.135, 0.18, 0.18 ; rapports instances/modele : iid 1.73-2.73 (b=1..50), permutation b=20,50 : 0.97, 1.14, 1.20, 1.34. Non relance. |
| threshold $\kappa<4.08$ (arithmetic only) | l. 608-609 | OK | ((kappa-1)/(kappa+1))^2 < e^-1 ssi kappa < (sqrt(e)+1)/(sqrt(e)-1) = 4.0830 (script section 9). La source citee n'a pas ete ouverte. |
| a fresh permutation per pass, which lifts the memory rate of SAGA from $e^{-1}$ to nearly $e^{-2}$, pays where memory limits | l. 680-681 (conclusion) | probleme (major, MO-1) | voir le constat |
| At most one nat per pass under independent sampling \finding{M3}, two under a permutation (note B4). | l. 637-638 (Table 2, ligne Memory) | probleme (minor, MO-2) | voir le constat |
| single runs under independent sampling lose about $1.6$ (measured, $n=1000$) | l. 603 | probleme (minor, MO-3) | voir le constat |
| ($n$ quadratics with a common Hessian, $c=\gamma n\lambda$): as $n\to\infty$ its expected squared error loses $\min\{2c,1\}$ nats per pass under independent sampling, the ceiling of \finding{M3}, and up to two under a fresh permutation, $\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$ (proved) | l. 599-603 | probleme (minor, MO-4) | voir le constat |
| the memory factor bounds the mean error, not a single run | l. 604-605 | probleme (minor, MO-5) | voir le constat |
| near the ceiling of SAGA under independent sampling, $e^{-1}$ per pass in expectation, since stale entries keep their error | l. 381-383 | probleme (minor, MO-6) | voir le constat |
| Linear SAGA explains part of it [...] On such a sum the recipe follows $1.2$--$1.3$ nats of $\mathrm{lr}^2$ per pass with independent indices and $1.7$--$1.9$ with a permutation [...] The instances need $1.7$ to $2.7$ times the passes of this model with independent batches but $1.0$ to $1.3$ times with a permutation at $b=20$ and $50$ | l. 599-600 et 603-606 | probleme (minor, MO-7) | voir le constat |
| $\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$ (proved) | l. 602-603 (statut '(proved)') ; source theory/B4_sampling_order.tex l. 378 et l. 525 | probleme (minor, MO-8) | voir le constat |

## Constats

### MO-1 (major, confiance high) : l. 680-681 (conclusion)

Phrase du papier : "a fresh permutation per pass, which lifts the memory rate of SAGA from $e^{-1}$ to nearly $e^{-2}$, pays where memory limits"

Probleme. Enonce general sur SAGA, sans reserve, alors que ce qui est prouve est (i) restreint a SAGA lineaire sur n quadratiques de meme hessienne, (ii) une limite n -> infini, (iii) un taux de l'ESPERANCE de l'erreur quadratique, (iv) atteint seulement pour un pas g = gamma*lambda proche de 0.24 sur le mode le plus raide. Pour ce qu'une trajectoire fait reellement, le passage de e^-1 a e^-2 est faux : les trajectoires i.i.d. perdent deja jusqu'a 1.6 nat (le papier le dit l. 603) et la note B4 conclut explicitement 'a factor 1.2 to 1.5 in the exponent [...], not 2'. De plus la relative est employee comme explication du gain de la recette ('pays where memory limits'), alors que sur le modele a hessienne commune la recette ne gagne rien a la permutation aux lots b=20 et 50 ou le gain est rapporte.

Preuve. theory/B4_sampling_order.tex l. 521 ('its statements are exact for identical Hessians only'), l. 531 ('the gain of reshuffling over what an i.i.d. run actually does is a factor 1.2 to 1.5 in the exponent on logistic regression with kappa <= n, not 2'), l. 440 ('against typical i.i.d. runs the gain of reshuffling is the smaller one'). Valeurs recalculees (math-order_exact_rates.py, math-order_typical_runs.py), n=1000 : esperance i.i.d. 0.9995 (c=1) ; trajectoires i.i.d. 1.61-1.64 (c=1, 2), 0.96 a g=1/4 ; permutation au mieux 1.995 (g=0.23) : meilleur contre meilleur 1.995/1.635 = 1.22. muon/results/M9b_summary.json, D_vs_M9 : sur le modele, rho_star = 0.22 pour les deux ordres a b=20 et 50, passes 8.7 -> 9.4 et 7.0 -> 9.1 avec la permutation ; muon/findings.md l. 613 ('No statement of the B4 note covers the fixed-length iteration, which is not linear') et l. 835-838 ('not a property of the table alone').

Correction proposee : `with minibatches the spectral map pays, and a fresh permutation per pass pays where memory limits (for linear SAGA it lifts the expected memory rate from $e^{-1}$ to nearly $e^{-2}$).`

### MO-2 (minor, confiance high) : l. 637-638 (Table 2, ligne Memory)

Phrase du papier : "At most one nat per pass under independent sampling \finding{M3}, two under a permutation (note B4)."

Probleme. Manquent 'in expectation' et la classe de problemes. Sous echantillonnage independant le plafond ne vaut que pour l'esperance (une trajectoire perd jusqu'a 1.6 nat, l. 603) ; sous permutation 'at most two' n'est etabli que sur le modele lineaire a hessienne commune (limite prouvee a g fixe ; r^rr(c) < 2 a c fixe et a n fini est numerique). Lue seule, la cellule est contredite par le papier lui-meme (1.6 > 1).

Preuve. results/B4b_summary.json typical.n=1000 : 1.61, 1.63, 1.56 > 1 ; note B4 l. 341 ('on a grid of 4001 values of c [...] smaller than min{2c,2}' : numerique) et l. 521 ; maxima exacts recalcules 1.949 (n=100), 1.995 (n=1000).

Correction proposee : `In expectation, at most one nat per pass under independent sampling \finding{M3}, two under a permutation (linear model, note B4).`

### MO-3 (minor, confiance high) : l. 603

Phrase du papier : "single runs under independent sampling lose about $1.6$ (measured, $n=1000$)"

Probleme. 1.6 est le MAXIMUM sur le pas, atteint pour 1 <= c <= 4 seulement ; l'exposant typique decroit ensuite avec le pas et tombe a 0.96 au pas du livre (g = 1/4), sous un nat. La phrase, sans pas, se lit comme une propriete generale des trajectoires. Elle juxtapose en outre trois meilleurs cas pris a des pas differents (1 nat a c fixe, 2 nats a g proche de 0.24, 1.6 nat a c entre 1 et 4) : au pas ou les trajectoires i.i.d. perdent 1.6 (c=1), la permutation perd 1.19 ; au pas ou la permutation perd 1.94 (g=1/4), les trajectoires i.i.d. perdent 0.96.

Preuve. results/B4b_summary.json typical.n=1000.<c>.median : 1.614, 1.635, 1.556, 1.484, 1.395, 1.352, 1.235, 1.128, 0.956 pour c = 1, 2, 4, 8, 16, 32, 64, 125, 250 ; reproduction independante (math-order_typical_runs.py) : 1.616, 1.639, 1.559, 1.516 (c=8), 1.342 (c=32), 1.111 (c=125), 0.958 (c=250) ; exposants exacts sous permutation au meme c : 1.188 (c=1), 1.935 (c=250).

Correction proposee : `single runs under independent sampling lose up to about $1.6$ (measured, $n=1000$, $1\le c\le4$; $0.96$ at $g=\frac14$)`

### MO-4 (minor, confiance high) : l. 599-603

Phrase du papier : "($n$ quadratics with a common Hessian, $c=\gamma n\lambda$): as $n\to\infty$ its expected squared error loses $\min\{2c,1\}$ nats per pass under independent sampling, the ceiling of \finding{M3}, and up to two under a fresh permutation, $\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$ (proved)"

Probleme. Deux limites differentes dans une meme phrase sans le dire : min{2c,1} est la limite a c fixe (pas par iteration g = c/n -> 0), la loi de la permutation est donnee a g fixe (c = ng -> infini). Au pas fixe g ou la permutation atteint 2 nats, l'esperance i.i.d. ne perd pas min{2c,1} = 1 mais 2(1-g)/(2-g) < 1 (Prop. B4-iid-fixed : 0.857 a g = 1/4), et a c fixe la permutation perd -2 log nu(c), inferieur au taux i.i.d. pour c < 0.73 (0.44 contre 0.5 a c = 1/4). Par ailleurs lambda designe partout ailleurs dans le papier le parametre ridge (l. 201-203) : ici c'est une valeur propre de la hessienne commune (loi par mode), ce qui n'est pas dit.

Preuve. theory/B4_sampling_order.tex l. 63 ('Two scalings'), l. 111-123 (Prop. B4-iid-fixed), l. 377 ('Reshuffling is not uniformly better'). Valeurs exactes recalculees, n=1000 : i.i.d. a g fixe 0.9746, 0.8890, 0.8572 (g=0.05, 0.2, 0.25) contre permutation 1.9738, 1.9939, 1.9350 ; a c fixe c=0.25 : i.i.d. 0.5000, permutation 0.4446 ; c=0.5 : 0.9850 contre 0.7775.

Correction proposee : `($n$ quadratics with a common Hessian, $\lambda$ one of its eigenvalues, $c=\gamma n\lambda$): as $n\to\infty$ at fixed $c$ its expected squared error loses $\min\{2c,1\}$ nats per pass under independent sampling, the ceiling of \finding{M3}, and up to two under a fresh permutation, $\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$, where independent sampling loses $\frac{2(1-g)}{2-g}<1$ (proved)`

### MO-5 (minor, confiance high) : l. 604-605

Phrase du papier : "the memory factor bounds the mean error, not a single run"

Probleme. 'the mean error' est impropre. Ce qui est borne par le facteur memoire est l'esperance de l'erreur QUADRATIQUE DE LA TABLE (F3, toute regle de pas) et, pour SAGA lineaire a hessienne commune seulement, l'esperance de l'erreur quadratique de l'iterat. L'erreur moyenne E[theta_t - eta_*] de SAGA sous tirage independant suit la descente de gradient sans aucun plafond memoire (le papier l'ecrit en (C4), l. 323 : 'its mean iterate being that of gradient descent'). Et pour la recette a longueur fixe, a laquelle la phrase est accolee, aucune borne sur l'erreur de l'iterat n'est prouvee.

Preuve. muon/notes_theory.md l. 104-118 (F3 : 'the *mean* memory error') et l. 136-137 ('That the iterate error and the memory error decay at the same rate is an *observation* on the runs of the study, not a theorem') ; muon/findings.md l. 613 ; dans le modele E e_t = (1-g)^t e_0 (estimateur sans biais), alors que E e_t^2 plafonne a un nat.

Correction proposee : `the memory factor bounds the mean squared error of the table, not a single run`

### MO-6 (minor, confiance medium) : l. 381-383

Phrase du papier : "near the ceiling of SAGA under independent sampling, $e^{-1}$ per pass in expectation, since stale entries keep their error"

Probleme. La quantite plafonnee n'est pas dite, et la phrase porte sur des passes jusqu'a 1e-9 de distance quadratique, pour 'the best family of every class' (dont des pas polaires). Le plafond est prouve en general pour l'erreur moyenne de la table (F3) et pour E[V_t] (remarque de la note B3) ; pour la distance quadratique il n'est prouve que pour SAGA lineaire a hessienne commune, n -> infini (note B4). Les medianes citees (16-21 passes pour 9 ln 10 = 20.7 nats, soit 0.99 a 1.30 nat par passe) sont d'ailleurs au niveau ou au-dessus d'un nat : coherent avec un plafond qui ne lie que l'esperance, mais 'near the ceiling' gagne a dire de quoi.

Preuve. theory/B3_prop59_tightness.tex l. 420 ('E_i[M_t] >= (1-1/n) M_{t-1}, hence E[V_t] >= Delta (1-1/n)^t M_0') ; theory/B4_sampling_order.tex l. 108 ('what (R1) adds is that the error itself [...] inherits this rate', dans le modele) ; muon/notes_theory.md l. 136-137.

Correction proposee : `near the ceiling of a SAGA table under independent sampling, whose mean squared error loses at most $e^{-1}$ per pass since stale entries keep their error.`

### MO-7 (minor, confiance high) : l. 599-600 et 603-606

Phrase du papier : "Linear SAGA explains part of it [...] On such a sum the recipe follows $1.2$--$1.3$ nats of $\mathrm{lr}^2$ per pass with independent indices and $1.7$--$1.9$ with a permutation [...] The instances need $1.7$ to $2.7$ times the passes of this model with independent batches but $1.0$ to $1.3$ times with a permutation at $b=20$ and $50$"

Probleme. Reserve manquante : les chiffres 1.2-1.3 et 1.7-1.9 sont mesures a lot un (non dit, le contexte est b = 5 a 50), et sur ce meme modele la recette ne gagne RIEN a la permutation aux lots b = 20 et 50 ou le papier rapporte le gain des instances (rho_star = 0.22 pour les deux ordres ; passes 8.7 -> 9.4 et 7.0 -> 9.1). Aux lots concernes, le modele lineaire n'explique donc pas le gain : celui-ci vient de ce que les instances perdent plus que le modele sous tirage independant. Le lecteur ne peut le deduire que de la figure 4(c).

Preuve. muon/results/M9b_summary.json, D_vs_M9 : modele, iid / reshuffle : rho_star 0.32/0.18 (b=1), 0.32/0.18 (b=5), 0.22/0.22 (b=20), 0.22/0.22 (b=50) ; passes 10.5/7.1, 9.4/9.1, 8.7/9.4, 7.0/9.1. L.cells : longues trajectoires a lot un. muon/findings.md l. 829-838.

Correction proposee : `On such a sum the recipe at batch one follows $1.2$--$1.3$ nats of $\mathrm{lr}^2$ per pass with independent indices and $1.7$--$1.9$ with a permutation, but at $b=20$ and $50$ its $\rho_\star$ is $0.22$ under both orders: the memory factor bounds the mean squared error of the table, not a single run.`

### MO-8 (minor, confiance high) : l. 602-603 (statut '(proved)') ; source theory/B4_sampling_order.tex l. 378 et l. 525

Phrase du papier : "$\min\{2,\log\frac{2-g}g\}$ at a fixed step $g=\gamma\lambda$ (proved)"

Probleme. Le papier a raison (la preuve de Prop. B4-rr-ms(ii) est complete et a ete refaite), mais la note citee se contredit sur ce statut : deux phrases anterieures a la preuve disent encore que la loi a g fixe sous permutation n'est pas prouvee. Un jury qui ouvre la note B4 a la rubrique 'Limits' y lit le contraire du papier.

Preuve. theory/B4_sampling_order.tex l. 525 : '(iii) For the reshuffled order the fixed-$g$ limit is derived and verified on the exact matrix, not proved' ; l. 378 : '(the first proved, the second verified on the exact matrix)' ; contre l. 25 (tableau de statut : 'proved'), l. 313 ('Fixed $g\in(0,2)$ (proved)'), l. 339, et results/findings.md l. 450-452 ('proved on 2026-11-19 after this section was first written').

Correction proposee : aucune dans le papier ; corriger la note (supprimer ou mettre a jour les deux phrases perimees de theory/B4_sampling_order.tex l. 378 et l. 525).

## Non verifie

- Chiffres '16-18 (softmax) and 17-21 passes (least squares) to 1e-9' de (M3) : donnees de l'experience M1 de l'etude Muon, hors perimetre, non retrouves.
- Experiences M9 / M9b (rho_star, passes, 1.2-1.3 et 1.7-1.9 nats) : valeurs relues dans muon/results/M9b_summary.json, experiences non relancees ; M9_summary.json non ouvert directement (les valeurs des instances sont lues dans D_vs_M9 de M9b).
- Phrase l. 607-609 (Finito, ying2017variance, park2020linear, mokhtari2018surpassing) : sources non ouvertes ; seule l'arithmetique kappa < 4.083 est verifiee. Le seuil 4.08 est un calcul des auteurs (note B4 l. 515 : 'a comparison that these papers do not make') et concerne une methode de type Finito (DIAG), pas SAGA : a faire confirmer par l'auditeur des citations.
- Parties de la note B4 que le papier ne reprend pas : ordre fixe (R2 : c_* = 17.70, theoreme de Hurwitz), courbure effective, composantes orthogonales, planchers sur l'instance dure, regression logistique (tableaux de l'Outcome, non relances), heuristique et conjecture sur l'exposant typique.
- Compilation de theory/main.pdf : seule l'inclusion de B4 dans theory/main.tex et les dates des fichiers ont ete controlees, pas le PDF lui-meme.
- Plafond d'un nat pour E||theta_t - eta_*||^2 de SAGA hors du modele a hessienne commune : aucun enonce prouve dans le depot (seulement E[M_t] et E[V_t]) ; non etabli ici non plus.

## Effet de bord a signaler
Le premier lancement de `math-order_exact_rates.py` (03:47) a importe `vrlab.order_model` sans l'option `-B` : Python a regenere trois caches de bytecode perimes dans `vrlab/__pycache__/` (`algorithms`, `order_model`, `solvers`, fichiers `.pyc` ignores par git). Aucun fichier source n'a ete touche et `git status` ne montre que `paper/audit/`. Les deux scripts posent desormais `sys.dont_write_bytecode = True` ; les relances n'ecrivent plus rien hors de `paper/audit/`.
