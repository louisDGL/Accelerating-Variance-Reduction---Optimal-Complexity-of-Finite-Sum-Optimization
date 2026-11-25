# Consigne de la session distante (audit de `paper/paper_saga_to_muon.tex`)

Écrite le 20 novembre 2026 vers 03h45 (heure de la machine, CET) par la session VSCode, pour la session qui la relaie
dans tmux.

## Situation

L'utilisateur a demandé : « Fais un audit complet du paper_saga_to_muon en t'assurant de la pertinence scientifique de
ce qui est avancé. Fais plusieurs agents, il faut avoir fini d'ici 4h max. Je vais me déconnecter de vscode, donc
prépare ta session à distance. »

Tu es cette session distante : une bifurcation (`--fork-session`) de la session VSCode, lancée dans la session tmux
`audit` avec Remote Control (nom `audit-papier-saga-muon`), en mode de permission auto. Elle ne dépend plus de VSCode.
L'utilisateur est absent : ne pose aucune question, décide et note tes choix dans le rapport. Tu as tout le contexte
de la session VSCode jusqu'à la bifurcation (papier lu en entier, README, mémoire). La session VSCode, si elle vit
encore, ne fait que surveiller : tu es seule à produire l'audit.

Échéance : **07h15**. Vise un rapport complet et écrit sur disque à **05h45**, puis améliore-le.

## Ce qu'il faut faire, dans l'ordre

1. **Lancer les auditeurs avec l'outil Agent, pas avec Workflow.** Mise à jour de 03h45 : dans ce terminal, l'outil
   Workflow ouvre un dialogue de confirmation (« Run a dynamic workflow? ») que seul l'utilisateur peut valider ; il
   est absent, et la session VSCode ne valide pas une permission à sa place. N'appelle donc plus Workflow.
   L'utilisateur a demandé plusieurs agents en toutes lettres : lance-les avec l'outil Agent (type
   `general-purpose`, en arrière-plan). Il y a 17 auditeurs ; la consigne complète de chacun est dans
   `paper/audit/process/prompts/<clé>.md`. Le prompt de chaque appel Agent tient en une phrase : « Lis
   /users/eleves-b/2023/louis.darrigol/projetSTATS/paper/audit/process/prompts/<clé>.md et exécute cette consigne en
   entier. » Chaque auditeur écrit `paper/audit/evidence/<clé>.md` et `<clé>.json` au fur et à mesure, ses scripts
   dans `paper/audit/checks/`, et répond par un résumé court.
   - Première vague, tout de suite et dans un seul message (12 appels) : `math-saga`, `math-nfl`, `math-smd`,
     `math-offset`, `math-schedule`, `math-order`, `num-f1`, `num-geometry`, `num-noise`, `num-m5`, `num-m6m7`,
     `consistency`.
   - Deuxième vague, dès que quatre auditeurs de la première ont rendu (5 appels) : `code-fidelity`, `cite-muon`,
     `cite-vr`, `cite-stat`, `novelty`.
   - Contre-vérification : quand un auditeur rend, lis son `<clé>.json` ; s'il contient des constats `blocker` ou
     `major`, lance un agent avec le prompt « Lis
     /users/eleves-b/2023/louis.darrigol/projetSTATS/paper/audit/process/prompts/_verificateur.md et exécute cette
     consigne pour la dimension CLE = <clé>. » Il écrit `paper/audit/evidence/<clé>.verif.json`.
   - Si un appel Agent ouvre lui aussi un dialogue ou est refusé, n'insiste pas : fais l'audit seule, dans l'ordre
     preuves, chiffres, cohérence, citations, et dis-le dans le rapport.
2. **Pendant que les auditeurs tournent**, sans les dupliquer : crée `paper/audit/AUDIT.md` avec un en-tête
   `Statut : en cours` ; relis les vérifications déjà faites ci-dessous ; prépare le plan du rapport.
3. **Quand les auditeurs rendent** : lis les fichiers `paper/audit/evidence/*.json` et, au besoin, les `.md`. Pour
   chaque constat bloquant ou majeur confirmé ou incertain, revérifie toi-même (relis le passage, refais le calcul)
   avant de l'inscrire : un constat faux dans le rapport coûte plus cher qu'un constat manquant. Déduplique entre
   dimensions. Si des auditeurs échouent (limite d'usage du compte, erreur), ne relance pas tout : lis leurs fichiers
   partiels, puis couvre toi-même en priorité les dimensions de preuves et de chiffres. N'attends pas un auditeur
   plus de 50 minutes : passe à la synthèse avec ce qui est écrit.
4. **Écrire le rapport** `paper/audit/AUDIT.md` (français ; citations du papier en anglais), pour un lecteur qui
   revient après quatre heures et doit décider quoi corriger avant de rendre le papier :
   - verdict d'ensemble en quelques lignes, puis les constats à traiter classés par gravité (bloquant, majeur,
     mineur), chacun avec : lignes du `.tex`, phrase du papier, problème, preuve (calcul, fichier:ligne, URL),
     correction proposée (texte LaTeX exact), verdict de la contre-vérification ;
   - tableau des énoncés principaux (Théorèmes 1-2, Propositions 3-5, (C1)-(C5), (M1)-(M7), chiffres du résumé) avec
     leur statut après audit : vérifié (comment), vérifié avec réserve, non vérifié (pourquoi) ;
   - ce qui a été contrôlé sans trouver de problème (bref, avec le type de preuve) ;
   - ce qui n'a pas pu être vérifié et pourquoi ;
   - méthode : auditeurs, périmètres, contre-vérification, limites de l'audit (dont celles de temps et de quota) ;
   - renvois vers `evidence/` et `checks/`.
5. **Corrections proposées, sans toucher au papier** : ne modifie pas `paper/paper_saga_to_muon.tex` ni aucun fichier
   hors de `paper/audit/`. Prépare `paper/audit/corrections.patch` (applicable par `git apply` depuis la racine) qui
   ne contient que les corrections des erreurs confirmées, minimales ; vérifie-le sur une copie (compilation
   pdflatex/bibtex, 12 pages au plus, aucune référence non résolue) et dépose le PDF obtenu dans
   `paper/audit/paper_saga_to_muon_corrige.pdf`. Les changements qui relèvent d'un choix d'auteur (reformuler une
   revendication, retirer un résultat) restent des recommandations dans le rapport, pas dans le patch. Pas de commit,
   pas de push.
6. **Clore** : mets `Statut : terminé le <date heure>` en tête du rapport ; si le temps le permet, publie le rapport
   en Artifact privé (lecture à distance) ; envoie une PushNotification d'une ligne (nombre de constats bloquants et
   majeurs, chemin du rapport) ; termine par un message de synthèse en français.

## Contraintes

- Quota : le compte a déjà atteint la limite d'usage Fable après de gros workflows (nuit du 18 au 19 novembre). Une
  seule passe d'auditeurs, pas de relance globale ; si des agents échouent pour limite d'usage, continue seule et
  priorise le rapport écrit. Écris le rapport sur disque tôt et complète-le, plutôt que de tout garder pour la fin.
- Permissions : ne valide jamais toi-même un dialogue de permission et ne contourne pas un refus ; si une action est
  refusée, fais autrement ou note-la comme non faite.
- Machine partagée (16 cœurs, charge 8 à 10) : BLAS mono-thread, petits calculs seulement.
- Fidélité : distingue dans le rapport ce qui a été revérifié indépendamment, ce qui repose sur les notes du dépôt, et
  ce qui n'a pas été vérifié. Ne présente pas comme vérifié ce qui ne l'a pas été.

## Vérifications déjà faites par la session VSCode (à croiser avec les auditeurs, pas à leur dicter)

Faites à la main en lisant le papier ; elles comptent comme une lecture indépendante des preuves.

- **Théorème 1** : la preuve est correcte. Minoration λ‖Δ‖_F² ≥ (λ/c_N²)N(Δ)² ; atteinte à la limite le long de W_t
  en Δ_0 maximiseur de N/‖·‖_F ; L_N(g) ≥ L_F(g)/c_N² pour g convexe ; L_F(f_i) = c_i‖x_i‖² + λ comme supremum.
  Exemple : logistique, il faut W_1x_i ≠ 0 pour tout i ; softmax, une coordonnée maximale unique de W_1x_i pour tout
  i ; vrai pour presque tout W_1. Le théorème est élémentaire : son contenu est l'observation de modélisation.
  κ_op ≥ min(K,d) est vrai pour toute F (moyenne sur les signes de Σ_k ε_k e_k e_kᵀ, chacun de norme spectrale 1) ;
  reste à vérifier ce que dit exactement la référence citée.
- **Identité (2)** : vérifiée en développant F(W − tO) pour une quadratique ; μ_PL ≤ J vient de F(W − tO) ≥ F_* pour
  tout t, μ_PL ≥ μ_op de l'inégalité de Polyak-Łojasiewicz duale.
- **Théorème 2** : (a) identité des trois points vérifiée, signes compris ; D_{F*}(G_t, G_{t+1}) = D_F(W_{t+1}, W_t)
  ≤ L_op η²/2 ; l'inégalité de sous-gradient ne demande aucun moment (|‖G+ξ‖_* − ‖ξ‖_*| ≤ ‖G‖_*, facteur polaire
  borné) ; coercivité sans moment : ψ(G) ≥ ‖G‖_*(2P(‖ξ‖_* ≤ M) − 1) − 2M dès que P(‖ξ‖_* ≤ M) > 1/2. (b) et (c)
  suivent l'argument standard. Contre-exemple scalaire (P(ξ=1)=0,4) vérifié : ψ minimale en g = 1, |Φ| ≥ 0,2.
- **Proposition 3** : (a), (b), (c), (d) corrects à la lecture ; (b) suppose E‖ξ‖_* fini (implicite dans E ξ = 0) ;
  (d) suppose l'existence d'un minimiseur de Λ et pas de terme ridge.
- **Proposition 4** : pour K = 1, a/c = 3/(d+2) vérifié analytiquement (E[R³] = (d+1)E[R]) ; d = 1 redonne le
  décalage scalaire κ_3/(6σ²b), signe cohérent. 0,464 pour 2×3 équivaut à a/c ≈ 0,568 : à confirmer par Monte-Carlo.
- **Proposition 5** : (a) vérifiée ; (b) vérifiée avec les constantes (il faut ρ^{A/2} ≥ 0,8, vrai car
  μ_op/L_eff ≤ 1/(10A) ; récurrence : φ(0) ≤ Cρη² si L_eff ≥ 2μ_op, φ(Cη²) = Cρη²) ; bornes d'âge correctes ;
  (c) vérifiée.
- **(C4), borne inférieure** : l'itérée moyenne est celle de la descente de gradient, donc
  E‖θ_t‖² ≥ ½((κ−1)/(κ+1))^{2t}‖θ_0‖² **pour θ_0 de poids égaux sur les deux directions propres**, d'où la borne.
  Telle qu'écrite (lignes 321-323), la phrase ne précise pas le point initial : pour θ_0 porté par la direction de
  courbure μ, γ = 1/μ et la table initialisée en θ_0, SAGA atteint le minimiseur en une itération. Constat probable
  (imprécision à corriger par « from θ_0 = (1,1) » ou « for some θ_0 ») : à confirmer par l'auditeur `math-saga`.
- **Tableau 2, ligne « Estimator, first »** (ligne 632) : la conséquence « The limit gradient is minus a nuclear
  median of the noise » renvoie à la Proposition 4 ; l'énoncé est celui du Théorème 2(b), la Proposition 4 n'en donne
  que la taille. Renvoi à vérifier.
