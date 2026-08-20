# Methods


## 0. Objectif & positionnement

Cette étude vise à déterminer si l'apprentissage par renforcement (RL, via GRPO) permet à un petit modèle de langage (1.5B) de développer de véritables capacités déductives sur des tâches de raisonnement logique, ou s'il amplifie principalement la mémorisation superficielle induite par l'apprentissage supervisé (SFT).

### Positionnement et différenciation vs K&K (Xie et al.)
Alors que les travaux antérieurs sur *Knights & Knaves* (K&K, Xie et al.) étudient des énigmes booléennes classiques, cette étude introduit un domaine logique inédit issu du jeu vidéo *Blue Prince* et structure l'analyse autour de deux axes fondamentaux :
1. **Comparaison systématique à 4 conditions** :
   - *Baseline* : Évaluation zero-shot sans entraînement (mesure de la capacité inhérente).
   - *RL seul (Cold-start)* : Optimisation GRPO directe depuis le modèle de base.
   - *SFT seul* : Fine-tuning supervisé sur des énigmes synthétiques canoniques.
   - *SFT + RL (Warm-start)* : Optimisation GRPO initialisée sur l'adaptateur SFT.
2. **Protocole de mémorisation LiMem adapté** : Quantification fine de l'écart entre généralisation logique et récitation textuelle via des mutations sémantiques minimales et des perturbations de surface isomorphes.

### Contraintes expérimentales
- **Échelle du modèle** : Modèle compact de $\sim 1.5\text{B}$ paramètres (`Qwen2.5-1.5B-Instruct`), adapté à l'étude de l'émergence du raisonnement sous faible capacité.
- **Ressources matérielles** : Entraînement et inférence exécutés sur GPU grand public personnel (NVIDIA RTX 3060 Ti, 8 Go VRAM).
- **Temporalité** : Projet de recherche mené sur un horizon compact d'environ un mois.

## 1. Domaine & règle du puzzle

### Description du jeu de base et mise en situation

Dans le jeu vidéo *Blue Prince*, le joueur explore un manoir aux pièces générées procéduralement et découvre régulièrement des salles d'énigmes logiques. Sur une table sont posées trois boîtes de couleurs distinctes : une boîte **Bleue** (à gauche), une boîte **Blanche** (au milieu) et une boîte **Noire** (à droite). 

L'objectif du joueur est d'ouvrir la boîte qui renferme les **gemmes** (la récompense). Une seule des trois boîtes contient les gemmes, les deux autres étant vides. Sur chaque boîte sont gravées une ou plusieurs affirmations textuelles. Le jeu impose une méta-règle universelle immuable :
> *« Au moins une boîte dit l'entière vérité (toutes ses affirmations sont vraies), et au moins une boîte ment complètement (toutes ses affirmations sont fausses). »*

#### Exemple 1 : Énigme à une affirmation par boîte (Puzzle canonique du jeu #11)
* **Boîte BLEUE (Gauche)** : `[B1]` *"THIS BOX IS BLUE"*
* **Boîte BLANCHE (Milieu)** : `[W1]` *"THE BLUE BOX IS TELLING THE TRUTH"*
* **Boîte NOIRE (Droite)** : `[K1]` *"THE BLUE BOX DOES NOT CONTAIN THE GEMS"*

*Résolution déductive :*
1. La boîte Bleue est physiquement bleue, donc `B1` est obligatoirement **Vraie** ($B_1 = \text{T}$).
2. La boîte Blanche affirme que la boîte Bleue dit la vérité ; comme $B_1 = \text{T}$, `W1` est également **Vraie** ($W_1 = \text{T}$).
3. L'invariant du jeu exige qu'au moins une boîte mente intégralement. Comme les boîtes Bleue et Blanche sont vraies, la boîte Noire est nécessairement **Fausse** ($K_1 = \text{F}$).
4. L'affirmation de la boîte Noire (*"La boîte bleue ne contient pas les gemmes"*) étant fausse, sa négation est vraie : **les gemmes se trouvent dans la boîte BLEUE**.

#### Exemple 2 : Énigme relationnelle avec comptage (Puzzle du jeu #12)
* **Boîte BLEUE (Gauche)** : `[B1]` *"EXACTLY TWO STATEMENTS ARE TRUE"*
* **Boîte BLANCHE (Milieu)** : `[W1]` *"THE BLUE BOX IS TELLING THE TRUTH"*
* **Boîte NOIRE (Droite)** : `[K1]` *"THE GEMS ARE IN A BOX THAT TELLS THE TRUTH"*

*Résolution déductive :*
1. Si `B1` était fausse, la boîte Bleue mentirait, ce qui rendrait `W1` fausse également. Pour satisfaire l'invariant ($\ge 1$ boîte toute-vraie), la boîte Noire devrait être vraie ($K_1 = \text{T}$). Le nombre total d'affirmations vraies serait alors de 1 (différent de 2), ce qui confirmerait $B_1 = \text{F}$. Mais si $K_1 = \text{T}$, les gemmes devraient être dans une boîte vraie (Noire), ce qui est consistant.
2. Si `B1` était vraie ($B_1 = \text{T}$), alors `W1` serait vraie ($W_1 = \text{T}$). Le total d'affirmations vraies étant de 2 (d'après $B_1$), `K1` devrait être fausse ($K_1 = \text{F}$). L'invariant serait respecté (Bleue et Blanche vraies, Noire fausse). D'après $K_1 = \text{F}$, les gemmes ne sont **pas** dans une boîte vraie, donc elles sont dans la boîte fausse : **les gemmes se trouvent dans la boîte NOIRE**.

### Règles fondamentales et invariant Blue Prince
- **Unicité de la gemme** : Exactement une boîte contient les gemmes ($g^* \in \mathcal{B}$ unique), les deux autres étant vides.
- **Garde de vacuité** : Chaque boîte porte un ensemble non vide d'affirmations $S(B)$ avec $|S(B)| \ge 1$, garantissant qu'aucune boîte n'échappe à l'évaluation de vérité par absence de déclaration.
- **Invariant Blue Prince** : Tout monde valide doit satisfaire simultanément :
  $$\exists B \in \mathcal{B}, \; \forall s \in S(B), \; \tau(s) = \text{True} \quad (\ge 1 \text{ boîte toute-vraie})$$
  $$\exists B \in \mathcal{B}, \; \forall s \in S(B), \; \tau(s) = \text{False} \quad (\ge 1 \text{ boîte toute-fausse})$$
  La troisième boîte peut être toute-vraie, toute-fausse ou mixte (si $|S(B)| \ge 2$).

### Différences structurelles avec Knights & Knaves (K&K)
- **Topologie fixée ($N=3$)** : Dans K&K, le nombre d'acteurs varie arbitrairement. Dans Blue Prince, le nombre de boîtes est strictement fixé à 3 ; la complexité s'exprime par le nombre total d'affirmations $N = \sum |S(B)|$, leur répartition et leur interdépendance.
- **Richesse référentielle et méta-textuelle** : Contrairement aux identités binaires locales de K&K (Chevalier/Valet), les affirmations combinent des faits physiques (`ContainsGems`, `IsEmpty`), des relations spatiales relatives (`AboveStatement`, `BelowStatement`), des prédicats d'agrégation (`CountTrueStatements`) et des assertions méta-logiques de boîte (`BoxIsTrue`), imposant une résolution par point fixe rigoureuse.

## 2. Représentation abstraite

L'ensemble du projet repose sur la construction d'un framework permettant de représenter de façon abstraite les énigmes logiques de ce domaine, de les résoudre avec un Oracle rigoureux, ainsi que de les générer et les muter pour évaluer la mémorisation (LiMem).

Les deux briques élémentaires de cette modélisation sont les structures de données `World` et `Puzzle` 
- La classe `World` représente un **monde candidat** spécifique (une instance d'état caractérisée par l'emplacement supposé de la gemme `gem_box` et une affectation de valeurs de vérité `box_truths` pour chaque affirmation). L'ensemble de tous les mondes possibles $\mathcal{W}$ réunit $3 \times 2^N$ configurations.
- La classe `Puzzle` représente l'**énigme elle-même**, c'est-à-dire l'ensemble des affirmations logiques inscrites sur chaque boîte (`box_statements`).

Les affirmations sont construites à partir de prédicats logiques formant la grammaire du problème. Par exemple, l'affirmation *"THE BLACK BOX CONTAINS THE GEMS"* est représentée par le prédicat `ContainsGems("BLACK")`.

On divise les prédicats logiques du système en deux catégories fondamentales :
1. **Les prédicats autocontenus (`SELF_CONTAINED`)** : Prédicats dont l'évaluation ne dépend que des propriétés physiques du monde candidat (ex: l'emplacement de la gemme `world.gem_box` ou les attributs géométriques d'une boîte). Ils ne lisent **pas** le dictionnaire de vérité `world.box_truths`. Exemple : `ContainsGems("BLACK")`, `BoxIsColor("WHITE", "WHITE")`, `IsEmpty("THIS")`.
2. **Les prédicats référentiels (`RELATIONAL`)** : Prédicats dont l'évaluation dépend explicitement de la lecture des valeurs de vérité d'autres affirmations dans `world.box_truths`. Exemple : `BoxIsTrue("BLUE")`, `CountTrueStatements("==", 2)`, `AboveStatement(...)`.

### Sémantique de référence et point fixe
Au niveau sémantique, une énigme est résolue sous une **sémantique de point fixe** : un monde candidat $w = (g, \tau) \in \mathcal{W}$ est consistant si et seulement si pour toute affirmation, la valeur de vérité calculée par son évaluation correspond exactement à la valeur de vérité postulée dans le monde candidat ($\text{eval}(s_{B,i}, w) == \tau(s_{B,i})$).

Chaque affirmation est notée $S(B,i)$, où $B \in \mathcal{B}$ désigne la boîte d'appartenance et $i \in \{1, \dots, |S(B)|\}$ le rang (indexation relative) de l'affirmation sur cette boîte. L'accès par rang est indispensable pour évaluer les prédicats autoréférentiels relatifs `AboveStatement` ($i-1$) et `BelowStatement` ($i+1$).

## 3. Solveur / Oracle

Le solveur prend en entrée un monde et un puzzle. Son principal objectif est de confronter les valeurs de vérité déclarées par le monde candidat avec les valeurs de vérité calculées par les prédicats logiques du puzzle en parcourant itérativement l'ensemble des mondes possibles ($3 \times 2^N$).
Il s'assure également que la sémantique du point fixe et que l'invariant Blue Prince sont respectés.
Cela permet de classer un puzzle dans l'une des trois catégories suivantes : 
- `PARADOX` ($|V| = 0$)
- `AMBIGUOUS` ($|V| \ge 2$)
- `SOLVABLE` ($|V| = 1$, condition `GOLD_STRICT_OK`)

Le solveur sert à la fois d'Oracle de vérité terrain pour l'évaluation et de filtre de qualité par échantillonnage avec rejet (*rejection sampling*) pour le générateur.

## 4. Générateur

Le générateur de problèmes prend en arguments des paramètres permettant de moduler la difficulté ainsi qu'un nombre d'essais :
- Nombre d'affirmations total $N$
- Ratio d'affirmations référentielles (`relational_ratio`)
- Profondeur de couche logique AST (`max_depth`), plafonnée à 2 afin de conserver une lisibilité grammaticale optimale en anglais naturel et d'éviter les ambiguïtés de parenthésage.

Le générateur produit des puzzles par force brute et confronte chacun d'eux à l'Oracle/Solveur, ne retenant que les problèmes possédant une solution unique et non ambigus.

On définit le taux d'acceptation comme le rapport entre le nombre de problèmes validés par l'Oracle et le nombre total de problèmes générés. Alors que l'intuition laisserait supposer un effondrement exponentiel du rendement avec $N$, le taux d'acceptation reste remarquablement stable entre 15 % et 25 %.

Un resultat surprenant a ete de constater d'abord par l'experience puis ensuite mathematiquement que ce n'est pas le cas.

### Démonstration mathématique : Télescopage et Loi de Poisson

La stabilité du rendement du générateur s'explique par un résultat d'analyse combinatoire en deux étapes : le télescopage exact du point fixe, suivi du filtrage par l'invariant.

#### 1. Télescopage de l'espérance de point fixe ($\mathbb{E}_{\text{fixpoint}} = 3$)

Soit $N$ le nombre total d'affirmations réparties sur les 3 boîtes.

- **Taille de l'espace d'états** :
  L'ensemble de tous les mondes candidats possibles $\Omega$ possède une cardinalité de :
  $$|\Omega| = 3 \times 2^N$$
  (3 emplacements possibles pour la gemme $\times$ $2^N$ configurations de vérité).

- **Probabilité de point fixe pour un monde candidat** :
  Sous une hypothèse de non-trivialité où chaque affirmation $s_i$ a une probabilité approchée de $1/2$ d'évaluer à sa valeur de vérité postulée $\tau_i$ :
  $$P(w \text{ est un point fixe}) = \prod_{i=1}^N P(\text{eval}(s_i, w) = \tau_i) \approx \left(\frac{1}{2}\right)^N = 2^{-N}$$

- **Télescopage invariant** :
  Par linéarité de l'espérance sur les $|\Omega|$ mondes candidats :
  $$\mathbb{E}[|V_{\text{fixpoint}}|] = \sum_{w \in \Omega} P(w \text{ point fixe}) = |\Omega| \times 2^{-N} = (3 \times 2^N) \times 2^{-N} = 3$$
  **Ce résultat est strictement invariant en $N$** : les facteurs $2^N$ et $2^{-N}$ se télescopent exactement. Quelle que soit la taille $N$, un système de contraintes booléennes aléatoires sur 3 boîtes admet en moyenne **exactement 3 mondes points fixes**.

#### 2. Filtrage par l'invariant Blue Prince et dépendance $q(N)$

L'invariant Blue Prince impose qu'au moins une boîte soit entièrement vraie et au moins une boîte entièrement fausse. 

Ce filtre intervient avec une probabilité $q(N) = P(\text{invariant satisfait} \mid \text{point fixe})$ qui dépend modérément de la taille des boîtes $k_B = |S(B)|$ ($\sum_B k_B = N$) :
- Pour $N=3$ ($k_B = 1$ affirmation par boîte) :
  $$q(3) = 1 - P(\text{3 boîtes toutes vraies ou 3 toutes fausses}) = 1 - \frac{2}{2^3} = 1 - \frac{2}{8} = 0.75$$
- À mesure que $N$ grandit et que les boîtes grossissent ($k_B \ge 2$), la condition d'avoir une boîte entièrement vraie ($P = 2^{-k_B}$) devient plus sélective, et $q(N)$ décroît graduellement vers $\sim 0.40 - 0.55$.

L'espérance finale du nombre de mondes valides est donc :
$$\mathbb{E}[|V|] = 3 \times q(N) \in [1.2, 2.25]$$

#### 3. Approximation par la Loi de Poisson et Analyse du Gap Théorie vs Réel

En modélisant le nombre de mondes valides $|V|$ par une loi de Poisson $\mathcal{P}(\lambda)$ de paramètre $\lambda = \mathbb{E}[|V|] = 3 \cdot q(N)$ :
$$P(|V| = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Pour un ordre de grandeur moyen $\lambda \approx 1.8$, la probabilité théorique d'obtenir une **solution unique stricte** ($|V| = 1$, condition `GOLD_STRICT_OK`) s'établit à :
$$P(|V| = 1) = \lambda e^{-\lambda} \approx 1.8 \times e^{-1.8} \approx 29.7\%$$

**Analyse du gap avec le rendement empirique (15 % – 25 %)** :
L'écart mesuré en pratique par rapport aux ~30 % théoriques découle de deux facteurs structurants :
1. *Corrélations référentielles* : Les prédicats relationnels (`BoxIsTrue`, `AboveStatement`, `CountTrueStatements`) brisent l'indépendance stricte des affirmations entre elles, augmentant la dispersion de la distribution et le taux de paradoxes ($|V|=0$).
2. *Contraintes mutuellement exclusives* : Les prédicats contradictoires sur la localisation de la gemme (`ContainsGems` vs `IsEmpty`) éliminent des sous-espaces entiers de mondes candidats d'un seul coup.

Néanmoins, le mécanisme fondamental de télescopage garantit que le taux de puzzles `GOLD_STRICT_OK` ne s'effondre pas exponentiellement avec $N$, mais reste ancré de façon prévisible et stable autour de **15 % à 25 %** pour tout $N \ge 3$.

Le generateur a ete complete par une gestion de l'aleatoire avec un système de seed, ainsi qu'un système de deduplication pour eviter les doublons de puzzle (utile pour eviter toute fuite dans les futurs datasets d'entrainement et de test)

## 5. Perturbateur (LiMem)

Afin de reproduire une expérience analogue au papier K&K (Xie et al.), un perturbateur de puzzle a été implémenté de telle sorte à générer des mutations d'un même puzzle. Cela permet d'évaluer le niveau de mémorisation du LLM à différentes étapes du projet (SFT seul vs SFT+RL). Le protocole repose sur deux axes d'évaluation complémentaires :

### A. Les deux axes d'évaluation LiMem
1. **Sur le Train synthétique (Instances vues)** : C'est le cœur de la mesure LiMem (Xie et al.). Le modèle ayant été entraîné sur ces exemples, l'écart entre la version originale et la version perturbée ($\text{Acc}_{\text{canonique}} - \text{Acc}_{\text{isomorphe}}$) mesure directement le **taux de mémorisation / récitation** installé par le SFT, et permet de vérifier si le RL (GRPO) amplifie ou réduit cette mémorisation.
2. **Sur les 57 Puzzles Réels (Instances inédites)** : Comme ces énigmes sont strictement exclues du train (anti-fuite), l'écart mesure la **robustesse lexicale de surface** et la **généralisation OOD**.

### B. Typologie des perturbations
Le puzzle est perturbé selon deux modalités :
- **Perturbations de surface (Isomorphes)** : On conserve rigoureusement la même logique et la même solution Oracle, mais on modifie le vocabulaire textuel (ex: `BOX` $\rightarrow$ `CHEST`, `GEMS` $\rightarrow$ `COIN` avec frontières de mots regex).
- **Mutations sémantiques (Mutants minimaux)** : On modifie un unique prédicat/argument dans une affirmation. L'Oracle vérifie que le nouveau puzzle admet toujours une unique solution stricte ($|V|=1$) et que la gemme a changé de boîte ($G_{\text{mut}} \neq G_{\text{orig}}$). Si un puzzle est intrinsèquement non mutable de façon minimale (22 cas parmi les 57 réels), il est proprement ignoré (`skipped`) pour ne pas biaiser la mesure par un puzzle synthétique substitué.

### C. Métriques Option B calculées
Pour éviter le plancher du hasard sur la gemme seule (~33 %), les écarts LiMem sont calculés sur les métriques fines :
- **$\text{LiMem}_{\text{bit}}$** : $\text{BitAcc}_{\text{canonique}} - \text{BitAcc}_{\text{isomorphe}}$
- **$\text{LiMem}_{\text{strict}}$** : $\text{StrictAcc}_{\text{canonique}} - \text{StrictAcc}_{\text{isomorphe}}$
- **Taux de fuite mémorielle (`Memory-Leakage`)** : Fraction des tirages sur les mutants minimaux où le modèle continue de prédire l'ancienne gemme canonique $G_{\text{orig}}$ au lieu de la nouvelle solution $G_{\text{mut}}$.

## 6. Harnais d'évaluation

Les modèles sont évalués en prompting zero-shot (sans démonstrateur pour éviter le biais d'imitation et la saturation de contexte).

### A. Format de réponse et analyseur tolérant
On impose au modèle de formater sa conclusion sur une ligne finale standardisée :
```text
FINAL: gem=<BLUE|WHITE|BLACK>; B1=<T|F>; B2=<T|F>; W1=<T|F>; ...
```
Un parseur tolérant aux variations mineures (espaces, casse, synonymes lexicaux) extrait la prédiction de la gemme et le vecteur booléen attribué à chaque affirmation.

### B. Système de notation Option B
Pour discriminer les bonnes réponses fortuites du raisonnement déductif réel, la notation repose sur le standard **Option B** (exigeant à la fois la boîte exacte et l'intégralité des valeurs de vérité) :
- **Taux de parsing (`parse_ok`)** : Fraction des réponses contenant une ligne `FINAL:` syntaxiquement valide.
- **Précision Gemme (`gem_correct`)** : Comparée au plancher du hasard pur ($33.3\%$).
- **Précision par bit (`bit_accuracy`)** : Fraction moyenne des affirmations correctement classées (hasard à $50.0\%$).
- **Exactitude stricte (`strict_correct`, Option B)** : Résolution simultanée de la boîte et de $100\%$ des bits de vérité (hasard à $\frac{1}{3 \times 2^N} \approx 2\%$).
- **Exploitabilité ($\text{pass}@G > 0$)** : Nombre de puzzles résolus au moins une fois sur $N=8$ tirages.

## 7. Entraînement

Le protocole d'entraînement est structuré selon une séquence en deux temps : **Supervised Fine-Tuning (SFT)** puis **Reinforcement Learning (GRPO)**.

### A. Supervised Fine-Tuning (Direct-FT)
Le SFT Direct apprend au modèle à associer directement l'énoncé du puzzle à la solution complète (gemme + vérités).

1. **Direct-FT v1 (Ablation Underfitting)** : Entraînement sur 2 époques avec LoRA $r=16$ ciblant uniquement l'attention (`q_proj, v_proj`). Le modèle a appris le format à 100 % mais est resté au niveau du hasard logique ($50.6\%$ bit-acc).
2. **Direct-FT v2 (Entraînement Capacitaire)** : Entraînement sur 8 époques avec LoRA $r=32$ ($\alpha=64$) sur l'ensemble des couches linéaires (`q,k,v,o,gate,up,down_proj`) via `trl` et `completion_only_loss=True`. Cette configuration a débloqué le signal logique avec $65.5\%$ de bit-acc en held-out synthétique et $12.5\%$ de résolution stricte sur les 57 réels.

Chaque puzzle généré ainsi que les puzzles du jeu de base sont stockés dans un set `seen_signatures` pour éviter toute fuite de données :
$$\text{Signatures}(\text{Train}) \cap \text{Signatures}(\text{Val}) \cap \text{Signatures}(\text{Test}) \cap \text{Signatures}(\text{Jeu}) = \emptyset$$

### B. Reinforcement Learning (GRPO)
L'étape de RL utilise l'algorithme GRPO (*Group Relative Policy Optimization*) via `trl.GRPOTrainer` en warm-start depuis l'adaptateur SFT v2 fusionné :
- **Configuration** : $G=8$ générations par prompt, $\beta_{\text{KL}}=0.04$, $LR=10^{-5}$, `per_device_batch_size=2`, `gradient_accumulation_steps=4`, 300 étapes d'optimisation.
- **Fonction de récompense graduée ($r \in [0, 2.8]$)** :
  $$r = \begin{cases} 0.0 & \text{si non parsable} \\ 1.0 \times \mathbb{I}(\text{gemme juste}) + 0.8 \times \text{frac\_bits} + 1.0 \times \mathbb{I}(\text{Option B strict}) & \text{sinon} \end{cases}$$

## 8. Journal des décisions structurantes

| Décision | Date | Alternative écartée | Justification principale |
| :--- | :--- | :--- | :--- |
| **Périmètre 57 GOLD** | 2026-08-10 | Inclure les 110 puzzles du jeu | Exclusion des monstres méta-textuels non modélisables formellement ($|V| \neq 1$). |
| **Notation Option B** | 2026-08-10 | Gem-only (boîte seule) | Élimination du plancher du hasard à 33 % ; discriminateur fin du raisonnement. |
| **Modèle Qwen2.5-1.5B-Instruct** | 2026-08-11 | DeepSeek-R1-Distill / Qwen-Math | R1 instable/dérive hors format ; Qwen-Math dégénère ; Qwen-Instruct respecte le format à 80 %+. |
| **Pivot SFT $\rightarrow$ RL** | 2026-08-11 | RL pur (Cold start) | Baseline zero-shot à 6.1 % strict : pas de rollouts positifs au départ sans amorçage supervisé. |
| **Direct-FT v2 capacitaire** | 2026-08-15 | SFT v1 (2 époques, attention-only) | SFT v1 sous-apprenait ($50.6\%$ bit-acc) ; v2 étend à 8 époques et all-linear ($65.5\%$ held-out). |
| **LiMem à double axe** | 2026-08-17 | LiMem uniquement sur les réels | Les réels mesurent la robustesse OOD ; le train synthétique mesure la vraie mémorisation SFT. |
