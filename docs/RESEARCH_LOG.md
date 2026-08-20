## Entrées

### 2026-08-10 - Calibration modèle


**Contexte** : A ce stade du projet j'avais lu les principaux papiers de recherche (Deepseek R1, k&k, small LLM) et pris des notes dessus. Une problematique en est ressortie et le cadrage initial etait fait. L'une des premières etapes pratiques est le choix du modèle de reference qu'il faudra finetune. Ce chois est important car il constitue le poont de depart et la base sur laquelle tout le reste allait s'appuyer.
A ce stade, la taille du modèle etait dejà choisit: 1.5B, car la litterature scientifique avait demontre la possibilite d'utiliser les methodes de RL sur de tels modèles. Il me restait à choisir quel modèle precis parmi deux candidats pouvait faire l'affaire entre : qwen 2.5 1.5B instruct et deepseek r1 distill qwen.

- **Fait** : Pour pouvoir les  departager, la première intuition est d'evaluer chacun des modèles sur les puzzles du jeu de base. L'implementation du système d'evaluation a un objectif double : permettre de discriminer les modèles et d'etablir une echelle de difficulte des problèmes en se basant sur le taux de reussite de chacun des puzzles.

Les modèles sont evalues en zero shot prompting, au detriment su one-shot. Cette decision permet:
- Éviter le biais d'imitation du démonstrateur : Le One-Shot fournit un exemple complet de raisonnement (CoT). Sur un petit modèle 1.5B, le modèle a tendance à copier la couleur ou la décision du démonstrateur plutôt qu'à déduire.
- Mesure pure de la capacité inhérente : Le Zero-Shot évalue si la capacité à respecter les règles de Blue Prince et la logique sont inhérentes aux poids du modèle.
- Contrainte de VRAM / Fenêtre de contexte : Évite d'alourdir le prompt d'entrée avec des traces de preuve sur une RTX 3060 Ti 8 Go.

- **Résultat** : Qwen2.5-1.5B-Instruct suit le format à ~80 % en zero-shot, tandis que DeepSeek-R1-Distill boucle et Qwen-Math dégénère.
- **Décision + pourquoi** : Sélectionner Qwen2.5-1.5B-Instruct comme modèle de base pour toute la suite expérimentale.
- **Ouvert** : Évaluer l'impact d'un SFT sur la capacité de suivi de format et de raisonnement.

### 2026-08-10 - Périmètre de la grammaire

**Contexte** : Avant de lancer l'évaluation des LLMs, il était nécessaire de faire un tri dans les puzzles du jeu de base (110 au total). La plupart sont constitués d'énoncés méta-textuels ("YOU WILL NOT GET THE GEMS") impossibles à résoudre sans contexte externe.

- **Fait** : Analyse et formalisation manuelle des 110 puzzles pour déterminer s'ils sont formalisables de façon déterministe.
- **Résultat** : 66 puzzles sont dans le scope grammatical défini, et 57 parmi eux sont qualifiés de "GOLD" ($|V| = 1$).
- **Décision + pourquoi** : Restreindre le benchmark de référence aux 57 puzzles GOLD pour garantir l'unicité stricte de la solution.
- **Ouvert** : Les puzzles hors-scope serviront de jeu de test OOD extrême pour mesurer le transfert.

### 2026-08-10 - Décision sur la notation

**Contexte** : Dans le papier K&K, le hasard pur est à 50 %. Dans Blue Prince, le hasard sur la gemme seule est de 33.3 %, ce qui reste trop élevé pour prouver un raisonnement logique réel sur de petits échantillons.

- **Fait** : Exiger dans le prompt la boîte finale ainsi que la valeur de vérité de chaque affirmation individuelle.
- **Résultat** : La probabilité de réussir au hasard tombe à $\frac{1}{3 \times 2^N}$ (ex: 2.08 % pour $N=4$).
- **Décision + pourquoi** : Adopter la notation stricte Option B (Gemme + toutes les vérités) comme métrique principale pour éliminer le plancher du hasard.
- **Ouvert** : Mesurer la précision par bit pour diagnostiquer l'apprentissage partiel.

### 2026-08-10 — Résultat négatif central

- **Contexte** : Une fois le harnais d'évaluation et la notation Option B en place, j'ai lancé l'évaluation de référence (zero-shot) de mon modèle candidat `qwen2.5-1.5b-instruct` sur l'ensemble des 57 puzzles GOLD du jeu, avec environ 8 essais par puzzle (456 générations au total).
- **Fait** : J'ai mesuré les performances du modèle pour voir s'il montrait déjà un début de capacité de raisonnement avant tout entraînement.
- **Résultat** : Les résultats montrent qu'il n'y a **strictement aucun signal de raisonnement en zero-shot** (le triptyque du hasard) :
  - Le modèle trouve la bonne boîte seulement dans 27.6 % des cas (ce qui est même inférieur aux 33.3 % du hasard pur).
  - La précision sur les valeurs de vérité des affirmations est de 49.7 % (exactement le résultat d'un pile ou face à 50 %).
  - Le taux de réussite strict (bonne boîte + toutes les bonnes vérités en même temps) plafonne à 6.1 %, et sur 41 des 57 puzzles, le modèle a fait 0 % de réussite sur 8 essais.
- **Décision + pourquoi** : Ce résultat confirme que les petits modèles 1.5B ne savent pas du tout résoudre ces énigmes au départ. Il est donc impossible de compter sur une capacité native du modèle pour démarrer le projet.
- **Ouvert** : Sans aucun signal initial, le modèle ne pourra pas s'améliorer tout seul en RL pur sans une première phase d'apprentissage guidé (SFT).

### 2026-08-11 — Élimination des modèles

- **Contexte** : Il fallait faire un choix définitif de modèle 1.5B pour la suite des expériences parmi les candidats identifiés : Qwen2.5-1.5B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B et Qwen2.5-Math-1.5B.
- **Fait** : J'ai testé chaque modèle sur le harnais d'évaluation pour observer leur comportement, leur respect du format de réponse imposé et leur stabilité sur ma configuration locale (RTX 3060 Ti 8 Go).
- **Résultat** :
  - *DeepSeek-R1-Distill* produit des traces de réflexion (`<think>`) interminables. Le modèle part souvent dans des boucles de répétition, oublie le format `FINAL:` imposé en utilisant des balises LaTeX `\boxed{}`, et sature rapidement la mémoire de ma carte graphique.
  - *Qwen-Math* n'a pas réussi à suivre des instructions en langage naturel simple et part en dégénérescence textuelle.
  - *Qwen2.5-1.5B-Instruct* est le seul à être stable, avec un taux de respect du format parsable de 79.6 %.
- **Décision + pourquoi** : J'ai éliminé R1-Distill et Qwen-Math, et j'ai retenu **Qwen2.5-1.5B-Instruct** comme modèle de référence pour tout le reste du projet.
- **Ouvert** : Je garde en tête l'idée de tester éventuellement la version non-instruct (Qwen2.5-1.5B de base) si besoin lors de l'étape de SFT.

### 2026-08-11 — Pivot méthodologique

- **Contexte** : Au départ, l'idée était de tester du renforcement pur (GRPO / RLVR) directement sur le modèle de base. Mais l'algorithme GRPO a besoin que le modèle trouve au moins de temps en temps la bonne solution pour recevoir une récompense positive et mettre à jour ses poids.
- **Fait** : Face au résultat négatif du zero-shot (6.1 % de réussite globale et 0 % sur la majorité des puzzles), le modèle ne produira presque jamais de rollouts positifs au départ (problème du "cold start").
- **Résultat** : Lancer du RL seul sur une politique totalement aveugle risque de faire tourner l'entraînement dans le vide sans jamais converger.
- **Décision + pourquoi** : J'ai décidé de faire pivoter l'approche vers un schéma **SFT $\rightarrow$ RL** : utiliser d'abord du SFT (Supervised Fine-Tuning) pour donner au modèle les bases du raisonnement et du format, puis utiliser le RL (GRPO) pour affiner la politique. Cela permet de comparer proprement 4 conditions : Baseline, RL seul, SFT seul, et SFT+RL.
- **Ouvert** : Déterminer si je ferai du SFT direct ou du SFT avec traces de pensée (CoT). La question reste ouverte sur la manière de générer ces traces CoT : soit de manière déterministe via l'Oracle, soit en les faisant générer par un modèle plus gros (SOTA) pour un style plus naturel.

### 2026-08-11 — Analyse de la difficulté

- **Contexte** : Pour pouvoir générer de futurs puzzles synthétiques de façon contrôlée, il fallait comprendre précisément ce qui rend un puzzle facile ou difficile pour le modèle.
- **Fait** : J'ai analysé les résultats puzzle par puzzle en croisant le taux de réussite du modèle avec les caractéristiques logiques de chaque énoncé (nombre de phrases, type de prédicats).
- **Résultat** : Le levier principal de difficulté n'est pas seulement le nombre de phrases $N$, mais surtout la présence d'**affirmations référentielles** (celles qui lisent la vérité d'autres boîtes). Dès qu'il y a des dépendances croisées entre les boîtes, le modèle décroche beaucoup plus vite.
- **Décision + pourquoi** : J'ai retenu trois paramètres clés pour contrôler la difficulté dans le futur générateur : le nombre total de statements $N$, la proportion de prédicats référentiels (`relational_ratio`), et le taux d'opérateurs logiques (`ast_prob`).
- **Ouvert** : Définir des seuils clairs pour classer les futurs puzzles générés en trois niveaux de difficulté (Facile, Moyen, Difficile).

### 2026-08-12 — Module Générateur Synthétique

- **Contexte** : Les 57 puzzles du jeu d'origine ne suffisent pas pour entraîner un modèle. Il me fallait un outil capable de fabriquer automatiquement des milliers de nouveaux puzzles valides avec leur solution garantie par l'Oracle.
- **Fait** : J'ai développé un générateur aléatoire combiné au solveur (`generate_random_puzzle` filtré par `is_world_valid`) qui construit des puzzles par force brute et ne retient que ceux qui ont une solution unique (GOLD).
- **Résultat** : 
  - J'ai plafonné la profondeur des formules logiques (AST) à 2 au maximum pour que les phrases restent en anglais naturel et parfaitement compréhensibles sans parenthèses.
  - J'ai constaté (puis démontré mathématiquement) que le taux d'acceptation du générateur reste stable autour de 15 à 25 % quel que soit le nombre de phrases, grâce au télescopage de l'espace d'états.
  - J'ai ajouté un système de signature pour dédupliquer les puzzles générés et éviter qu'un même problème ne se retrouve à la fois dans le jeu d'entraînement et de test.
- **Décision + pourquoi** : Le générateur est validé et prêt à produire les datasets d'entraînement synthétiques à grande échelle.
- **Ouvert** : Construire les scripts pour exporter les datasets finaux au format JSONL.

### 2026-08-13 — Perturbateur & LiMem

- **Contexte** : Pour évaluer si le modèle entraîné a vraiment appris à raisonner ou s'il se contente de mémoriser les phrases du jeu d'entraînement, je voulais implémenter une méthode similaire à celle du papier K&K (Xie et al.) avec des puzzles mutés.
- **Fait** : J'ai codé deux types de perturbations distincts dans `perturbator.py` :
  1. *La mutation sémantique* : on modifie un seul petit paramètre logique dans une affirmation (par exemple la boîte ciblée ou le comparateur). L'Oracle vérifie que le nouveau puzzle a toujours une solution unique, mais que la gemme gagnante a changé de boîte.
  2. *Les perturbations de surface* : on conserve la même logique exacte et la même solution, mais on change le vocabulaire au moment du rendu textuel (remplacer `BOX` par `CHEST`, `GEMS` par `RUBY`, ou permuter les couleurs des boîtes comme `BLUE` devenant `CYAN`).
- **Résultat** : Ces deux fonctions permettent de séparer proprement la mémorisation de surface (mots-clés) de la mémorisation sémantique (raisonnement logique).
- **Décision + pourquoi** : J'ai gardé les perturbations de surface strictement au niveau du texte généré pour ne pas casser la logique interne du solveur Oracle.
- **Ouvert** : Tester le futur modèle finetuné sur cette grille de perturbations pour quantifier son score LiMem.


### 2026-08-15 — Constructeur de dataset Direct SFT 

- **Contexte** : Pour commencer les différentes phases du projet (SFT + RL), la conception de datasets d'entrainement, de test et d'évaluation est nécessaire.
- **Fait** : j'ai implémenté dataset_builder permettant de générer de tels datasets en respectant les règles usuelles (principe anti fuite, format jsonl, équilibrage strict)
- **Décision + pourquoi** : La décision de générer un dataset équilibré par strates a été prise au détriment d'un dataset réaliste suivant la repartition reelle du jeu de base afin de faciliter et faire gagner en robustesse le score Limem.
- **Ouvert** : la décision concernant la création des CoT synthetiques (utilisation d'une API SOTA vs résolution déterministe par bruteforce) reste encore ouverte. L'API SOTA permettrait d'avoir des raisonnements plus organiques et concis, mais le proncipal facteur limitant reste le cout.

### 2026-08-15 — Direct SFT et tests d'évaluation 

- **Contexte** : début de la phase SFT (Direct, uniquement donner la paire question/réponse)
- **Fait** : 
  - j'ai implémenté un pipeline de fine tuning en qLoRA avec des hyperparamètres bien choisis, puis ai testé le modèle finetuné sur les 57 puzzle gold et les puzzles du dataset de test généré à partir du générateur synthétique
  - Une fois les éval sur des dataset in-distribution et out-distribution, j'en déduis que le modèle ne raisonne pas du tout. Avant de poursuivre, je lance un test d'évaluation sur le dataset train directement pour diagnostiquer l'absence de signal: si le taux est sensiblement au dessus du hasard, rassurant, le modèle a appris les exemples mais ne parvient pas à généraliser. Si là aussi résultat proche du hasard, alors le modèle a simplement sous-appris, et il faudra réentrainer avec plus d'epochs/changer les HP.
- **Décision + pourquoi** : J'ai décidé de ne modifier que les matrives d'attention Q,K,V, correspondant à une modofication de surface. Cela devrait être (et a été) suffisant pour faire apprendre le format de parsing au modèle
- **Ouvert** : Les datasets gold (out-distribution) et test (in-distribution) sont tous les deux utilisés pour l'évaluation de façon séparée. J'ai remarqué cependant de légères différences (présence ou non de points à la fin des statements par exemple) entre les deux. A voir si cela ne rend pas le résultat ambigu.

### 2026-08-15 — Direct SFT , sous apprentissage et réitération

- **Contexte** : une eval sur le dataset de train directement a conduit a diagnostiquer que le modèle a sous appris (résultats à peine au dessus du hasard)
- **Fait** : 
  - j'ai relancé le SFT en chngeant les hypereramètres. J'ai augmenté le nb d'epochs (de 2 à 8) et ajouté les matrices `gate_proj`, `up_proj`, `down_proj` à modifier via le qLoRA.
- **Décision + pourquoi** : avec seulement 2 epochs, il est probable que le modèle n'ait pas eu le temps d'apprendre. De plus, la modification des matrices d'attention Q,K,V seulement  ne permet pas de modifier en profondeur le réseau, résultat en des améliorations faibles.
- **Ouvert** : en l'attente des premiers résultat. Les sessions de finetuning durent de l'ordre de 3h sur carte.

### 2026-08-16 — Validation de Direct-FT v2 (Déblocage du signal capacitaire)

- **Contexte** : L'entraînement Direct-FT v2 (8 époques, LoRA $r=32$ all-linear) s'est terminé proprement. Il fallait vérifier si l'augmentation de capacité et de temps d'exposition permettait de sortir de la zone de hasard.
- **Fait** : J'ai évalué le modèle v2 sur les 3 splits de référence : in-sample Train ($N=840$), test held-out synthétique ($N=1680$) et les 57 réels GOLD du jeu ($N=456$).
- **Résultat** : Les résultats confirment un déblocage net du signal logique :
  - Sur le train : la précision par bit monte à **70.7 %** (+23 points vs v1) et la résolution stricte Option B atteint **20.2 %** ($170 / 840$).
  - En held-out synthétique : le modèle généralise avec **65.5 %** de bit-accuracy (+15.5 pts au-dessus du hasard) et **9.3 %** de succès strict ($\times 6.6$ par rapport aux $1.4\%$ de la v1).
  - Sur les 57 réels : la bit-accuracy atteint **60.3 %** et le succès strict **12.5 %** (doublé par rapport aux $6.1\%$ de la baseline zero-shot, 18/57 puzzles exploitables).
- **Décision + pourquoi** : Le Direct-FT v2 est validé comme adaptateur de référence pour initialiser la phase de RL (warm-start). L'hypothèse d'absence totale de signal est révisée : un signal logique réel mais plafonné émerge en direct prediction.
- **Ouvert** : Passer à l'étape RL (GRPO) pour voir si le gradient par renforcement peut amplifier ce signal.

### 2026-08-16 — Entraînement RL GRPO et diagnostic de convergence

- **Contexte** : Avec l'adaptateur Direct-FT v2 comme point de départ, j'ai implémenté le pipeline de RL avec l'algorithme GRPO via TRL (`train_grpo.py`) sur 300 steps.
- **Fait** : 
  - J'ai configuré une reward graduée ($r \in [0, 2.8]$) récompensant la gemme, la fraction de bits justes et un bonus pour la résolution stricte Option B.
  - J'ai d'abord testé un learning rate conservateur ($LR = 10^{-6}$, $\beta=0.04$, $G=8$). Constatant que la reward stagnait à $\sim 1.05$ et que la divergence KL restait figée à $0.0007$, j'ai relancé une passe avec $LR = 10^{-5}$.
- **Résultat** : Avec $LR = 10^{-5}$, le modèle bouge enfin (la divergence KL monte à $0.038 - 0.073$, soit $\times 50$), mais la reward moyenne continue d'osciller en plateau autour de $\sim 0.99 - 1.15$ sans tendance haussière.
- **Décision + pourquoi** : Ce résultat négatif est fondamental : en l'absence de chaîne de pensée (CoT), le RL sur 25 tokens directs ne dispose d'aucun espace computationnel pour construire des inférences logiques. Les variations de score au sein d'un groupe $G=8$ relèvent du bruit d'échantillonnage de surface.
- **Ouvert** : Cela démontre la nécessité d'introduire des traces de raisonnement (CoT-FT / CoT-RL).

### 2026-08-17 — Protocole LiMem Option B et sonde de mémorisation

- **Contexte** : Pour mesurer formellement la mémorisation et l'invariance logique du modèle aux différentes étapes (Baseline, SFT seul, SFT+RL), il fallait implémenter un script d'évaluation LiMem complet et robuste (`eval_limem.py`).
- **Fait** : 
  - Suite à une analyse méthodologique rigoureuse (protocole Xie et al.), j'ai structuré l'évaluation en deux axes distincts : la vraie mesure de mémorisation sur les instances vues (Train SFT), et la mesure de robustesse lexicale OOD sur les instances inédites (57 Réels).
  - J'ai intégré la notation complète Option B via `grade()` pour évaluer le gap sur la Bit-Accuracy et le Strict, éliminant l'écueil du gem-only qui masquait la dynamique dans le bruit du hasard.
  - J'ai éliminé tout remplacement synthétique étranger : les 22 puzzles non mutables de façon minimale sont proprement ignorés (`skipped`) pour que le taux de fuite mémorielle (`memory_leak`) reste non biaisé.
  - J'ai corrigé le formatage des prompts pour le modèle Instruct en appliquant le template ChatML (`apply_chat_template`) et en isolant la perturbation lexicale des mots-clés de syntaxe de sortie.
- **Décision + pourquoi** : Utiliser ce harnais pour produire les tableaux comparatifs de mémorisation SFT seul vs SFT+RL et prouver expérimentalement l'impact du RL sur la récitation.
- **Ouvert** : Lancer l'évaluation complète sur le train et les 57 réels pour consigner les chiffres définitifs.