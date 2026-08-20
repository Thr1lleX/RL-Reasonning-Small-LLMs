# RL/GRPO pour le raisonnement logique, puzzles « Blue Prince »

L'objectif de ce projet de recherche est d'entraîner un petit LLM (Qwen2.5-1.5B) au raisonnement
logique sur un environnement de puzzles fait maison (inspiré de l'énigme des boîtes de *Blue Prince*,
architecturalement proche de Knights & Knaves), et **mesurer la mémorisation** (LiMem) à
travers quatre conditions d'entraînement (baseline / RL-seul / SFT / SFT+RL).

Plus d'informations sur l'énigme et son fonctionnement: https://blue-prince.fandom.com/wiki/Parlor_Puzzle


> **Statut : work in progress.** Premières évaluations du score Limem du modèle en Direct-SFT.

## Documentation
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) — journal de bord daté (décisions + pourquoi).
- [`docs/METHODS.md`](docs/METHODS.md) — le système et les choix de conception.
- [`docs/RESULTS.md`](docs/RESULTS.md) — index des expériences (config + seed + artefact). [WIP]


## Structure
```
Solver/
  models.py         # World, Puzzle
  predicates.py     # grammaire de prédicats (+ to_english)
  solver.py         # oracle : énumération + consistance + invariant, unicité stricte
  generator.py      # generate-and-filter de puzzles GOLD
  perturbator.py    # mutation sémantique + perturbations de surface (LiMem)
  evaluate_batch.py # benchmark de l'oracle sur les 66 puzzles in-scope
  eval_lib.py       # noyau d'éval (oracle, prompt, parsing, notation Option B)
  run_eval.py       # harnais d'éval LLM (zero-shot, reprenable)
  aggregate.py      # agrégation des résultats d'éval
  results/          # artefacts (JSONL, rapports MD)
Parlor puzzles.csv  # 110 puzzles réels du jeu (éval hors-distribution)
coverage_worksheetv3.csv  # classification in-scope / GOLD des 110
```

## Dépendances (éval LLM)
`torch` (build CUDA), `transformers`. Les modèles 1.5B se téléchargent au premier chargement.

## Références Bibliographiques
- Shao et al. 2024, *DeepSeekMath* — papier source de GRPO.
- DeepSeek-AI (Guo et al.) 2025, *DeepSeek-R1* — GRPO + RLVR à grande échelle.
- Dang & Ngo 2026 (AAAI), *Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't* — précédent direct pour l'échelle de ressources, config Table 5, insights sur instabilité/dérive de longueur et de langue.
- Xie et al. 2024/2025, *On Memorization of Large Language Models in Logical Reasoning* — benchmark K&K dynamique, métrique LiMem, patron architectural Generator/Solver/Reasoner/Perturber, résultats zero-shot par modèle (Fig. 3).
