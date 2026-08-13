# RL/GRPO pour le raisonnement logique — puzzles « Blue Prince »

Projet de recherche : entraîner un petit LLM (Qwen2.5-1.5B) au raisonnement
logique sur un environnement de puzzles fait maison (inspiré des boîtes de *Blue Prince*,
architecturalement proche de Knights & Knaves), et **mesurer la mémorisation** (LiMem) à
travers quatre conditions d'entraînement (baseline / RL-seul / SFT / SFT+RL).

> **Statut : work in progress.** Toolchain (générateur / solveur-oracle / traducteur NL /
> perturbateur / harnais d'éval) construite et validée. Calibration terminée : *aucun signal
> zero-shot* → pivot vers SFT→RL. Prochaine étape : assemblage des datasets.

## Documentation
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) — journal de bord daté (décisions + pourquoi).
- [`docs/METHODS.md`](docs/METHODS.md) — le système et les choix de conception.
- [`docs/RESULTS.md`](docs/RESULTS.md) — index des expériences (config + seed + artefact).


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

## Lancer (depuis la racine du dépôt)

Benchmark de l'oracle (57/66 GOLD) :
```bash
python Solver/evaluate_batch.py
```

Éval zero-shot d'un modèle (reprenable ; Ctrl-C puis relancer la même commande) :
```bash
python Solver/run_eval.py --model qwen2.5-1.5b-instruct --n-samples 8
```
```bash
python Solver/aggregate.py
```

Démo générateur / perturbateur :
```bash
python Solver/generator.py
```
```bash
python Solver/perturbator.py
```

## Dépendances (éval LLM)
`torch` (build CUDA), `transformers`. Les modèles 1.5B se téléchargent au premier chargement.

## Références
- Shao et al. 2024, *DeepSeekMath* — papier source de GRPO.
- DeepSeek-AI (Guo et al.) 2025, *DeepSeek-R1* — GRPO + RLVR à grande échelle.
- Dang & Ngo 2026 (AAAI), *Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't* — précédent direct pour l'échelle de ressources, config Table 5, insights sur instabilité/dérive de longueur et de langue.
- Xie et al. 2024/2025, *On Memorization of Large Language Models in Logical Reasoning* — benchmark K&K dynamique, métrique LiMem, patron architectural Generator/Solver/Reasoner/Perturber, résultats zero-shot par modèle (Fig. 3).
