import os
import sys
import json
import random
from typing import Dict, List, Set, Tuple

# Ajouter le dossier Solver pour importer les modules existants
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Solver"))

from models import World, Puzzle, BOXES
from solver import generate_all_worlds, is_world_valid, solve_puzzle
from generator import generate_gold_puzzle, puzzle_signature
from evaluate_batch import PUZZLES_REGISTER
from eval_lib import RULES, BOX_LABEL, BOX_PREFIX


from eval_lib import render_prompt, BOX_PREFIX   # (remplace l'import de RULES, BOX_LABEL)

def format_chat_prompt(puzzle, world, example_id, split_name, metadata=None):
    if metadata is None:
        metadata = {}

    # même template que l'éval, seul le contenu du puzzle change
    box_texts = {b: [s.to_english() for s in puzzle.box_statements[b]] for b in BOXES}
    user_content = render_prompt(box_texts)

    # cible Direct-FT : réponse seule
    solution_parts = [f"gem={world.gem_box}"]
    for b in BOXES:
        for i, truth in enumerate(world.box_truths[b], 1):
            solution_parts.append(f"{BOX_PREFIX[b]}{i}={'T' if truth else 'F'}")
    assistant_content = f"FINAL: {'; '.join(solution_parts)}"

    return {
        "id": example_id,
        "split": split_name,
        "signature": puzzle_signature(puzzle),
        "metadata": {
            "n_statements": sum(len(s) for s in puzzle.box_statements.values()),
            "gem_box": world.gem_box,
            **metadata,
        },
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_direct_ft_dataset(train_size: int = 2100, val_size: int = 210, test_size: int = 210, seed: int = 42):
    """
    Génère un dataset SFT équilibré par taille N et par boîte (gemme),
    avec des splits strictement disjoints et anti-fuite par rapport aux 57 puzzles GOLD.
    """
    random.seed(seed)

    # 1. Initialisation du filtre anti-fuite avec les puzzles canoniques du jeu
    seen_puzzles = set()
    for p in PUZZLES_REGISTER.values():
        seen_puzzles.add(puzzle_signature(p))
    print(f"[OK] {len(seen_puzzles)} signatures du jeu original chargees dans le filtre anti-fuite.")

    # 2. Grille de stratification par taille
    N_SIZES = [3, 4, 5, 6, 7, 8, 9]
    num_sizes = len(N_SIZES)
    num_boxes = len(BOXES)

    # Quotas par (split, taille, boîte)
    split_configs = [
        ('train', train_size),
        ('val', val_size),
        ('test', test_size)
    ]

    stats = {
        'train': {'total': 0, 'by_size': {n: 0 for n in N_SIZES}, 'by_box': {b: 0 for b in BOXES}},
        'val': {'total': 0, 'by_size': {n: 0 for n in N_SIZES}, 'by_box': {b: 0 for b in BOXES}},
        'test': {'total': 0, 'by_size': {n: 0 for n in N_SIZES}, 'by_box': {b: 0 for b in BOXES}},
    }

    output_dir = os.path.join(os.path.dirname(__file__), "ft_direct")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Boucle de génération par split
    for split_name, total_target in split_configs:
        print(f"\n==========================================")
        print(f"GENERATION DU SPLIT : {split_name.upper()} (Cible : {total_target})")
        print(f"==========================================")

        current_stats = stats[split_name]
        chat_buffer = []
        target_per_size_box = max(1, total_target // (num_sizes * num_boxes))
        example_idx = 1

        for size in N_SIZES:
            for box in BOXES:
                box_count = 0
                while box_count < target_per_size_box:
                    # Paramètres aléatoires de difficulté
                    relational_ratio = random.uniform(0.2, 0.8)
                    ast_prob = random.uniform(0.1, 0.5)
                    max_depth = random.randint(1, 2)

                    # Génération du puzzle Or strict
                    puzzle = generate_gold_puzzle(
                        total_statements=size,
                        relational_ratio=relational_ratio,
                        ast_prob=ast_prob,
                        max_depth=max_depth,
                        seen_signatures=seen_puzzles
                    )

                    # Résolution par l'Oracle
                    worlds = [w for w in generate_all_worlds(puzzle) if is_world_valid(puzzle, w)]
                    if len(worlds) != 1:
                        continue
                    world = worlds[0]

                    # Contrôle d'équilibrage de la gemme
                    if world.gem_box == box:
                        sig = puzzle_signature(puzzle)
                        seen_puzzles.add(sig)

                        box_count += 1
                        current_stats['total'] += 1
                        current_stats['by_size'][size] += 1
                        current_stats['by_box'][box] += 1

                        example_id = f"direct_ft_{split_name}_{example_idx:05d}"
                        meta = {
                            "relational_ratio": round(relational_ratio, 3),
                            "ast_prob": round(ast_prob, 3),
                            "max_depth": max_depth
                        }
                        chat_data = format_chat_prompt(puzzle, world, example_id, split_name, meta)
                        chat_buffer.append(chat_data)
                        example_idx += 1

            print(f"  -> Taille N={size} terminee : {current_stats['by_size'][size]} puzzles generes.")

        # 4. Écriture du fichier JSONL pour ce split
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in chat_buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"\n[OK] Fichier ecrit : {output_path} ({len(chat_buffer)} exemples)")
        print(f"  Distribution par taille : {current_stats['by_size']}")
        print(f"  Distribution par boite  : {current_stats['by_box']}")

    print("\n==========================================")
    print("GENERATION DE TOUS LES SPLITS TERMINEE AVEC SUCCES !")
    print("==========================================")


if __name__ == "__main__":
    # Test de génération standard
    build_direct_ft_dataset(train_size=2100, val_size=210, test_size=210, seed=42)
