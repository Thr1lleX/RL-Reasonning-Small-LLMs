"""
Genere le dataset de PROMPTS pour le GRPO : puzzles GOLD synthetiques, equilibres
par taille N et par gemme, DISJOINTS du SFT-train/val/test ET des 57 puzzles reels.

Format par ligne (attendu par train_grpo.py) :
  {"prompt": [{"role":"user","content": <prompt complet>}],
   "gem": "WHITE", "box_truths": {"BLUE":[...], "WHITE":[...], "BLACK":[...]},
   "signature": "...", "metadata": {...}}

Usage : python Dataset/build_grpo_prompts.py [--total 2100] [--seed 42]
"""
import os
import sys
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "Solver"))

from models import BOXES
from solver import generate_all_worlds, is_world_valid
from generator import generate_gold_puzzle, puzzle_signature
from evaluate_batch import PUZZLES_REGISTER
from eval_lib import render_prompt

FT_DIR = os.path.join(HERE, "ft_direct")
OUT_PATH = os.path.join(HERE, "grpo_prompts.jsonl")
N_SIZES = [3, 4, 5, 6, 7, 8, 9]


def load_leak_signatures():
    """Signatures a exclure : 57 reels + tout le SFT (train/val/test)."""
    sigs = set(puzzle_signature(p) for p in PUZZLES_REGISTER.values())
    n_real = len(sigs)
    n_sft = 0
    for split in ("train", "val", "test"):
        path = os.path.join(FT_DIR, f"{split}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "signature" in r:
                        sigs.add(r["signature"])
                        n_sft += 1
    print(f"[anti-fuite] {n_real} reels + {n_sft} SFT -> {len(sigs)} signatures uniques exclues.")
    return sigs


def build(total, seed):
    random.seed(seed)
    seen = load_leak_signatures()
    per_cell = max(1, total // (len(N_SIZES) * len(BOXES)))

    out = []
    for size in N_SIZES:
        for box in BOXES:
            c = 0
            while c < per_cell:
                rr = random.uniform(0.2, 0.8)
                ap = random.uniform(0.1, 0.5)
                md = random.randint(1, 2)
                puzzle = generate_gold_puzzle(
                    total_statements=size, relational_ratio=rr, ast_prob=ap,
                    max_depth=md, seen_signatures=seen,
                )
                worlds = [w for w in generate_all_worlds(puzzle) if is_world_valid(puzzle, w)]
                if len(worlds) != 1:
                    continue
                w = worlds[0]
                if w.gem_box != box:      # equilibrage gemme
                    continue
                seen.add(puzzle_signature(puzzle))
                box_texts = {b: [s.to_english() for s in puzzle.box_statements[b]] for b in BOXES}
                out.append({
                    "prompt": [{"role": "user", "content": render_prompt(box_texts)}],
                    "gem": w.gem_box,
                    "box_truths": {b: list(w.box_truths[b]) for b in BOXES},
                    "signature": puzzle_signature(puzzle),
                    "metadata": {
                        "n_statements": size,
                        "relational_ratio": round(rr, 3),
                        "ast_prob": round(ap, 3),
                        "max_depth": md,
                    },
                })
                c += 1
        print(f"  N={size} : {sum(1 for r in out if r['metadata']['n_statements'] == size)} prompts")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[OK] {len(out)} prompts ecrits dans {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=2100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.total, args.seed)
