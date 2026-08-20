"""
Évaluation de robustesse et mémorisation LiMem (Linear Memorization Probe).
Protocole inspiré de Xie et al. adapté à Blue Prince (Option B).

Axes d'évaluation :
  1. SUR LE TRAIN SYNTHÉTIQUE (Instances vues) :
     Mesure la VRAIE mémorisation SFT vs désapprentissage par RL (GRPO).
     Gap = Acc(canonique vue) - Acc(isomorphe) => Score LiMem pur.
  2. SUR LES 57 RÉELS / HELD-OUT (Instances inédites) :
     Mesure la robustesse de surface et la généralisation OOD.

Métriques calculées (Option B) :
  - Bit-Accuracy (fraction de propositions T/F correctes)
  - Strict-Accuracy (Gemme ET toutes les propositions correctes)
  - Gem-Accuracy
  - LiMem Bit / Strict Delta
  - Memory-Leakage Rate sur les mutants minimaux (P(pred == orig_gem | mutant))

Usage :
  # 1. Mesure LiMem Mémorisation sur le Train SFT :
  python Solver/eval_limem.py --dataset Dataset/ft_direct/train.jsonl --limit 100 --model Qwen/Qwen2.5-1.5B-Instruct --sft-adapter Training/sft_direct_2_epochs_8_lorar32 --label qwen-sft-v2-train-limem

  # 2. Mesure Robustesse OOD sur les 57 Réels :
  python Solver/eval_limem.py --split real57 --model Qwen/Qwen2.5-1.5B-Instruct --sft-adapter Training/sft_direct_2_epochs_8_lorar32 --label qwen-sft-v2-real57-limem
"""

import os
import re
import sys
import json
import zlib
import random
import argparse
import pickle
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from models import Puzzle, BOXES
from evaluate_batch import PUZZLES_REGISTER
from perturbator import mutate_gold_puzzle
from solver import generate_all_worlds, is_world_valid
from eval_lib import build_oracle, load_statement_texts, build_prompt, parse_final, grade

RESULTS_DIR = os.path.join(HERE, "results")


# --------------------------------------------------------------------------
# 1. Chargement universel du modèle (Base, SFT ou Post-GRPO)
# --------------------------------------------------------------------------

def load_model(base_model_id: str, sft_adapter: str = None, grpo_adapter: str = None):
    print(f"[*] Chargement du modèle de base : {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    if sft_adapter:
        print(f"[*] Application de l'adaptateur SFT : {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    if grpo_adapter:
        print(f"[*] Application de l'adaptateur GRPO : {grpo_adapter}")
        model = PeftModel.from_pretrained(model, grpo_adapter)
        model = model.merge_and_unload()

    model.config.use_cache = False
    model.eval()
    return tokenizer, model


# --------------------------------------------------------------------------
# 2. Perturbation lexicale robuste (Regex Word Boundary)
# --------------------------------------------------------------------------

def apply_lexical_perturbation(prompt_text: str) -> str:
    """Remplace les termes de surface dans les énoncés tout en préservant la ligne de syntaxe FINAL: gem=."""
    # Sépare les énoncés de la consigne de formatage final
    parts = prompt_text.split("Think step by step.")
    body = parts[0]
    tail = "Think step by step." + parts[1] if len(parts) > 1 else ""

    replacements = [
        (r"\bBOXES\b", "CHESTS"),
        (r"\bBoxes\b", "Chests"),
        (r"\bboxes\b", "chests"),
        (r"\bBOX\b", "CHEST"),
        (r"\bBox\b", "Chest"),
        (r"\bbox\b", "chest"),
        (r"\bGEMS\b", "COINS"),
        (r"\bGems\b", "Coins"),
        (r"\bgems\b", "coins"),
        (r"\bGEM\b", "COIN"),
        (r"\bGem\b", "Coin"),
        (r"\bgem\b", "coin"),
    ]
    for pattern, repl in replacements:
        body = re.sub(pattern, repl, body)

    return body + tail


# --------------------------------------------------------------------------
# 3. Préparation des Triplets (57 Réels vs Dataset Synthétique)
# --------------------------------------------------------------------------

def build_triplets_real57(seed: int = 42):
    """Génère les triplets pour les 57 puzzles réels GOLD."""
    random.seed(seed)
    triplets = []
    oracle = build_oracle()
    texts = load_statement_texts()
    skipped_mutants = 0

    for pid in sorted(oracle.keys()):
        puzzle = PUZZLES_REGISTER[pid]
        gold_orig = oracle[pid]

        # 1. Canonique
        prompt_canonique = build_prompt(pid, texts)
        canonique_data = {"prompt": prompt_canonique, "gold": gold_orig}

        # 2. Isomorphe (perturbation lexicale de surface)
        prompt_iso = apply_lexical_perturbation(prompt_canonique)
        iso_data = {"prompt": prompt_iso, "gold": gold_orig}

        # 3. Mutant (mutation minimale uniquement — pas de substitution)
        mut_data = None
        try:
            mutated_puzzle = mutate_gold_puzzle(puzzle, max_attempts=500)
            valid_worlds = [w for w in generate_all_worlds(mutated_puzzle) if is_world_valid(mutated_puzzle, w)]
            if len(valid_worlds) == 1 and valid_worlds[0].gem_box != gold_orig["gem"]:
                new_world = valid_worlds[0]
                mut_gold = {
                    "gem": new_world.gem_box,
                    "box_truths": {b: list(new_world.box_truths[b]) for b in BOXES},
                }
                mut_texts = {
                    pid: {
                        b: [stmt.to_english() for stmt in mutated_puzzle.box_statements[b]]
                        for b in BOXES
                    }
                }
                prompt_mut = build_prompt(pid, mut_texts)
                mut_data = {
                    "prompt": prompt_mut,
                    "gold": mut_gold,
                    "canonical_orig_gem": gold_orig["gem"],
                }
        except Exception:
            mut_data = None

        if mut_data is None:
            skipped_mutants += 1

        triplets.append((f"real_gold_{pid:03d}", canonique_data, iso_data, mut_data))

    print(f"[*] Triplets Réels générés : {len(triplets)} puzzles ({len(triplets) - skipped_mutants} mutants valides, {skipped_mutants} mutants sautés)")
    return triplets


def build_triplets_dataset(jsonl_path: str, limit: int = 100, seed: int = 42):
    """Charge les VRAIES instances vues du train.jsonl + les mutants issus de leurs objets AST."""
    random.seed(seed)
    triplets = []
    skipped_mutants = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Chargement des objets Puzzle AST correspondants si disponibles
    pkl_path = jsonl_path.replace(".jsonl", "_puzzles.pkl")
    puzzles_ast = None
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            puzzles_ast = pickle.load(f)
        print(f"[*] Objets Puzzle AST chargés depuis {pkl_path} ({len(puzzles_ast)} objets)")

    indices = list(range(len(lines)))
    if limit and len(lines) > limit:
        random.shuffle(indices)
        indices = indices[:limit]

    for idx in indices:
        item = lines[idx]
        pid_str = item.get("id", f"train_{idx:05d}")
        user_prompt = item["messages"][0]["content"]
        assistant_ans = item["messages"][1]["content"]

        parsed_gold = parse_final(assistant_ans)
        if not parsed_gold or parsed_gold["gem"] is None:
            continue

        gold_orig = {
            "gem": parsed_gold["gem"],
            "box_truths": {
                b: [parsed_gold["box_idx"][b].get(i, False) for i in range(1, len(parsed_gold["box_idx"][b]) + 1)]
                for b in BOXES
            },
        }

        # 1. Canonique (VRAI texte vu pendant l'entraînement SFT)
        canonique_data = {"prompt": user_prompt, "gold": gold_orig}

        # 2. Isomorphe (perturbation lexicale de surface)
        prompt_iso = apply_lexical_perturbation(user_prompt)
        iso_data = {"prompt": prompt_iso, "gold": gold_orig}

        # 3. Mutant (mutation sémantique sur l'objet AST correspondant)
        mut_data = None
        if puzzles_ast and idx < len(puzzles_ast):
            puzzle = puzzles_ast[idx]
            try:
                mutated_puzzle = mutate_gold_puzzle(puzzle, max_attempts=500)
                valid_worlds = [w for w in generate_all_worlds(mutated_puzzle) if is_world_valid(mutated_puzzle, w)]
                if len(valid_worlds) == 1 and valid_worlds[0].gem_box != gold_orig["gem"]:
                    new_world = valid_worlds[0]
                    mut_gold = {
                        "gem": new_world.gem_box,
                        "box_truths": {b: list(new_world.box_truths[b]) for b in BOXES},
                    }
                    mut_texts = {
                        idx: {
                            b: [stmt.to_english() for stmt in mutated_puzzle.box_statements[b]]
                            for b in BOXES
                        }
                    }
                    prompt_mut = build_prompt(idx, mut_texts)
                    mut_data = {
                        "prompt": prompt_mut,
                        "gold": mut_gold,
                        "canonical_orig_gem": gold_orig["gem"],
                    }
            except Exception:
                mut_data = None

        if mut_data is None:
            skipped_mutants += 1

        triplets.append((pid_str, canonique_data, iso_data, mut_data))

    print(f"[*] Triplets VRAI Train prêts : {len(triplets)} instances ({len(triplets) - skipped_mutants} mutants valides, {skipped_mutants} sautés)")
    return triplets


# --------------------------------------------------------------------------
# 4. Calcul de métriques fines (Option B)
# --------------------------------------------------------------------------

def compute_frac_bits(parsed_truths: dict, gold_box_truths: dict) -> float:
    """Calcule la fraction exacte de propositions correctes (bit-accuracy)."""
    if not parsed_truths:
        return 0.0
    total = 0
    correct = 0
    for b in BOXES:
        g_list = gold_box_truths.get(b, [])
        p_list = parsed_truths.get(b, [])
        for i in range(len(g_list)):
            total += 1
            if i < len(p_list) and p_list[i] is not None and p_list[i] == g_list[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


def seed_for(model_label: str, condition: str, pid: str, sample_idx: int, base_seed: int) -> int:
    return zlib.crc32(f"{model_label}|{condition}|{pid}|{sample_idx}|{base_seed}".encode()) & 0x7FFFFFFF


def generate_single_response(tokenizer, model, prompt: str, temperature: float, max_new_tokens: int, seed: int) -> str:
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def append_record(fh, rec: dict):
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def load_done_samples(path):
    """Charge les triplets (condition, puzzle_id, sample_idx) déjà enregistrés."""
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    done.add((r["condition"], str(r["puzzle_id"]), r["sample_idx"]))
                except Exception:
                    continue
    return done
# --------------------------------------------------------------------------
# 5. Boucle Principale d'Évaluation LiMem
# --------------------------------------------------------------------------

def run_eval_limem(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out if args.out else os.path.join(RESULTS_DIR, f"limem_{args.label}.jsonl")
    print(f"[*] Fichier de sortie : {out_path}")

    if args.dataset:
        triplets = build_triplets_dataset(args.dataset, limit=args.limit, seed=args.seed)
        mode_label = f"VRAI Dataset Train ({args.dataset})"
    else:
        triplets = build_triplets_real57(seed=args.seed)
        mode_label = "57 Puzzles Réels GOLD (Blue Prince)"

    tokenizer, model = load_model(args.model, args.sft_adapter, args.grpo_adapter)

    # Accumulateurs de métriques
    stats = {
        "canonique": {"total": 0, "parse_ok": 0, "gem_correct": 0, "strict_correct": 0, "bit_acc_sum": 0.0},
        "isomorphe": {"total": 0, "parse_ok": 0, "gem_correct": 0, "strict_correct": 0, "bit_acc_sum": 0.0},
        "mutant":    {"total": 0, "parse_ok": 0, "gem_correct": 0, "strict_correct": 0, "bit_acc_sum": 0.0, "memory_leaks": 0},
    }
    done_samples = load_done_samples(out_path)
    print(f"[*] Début de l'évaluation LiMem [{mode_label}] avec {len(triplets)} puzzles x {args.n_samples} tirages...")

    with open(out_path, "a", encoding="utf-8") as fh:
        for idx, (pid, can_data, iso_data, mut_data) in enumerate(triplets, 1):
            if idx % 10 == 0 or idx == len(triplets):
                print(f"  -> Progression : Puzzle {idx}/{len(triplets)}...")

            conditions = [
                ("canonique", can_data["prompt"], can_data["gold"], can_data["gold"]["gem"]),
                ("isomorphe", iso_data["prompt"], iso_data["gold"], can_data["gold"]["gem"]),
            ]
            if mut_data is not None:
                conditions.append(("mutant", mut_data["prompt"], mut_data["gold"], mut_data["canonical_orig_gem"]))
            
            for cond_name, prompt, gold, orig_gem in conditions:
                for s in range(args.n_samples):
                    if (cond_name, str(pid), s) in done_samples:
                        continue
                    sample_seed = seed_for(args.label, cond_name, str(pid), s, args.seed)
                    raw_text = generate_single_response(
                        tokenizer, model, prompt, args.temperature, args.max_new_tokens, sample_seed
                    )
                    
                    parsed = parse_final(raw_text)
                    g = grade(parsed, gold)
                    
                    parse_ok = g["parse_ok"]
                    gem_correct = g["gem_correct"]
                    strict_correct = g["correct"]
                    frac_bits = compute_frac_bits(g.get("parsed_truths"), gold["box_truths"])
                    
                    is_memory_leak = (cond_name == "mutant" and parse_ok and g["gem"] == orig_gem)

                    # Mise à jour des stats
                    st = stats[cond_name]
                    st["total"] += 1
                    if parse_ok:
                        st["parse_ok"] += 1
                    if gem_correct:
                        st["gem_correct"] += 1
                    if strict_correct:
                        st["strict_correct"] += 1
                    st["bit_acc_sum"] += frac_bits

                    if cond_name == "mutant" and is_memory_leak:
                        st["memory_leaks"] += 1

                    # Enregistrement
                    record = {
                        "model": args.label,
                        "puzzle_id": pid,
                        "condition": cond_name,
                        "sample_idx": s,
                        "seed": sample_seed,
                        "temperature": args.temperature,
                        "parse_ok": parse_ok,
                        "predicted_gem": g["gem"],
                        "target_gem": gold["gem"],
                        "canonical_orig_gem": orig_gem,
                        "gem_correct": gem_correct,
                        "strict_correct": strict_correct,
                        "bit_accuracy": frac_bits,
                        "memory_leak": is_memory_leak,
                        "raw": raw_text,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    }
                    append_record(fh, record)




    # ----------------------------------------------------------------------
    # 6. Rapport Final LiMem (Option B)
    # ----------------------------------------------------------------------
    def get_rates(k):
        tot = stats[k]["total"]
        if tot == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            stats[k]["parse_ok"] / tot,
            stats[k]["gem_correct"] / tot,
            stats[k]["strict_correct"] / tot,
            stats[k]["bit_acc_sum"] / tot,
        )

    p_can, gem_can, strict_can, bit_can = get_rates("canonique")
    p_iso, gem_iso, strict_iso, bit_iso = get_rates("isomorphe")
    p_mut, gem_mut, strict_mut, bit_mut = get_rates("mutant")

    limem_bit_gap = max(0.0, bit_can - bit_iso)
    limem_strict_gap = max(0.0, strict_can - strict_iso)
# Un modèle aléatoire sort l'ancienne gemme 1 fois sur 3 (~33.3 %)
    chance_leak = 33.33
    raw_leak_rate = (stats["mutant"]["memory_leaks"] / stats["mutant"]["total"] * 100) if stats["mutant"]["total"] > 0 else 0.0
    excess_leak = max(0.0, raw_leak_rate - chance_leak)

   
    print("\n" + "=" * 78)
    print(f"               RAPPORT D'ÉVALUATION LIMEM (Option B) : {args.label}")
    print(f"               Contexte : {mode_label}")
    print("=" * 78)
    print(f"  Puzzles évalués : {len(triplets)} | Tirages/condition (N) : {args.n_samples}")
    print("-" * 78)
    print(f"  {'Condition':<15} | {'Parse OK':<10} | {'Gem Acc':<10} | {'Bit-Acc':<10} | {'Strict (Option B)':<18}")
    print("-" * 78)
    print(f"  {'Canonique':<15} | {p_can*100:6.1f} %   | {gem_can*100:6.1f} %   | {bit_can*100:6.1f} %   | {strict_can*100:6.1f} %")
    print(f"  {'Isomorphe':<15} | {p_iso*100:6.1f} %   | {gem_iso*100:6.1f} %   | {bit_iso*100:6.1f} %   | {strict_iso*100:6.1f} %")
    if stats["mutant"]["total"] > 0:
        print(f"  {'Mutant':<15} | {p_mut*100:6.1f} %   | {gem_mut*100:6.1f} %   | {bit_mut*100:6.1f} %   | {strict_mut*100:6.1f} %")
    print("-" * 78)
    print(f"  * LiMem Bit-Accuracy Gap    : {limem_bit_gap * 100:5.1f} %  (BitAcc_can - BitAcc_iso)")
    print(f"  * LiMem Strict Gap (Opt B)  : {limem_strict_gap * 100:5.1f} %  (Strict_can - Strict_iso)")
    if stats["mutant"]["total"] > 0:
        print(f"  * Memory-Leakage Rate       : {raw_leak_rate:5.1f} %  (Hasard théorique : ~33.3 % | Excès de récitation : {excess_leak:+5.1f} %)")
    print("=" * 78)
    print(f"[*] Résultats complets sauvegardés dans : {out_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation de robustesse et mémorisation LiMem (Option B)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Identifiant HF du modèle de base")
    parser.add_argument("--sft-adapter", type=str, default=None, help="Chemin vers l'adaptateur LoRA SFT")
    parser.add_argument("--grpo-adapter", type=str, default=None, help="Chemin vers l'adaptateur LoRA GRPO")
    parser.add_argument("--label", type=str, default="qwen-limem", help="Nom du modèle/run pour les logs")
    parser.add_argument("--dataset", type=str, default=None, help="Chemin vers un dataset JSONL (ex: Dataset/ft_direct/train.jsonl pour vraie mémorisation)")
    parser.add_argument("--split", type=str, default="real57", help="Split par défaut si dataset non spécifié ('real57')")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre de puzzles à évaluer")
    parser.add_argument("--n-samples", type=int, default=8, help="Nombre de tirages par puzzle et par condition")
    parser.add_argument("--temperature", type=float, default=0.7, help="Température de génération")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Longueur maximale générée")
    parser.add_argument("--seed", type=int, default=42, help="Seed de base pour la reproductibilité")
    parser.add_argument("--out", type=str, default=None, help="Chemin du fichier JSONL de sortie")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval_limem(args)
