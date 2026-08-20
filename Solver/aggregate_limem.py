"""
Agrege un JSONL produit par eval_limem.py (meme si partiel/interrompu) et
reimprime le tableau LiMem (canonique / isomorphe / mutant + gap + leak).
Utile pour lire un run coupe en cours, et pour comparer plusieurs conditions.

Usage :
  python Solver/aggregate_limem.py Solver/results/limem_qwen-base-train-limem.jsonl
  python Solver/aggregate_limem.py Solver/results/limem_*.jsonl        # plusieurs fichiers
"""
import os
import sys
import json
import glob
import collections

CHANCE_LEAK = 100.0 / 3.0   # ~33.3 % : un modele au hasard ressort l'ancienne gemme 1 fois sur 3


def aggregate(path):
    conds = ("canonique", "isomorphe", "mutant")
    acc = {c: {"n": 0, "parse": 0, "gem": 0, "strict": 0, "bit": 0.0,
               "leak": 0, "puzzles": set()} for c in conds}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("condition")
            if c not in acc:
                continue
            a = acc[c]
            a["n"] += 1
            a["puzzles"].add(str(r.get("puzzle_id")))
            a["parse"] += int(bool(r.get("parse_ok")))
            a["gem"] += int(bool(r.get("gem_correct")))
            a["strict"] += int(bool(r.get("strict_correct")))
            a["bit"] += float(r.get("bit_accuracy") or 0.0)
            a["leak"] += int(bool(r.get("memory_leak")))

    def rates(c):
        a = acc[c]
        n = a["n"]
        if n == 0:
            return None
        return {
            "n": n, "puzzles": len(a["puzzles"]),
            "parse": a["parse"] / n, "gem": a["gem"] / n,
            "strict": a["strict"] / n, "bit": a["bit"] / n,
            "leak": a["leak"] / n,
        }

    label = os.path.basename(path)
    print("\n" + "=" * 78)
    print(f"  LiMem (agrege) : {label}")
    print("=" * 78)
    print(f"  {'Condition':<12} | {'#puz':>4} | {'#gen':>5} | {'Parse':>6} | {'Gem':>6} | {'Bit':>6} | {'Strict':>7}")
    print("-" * 78)
    R = {}
    for c in conds:
        rr = rates(c)
        R[c] = rr
        if rr is None:
            continue
        print(f"  {c:<12} | {rr['puzzles']:>4} | {rr['n']:>5} | "
              f"{rr['parse']*100:5.1f}% | {rr['gem']*100:5.1f}% | "
              f"{rr['bit']*100:5.1f}% | {rr['strict']*100:6.1f}%")
    print("-" * 78)
    if R["canonique"] and R["isomorphe"]:
        bit_gap = max(0.0, R["canonique"]["bit"] - R["isomorphe"]["bit"])
        strict_gap = max(0.0, R["canonique"]["strict"] - R["isomorphe"]["strict"])
        print(f"  * LiMem Bit-Gap    : {bit_gap*100:5.1f} %  (BitAcc_can - BitAcc_iso)")
        print(f"  * LiMem Strict-Gap : {strict_gap*100:5.1f} %  (Strict_can - Strict_iso)")
    if R["mutant"]:
        leak = R["mutant"]["leak"] * 100
        print(f"  * Memory-Leak      : {leak:5.1f} %  "
              f"(hasard ~{CHANCE_LEAK:.1f} % | exces {max(0.0, leak - CHANCE_LEAK):+5.1f} %)")
    print("=" * 78)


def main():
    if len(sys.argv) < 2:
        print("usage: python Solver/aggregate_limem.py <fichier.jsonl> [autres...]")
        sys.exit(1)
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)) or [arg])
    for p in paths:
        if os.path.exists(p):
            aggregate(p)
        else:
            print(f"[!] introuvable : {p}")


if __name__ == "__main__":
    main()
