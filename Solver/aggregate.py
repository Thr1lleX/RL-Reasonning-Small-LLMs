"""
Agregation des resultats d'eval (lit le JSONL, calcule les stats).
Peut tourner a tout moment, meme sur un run partiel.

Usage : python Solver/aggregate.py [--in results/eval_runs.jsonl]
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
from eval_lib import build_oracle

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_IN = os.path.join(RESULTS_DIR, "eval_runs.jsonl")

# Baseline de reference (distribution des solutions sur les GOLD, cf. discussion)
MAJORITY_LABEL = "WHITE"


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        sys.exit(f"Aucun fichier de resultats : {args.inp}")

    oracle = build_oracle()
    gold_ids = set(oracle)
    recs = load_records(args.inp)

    # regroupement par modele
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    # baseline majoritaire : proportion de GOLD dont la solution == MAJORITY_LABEL
    maj = sum(1 for pid in gold_ids if oracle[pid]["gem"] == MAJORITY_LABEL) / len(gold_ids)

    lines = ["# Agregation eval zero-shot\n"]
    print("=" * 90)
    print(f"GOLD puzzles : {len(gold_ids)} | baseline majoritaire ('{MAJORITY_LABEL}') : {maj*100:.1f}%")
    print("=" * 90)
    lines.append(f"- GOLD puzzles : {len(gold_ids)}")
    lines.append(f"- Baseline majoritaire (toujours '{MAJORITY_LABEL}') : **{maj*100:.1f}%** (plancher gem-only)\n")

    for mk in sorted(by_model):
        rs = by_model[mk]
        n = len(rs)
        n_parsed = sum(1 for r in rs if r["parse_ok"])
        parse_ok = n_parsed / n
        count_ok = sum(1 for r in rs if r.get("count_ok")) / n
        gem_acc = sum(1 for r in rs if r["gem_correct"]) / n
        gem_given_parsed = (sum(1 for r in rs if r["gem_correct"]) / n_parsed) if n_parsed else 0.0
        truths_acc = sum(1 for r in rs if r["truths_correct"]) / n
        strict_acc = sum(1 for r in rs if r["correct"]) / n

        # Precision par bit de verite, sur les reponses parsables (hasard = 50%).
        # Mesure neutre : revele le signal latent que le strict tout-ou-rien ecrase.
        bits_tot = bits_ok = 0
        for r in rs:
            pt = r.get("parsed_truths")
            if not pt:
                continue
            ot = oracle[r["puzzle_id"]]["box_truths"]
            for bx in ("BLUE", "WHITE", "BLACK"):
                got = pt.get(bx) or []
                for i, ev in enumerate(ot[bx]):
                    bits_tot += 1
                    if i < len(got) and got[i] == ev:
                        bits_ok += 1
        per_stmt = (bits_ok / bits_tot) if bits_tot else 0.0

        # pass-rate par puzzle (strict) -> difficulte
        per_puzzle = defaultdict(list)
        for r in rs:
            per_puzzle[r["puzzle_id"]].append(1 if r["correct"] else 0)
        passrates = {pid: sum(v) / len(v) for pid, v in per_puzzle.items()}
        mean_pr = sum(passrates.values()) / len(passrates) if passrates else 0.0
        n_puzzles = len(passrates)
        samples_per = n / n_puzzles if n_puzzles else 0

        # Signal exploitable pour le bootstrap RLVR (variance de reward dans le groupe)
        never = sum(1 for pr in passrates.values() if pr == 0.0)
        always = sum(1 for pr in passrates.values() if pr == 1.0)
        exploitable = sum(1 for pr in passrates.values() if 0.0 < pr < 1.0)
        pass_at_g = sum(1 for pr in passrates.values() if pr > 0.0)

        print(f"\n### {mk}")
        print(f"  echantillons          : {n}  (~{samples_per:.0f}/puzzle sur {n_puzzles} puzzles)")
        print(f"  format parsable       : {parse_ok*100:5.1f}%   (FINAL + gem lisibles)")
        print(f"  compte statements OK  : {count_ok*100:5.1f}%   (bon nombre de T/F par boite)")
        print(f"  accuracy gem-only     : {gem_acc*100:5.1f}%   (vs {maj*100:.1f}% majoritaire, 33% hasard)")
        print(f"  gem-only si parsable  : {gem_given_parsed*100:5.1f}%   (hasard uniforme 33%)")
        print(f"  precision / bit verite: {per_stmt*100:5.1f}%   (parsables, hasard 50%)  <-- signal latent")
        print(f"  verites toutes justes : {truths_acc*100:5.1f}%")
        print(f"  accuracy STRICTE (B)  : {strict_acc*100:5.1f}%")
        print(f"  pass-rate moyen/puzzle: {mean_pr*100:5.1f}%")
        print(f"  --- signal bootstrap RLVR (sur {n_puzzles} puzzles) ---")
        print(f"  jamais resolus (pr=0) : {never}")
        print(f"  toujours resolus (pr=1): {always}")
        print(f"  EXPLOITABLES (0<pr<1) : {exploitable}   <-- variance de reward = signal RL")
        print(f"  pass@G (resolu >=1 fois): {pass_at_g}/{n_puzzles} ({pass_at_g/n_puzzles*100:.0f}%)")

        lines.append(f"## {mk}\n")
        lines.append(f"- echantillons : {n} (~{samples_per:.0f}/puzzle sur {n_puzzles} puzzles)")
        lines.append(f"- format parsable : {parse_ok*100:.1f}%")
        lines.append(f"- accuracy gem-only : {gem_acc*100:.1f}% (parsable {gem_given_parsed*100:.1f}%, hasard 33%)")
        lines.append(f"- precision/bit de verite : {per_stmt*100:.1f}% (hasard 50%)")
        lines.append(f"- **accuracy stricte (Option B) : {strict_acc*100:.1f}%**")
        lines.append(f"- pass-rate moyen par puzzle : {mean_pr*100:.1f}% | exploitables {exploitable}/{n_puzzles} | pass@G {pass_at_g}/{n_puzzles}\n")

        # difficulte par puzzle (les plus durs d'abord)
        hard = sorted(passrates.items(), key=lambda kv: kv[1])
        lines.append("| puzzle | pass-rate | difficulte (1-pr) |")
        lines.append("|---|---|---|")
        for pid, pr in hard:
            lines.append(f"| {pid} | {pr*100:.0f}% | {(1-pr)*100:.0f}% |")
        lines.append("")

    out_md = os.path.join(RESULTS_DIR, "eval_aggregate.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[INFO] Rapport ecrit : {out_md}")


if __name__ == "__main__":
    main()
