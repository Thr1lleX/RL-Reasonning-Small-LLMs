"""
Evaluation d'un modele sur un dataset JSONL (val/test synthetique) dont le gold
est embarque dans le message assistant. Complement de run_eval.py (qui evalue sur
les 57 GOLD reels). Memes metriques, reprenable, supporte --adapter (LoRA).

Usage :
  python Solver/run_eval_dataset.py --dataset Dataset/ft_direct/test.jsonl \\
      --model qwen2.5-1.5b-instruct --adapter Training/sft_direct_out \\
      --label qwen-sft-direct --n-samples 8 --max-new-tokens 128 \\
      --out Solver/results/sft_test_eval.jsonl
"""
import os
import sys
import json
import zlib
import argparse
import collections
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from models import BOXES
from eval_lib import parse_final, grade
from run_eval import load_model, generate_once, append_record, MODELS


def seed_for(label, ex_id, s):
    return zlib.crc32(f"{label}|{ex_id}|{s}".encode()) & 0x7FFFFFFF


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["example_id"], r["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def gold_from_final(final_line):
    """Reconstruit {'gem', 'box_truths'} depuis la ligne FINAL de l'assistant."""
    parsed = parse_final(final_line)
    if parsed is None:
        return None
    box_truths = {}
    for b in BOXES:
        idx_map = parsed["box_idx"][b]
        n = max(idx_map) if idx_map else 0
        box_truths[b] = [idx_map.get(i) for i in range(1, n + 1)]
    return {"gem": parsed["gem"], "box_truths": box_truths}


def dry_stub(gold, s):
    """s%3 : 0 -> correct, 1 -> gemme ok mais verites fausses, 2 -> non parsable."""
    if s % 3 == 2:
        return "no final line here"
    prefix = {"BLUE": "B", "WHITE": "W", "BLACK": "K"}
    bt = gold["box_truths"] if s % 3 == 0 else {b: [not x for x in gold["box_truths"][b]] for b in BOXES}
    segs = [f"gem={gold['gem']}"]
    for b in BOXES:
        for i, x in enumerate(bt[b], 1):
            segs.append(f"{prefix[b]}{i}={'T' if x else 'F'}")
    return "FINAL: " + "; ".join(segs)


def aggregate(path, label):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["model"] == label]
    n = len(rows)
    if n == 0:
        print("aucun record")
        return
    parse_ok = sum(r["parse_ok"] for r in rows) / n
    gem = sum(r["gem_correct"] for r in rows) / n
    strict = sum(r["correct"] for r in rows) / n
    bits_tot = bits_ok = 0
    for r in rows:
        pt = r.get("parsed_truths")
        if not pt:
            continue
        for b in BOXES:
            got = pt.get(b) or []
            for i, ev in enumerate(r["gold"]["box_truths"][b]):
                bits_tot += 1
                if i < len(got) and got[i] == ev:
                    bits_ok += 1
    per_bit = bits_ok / bits_tot if bits_tot else 0.0
    per = collections.defaultdict(list)
    for r in rows:
        per[r["example_id"]].append(1 if r["correct"] else 0)
    passg = sum(1 for v in per.values() if sum(v) > 0)
    print(f"\n### {label}  (IN-DISTRIBUTION, {len(per)} exemples, n={n})")
    print(f"  format parsable : {parse_ok*100:5.1f}%")
    print(f"  gem-only        : {gem*100:5.1f}%   (hasard 33%)")
    print(f"  precision / bit : {per_bit*100:5.1f}%   (hasard 50%)")
    print(f"  strict Option B : {strict*100:5.1f}%")
    print(f"  pass@G          : {passg}/{len(per)} ({passg/len(per)*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="cle MODELS ou id HF direct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = args.label or (f"{args.model}-sft" if args.adapter else args.model)

    examples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
            gold = gold_from_final(asst)
            if gold is not None:
                examples.append((r.get("id", str(len(examples))), user, gold))
    if args.limit:
        examples = examples[:args.limit]

    done = load_done(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    todo = [(e, s) for e, _, _ in examples for s in range(args.n_samples)
            if (label, e, s) not in done]
    print(f"Dataset : {len(examples)} exemples | label : {label} | N : {args.n_samples} | restantes : {len(todo)}")

    tok = model = None
    if todo and not args.dry_run:
        base_id = MODELS.get(args.model, args.model)
        print(f"Chargement {base_id}" + (f"  +adapter:{args.adapter}" if args.adapter else ""))
        tok, model = load_model(base_id, args.adapter)

    with open(args.out, "a", encoding="utf-8") as fh:
        for ex_id, user, gold in examples:
            for s in range(args.n_samples):
                if (label, ex_id, s) in done:
                    continue
                seed = seed_for(label, ex_id, s)
                if args.dry_run:
                    raw = dry_stub(gold, s)
                else:
                    raw = generate_once(tok, model, user, args.temperature,
                                        args.max_new_tokens, seed)
                g = grade(parse_final(raw), gold)
                rec = {
                    "model": label, "example_id": ex_id, "sample_idx": s, "seed": seed,
                    **g, "gold": gold, "raw": raw,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                append_record(fh, rec)
                done.add((label, ex_id, s))

    aggregate(args.out, label)


if __name__ == "__main__":
    main()
