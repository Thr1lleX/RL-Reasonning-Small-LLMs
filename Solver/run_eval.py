"""
Harnais d'evaluation zero-shot des LLM sur les 57 puzzles GOLD.
- Reprise entre sessions : append-only JSONL, on saute les (model, puzzle, sample) deja faits.
- torch/transformers importes en lazy : --dry-run teste toute la tuyauterie sans GPU.

Usage :
  python Solver/run_eval.py --model all --n-samples 16 --temperature 0.7
  python Solver/run_eval.py --dry-run            # test a sec (aucun GPU requis)
  (Ctrl-C a tout moment, relancer la meme commande -> reprend.)
"""
import os
import sys
import json
import zlib
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from eval_lib import build_oracle, load_statement_texts, build_prompt, parse_final, grade

MODELS = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-math-1.5b-instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "eval_runs.jsonl")


def seed_for(model_key, pid, s):
    return zlib.crc32(f"{model_key}|{pid}|{s}".encode()) & 0x7FFFFFFF


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
                    done.add((r["model"], r["puzzle_id"], r["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_record(fh, rec):
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


# --------------------------------------------------------------------------
# Backend modele (lazy) — non importe en dry-run
# --------------------------------------------------------------------------

def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return tok, model


def generate_once(tok, model, prompt, temperature, max_new_tokens, seed):
    import torch
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    encoded = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **encoded, do_sample=True, temperature=temperature, top_p=0.95,
            max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def dry_run_output(pid, s, oracle):
    """Stub deterministe pour tester la chaine sans modele.
    - s % 3 == 0 : reponse correcte
    - s % 3 == 1 : bonne boite, verites fausses
    - s % 3 == 2 : format casse (non parsable)
    """
    o = oracle[pid]
    prefix = {"BLUE": "B", "WHITE": "W", "BLACK": "K"}
    def fmt(bt):
        segs = []
        for b in ["BLUE", "WHITE", "BLACK"]:
            for i, x in enumerate(bt[b], 1):
                segs.append(f"{prefix[b]}{i}={'T' if x else 'F'}")
        return "; ".join(segs)
    if s % 3 == 2:
        return "I think the answer is the white box but I'm not sure."
    if s % 3 == 0:
        return f"Reasoning...\nFINAL: gem={o['gem']}; {fmt(o['box_truths'])}"
    flipped = {b: [not x for x in o['box_truths'][b]] for b in o['box_truths']}
    return f"Reasoning...\nFINAL: gem={o['gem']}; {fmt(flipped)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="cle de MODELS ou 'all'")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="ne traiter que les N premiers puzzles GOLD (sonde)")
    ap.add_argument("--puzzles", default=None,
                    help="ids separes par des virgules, ex: 1,8,63 (sonde ciblee)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_keys = list(MODELS) if args.model == "all" else [args.model]
    for mk in model_keys:
        if mk not in MODELS:
            sys.exit(f"Modele inconnu : {mk}. Choix : {list(MODELS)} ou 'all'")

    oracle = build_oracle()
    texts = load_statement_texts()
    gold_ids = sorted(oracle)

    if args.puzzles:
        wanted = {int(x) for x in args.puzzles.split(",") if x.strip()}
        missing = wanted - set(gold_ids)
        if missing:
            print(f"[!] ids ignores (absents du GOLD set) : {sorted(missing)}")
        gold_ids = [pid for pid in gold_ids if pid in wanted]
    elif args.limit:
        gold_ids = gold_ids[:args.limit]
    if not gold_ids:
        sys.exit("Aucun puzzle a traiter apres filtrage.")

    done = load_done(args.out)

    total_units = len(model_keys) * len(gold_ids) * args.n_samples
    print(f"GOLD puzzles : {len(gold_ids)} | modeles : {model_keys} | N : {args.n_samples}")
    print(f"Unites totales : {total_units} | deja faites : {len(done)} | restantes : {total_units - len([1 for mk in model_keys for pid in gold_ids for s in range(args.n_samples) if (mk,pid,s) in done])}")
    print(f"Sortie : {args.out}" + ("  [DRY-RUN]" if args.dry_run else ""))

    with open(args.out, "a", encoding="utf-8") as fh:
        for mk in model_keys:
            todo = [(pid, s) for pid in gold_ids for s in range(args.n_samples)
                    if (mk, pid, s) not in done]
            if not todo:
                print(f"[{mk}] deja complet, saute.")
                continue

            tok = model = None
            if not args.dry_run:
                print(f"[{mk}] chargement du modele {MODELS[mk]} ...")
                tok, model = load_model(MODELS[mk])

            print(f"[{mk}] {len(todo)} generations a faire.")
            n_done = 0
            for pid in gold_ids:
                prompt = build_prompt(pid, texts)
                for s in range(args.n_samples):
                    if (mk, pid, s) in done:
                        continue
                    seed = seed_for(mk, pid, s)
                    if args.dry_run:
                        raw = dry_run_output(pid, s, oracle)
                    else:
                        raw = generate_once(tok, model, prompt, args.temperature,
                                            args.max_new_tokens, seed)
                    g = grade(parse_final(raw), oracle[pid])
                    rec = {
                        "model": mk, "puzzle_id": pid, "sample_idx": s, "seed": seed,
                        "temperature": args.temperature,
                        **g,
                        "raw": raw,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    }
                    append_record(fh, rec)
                    done.add((mk, pid, s))
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"  [{mk}] {n_done}/{len(todo)}")

            if not args.dry_run:
                import torch
                del model
                torch.cuda.empty_cache()
            print(f"[{mk}] termine ({n_done} nouvelles generations).")

    print("Fini.")


if __name__ == "__main__":
    main()
