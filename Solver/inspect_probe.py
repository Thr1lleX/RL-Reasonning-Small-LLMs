"""Lecture lisible d'un fichier de resultats d'eval (probe/smoke/run).
Usage : python Solver/inspect_probe.py --in Solver/results/probe_r1.jsonl [--tail 400]
"""
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--tail", type=int, default=0, help="afficher les N derniers chars du raw")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
print(f"n = {len(rows)}\n")
hdr = f"{'id':>4} | parse | gem | truths | count | think_end | boxed | chars"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda r: (r["puzzle_id"], r["sample_idx"])):
    raw = r.get("raw", "")
    def b(x):
        return {True: "Y", False: "-", None: "?"}.get(x, "?")
    print(f"{r['puzzle_id']:>4} |   {b(r['parse_ok'])}   |  {b(r['gem_correct'])}  |"
          f"   {b(r['truths_correct'])}    |   {b(r.get('count_ok'))}   |"
          f"     {b('</think>' in raw)}     |   {b('boxed' in raw)}   | {len(raw)}")
    if args.tail:
        print("     ...", repr(raw[-args.tail:]), "\n")

n = len(rows)
print(f"\nparse_ok      : {sum(r['parse_ok'] for r in rows)}/{n}")
print(f"think ferme   : {sum('</think>' in r.get('raw','') for r in rows)}/{n}")
print(f"boxed present : {sum('boxed' in r.get('raw','') for r in rows)}/{n}")
print(f"chars med/max : {sorted(len(r.get('raw','')) for r in rows)[n//2]} / {max(len(r.get('raw','')) for r in rows)}")
