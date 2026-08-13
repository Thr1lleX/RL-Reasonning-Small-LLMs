import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))

from models import Puzzle, BOXES
from evaluate_batch import PUZZLES_REGISTER, SOL_MAPPING
from solver import generate_all_worlds, is_world_valid

failed_ids = [7, 28, 35, 41, 42, 43, 78, 96]
csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")

ground_truths = {}
statement_texts = {}

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row["ID"])
        if pid in failed_ids:
            ground_truths[pid] = SOL_MAPPING.get(int(row["Sol"]), "UNKNOWN")
            statement_texts[pid] = (row["Blue"], row["White"], row["Black"])

all_worlds = generate_all_worlds()

print("=" * 100)
print(" DIAGNOSTIC DE RESOLUTION POUR LES 8 PUZZLES EN ECHEC / AMBIGUS")
print("=" * 100)

for pid in failed_ids:
    expected = ground_truths[pid]
    texts = statement_texts[pid]
    puzzle = PUZZLES_REGISTER[pid]
    
    valid_worlds = [w for w in all_worlds if is_world_valid(puzzle, w)]
    gem_boxes = set(w.gem_box for w in valid_worlds)
    
    print(f"\n--- PUZZLE ID {pid} (Attendu: {expected}) ---")
    print(f"  Blue : \"{texts[0]}\"")
    print(f"  White: \"{texts[1]}\"")
    print(f"  Black: \"{texts[2]}\"")
    print(f"  Mondes valides trouvés ({len(valid_worlds)}) :")
    for w in valid_worlds:
        print(f"    -> gem_box={w.gem_box:<6} truths={w.truths}")
