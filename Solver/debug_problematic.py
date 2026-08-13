import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))

from evaluate_batch import PUZZLES_REGISTER
from solver import generate_all_worlds, is_world_valid

csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")

problem_ids = [81, 88, 58, 65, 70]

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row["ID"])
        if pid in problem_ids:
            print("=" * 80)
            print(f"ID {pid} | Sol attendue: {row['Sol']}")
            print(f"  Blue : {row['Blue']}")
            print(f"  White: {row['White']}")
            print(f"  Black: {row['Black']}")
            
            if pid in PUZZLES_REGISTER:
                puzzle = PUZZLES_REGISTER[pid]
                all_w = generate_all_worlds(puzzle)
                valid_w = [w for w in all_w if is_world_valid(puzzle, w)]
                print(f"  Mondes totaux: {len(all_w)} | Mondes valides: {len(valid_w)}")
                for w in valid_w:
                    print(f"    -> gem_box: {w.gem_box}, box_truths: {w.box_truths}")
