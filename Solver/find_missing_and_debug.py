import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))

from evaluate_batch import PUZZLES_REGISTER
from solver import generate_all_worlds, is_world_valid, solve_puzzle

csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")

all_in_scope = []
with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("in_scope") == "Y":
            all_in_scope.append(int(row["ID"]))

print(f"Total in-scope puzzles in CSV: {len(all_in_scope)}")
print(f"Total registered puzzles in PUZZLES_REGISTER: {len(PUZZLES_REGISTER)}")

missing = set(all_in_scope) - set(PUZZLES_REGISTER.keys())
print(f"Missing puzzle ID(s): {missing}")

for pid in missing:
    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["ID"]) == pid:
                print(f"\nMissing ID {pid}:")
                print(f"  Blue : {row['Blue']}")
                print(f"  White: {row['White']}")
                print(f"  Black: {row['Black']}")
