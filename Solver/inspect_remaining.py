import csv
import os

already_done = {1, 3, 6, 7, 8, 11, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 97}
csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("in_scope") == "Y" and row.get("n_stmts") == "3":
            pid = int(row["ID"])
            if pid not in already_done:
                print(f"ID {pid:<3} | Blue: {row['Blue']:<45} | White: {row['White']:<45} | Black: {row['Black']}")
