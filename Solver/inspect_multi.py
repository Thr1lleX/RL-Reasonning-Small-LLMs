import csv
import os

multi_ids = {49, 58, 62, 63, 64, 65, 66, 70, 71, 76, 80, 93, 106}
csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row["ID"])
        if pid in multi_ids:
            print(f"ID {pid:<3} (n={row['n_stmts']}) | Sol: {row['Sol']}")
            print(f"  Blue : {row['Blue']}")
            print(f"  White: {row['White']}")
            print(f"  Black: {row['Black']}")
