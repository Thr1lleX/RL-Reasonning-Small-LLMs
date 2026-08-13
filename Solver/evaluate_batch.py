import sys
import os
import csv
import json
from datetime import datetime
from typing import Dict, List

# Inclusion du chemin courant pour les imports
sys.path.append(os.path.dirname(__file__))

from models import Puzzle, BOXES
from predicates import (
    ContainsGems, BoxIsColor, NotPredicate, AndPredicate, OrPredicate,
    CountTrueBoxes, CountTrueStatements, CountFalseStatements,
    BoxIsTrue, NeighborContainsGems, BothNeighborsContainGems, NeighborIsTrue,
    GemsInBoxWithCondition, IsEmpty, OtherTwoBoxesAreEmpty, IsMiddle,
    ThisIsTheOnlyTrueStatement, OtherTwoBoxesAreTrue, OtherTwoBoxesAreFalse,
    AllBoxesAreEmpty, AllBoxesContainGems, OtherTwoBoxesAreColor,
    EmptyBoxesHaveTruth, OneOtherBoxIsFalse, OneOtherBoxIsTrue,
    StatementsAreEquallyTrue, GemsInOnlyTruthBox,
    OnlyOneNeighborIsFalseAndContainsGems, GemsInBothBoxes,
    ABoxWithAFalseStatementIsEmpty, GemsInOtherBoxWithTrueStatement,
    FalseBoxesAreBothTrue, OnlyOneBoxContainsGems, AboveStatementIsTrue,
    TopStatementOfEachBoxIsTrue, CountTrueStatementsOnBox, CountFalseStatementsOnBox,
    AllStatementsOnBoxAreTrue, AlwaysTrue, AlwaysFalse, ThisBoxStatementsAreEqual
)
from solver import generate_all_worlds, is_world_valid, solve_puzzle

SOL_MAPPING = {
    1: "BLUE",
    2: "WHITE",
    3: "BLACK"
}

def make_single_puzzle(blue_stmt, white_stmt, black_stmt) -> Puzzle:
    return Puzzle(box_statements={
        "BLUE": [blue_stmt],
        "WHITE": [white_stmt],
        "BLACK": [black_stmt]
    })

PUZZLES_REGISTER: Dict[int, Puzzle] = {
    # --- JALON 1 (53 Puzzles Single-Statement) ---
    1: make_single_puzzle(BoxIsColor("THIS", "WHITE"), BoxIsColor("THIS", "BLACK"), ContainsGems("BLUE")),
    3: make_single_puzzle(IsMiddle("THIS"), ContainsGems("WHITE"), IsMiddle("THIS")),
    6: make_single_puzzle(ContainsGems("BLACK"), AndPredicate(IsEmpty("THIS"), IsEmpty("BLUE")), AllBoxesAreEmpty()),
    7: make_single_puzzle(AllBoxesContainGems(), GemsInBothBoxes("BLUE", "BLACK"), ContainsGems("THIS")),
    8: make_single_puzzle(NotPredicate(ContainsGems("WHITE")), NotPredicate(ContainsGems("THIS")), ContainsGems("THIS")),
    11: make_single_puzzle(BoxIsColor("THIS", "BLUE"), BoxIsTrue("BLUE"), NotPredicate(ContainsGems("BLUE"))),
    12: make_single_puzzle(CountTrueStatements("==", 2), BoxIsTrue("BLUE"), GemsInBoxWithCondition(expected_truth=True)),
    14: make_single_puzzle(BoxIsColor("THIS", "BLACK"), BoxIsColor("THIS", "BLACK"), NotPredicate(IsEmpty("BLACK"))),
    15: make_single_puzzle(NeighborContainsGems("THIS"), BothNeighborsContainGems("THIS"), NeighborIsTrue("THIS")),
    16: make_single_puzzle(CountFalseStatements("==", 3), CountFalseStatements("==", 2), GemsInBoxWithCondition(expected_truth=False)),
    20: make_single_puzzle(OtherTwoBoxesAreTrue(), OtherTwoBoxesAreColor("BLUE"), OtherTwoBoxesAreEmpty()),
    21: make_single_puzzle(NotPredicate(ContainsGems("THIS")), BoxIsTrue("BLUE"), ContainsGems("THIS")),
    22: make_single_puzzle(GemsInBoxWithCondition(expected_truth=False), NotPredicate(BoxIsTrue("BLUE")), NotPredicate(BoxIsTrue("WHITE"))),
    23: make_single_puzzle(ContainsGems("THIS"), NotPredicate(ContainsGems("THIS")), OtherTwoBoxesAreTrue()),
    24: make_single_puzzle(GemsInBoxWithCondition(expected_truth=False), BoxIsTrue("BLUE"), ContainsGems("BLUE")),
    25: make_single_puzzle(NotPredicate(IsEmpty("BLACK")), BoxIsTrue("BLACK"), GemsInBoxWithCondition(expected_truth=False)),
    26: make_single_puzzle(NotPredicate(ContainsGems("THIS")), ContainsGems("THIS"), BoxIsTrue("WHITE")),
    27: make_single_puzzle(CountFalseStatements("==", 2), ThisIsTheOnlyTrueStatement(), IsEmpty("WHITE")),
    28: make_single_puzzle(IsEmpty("THIS"), ABoxWithAFalseStatementIsEmpty(), CountFalseStatements("==", 2)),
    29: make_single_puzzle(EmptyBoxesHaveTruth(expected_truth=True), BoxIsTrue("BLUE"), ContainsGems("THIS")),
    30: make_single_puzzle(OneOtherBoxIsFalse("THIS"), IsEmpty("THIS"), BoxIsTrue("BLUE")),
    31: make_single_puzzle(ContainsGems("BLACK"), NotPredicate(ContainsGems("THIS")), OtherTwoBoxesAreTrue()),
    32: make_single_puzzle(ContainsGems("WHITE"), OtherTwoBoxesAreTrue(), NotPredicate(ContainsGems("BLUE"))),
    33: make_single_puzzle(NotPredicate(GemsInBoxWithCondition(expected_truth=True)), StatementsAreEquallyTrue(BoxIsTrue("BLUE"), BoxIsTrue("THIS")), NotPredicate(GemsInBoxWithCondition(expected_truth=True))),
    34: make_single_puzzle(CountTrueStatements("==", 1), GemsInBoxWithCondition(expected_truth=False), CountTrueStatements("==", 2)),
    35: make_single_puzzle(ThisIsTheOnlyTrueStatement(), GemsInOtherBoxWithTrueStatement("THIS"), OneOtherBoxIsTrue()),
    36: make_single_puzzle(GemsInOnlyTruthBox(expected_truth=True), OneOtherBoxIsTrue(), GemsInOnlyTruthBox(expected_truth=False)),
    37: make_single_puzzle(CountFalseStatements("==", 1), CountTrueStatements("==", 1), GemsInBoxWithCondition(expected_truth=False)),
    38: make_single_puzzle(GemsInOnlyTruthBox(expected_truth=True), CountTrueStatements("==", 1), ContainsGems("BLUE")),
    41: make_single_puzzle(EmptyBoxesHaveTruth(expected_truth=True), FalseBoxesAreBothTrue(), ContainsGems("BLUE")),
    42: make_single_puzzle(BothNeighborsContainGems("THIS"), BothNeighborsContainGems("THIS"), BothNeighborsContainGems("THIS")),
    43: make_single_puzzle(CountTrueStatements("==", 2), OnlyOneBoxContainsGems(), IsEmpty("THIS")),
    47: make_single_puzzle(EmptyBoxesHaveTruth(expected_truth=True), ContainsGems("BLACK"), NotPredicate(IsEmpty("THIS"))),
    51: make_single_puzzle(IsEmpty("THIS"), GemsInBoxWithCondition(expected_truth=True), AndPredicate(NotPredicate(BoxIsTrue("THIS")), NotPredicate(BoxIsTrue("WHITE")))),
    52: make_single_puzzle(IsEmpty("WHITE"), NotPredicate(BoxIsTrue("BLUE")), EmptyBoxesHaveTruth(expected_truth=False)),
    55: make_single_puzzle(EmptyBoxesHaveTruth(expected_truth=True), EmptyBoxesHaveTruth(expected_truth=False), NotPredicate(IsEmpty("THIS"))),
    56: make_single_puzzle(AndPredicate(BoxIsTrue("THIS"), NotPredicate(BoxIsTrue("THIS"))), OrPredicate(ContainsGems("BLUE"), BoxIsTrue("BLUE")), StatementsAreEquallyTrue(BoxIsTrue("THIS"), BoxIsTrue("WHITE"))),
    59: make_single_puzzle(NotPredicate(ContainsGems("BLACK")), StatementsAreEquallyTrue(BoxIsTrue("THIS"), BoxIsTrue("BLACK")), ContainsGems("WHITE")),
    78: make_single_puzzle(OnlyOneNeighborIsFalseAndContainsGems(), OnlyOneNeighborIsFalseAndContainsGems(), OnlyOneNeighborIsFalseAndContainsGems()),
    79: make_single_puzzle(CountTrueStatements("==", 1), NeighborIsTrue("THIS"), NeighborContainsGems("THIS")),
    91: make_single_puzzle(BoxIsTrue("WHITE"), OtherTwoBoxesAreFalse(), OtherTwoBoxesAreEmpty()),
    95: make_single_puzzle(ContainsGems("BLACK"), ContainsGems("WHITE"), CountTrueStatements("==", 1)),
    96: make_single_puzzle(CountTrueStatements("==", 1), OnlyOneBoxContainsGems(), ContainsGems("WHITE")),
    97: make_single_puzzle(CountTrueStatements("==", 1), ContainsGems("BLACK"), ContainsGems("BLUE")),
    98: make_single_puzzle(NotPredicate(NeighborIsTrue("THIS")), NeighborIsTrue("THIS"), NeighborContainsGems("THIS")),
    99: make_single_puzzle(BoxIsTrue("BLACK"), GemsInBoxWithCondition(expected_truth=True), BoxIsTrue("BLUE")),
    100: make_single_puzzle(GemsInBoxWithCondition(expected_truth=False), GemsInBoxWithCondition(expected_truth=False), OneOtherBoxIsFalse()),
    102: make_single_puzzle(ContainsGems("WHITE"), GemsInBoxWithCondition(expected_truth=False), BoxIsTrue("WHITE")),
    103: Puzzle({
        "BLUE": [NeighborContainsGems("THIS")],
        "WHITE": [],
        "BLACK": [AlwaysFalse()]
    }),
    104: make_single_puzzle(ContainsGems("THIS"), NeighborContainsGems("THIS"), ContainsGems("BLUE")),
    105: make_single_puzzle(BoxIsColor("THIS", "BLUE"), GemsInBoxWithCondition(expected_truth=True), BoxIsColor("THIS", "BLACK")),
    107: make_single_puzzle(NotPredicate(BoxIsTrue("BLACK")), BoxIsTrue("BLACK"), GemsInBoxWithCondition(expected_truth=False)),
    108: make_single_puzzle(BoxIsColor("BLACK", "BLACK"), BoxIsTrue("BLACK"), IsEmpty("BLACK")),
    110: make_single_puzzle(GemsInBothBoxes("BLACK", "WHITE"), GemsInBothBoxes("BLUE", "BLACK"), AndPredicate(NotPredicate(ContainsGems("WHITE")), NotPredicate(ContainsGems("BLUE")))),

    # --- JALON 2 (13 Puzzles Multi-Statements In-Scope) ---
    58: Puzzle({
        "BLUE": [NotPredicate(ContainsGems("BLACK"))],
        "WHITE": [ContainsGems("BLUE")],
        "BLACK": [
            CountFalseStatementsOnBox("THIS", "==", 2),
            NotPredicate(BoxIsTrue("WHITE"))
        ]
    }),
    62: Puzzle({
        "BLUE": [ContainsGems("THIS"), AboveStatementIsTrue()],
        "WHITE": [NotPredicate(ContainsGems("THIS")), NotPredicate(AboveStatementIsTrue())],
        "BLACK": [TopStatementOfEachBoxIsTrue(), AboveStatementIsTrue()]
    }),
    63: Puzzle({
        "BLUE": [AllStatementsOnBoxAreTrue("THIS"), ContainsGems("THIS")],
        "WHITE": [NotPredicate(ContainsGems("BLUE")), NotPredicate(ContainsGems("BLACK"))],
        "BLACK": [GemsInBoxWithCondition(expected_truth=False), AllStatementsOnBoxAreTrue("BLUE")]
    }),
    64: Puzzle({
        "BLUE": [NotPredicate(ContainsGems("THIS")), NotPredicate(ContainsGems("WHITE"))],
        "WHITE": [NotPredicate(ContainsGems("THIS")), ContainsGems("BLUE")],
        "BLACK": [NotPredicate(ContainsGems("THIS")), ContainsGems("BLUE")]
    }),
    65: Puzzle({
        "BLUE": [
            ThisBoxStatementsAreEqual(),
            IsEmpty("THIS")
        ],
        "WHITE": [AlwaysTrue()],
        "BLACK": [
            AllStatementsOnBoxAreTrue("BLUE"),
            GemsInBoxWithCondition(expected_truth=True)
        ]
    }),

    66: Puzzle({
        "BLUE": [CountTrueStatementsOnBox("WHITE", "==", 1), CountTrueStatementsOnBox("BLACK", "==", 0)],
        "WHITE": [NotPredicate(ContainsGems("BLACK")), AboveStatementIsTrue()],
        "BLACK": [CountTrueStatementsOnBox("BLUE", "==", 1), ContainsGems("BLUE")]
    }),
    70: Puzzle({
        "BLUE": [
            CountTrueStatementsOnBox("WHITE", ">=", 1),
            CountTrueStatementsOnBox("BLACK", ">=", 1)
        ],
        "WHITE": [
            CountFalseStatementsOnBox("BLACK", ">=", 1),
            NotPredicate(ContainsGems("BLACK"))
        ],
        "BLACK": [
            CountFalseStatementsOnBox("BLUE", "==", 2),
            GemsInOnlyTruthBox(expected_truth=True) 
        ]
    }),

    71: Puzzle({
        "BLUE": [NotPredicate(IsEmpty("THIS")), NotPredicate(ContainsGems("BLACK"))],
        "WHITE": [BoxIsTrue("THIS"), NotPredicate(AboveStatementIsTrue())],
        "BLACK": [ContainsGems("BLACK"), ContainsGems("WHITE")]
    }),
    76: Puzzle({
        "BLUE": [AndPredicate(BoxIsTrue("BLACK"), ContainsGems("BLACK"))],
        "WHITE": [NotPredicate(BoxIsTrue("BLACK")), NotPredicate(AboveStatementIsTrue())],
        "BLACK": [OrPredicate(IsEmpty("BLUE"), BoxIsTrue("BLUE"))]
    }),
    80: Puzzle({
        "BLUE": [CountFalseStatementsOnBox("THIS", "==", 1), ContainsGems("WHITE")],
        "WHITE": [CountTrueStatementsOnBox("THIS", "==", 1), GemsInBoxWithCondition(expected_truth=True)],
        "BLACK": [CountFalseStatements("==", 3)]
    }),
    81: Puzzle({
        "BLUE": [
            CountTrueStatementsOnBox("THIS", ">=", 2),
            CountTrueStatementsOnBox("BLACK", ">=", 2),
            ContainsGems("THIS")
        ],
        "WHITE": [
            CountTrueStatementsOnBox("BLUE", "==", 0),
            CountFalseStatementsOnBox("BLACK", "==", 0),
            ContainsGems("THIS")
        ],
        "BLACK": [
            AlwaysTrue(),
            AlwaysTrue(),
            ContainsGems("THIS")
        ]
    }),
    88: Puzzle({
        "BLUE": [
            CountTrueStatementsOnBox("WHITE", "==", 1),
            ContainsGems("BLACK"),
            IsEmpty("THIS")
        ],
        "WHITE": [
            CountTrueStatementsOnBox("BLUE", "==", 1),
            IsEmpty("BLACK"),
            GemsInBoxWithCondition(expected_truth=True)
        ],
        "BLACK": [
            CountTrueStatementsOnBox("THIS", "==", 1),
            ContainsGems("BLUE"),
            IsEmpty("WHITE")
        ]
    })
}

def evaluate_batch(csv_path: str):
    ground_truths: Dict[int, str] = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    p_id = int(row["ID"])
                    sol_num = int(row["Sol"])
                    ground_truths[p_id] = SOL_MAPPING.get(sol_num, "UNKNOWN")
                except (ValueError, KeyError):
                    continue

    print("=" * 125)
    print(" BENCHMARK TOTAL RECONCILIE ET STRICT (OPTION B : TOUS LES 66 PUZZLES IN-SCOPE DEPUIS LE CSV)")
    print("=" * 125)
    
    total_tested = 0
    strict_gold_count = 0
    multi_truths_count = 0
    ambiguous_gem_count = 0
    paradox_count = 0

    header = f"{'ID':<5} | {'Attendu':<8} | {'Prédit':<10} | {'Statut Oracle':<18} | {'Unicité Monde':<15} | {'Combinaisons Valides (Boîte & truths)'}"
    print(header)
    print("-" * 125)

    detailed_results = []

    for puzzle_id, puzzle in sorted(PUZZLES_REGISTER.items()):
        total_tested += 1
        expected_gem_box = ground_truths.get(puzzle_id, "N/A")
        
        all_worlds = generate_all_worlds(puzzle)
        valid_worlds = [w for w in all_worlds if is_world_valid(puzzle, w)]
        gem_boxes = set(w.gem_box for w in valid_worlds)

        # Évaluation stricte du statut Oracle (Option B)
        if len(valid_worlds) == 1:
            predicted_gem_box = valid_worlds[0].gem_box
            if predicted_gem_box == expected_gem_box:
                oracle_status = "GOLD_STRICT_OK"
                strict_gold_count += 1
            else:
                oracle_status = "ERR_GEM_MISMATCH"
            unicity_str = f"Unique (1/{len(all_worlds)})"
        elif len(gem_boxes) == 1:
            predicted_gem_box = list(gem_boxes)[0]
            oracle_status = "EXCLUDED_MULTI_TRUTHS"
            multi_truths_count += 1
            unicity_str = f"Multi ({len(valid_worlds)}/{len(all_worlds)})"
        elif len(gem_boxes) == 0:
            predicted_gem_box = "PARADOX"
            oracle_status = "EXCLUDED_PARADOX"
            paradox_count += 1
            unicity_str = f"Aucun (0/{len(all_worlds)})"
        else:
            predicted_gem_box = "AMBIGUOUS"
            oracle_status = "EXCLUDED_AMBIGUOUS"
            ambiguous_gem_count += 1
            unicity_str = f"Multi ({len(valid_worlds)}/{len(all_worlds)})"

        combos = [f"({w.gem_box}: box_truths={w.box_truths})" for w in valid_worlds]
        combos_str = ", ".join(combos)

        print(f"{puzzle_id:<5} | {expected_gem_box:<8} | {predicted_gem_box:<10} | {oracle_status:<18} | {unicity_str:<15} | {combos_str}")

        detailed_results.append({
            "puzzle_id": puzzle_id,
            "expected": expected_gem_box,
            "predicted": predicted_gem_box,
            "oracle_status": oracle_status,
            "is_strict_gold": (oracle_status == "GOLD_STRICT_OK"),
            "unicity_status": unicity_str,
            "valid_worlds_count": len(valid_worlds),
            "valid_worlds": [{"gem_box": w.gem_box, "box_truths": {b: list(t) for b, t in w.box_truths.items()}} for w in valid_worlds]
        })

    strict_gold_percent = (strict_gold_count / total_tested * 100) if total_tested > 0 else 0
    print("=" * 125)
    print(f" Total puzzles in-scope évalués            : {total_tested} / 66")
    print(f" Oracle Strict Gold (Monde Unique + Gem OK): {strict_gold_count} / {total_tested} ({strict_gold_percent:.1f}%)")
    print(f" Puzzles Exclus (Vérités Multiples)       : {multi_truths_count} / {total_tested}")
    print(f" Puzzles Exclus (Gemme Ambiguë / Symétrie) : {ambiguous_gem_count} / {total_tested}")
    print(f" Puzzles Exclus (Paradoxe / 0 mondes)      : {paradox_count} / {total_tested}")
    print(f" Total Décompte Exclusions                 : {strict_gold_count} (Gold) + {multi_truths_count} (Multi) + {ambiguous_gem_count} (Ambig) + {paradox_count} (Paradox) = {strict_gold_count + multi_truths_count + ambiguous_gem_count + paradox_count} / {total_tested}")
    print("=" * 125)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "benchmark_full_66_results.json")
    json_payload = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_mode": "Full Benchmark 66 Puzzles In-Scope (Jalon 1 + Jalon 2)",
        "summary": {
            "total_tested": total_tested,
            "strict_gold_count": strict_gold_count,
            "strict_gold_percent": round(strict_gold_percent, 1),
            "multi_truths_count": multi_truths_count,
            "ambiguous_gem_count": ambiguous_gem_count,
            "paradox_count": paradox_count
        },
        "results": detailed_results
    }
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(results_dir, "benchmark_full_66_report.md")
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# Rapport Benchmark Complet Réconcilié (66 Puzzles In-Scope)\n\n")
        f.write(f"**Date d'exécution** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Synthèse Exhaustive (Option B Strict)\n\n")
        f.write(f"- **Total Puzzles In-Scope** : {total_tested} / 66\n")
        f.write(f"- **Oracle Strict Gold Standard** : **{strict_gold_percent:.1f}%** ({strict_gold_count} / {total_tested})\n")
        f.write(f"- **Exclus (Vérités Multiples)** : {multi_truths_count} / {total_tested}\n")
        f.write(f"- **Exclus (Gemme Ambiguë / Symétrie)** : {ambiguous_gem_count} / {total_tested}\n")
        f.write(f"- **Exclus (Paradoxe / 0 Mondes)** : {paradox_count} / {total_tested}\n")
        f.write(f"- **Vérification du Décompte** : {strict_gold_count} + {multi_truths_count} + {ambiguous_gem_count} + {paradox_count} = {total_tested}\n\n")
        f.write("## Tableau de Qualification Complet\n\n")
        f.write("| ID | Attendu | Prédit | Statut Oracle | Unicité | Mondes Valides |\n")
        f.write("|---|---|---|---|---|---|\n")
        for item in detailed_results:
            f.write(f"| {item['puzzle_id']} | {item['expected']} | {item['predicted']} | {item['oracle_status']} | {item['unicity_status']} | {item['valid_worlds_count']} mondes |\n")
    
    print(f"\n[INFO] Résultats du benchmark réconcilié enregistrés avec succès dans :")
    print(f"       -> JSON : {json_path}")
    print(f"       -> MD   : {md_path}")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coverage_worksheetv3.csv")
    evaluate_batch(csv_file)
