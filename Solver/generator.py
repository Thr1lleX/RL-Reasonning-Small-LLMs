from typing import List, Callable,Dict
import random
import time
from models import Puzzle, BOXES
from predicates import (
    Predicate, ContainsGems, BoxIsColor, IsEmpty, CountTrueStatements,
    CountFalseStatements, OtherTwoBoxesAreEmpty, IsMiddle, AlwaysTrue, AlwaysFalse,
    BoxIsTrue, AboveStatementIsTrue, TopStatementOfEachBoxIsTrue,
    AllStatementsOnBoxAreTrue, NeighborContainsGems, BothNeighborsContainGems,
    NeighborIsTrue, CountTrueStatementsOnBox, CountFalseStatementsOnBox,
    GemsInOnlyTruthBox, GemsInBoxWithCondition, NotPredicate, OrPredicate, AndPredicate
)
from solver import solve_puzzle, generate_all_worlds, is_world_valid


# Ensembles de valeurs pour les tirages
TARGET_BOXES = ["THIS", "BLUE", "WHITE", "BLACK"]
COLORS = ["BLUE", "WHITE", "BLACK"]
OPERATORS = ["==", ">=", "<="]


# 1. Banque de Prédicats Auto-Contenus (Self-Contained)
SELF_CONTAINED_FACTORIES: List[Callable[[], Predicate]] = [
    lambda: ContainsGems(random.choice(TARGET_BOXES)),
    lambda: IsEmpty(random.choice(TARGET_BOXES)),
    lambda: BoxIsColor(random.choice(TARGET_BOXES), random.choice(COLORS)),
    lambda: OtherTwoBoxesAreEmpty(random.choice(TARGET_BOXES)),
    lambda: NeighborContainsGems(random.choice(TARGET_BOXES)),
    lambda: BothNeighborsContainGems(random.choice(TARGET_BOXES)),
    lambda: IsMiddle(random.choice(TARGET_BOXES))
    
]
# 2. Banque de Prédicats de Référentiels / Relatifs (Relational)
RELATIONAL_FACTORIES: List[Callable[[], Predicate]] = [
    lambda: BoxIsTrue(random.choice(TARGET_BOXES)),
    lambda: AboveStatementIsTrue(),
    lambda: TopStatementOfEachBoxIsTrue(),
    lambda: AllStatementsOnBoxAreTrue(random.choice(TARGET_BOXES)),
    lambda: NeighborIsTrue(random.choice(TARGET_BOXES)),
    lambda: CountTrueStatementsOnBox(random.choice(TARGET_BOXES), random.choice(OPERATORS), random.randint(1, 2)),
    lambda: CountFalseStatementsOnBox(random.choice(TARGET_BOXES), random.choice(OPERATORS), random.randint(1, 2)),
    lambda: GemsInOnlyTruthBox(random.choice([True, False])),
    lambda: CountTrueStatements(random.choice(OPERATORS), random.randint(1, 3)),
    lambda: CountFalseStatements(random.choice(OPERATORS), random.randint(1, 3)),
    lambda: GemsInBoxWithCondition(random.choice([True, False]))
]


def apply_ast_transform(pred, relational_ratio,ast_prob, current_depth: int = 1, max_depth: int = 2):
    
    if current_depth >= max_depth or random.random() >= ast_prob:
        return pred
    else:
        r = random.random()
        if r < 0.33:
            sub_pred = apply_ast_transform(pred, relational_ratio, ast_prob, current_depth + 1, max_depth)
            return NotPredicate(sub_pred)
        elif r < 0.66:
            pred2 = random.choice(SELF_CONTAINED_FACTORIES)() if random.random() > relational_ratio else random.choice(RELATIONAL_FACTORIES)()
            sub1 = apply_ast_transform(pred, relational_ratio, ast_prob, current_depth + 1, max_depth)
            sub2 = apply_ast_transform(pred2, relational_ratio, ast_prob, current_depth + 1, max_depth)
            return AndPredicate(sub1, sub2)
        else:
            pred2 = random.choice(SELF_CONTAINED_FACTORIES)() if random.random() > relational_ratio else random.choice(RELATIONAL_FACTORIES)()
            sub1 = apply_ast_transform(pred, relational_ratio, ast_prob, current_depth + 1, max_depth)
            sub2 = apply_ast_transform(pred2, relational_ratio, ast_prob, current_depth + 1, max_depth)
            return OrPredicate(sub1, sub2)
    return pred

def generate_random_puzzle(total_statements : int, relational_ratio: float,ast_prob: float, max_depth : int = 2 ) -> Puzzle:
    assert total_statements >= 3, f"total_statements doit être >= 3 (valeur reçue: {total_statements})"
    box_statements = {}
    # 1. Répartition de base équilibrée
    q = total_statements // 3
    r = total_statements % 3

    counts = {"BLUE": q, "WHITE": q, "BLACK": q}

    # 2. Distribuer le reste R au hasard sur R boîtes différentes
    if r > 0:
        extra_boxes = random.sample(BOXES, r)  # Choisit R boîtes uniques sans remise
        for box in extra_boxes:
            counts[box] += 1

    for box in BOXES:
        statements = []
        for _ in range(counts[box]):
            if random.random() < relational_ratio:
                factory = random.choice(RELATIONAL_FACTORIES)
            else:
                factory = random.choice(SELF_CONTAINED_FACTORIES)
            pred = factory()
            #AST PREDICATES
            pred = apply_ast_transform(pred, relational_ratio, ast_prob, current_depth=1, max_depth=max_depth)
            statements.append(pred)
        box_statements[box] = statements

    return Puzzle(box_statements = box_statements)


def puzzle_signature(puzzle: Puzzle) -> str:
    return repr(puzzle)


def benchmark_generator(k : int, N : int, relational_ratio: float, ast_prob: float, seed:int = None, max_depth : int = 2):
    t_0 = time.time()
    results = {
    "GOLD_STRICT_OK": 0,
    "EXCLUDED_MULTI_TRUTHS": 0,
    "EXCLUDED_AMBIGUOUS": 0,
    "EXCLUDED_PARADOX": 0
    }
    if seed is not None:
        random.seed(seed)

    for i in range (k):
        puzzle = generate_random_puzzle(total_statements=N, relational_ratio=relational_ratio, ast_prob=ast_prob, max_depth=max_depth)
        all_worlds = generate_all_worlds(puzzle)
        valid_worlds = [w for w in all_worlds if is_world_valid(puzzle, w)]
        gem_boxes = set(w.gem_box for w in valid_worlds)

        if len(valid_worlds) == 1:
            results["GOLD_STRICT_OK"] += 1
        elif len(gem_boxes) == 1:
            results["EXCLUDED_MULTI_TRUTHS"] += 1
        elif len(gem_boxes) > 1:
            results["EXCLUDED_AMBIGUOUS"] += 1
        else:
            results["EXCLUDED_PARADOX"] += 1
    results = f" {k} PUZZLES\n {N} STATEMENTS\n RELATIONAL RATIO = {relational_ratio}\n AST_PROBABILITY = {ast_prob}\n Temps = {time.time() - t_0:.3f} s \n GOLD_STRICT_OK: {results['GOLD_STRICT_OK']}\n EXCLUDED_MULTI_TRUTHS: {results['EXCLUDED_MULTI_TRUTHS']}\n EXCLUDED_AMBIGUOUS: {results['EXCLUDED_AMBIGUOUS']}\n EXCLUDED_PARADOX: {results['EXCLUDED_PARADOX']}"
    return results

def generate_gold_puzzle(total_statements: int, relational_ratio: float, 
                        ast_prob: float, max_attempts: int = 1000, max_depth : int = 2, 
                        seed : int = None, seen_signatures: set = None) -> Puzzle:

    if seen_signatures is None:
        seen_signatures = set()

    if seed is not None:
        random.seed(seed)
    for _ in range(max_attempts):
        puzzle = generate_random_puzzle(total_statements, relational_ratio, ast_prob, max_depth=max_depth)
        sig = puzzle_signature(puzzle)
        if sig in seen_signatures:
            continue 
        all_worlds = generate_all_worlds(puzzle)
        valid_worlds = [w for w in all_worlds if is_world_valid(puzzle, w)]
        if len(valid_worlds) == 1:
            seen_signatures.add(sig)
            return puzzle
    raise RuntimeError("Impossible de générer un puzzle Gold Strict dans la limite de tentatives.")



if __name__ == "__main__":
    # 1. Générer un puzzle Or Strict (N=5 statements)
    puzzle = generate_gold_puzzle(total_statements=3, relational_ratio=0, ast_prob=0, max_depth=3)
    
    # 2. Récupérer le monde valide unique (l'Oracle nous donne la boîte gagnante)
    all_worlds = generate_all_worlds(puzzle)
    gold_world = [w for w in all_worlds if is_world_valid(puzzle, w)][0]
    
    # 3. Afficher le puzzle formé sous sa forme finale en anglais
    print("==========================================")
    print("      PUZZLE BLUE PRINCE (SYNTHÉTIQUE)    ")
    print("==========================================")
    
    for box, statements in puzzle.box_statements.items():
        print(f"\n--- {box} BOX ---")
        if not statements:
            print("  (No statements)")
        for idx, stmt in enumerate(statements, 1):
            print(f"  {idx}. {stmt.to_english()}")
            
    print("\n==========================================")
    print(f" SOLUTION EXPLICITE (ORACLE) : {gold_world.gem_box}")
    print("==========================================")

