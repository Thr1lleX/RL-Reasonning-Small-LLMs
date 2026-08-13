import itertools
from typing import List
from models import World, Puzzle, BOXES

def generate_all_worlds(puzzle: Puzzle) -> List[World]:
    worlds = []
    # vérités possibles par boîte
    blue_truths = list(itertools.product([True, False], repeat=len(puzzle.box_statements["BLUE"])))
    white_truths = list(itertools.product([True, False], repeat=len(puzzle.box_statements["WHITE"])))
    black_truths = list(itertools.product([True, False], repeat=len(puzzle.box_statements["BLACK"])))
    
    for gem_box in BOXES:
        for b_t in blue_truths:
            for w_t in white_truths:
                for k_t in black_truths:
                    box_truths = {"BLUE": b_t, "WHITE": w_t, "BLACK": k_t}
                    worlds.append(World(gem_box=gem_box, box_truths=box_truths))
    return worlds


def is_world_valid(puzzle: Puzzle, world: World) -> bool:
    # 1. Point fixe sur chaque statement de chaque boîte
    for box in BOXES:
        stmts = puzzle.box_statements[box]
        truths = world.box_truths[box]
        for i, stmt in enumerate(stmts):
            expected = truths[i]
            actual = stmt.evaluate(world, current_box=box, statement_index=i)
            if actual != expected:
                return False

    # 2. Invariant Blue Prince (agrégation au niveau boîte)
    has_completely_true = any(
        len(world.box_truths[b]) >= 1 and all(world.box_truths[b])
        for b in BOXES
    )
    has_completely_false = any(
        len(world.box_truths[b]) >= 1 and all(not t for t in world.box_truths[b])
        for b in BOXES
    )

    return has_completely_true and has_completely_false

def solve_puzzle(puzzle: Puzzle):
    all_worlds = generate_all_worlds(puzzle)
    valid_worlds = [w for w in all_worlds if is_world_valid(puzzle, w)]
    
    # On récupère l'ensemble des boîtes aux gemmes possibles
    gem_boxes = set(w.gem_box for w in valid_worlds)
    
    if len(gem_boxes) == 1:
        # Solution unique pour la gemme
        gem_box = list(gem_boxes)[0]
        return f"SOLVABLE: Gems in {gem_box}"
    elif len(gem_boxes) == 0:
        return "PARADOX"
    return "AMBIGUOUS"