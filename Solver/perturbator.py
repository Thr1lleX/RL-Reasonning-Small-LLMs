from models import World
import random
import time
from typing import List, Dict, Set, Optional, Tuple
import re
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
from generator import generate_gold_puzzle, puzzle_signature, generate_random_puzzle, SELF_CONTAINED_FACTORIES, RELATIONAL_FACTORIES


REWARDS = ["COIN","RUBY","SAPPHIRE","EMERALD","TALISMAN","AMULET","RUNE",
            "POTION","CRYSTAL","KEY","ARTIFACT","RELIC","TREASURE","MEDAL","PEARL","ORB","CROWN","RING"]            
REWARDS_PLURAL = ["COINS","RUBIES","SAPPHIRES","EMERALDS","TALISMANS","AMULETS",
                    "RUNES","POTIONS","CRYSTALS","KEYS","ARTIFACTS","RELICS","TREASURES","MEDALS","PEARLS",
                    "ORBS","CROWNS","RINGS"]

CONTAINER_SING = [ "CHEST", "CRATE", "BAG", "TRUNK", "VAULT","POUCH"]
CONTAINER_PLURAL = [ "CHESTS", "CRATES", "BAGS", "TRUNKS", "VAULTS","POUCHES"]


COLORS = ["CRIMSON","VERMILION","RUBY","CARMINE","ORANGE","AMBER","GOLDEN","YELLOW","LIME",
        "EMERALD","JADE","MINT","TEAL","CYAN","CERULEAN","COBALT","INDIGO","VIOLET","PURPLE","MAGENTA",
        "AMETHYST","LAVENDER","MAUVE","PINK","ROSE","SILVER","IVORY","OBSIDIAN","CHARCOAL"]

def mutate_predicate(pred: Predicate) -> Optional[Predicate]:
    # --- Cas 1 : Nœuds logiques AST ---
    if isinstance(pred, NotPredicate):
        if random.random() < 0.5:
            return pred.sub_predicate
        else:
            sub = mutate_predicate(pred.sub_predicate)
            return NotPredicate(sub) if sub is not None else None

    elif isinstance(pred, AndPredicate):
        r = random.random()
        if r < 0.33:
            return OrPredicate(pred.sub_predicate, pred.sub_predicate2)
        elif r < 0.66:
            sub = mutate_predicate(pred.sub_predicate)
            return AndPredicate(sub, pred.sub_predicate2) if sub is not None else None
        else:
            sub = mutate_predicate(pred.sub_predicate2)
            return AndPredicate(pred.sub_predicate, sub) if sub is not None else None

    elif isinstance(pred, OrPredicate):
        r = random.random()
        if r < 0.33:
            return AndPredicate(pred.sub_predicate, pred.sub_predicate2)
        elif r < 0.66:
            sub = mutate_predicate(pred.sub_predicate)
            return OrPredicate(sub, pred.sub_predicate2) if sub is not None else None
        else:
            sub = mutate_predicate(pred.sub_predicate2)
            return OrPredicate(pred.sub_predicate, sub) if sub is not None else None

    # --- Cas 2 : Prédicats Atomiques (Collecte des attributs mutables) ---
    mutable_attrs = []
    if hasattr(pred, "target_box"):
        mutable_attrs.append("target_box")
    if hasattr(pred, "color"):
        mutable_attrs.append("color")
    if hasattr(pred, "op"):
        mutable_attrs.append("op")
    if hasattr(pred, "num"):
        mutable_attrs.append("num")
    if hasattr(pred, "expected_truth"):
        mutable_attrs.append("expected_truth")

    # Si le prédicat n'a aucun attribut modifiable (ex: IsMiddle), on annule cette tentative
    if not mutable_attrs:
        return None

    # Tirage équitable d'UN SEUL attribut à muter
    chosen_attr = random.choice(mutable_attrs)
    kwargs = dict(pred.__dict__)

    if chosen_attr == "target_box":
        other_boxes = [b for b in ["THIS", "BLUE", "WHITE", "BLACK"] if b != pred.target_box]
        kwargs["target_box"] = random.choice(other_boxes)
    elif chosen_attr == "color":
        other_colors = [c for c in ["BLUE", "WHITE", "BLACK"] if c != pred.color]
        kwargs["color"] = random.choice(other_colors)
    elif chosen_attr == "op":
        other_ops = [op for op in ["==", ">=", "<="] if op != pred.op]
        kwargs["op"] = random.choice(other_ops)
    elif chosen_attr == "num":
        kwargs["num"] = random.randint(1, 3)
    elif chosen_attr == "expected_truth":
        kwargs["expected_truth"] = not pred.expected_truth

    return type(pred)(**kwargs)



def mutate_gold_puzzle(puzzle: Puzzle, train_signatures: Optional[Set[str]] = None, max_attempts: int = 200) -> Puzzle:
    if train_signatures is None:
        train_signatures = set()

    # 1. Obtenir la gemme du puzzle d'origine via l'Oracle
    orig_worlds = [w for w in generate_all_worlds(puzzle) if is_world_valid(puzzle, w)]
    if not orig_worlds:
        raise ValueError("Le puzzle d'origine n'est pas un puzzle valide.")
    orig_gem = orig_worlds[0].gem_box

    # 2. Boucle de tentative de mutation
    for _ in range(max_attempts):
        # Choisir une boîte non vide au hasard
        eligible_boxes = [b for b in BOXES if len(puzzle.box_statements[b]) > 0]
        mutation_box = random.choice(eligible_boxes)

        # Choisir un statement au hasard à l'index de la boîte
        idx = random.randint(0, len(puzzle.box_statements[mutation_box]) - 1)
        old_statement = puzzle.box_statements[mutation_box][idx]

        # Muter uniquement ce prédicat
        new_statement = mutate_predicate(old_statement)
        if new_statement is None:
            continue  # Si ce statement ne peut pas être muté, la boucle passe à une autre tentative

        # Créer une copie fraîche avec la mutation
        mutated_statements = {b: list(stmts) for b, stmts in puzzle.box_statements.items()}
        mutated_statements[mutation_box][idx] = new_statement
        mutated_puzzle = Puzzle(box_statements=mutated_statements)

        # Contrôle 1 : Anti-fuite
        sig = puzzle_signature(mutated_puzzle)
        if sig in train_signatures:
            continue

        # Contrôle 2 : Oracle Gold (Unicité)
        valid_worlds = [w for w in generate_all_worlds(mutated_puzzle) if is_world_valid(mutated_puzzle, w)]
        if len(valid_worlds) != 1:
            continue

        # Contrôle 3 : Changement de gemme
        new_gem = valid_worlds[0].gem_box
        if new_gem == orig_gem:
            continue

        # Tout est validé !
        return mutated_puzzle

    raise RuntimeError("Impossible de trouver une mutation Gold valide avec gemme différente dans la limite de tentatives.")


def render_surface_identity(puzzle: Puzzle, gold_gem_box: str, color_mapping: Dict[str, str] = None) -> Tuple[str, str]:
    # 1. Si aucun mapping n'est fourni, en choisir un au hasard
    if color_mapping is None:
        new_colors = random.sample(COLORS, 3)
        color_mapping = {
            "BLUE": new_colors[0],
            "WHITE": new_colors[1],
            "BLACK": new_colors[2]
        }

    # 2. Construire le texte en langage naturel
    text = ""
    for box_color in ["BLUE", "WHITE", "BLACK"]:
        statements = puzzle.box_statements[box_color]
        surface_box_name = color_mapping[box_color]
        
        text += f"\n--- {surface_box_name} BOX ---"
        if not statements:
            text += "\n  (No statements)"
        for idx, stmt in enumerate(statements, 1):
            text += f"\n  {idx}. {stmt.to_english()}."

    # 3. Remplacer les occurrences de couleurs dans le texte
    
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, color_mapping)) + r")\b")
    text = pattern.sub(lambda m: color_mapping[m.group(0)], text)

    # 4. Solution remappée pour le LLM
    surface_solution = color_mapping[gold_gem_box]

    return text, surface_solution


def relabel_predicate(pred: Predicate, mapping: Dict[str, str]) -> Predicate:
    if isinstance(pred, AndPredicate):
        return AndPredicate(relabel_predicate(pred.sub_predicate, mapping), relabel_predicate(pred.sub_predicate2, mapping))
    if isinstance(pred, OrPredicate):
        return OrPredicate(relabel_predicate(pred.sub_predicate, mapping), relabel_predicate(pred.sub_predicate2, mapping))
    if isinstance(pred, NotPredicate):
        return NotPredicate(relabel_predicate(pred.sub_predicate, mapping))

    kwargs = dict(pred.__dict__)
    if "color" in kwargs and kwargs["color"] in mapping:
        kwargs["color"] = mapping[kwargs["color"]]
    if "target_box" in kwargs and kwargs["target_box"] in mapping:
        kwargs["target_box"] = mapping[kwargs["target_box"]]
    
    return type(pred)(**kwargs)

def render_surface_lexical(puzzle: Puzzle) -> str:
    text = ""
    for box, statements in puzzle.box_statements.items():
        text += f"\n--- {box} BOX ---"
        if not statements:
            text += "\n  (No statements)"
        for idx, stmt in enumerate(statements, 1):
            text += f"\n  {idx}. {stmt.to_english()}."

    # On sélectionne un contenant différent
    new_container_name_singular = random.choice(CONTAINER_SING)
    container_sing_index = CONTAINER_SING.index(new_container_name_singular)
    new_container_name_plural = CONTAINER_PLURAL[container_sing_index]

    # On sélectionne un type de gemme différent
    new_gem_singular = random.choice(REWARDS)
    gem_singular_index = REWARDS.index(new_gem_singular)
    new_gem_plural = REWARDS_PLURAL[gem_singular_index]

    # Remplacements des occurrences (pluriels d'abord, puis singuliers)
    text = text.replace("BOXES", new_container_name_plural)
    text = text.replace("GEMS", new_gem_plural)
    text = text.replace("BOX", new_container_name_singular)
    text = text.replace("GEM", new_gem_singular)

    return text


if __name__ == "__main__":
    orig_puzzle = generate_gold_puzzle(total_statements=5, relational_ratio=0.5, ast_prob=0.3, max_depth=2)
    orig_world = [w for w in generate_all_worlds(orig_puzzle) if is_world_valid(orig_puzzle, w)][0]
    
    # Générer le texte perturbé en langage naturel
    text, surface_solution = render_surface_identity(orig_puzzle, orig_world.gem_box)
    text_lexical = render_surface_lexical(orig_puzzle)

    print("==========================================")
    print("   PROMPT PERTURBÉ EN LANGAGE NATUREL    ")
    print("==========================================")
    print(text)
    print("\n==========================================")
    print(text_lexical)
    print("\n==========================================")
    print(f" 💎 SOLUTION (INCHANGÉE) : {surface_solution}")
    print("==========================================")
