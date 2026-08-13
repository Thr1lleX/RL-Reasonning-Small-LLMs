from models import World, BOXES

NUM_WORDS = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX"}

class Predicate:
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        raise NotImplementedError("Chaque prédicat doit implémenter evaluate()")

    def __repr__(self) -> str:
        attrs = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def to_english(self) -> str:
        raise NotImplementedError(f"to_english() non implémenté pour {self.__class__.__name__}")

class AlwaysTrue(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return True

    def to_english(self) -> str:
        return "THIS STATEMENT IS TRUE"

class AlwaysFalse(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return False

    def to_english(self) -> str:
        return "THIS STATEMENT IS FALSE"

class AboveStatement(Predicate):
    def __init__(self, sub_predicate: Predicate):
        self.sub_predicate = sub_predicate

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return self.sub_predicate.evaluate(world, current_box, statement_index - 1)

    def to_english(self) -> str:
        sub_str = self.sub_predicate.to_english().rstrip('.')
        return f"THE STATEMENT ABOVE SAYS THAT {sub_str}"

class BelowStatement(Predicate):
    def __init__(self, sub_predicate: Predicate):
        self.sub_predicate = sub_predicate

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return self.sub_predicate.evaluate(world, current_box, statement_index + 1)

    def to_english(self) -> str:
        sub_str = self.sub_predicate.to_english().rstrip('.')
        return f"THE STATEMENT BELOW SAYS THAT {sub_str}"

class ContainsGems(Predicate):
    def __init__(self, target_box: str):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return world.gem_box == actual_box

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "THIS BOX CONTAINS THE GEMS"
        return f"THE {self.target_box} BOX CONTAINS THE GEMS"

class BoxIsColor(Predicate):
    def __init__(self, target_box: str, color: str):
        self.target_box = target_box
        self.color = color

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return actual_box == self.color

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return f"THIS BOX IS {self.color}"
        return f"THE {self.target_box} BOX IS {self.color}"

class NotPredicate(Predicate):
    def __init__(self, sub_predicate: Predicate):
        self.sub_predicate = sub_predicate

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return not self.sub_predicate.evaluate(world, current_box, statement_index)

    def to_english(self) -> str:
        if isinstance(self.sub_predicate, ContainsGems):
            target = self.sub_predicate.target_box
            if target == "THIS":
                return "THE GEMS ARE NOT IN THIS BOX"
            return f"THE GEMS ARE NOT IN THE {target} BOX"
        elif isinstance(self.sub_predicate, IsEmpty):
            target = self.sub_predicate.target_box
            if target == "THIS":
                return "THIS BOX IS NOT EMPTY"
            return f"THE {target} BOX IS NOT EMPTY"
        elif isinstance(self.sub_predicate, BoxIsTrue):
            target = self.sub_predicate.target_box
            if target == "THIS":
                return "THIS BOX HAS A FALSE STATEMENT"
            return f"THE {target} BOX HAS A FALSE STATEMENT"
        sub_str = self.sub_predicate.to_english().rstrip('.')
        return f"IT IS NOT TRUE THAT {sub_str}"

class AndPredicate(Predicate):
    def __init__(self, sub_predicate: Predicate, sub_predicate2: Predicate):
        self.sub_predicate = sub_predicate
        self.sub_predicate2 = sub_predicate2

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return self.sub_predicate.evaluate(world, current_box, statement_index) and self.sub_predicate2.evaluate(world, current_box, statement_index)

    def to_english(self) -> str:
        sub1 = self.sub_predicate.to_english().rstrip('.')
        sub2 = self.sub_predicate2.to_english().rstrip('.')
        return f"{sub1} AND {sub2}"

class OrPredicate(Predicate):
    def __init__(self, sub_predicate: Predicate, sub_predicate2: Predicate):
        self.sub_predicate = sub_predicate
        self.sub_predicate2 = sub_predicate2

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return self.sub_predicate.evaluate(world, current_box, statement_index) or self.sub_predicate2.evaluate(world, current_box, statement_index)

    def to_english(self) -> str:
        sub1 = self.sub_predicate.to_english().rstrip('.')
        sub2 = self.sub_predicate2.to_english().rstrip('.')
        return f"EITHER {sub1} OR {sub2}"

class CountTrueBoxes(Predicate):
    def __init__(self, target_count: str):
        self.target_count = target_count

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        if self.target_count == "THIS":
            actual_count = 1
        elif self.target_count == "ONE":
            actual_count = 1
        elif self.target_count == "TWO":
            actual_count = 2
        elif self.target_count == "THREE":
            actual_count = 3
        elif self.target_count == "FOUR":
            actual_count = 4
        else:
            raise ValueError(f"Unsupported target count: {self.target_count}")
        real_true_count = sum(sum(t) for t in world.box_truths.values())
        return actual_count == real_true_count

    def to_english(self) -> str:
        return f"EXACTLY {self.target_count} BOXES ARE TRUE"

class CountTrueStatements(Predicate):
    def __init__(self, op: str, num: int):
        self.op = op
        self.num = num

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        real_count = sum(sum(t) for t in world.box_truths.values())
        if self.op == "==":
            return real_count == self.num
        elif self.op == ">":
            return real_count > self.num
        elif self.op == "<":
            return real_count < self.num
        elif self.op == "<=":
            return real_count <= self.num
        elif self.op == ">=":
            return real_count >= self.num
        elif self.op == "!=":
            return real_count != self.num
        else:
            raise ValueError(f"Unsupported operator: {self.op}")

    def to_english(self) -> str:
        num_str = NUM_WORDS.get(self.num, str(self.num))
        if self.op == "==":
            return f"EXACTLY {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} TRUE"
        elif self.op == ">=":
            return f"AT LEAST {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} TRUE"
        elif self.op == "<=":
            return f"AT MOST {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} TRUE"
        elif self.op == ">":
            return f"MORE THAN {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} TRUE"
        elif self.op == "<":
            return f"LESS THAN {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} TRUE"
        return f"STATEMENT COUNT IS {self.op} {num_str} TRUE"

class CountFalseStatements(Predicate):
    def __init__(self, op: str, num: int):
        self.op = op
        self.num = num

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        real_count = sum(sum(not x for x in t) for t in world.box_truths.values())
        if self.op == "==":
            return real_count == self.num
        elif self.op == ">":
            return real_count > self.num
        elif self.op == "<":
            return real_count < self.num
        elif self.op == "<=":
            return real_count <= self.num
        elif self.op == ">=":
            return real_count >= self.num
        elif self.op == "!=":
            return real_count != self.num
        else:
            raise ValueError(f"Unsupported operator: {self.op}")

    def to_english(self) -> str:
        num_str = NUM_WORDS.get(self.num, str(self.num))
        if self.op == "==":
            return f"EXACTLY {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} FALSE"
        elif self.op == ">=":
            return f"AT LEAST {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} FALSE"
        elif self.op == "<=":
            return f"AT MOST {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} FALSE"
        elif self.op == ">":
            return f"MORE THAN {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} FALSE"
        elif self.op == "<":
            return f"LESS THAN {num_str} STATEMENT{'S ARE' if self.num != 1 else ' IS'} FALSE"
        return f"STATEMENT COUNT IS {self.op} {num_str} FALSE"

class BoxIsTrue(Predicate):
    def __init__(self, target_box: str):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return all(world.box_truths[actual_box])

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "THIS BOX IS TRUE"
        return f"THE {self.target_box} BOX IS TRUE"

def get_neighbours(box):
    if box == "BLUE":
        return ["WHITE"]
    elif box == "WHITE":
        return ["BLUE", "BLACK"]
    elif box == "BLACK":
        return ["WHITE"]
    raise ValueError(f"Boîte inconnue : {box}")

class NeighborContainsGems(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        neighbors = get_neighbours(actual_box)
        for neighbor in neighbors:
            if world.gem_box == neighbor:
                return True
        return False

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "A BOX NEXT TO THIS BOX CONTAINS THE GEMS"
        return f"A BOX NEXT TO THE {self.target_box} BOX CONTAINS THE GEMS"

class BothNeighborsContainGems(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        neighbors = get_neighbours(actual_box)
        return all(world.gem_box == n for n in neighbors)

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "BOXES NEXT TO THIS BOX CONTAIN GEMS"
        return f"BOXES NEXT TO THE {self.target_box} BOX CONTAIN GEMS"

class NeighborIsTrue(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        neighbors = get_neighbours(actual_box)
        return any(all(world.box_truths[n]) for n in neighbors)

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "A BOX NEXT TO THIS BOX IS TRUE"
        return f"A BOX NEXT TO THE {self.target_box} BOX IS TRUE"

class GemsInBoxWithCondition(Predicate):
    def __init__(self, expected_truth: bool):
        self.expected_truth = expected_truth

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        gem_box_truth = all(world.box_truths[world.gem_box])
        return gem_box_truth == self.expected_truth

    def to_english(self) -> str:
        if self.expected_truth:
            return "THE GEMS ARE IN A BOX THAT IS TRUE"
        return "THE GEMS ARE IN A BOX THAT IS FALSE"

class IsEmpty(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return world.gem_box != actual_box

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "THIS BOX IS EMPTY"
        return f"THE {self.target_box} BOX IS EMPTY"

class OtherTwoBoxesAreEmpty(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_boxes = [b for b in BOXES if b != actual_box]
        return all(world.gem_box != b for b in other_boxes)

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "THE OTHER TWO BOXES ARE EMPTY"
        return f"THE OTHER TWO BOXES RELATIVE TO THE {self.target_box} BOX ARE EMPTY"


class IsMiddle(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return actual_box == "WHITE"

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "THIS BOX IS THE MIDDLE BOX"
        return f"THE {self.target_box} BOX IS THE MIDDLE BOX"

class ThisIsTheOnlyTrueStatement(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        total_statements = sum(len(t) for t in world.box_truths.values())
        total_true = sum(sum(t) for t in world.box_truths.values())
        current_stmt_truth = world.box_truths[current_box][statement_index]
        return total_true == 1 and current_stmt_truth is True

    def to_english(self) -> str:
        return "THIS IS THE ONLY TRUE STATEMENT"

class OtherTwoBoxesAreTrue(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_boxes = [b for b in BOXES if b != actual_box]
        return all(all(world.box_truths[b]) for b in other_boxes)

    def to_english(self) -> str:
        return "THE OTHER TWO BOXES ARE TRUE"

class OtherTwoBoxesAreFalse(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_boxes = [b for b in BOXES if b != actual_box]
        return all(all(not t for t in world.box_truths[b]) for b in other_boxes)

    def to_english(self) -> str:
        return "THE OTHER TWO BOXES ARE FALSE"

class AllBoxesAreEmpty(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return all(world.gem_box != b for b in BOXES)

    def to_english(self) -> str:
        return "ALL BOXES ARE EMPTY"

class AllBoxesContainGems(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return all(world.gem_box == b for b in BOXES)

    def to_english(self) -> str:
        return "ALL BOXES CONTAIN GEMS"

class OtherTwoBoxesAreColor(Predicate):
    def __init__(self, color: str):
        self.color = color

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        other_boxes = [b for b in BOXES if b != current_box]
        return all(b == self.color for b in other_boxes)

    def to_english(self) -> str:
        return f"THE OTHER TWO BOXES ARE {self.color}"

class EmptyBoxesHaveTruth(Predicate):
    def __init__(self, expected_truth: bool):
        self.expected_truth = expected_truth

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        empty_boxes = [b for b in BOXES if b != world.gem_box]
        return all(all(t == self.expected_truth for t in world.box_truths[b]) for b in empty_boxes)

    def to_english(self) -> str:
        if self.expected_truth:
            return "EMPTY BOXES ARE ALL TRUE"
        return "EMPTY BOXES ARE ALL FALSE"

class OneOtherBoxIsFalse(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_boxes = [b for b in BOXES if b != actual_box]
        return sum(1 for b in other_boxes if not all(world.box_truths[b])) == 1

    def to_english(self) -> str:
        return "ONE OTHER BOX IS FALSE"

class OneOtherBoxIsTrue(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_boxes = [b for b in BOXES if b != actual_box]
        return sum(1 for b in other_boxes if all(world.box_truths[b])) == 1

    def to_english(self) -> str:
        return "ONE OTHER BOX IS TRUE"

class StatementsAreEquallyTrue(Predicate):
    def __init__(self, pred1: Predicate, pred2: Predicate):
        self.pred1 = pred1
        self.pred2 = pred2

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return self.pred1.evaluate(world, current_box, statement_index) == self.pred2.evaluate(world, current_box, statement_index)

    def to_english(self) -> str:
        p1 = self.pred1.to_english().rstrip('.')
        p2 = self.pred2.to_english().rstrip('.')
        return f"THE STATEMENT ({p1}) AND ({p2}) ARE EITHER BOTH TRUE OR BOTH FALSE"

class GemsInOnlyTruthBox(Predicate):
    def __init__(self, expected_truth: bool):
        self.expected_truth = expected_truth

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        matching_boxes = [b for b in BOXES if all(t == self.expected_truth for t in world.box_truths[b])]
        return len(matching_boxes) == 1 and world.gem_box == matching_boxes[0]

    def to_english(self) -> str:
        if self.expected_truth:
            return "THE BOX WITH NO FALSE STATEMENTS HAS THE GEMS"
        return "THE BOX WITH TWO FALSE STATEMENTS HAS THE GEMS"

class OnlyOneNeighborIsFalseAndContainsGems(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        neighbors = get_neighbours(current_box)
        false_neighbors = [n for n in neighbors if not all(world.box_truths[n])]
        return len(false_neighbors) == 1 and world.gem_box == false_neighbors[0]

    def to_english(self) -> str:
        return "ONLY ONE NEIGHBOR IS FALSE AND CONTAINS GEMS"

class GemsInBothBoxes(Predicate):
    def __init__(self, box1: str, box2: str):
        self.box1 = box1
        self.box2 = box2

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return False

    def to_english(self) -> str:
        return f"GEMS ARE IN BOTH {self.box1} AND {self.box2}"

class ABoxWithAFalseStatementIsEmpty(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        for b in BOXES:
            if any(not t for t in world.box_truths[b]) and b != world.gem_box:
                return True
        return False

    def to_english(self) -> str:
        return "A BOX WITH A FALSE STATEMENT IS EMPTY"

class GemsInOtherBoxWithTrueStatement(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        other_true_boxes = [
            b for b in BOXES
            if b != actual_box and all(world.box_truths[b])
        ]
        return world.gem_box in other_true_boxes

    def to_english(self) -> str:
        return "THE GEMS ARE IN ANOTHER BOX THAT IS TRUE"

class FalseBoxesAreBothTrue(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return False

    def to_english(self) -> str:
        return "FALSE BOXES ARE BOTH TRUE"

class OnlyOneBoxContainsGems(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        return True

    def to_english(self) -> str:
        return "ONLY ONE BOX CONTAINS THE GEMS"

class AllStatementsOnBoxAreTrue(Predicate):
    def __init__(self, target_box: str = "THIS"):
        self.target_box = target_box

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        return all(world.box_truths[actual_box])

    def to_english(self) -> str:
        if self.target_box == "THIS":
            return "ALL STATEMENTS ON THIS BOX ARE TRUE"
        return f"ALL STATEMENTS ON THE {self.target_box} BOX ARE TRUE"

class AboveStatementIsTrue(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        if statement_index == 0:
            return False
        return world.box_truths[current_box][statement_index - 1]

    def to_english(self) -> str:
        return "THE STATEMENT ABOVE THIS ONE IS TRUE"

class TopStatementOfEachBoxIsTrue(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        non_empty_boxes = [b for b in BOXES if len(world.box_truths[b]) > 0]
        if not non_empty_boxes:
            return True
        return all(world.box_truths[b][0] for b in non_empty_boxes)

    def to_english(self) -> str:
        return "THE TOP STATEMENT OF EACH BOX IS TRUE"

class CountTrueStatementsOnBox(Predicate):
    def __init__(self, target_box: str, op: str, num: int):
        self.target_box = target_box
        self.op = op
        self.num = num

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        real_count = sum(world.box_truths[actual_box])
        if self.op == "==":
            return real_count == self.num
        elif self.op == ">=":
            return real_count >= self.num
        elif self.op == "<=":
            return real_count <= self.num
        return False

    def to_english(self) -> str:
        num_str = NUM_WORDS.get(self.num, str(self.num))
        box_str = "THIS BOX" if self.target_box == "THIS" else f"THE {self.target_box} BOX"
        if self.op == "==":
            return f"{box_str} HAS EXACTLY {num_str} TRUE STATEMENT{'S' if self.num != 1 else ''}"
        elif self.op == ">=":
            return f"{box_str} HAS AT LEAST {num_str} TRUE STATEMENT{'S' if self.num != 1 else ''}"
        elif self.op == "<=":
            return f"{box_str} HAS AT MOST {num_str} TRUE STATEMENT{'S' if self.num != 1 else ''}"
        return f"{box_str} HAS {self.op} {num_str} TRUE STATEMENTS"

class CountFalseStatementsOnBox(Predicate):
    def __init__(self, target_box: str, op: str, num: int):
        self.target_box = target_box
        self.op = op
        self.num = num

    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        actual_box = current_box if self.target_box == "THIS" else self.target_box
        real_count = sum(not t for t in world.box_truths[actual_box])
        if self.op == "==":
            return real_count == self.num
        elif self.op == ">=":
            return real_count >= self.num
        elif self.op == "<=":
            return real_count <= self.num
        return False

    def to_english(self) -> str:
        num_str = NUM_WORDS.get(self.num, str(self.num))
        box_str = "THIS BOX" if self.target_box == "THIS" else f"THE {self.target_box} BOX"
        if self.op == "==":
            return f"{box_str} HAS EXACTLY {num_str} FALSE STATEMENT{'S' if self.num != 1 else ''}"
        elif self.op == ">=":
            return f"{box_str} HAS AT LEAST {num_str} FALSE STATEMENT{'S' if self.num != 1 else ''}"
        elif self.op == "<=":
            return f"{box_str} HAS AT MOST {num_str} FALSE STATEMENT{'S' if self.num != 1 else ''}"
        return f"{box_str} HAS {self.op} {num_str} FALSE STATEMENTS"

class ThisBoxStatementsAreEqual(Predicate):
    def evaluate(self, world: World, current_box: str, statement_index: int = 0) -> bool:
        truths = world.box_truths[current_box]
        if len(truths) < 2:
            return True
        return truths[0] == truths[1]

    def to_english(self) -> str:
        return "THE STATEMENTS ON THIS BOX ARE EITHER BOTH TRUE OR BOTH FALSE"
