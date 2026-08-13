from dataclasses import dataclass, field
from typing import Tuple, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from predicates import Predicate

BOXES = ["BLUE", "WHITE", "BLACK"]

@dataclass(frozen=True)
class World:
    gem_box: str                                  # 'BLUE', 'WHITE', ou 'BLACK'
    box_truths: Dict[str, Tuple[bool, ...]]       # ex: {'BLUE': (True, False), 'WHITE': (True, True), 'BLACK': (False, False)}


@dataclass
class Puzzle:
    box_statements: Dict[str, List['Predicate']]   # ex: {'BLUE': [stmt0, stmt1], 'WHITE': [...], 'BLACK': [...]}
    def __repr__(self) -> str:
        res = "Puzzle:\n"
        for box, stmts in self.box_statements.items():
            res += f"  [{box}]:\n"
            for stmt in stmts:
                res += f"    - {stmt}\n"
        return res

