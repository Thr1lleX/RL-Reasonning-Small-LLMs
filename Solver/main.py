from models import Puzzle
from predicates import (ContainsGems, BoxIsColor, NotPredicate, CountTrueBoxes, 
BoxIsTrue, GemsInBoxWithCondition, IsMiddle, CountTrueStatements, CountFalseStatements, 
IsEmpty, OtherTwoBoxesAreEmpty, ThisIsTheOnlyTrueStatement, OtherTwoBoxesAreTrue, 
AllBoxesContainGems, AllBoxesAreEmpty, OtherTwoBoxesAreColor,AllStatementsOnBoxAreTrue, AboveStatementIsTrue,
TopStatementOfEachBoxIsTrue)
from solver import solve_puzzle

if __name__ == "__main__":


    puzzle63 = Puzzle(
        box_statements={
            "BLUE": [
                AllStatementsOnBoxAreTrue("THIS"),
                ContainsGems("THIS")
            ],
            "WHITE": [
                NotPredicate(ContainsGems("BLUE")),
                NotPredicate(ContainsGems("BLACK"))
            ],
            "BLACK": [
                GemsInBoxWithCondition(expected_truth=False),
                AllStatementsOnBoxAreTrue("BLUE")
            ]
        }
    )
    print("Puzzle 63 Solution:", solve_puzzle(puzzle63))
    
    puzzle62 = Puzzle(
        box_statements={
            "BLUE": [
                ContainsGems("THIS"),
                AboveStatementIsTrue()
            ],
            "WHITE": [
                NotPredicate(ContainsGems("THIS")),
                NotPredicate(AboveStatementIsTrue())
            ],
            "BLACK": [
                TopStatementOfEachBoxIsTrue(),
                AboveStatementIsTrue()
            ]
        }
    )

    print("Puzzle 62 Solution:", solve_puzzle(puzzle62))
