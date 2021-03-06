# import the necessary packages
import numpy as np

# import modules
from solve_sudoku import sudoku_board, sudoku_solver

test_board_valid = [[8, -99, -99, -99, 1, -99, -99, -99, 9],
                    [-99, 5, -99, 8, -99, 7, -99, 1, -99],
                    [-99, -99, 4, -99, 9, -99, 7, -99, -99],
                    [-99, 6, -99, 7, -99, 1, -99, 2, -99],
                    [5, -99, 8, -99, 6, -99, 1, -99, 7],
                    [-99, 1, -99, 5, -99, 2, -99, 9, -99],
                    [-99, -99, 7, -99, 4, -99, 6, -99, -99],
                    [-99, 8, -99, 3, -99, 9, -99, 4, -99],
                    [3, -99, -99, -99, 5, -99, None, -99, 8]]  # valid board
test_board_invalid_1 = [[8, -99, -99, -99, 1, -99, -99, -99, 9],
                        [-99, 5, -99, 8, -99, 7, -99, 1, 9],
                        [-99, -99, 4, -99, 9, -99, 7, -99, -99],
                        [-99, 6, -99, 7, -99, 1, -99, 2, -99],
                        [5, -99, 8, -99, 6, -99, 1, -99, 7],
                        [-99, 1, -99, 5, -99, 2, -99, 9, -99],
                        [-99, -99, 7, -99, 4, -99, 6, -99, -99],
                        [-99, 8, -99, 3, -99, 9, -99, 4, -99],
                        [3, -99, -99, -99, 5, -99, None, -99, 8]]  # invalid board at column 8
test_board_invalid_2 = [[8, -99, -99, -99, 1, -99, -99, -99, 9],
                        [-99, 5, -99, 8, -99, 7, -99, 1, -99],
                        [-99, -99, 4, -99, 9, -99, 7, -99, -99],
                        [-99, 6, -99, 7, -99, 1, -99, 2, -99],
                        [5, -99, 8, -99, 6, -99, 1, -99, 7],
                        [-99, 1, -99, 5, -99, 2, -99, 9, -99],
                        [-99, -99, 7, -99, 4, -99, 6, -99, -99],
                        [-99, 8, -99, 3, -99, 9, -99, 4, -99],
                        [3, -99, -99, -99, 5, -99, None, 8, 8]]  # invalid board at row 8
test_board_invalid_3 = [[8, -99, -99, -99, 1, -99, -99, -99, 9],
                        [-99, 5, -99, 8, -99, 7, -99, 1, -99],
                        [-99, -99, 4, -99, 9, -99, 7, -99, -99],
                        [-99, 6, -99, 7, -99, 1, -99, 2, -99],
                        [5, -99, 8, -99, 6, -99, 1, -99, 7],
                        [-99, 1, -99, 5, -99, 2, -99, 9, -99],
                        [-99, -99, 7, -99, 4, -99, 6, -99, -99],
                        [-99, 2, -99, 3, -99, 9, -99, 8, -99],
                        [3, -99, -99, -99, 5, -99, None, None, 8]]  # invalid board at box (2, 2)
test_board_easy_1 = [[8, 7, 2, 4, 1, 3, 5, 6, 9],
                     [9, 5, 6, 8, 2, 7, 3, 1, 4],
                     [1, 3, 4, 6, 9, 5, 7, 8, 2],
                     [4, 6, 9, 7, 3, 1, 8, 2, 5],
                     [5, 2, 8, 9, 6, 4, 1, 3, 7],
                     [7, 1, 3, 5, 8, 2, 4, 9, 6],
                     [2, 9, 7, 1, 4, 8, 6, 5, 3],
                     [6, 8, 5, 3, 7, 9, 2, 4, 1],
                     [3, 4, 1, 2, 5, 6, 9, 7, 8]]  # valid full board
test_board_easy_2 = [[8, 7, 2, 4, 1, 3, 5, 6, 9],
                     [9, 5, 6, 8, 2, 7, 3, 1, 4],
                     [1, 3, 4, 6, 9, 5, 7, 8, 2],
                     [4, 6, 9, 7, 3, 1, 8, 2, 5],
                     [5, 2, 8, 9, 6, 4, 1, 3, 7],
                     [7, 1, 3, 5, 8, 2, 4, 9, 6],
                     [2, 9, 7, 1, 4, 8, 6, 5, 3],
                     [6, 8, 5, 3, 7, 9, 2, 4, None],
                     [3, 4, 1, 2, 5, 6, 9, None, 8]]  # valid to-be-filled board

# test_board = np.array(test_board_valid)
test_board = np.array(test_board_invalid_1)  # Failed at Column #8
# test_board = np.array(test_board_invalid_2)  # Failed at Row #8
# test_board = np.array(test_board_invalid_3)  # Failed at Box (row, col) = (2, 2)
# test_board = np.array(test_board_easy_1)
# test_board = np.array(test_board_easy_2)

# === BOARD ===
# sb = sudoku_board.SudokuBoard(board=test_board)
sb = sudoku_board.SudokuBoard(board=test_board, invalid_tolerable=True)
# Stored Board
print("Stored Board:")
print(sb.board)
# Possible Numbers of an Empty Cell
print("Possible Numbers of Empty Cell (%d, %d): " % (8, 1), end="")
print(sb.find_cell_possible_nums(row_idx=8, col_idx=1))  # [2, 4, 9]
# Output the Formatted Board
print("Formatted Board (Indent by \"\\t\")")
print(sb.output_board_as_str(line_prefix="\t"))
print()

# === SOLVER ===
sb_solver = sudoku_solver.SudokuSolver()
empties_cnt, solved, solved_board = sb_solver.solve(board=sb, method="backtrack")
print("Solved = %d (Emptied %d Cell(s))" % (solved, empties_cnt))
print("Solved Board:")
print(solved_board.output_board_as_str(line_prefix="\t"))
