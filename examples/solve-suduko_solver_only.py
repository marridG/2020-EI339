# import the necessary packages
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# import modules
from solve_suduko import sudoku_board, sudoku_solver

test_board_valid = [[8, -99, -99, -99, 1, -99, -99, -99, 9],
                    [-99, 5, -99, 8, -99, 7, -99, 1, -99],
                    [-99, -99, 4, -99, 9, -99, 7, -99, -99],
                    [-99, 6, -99, 7, -99, 1, -99, 2, -99],
                    [5, -99, 8, -99, 6, -99, 1, -99, 7],
                    [-99, 1, -99, 5, -99, 2, -99, 9, -99],
                    [-99, -99, 7, -99, 4, -99, 6, -99, -99],
                    [-99, 8, -99, 3, -99, 9, -99, 4, -99],
                    [3, -99, -99, -99, 5, -99, None, -99, 8]]  # valid board
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
test_board = np.array(test_board_valid)
# test_board = np.array(test_board_easy_1)
# test_board = np.array(test_board_easy_2)
# test_board = np.array(test_board_invalid_1)
# test_board = np.array(test_board_invalid_2)
# test_board = np.array(test_board_invalid_3)
sb = sudoku_board.SudokuBoard(board=test_board,
                              invalid_tolerable=False, show_info=True)
sb_solver = sudoku_solver.SudokuSolver()
solved, solved_board = sb_solver.backtrack(board=sb, recheck=True)
print(solved)
print(solved_board.output_board_as_str(line_prefix="\t"))
