# import the necessary packages
import os
import sys
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# import modules
from solve_suduko import sudoku_board

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
test_board = np.array(test_board_valid)
# test_board = np.array(test_board_invalid_1)  # Failed at Column #8
# test_board = np.array(test_board_invalid_2)  # Failed at Row #8
# test_board = np.array(test_board_invalid_3)  # Failed at Box (row, col) = (2, 2)
sb = sudoku_board.SudokuBoard(board=test_board)
# Stored Board
print("Stored Board")
print(sb.board)
# Possible Numbers of an Empty Cell
print("Possible Numbers of Empty Cell (%d, %d): " % (8, 1), end="")
print(sb.find_cell_possible_nums(row_idx=8, col_idx=1))  # [2, 4, 9]
# Output the Formatted Board
print("Formatted Board (Indent by \"\\t\")")
print(sb.output_board_as_str(line_prefix="\t"))
