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
# test_board = np.array(test_board_invalid_1)
# test_board = np.array(test_board_invalid_2)
# test_board = np.array(test_board_invalid_3)
sb = sudoku_board.SudokuBoard(board=test_board)
# Stored Board
print(sb.board)
# Possible Numbers of a Non-empty Cell
print(sb.find_cell_possible_nums(row_idx=8, col_idx=1))  # [2, 4, 9]
