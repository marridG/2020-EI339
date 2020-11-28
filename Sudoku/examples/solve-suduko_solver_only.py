# import the necessary packages
import os
import sys
import numpy as np
import cv2

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
test_board = np.array(test_board_valid)
# test_board = np.array(test_board_invalid_1)
# test_board = np.array(test_board_invalid_2)
# test_board = np.array(test_board_invalid_3)
sb = sudoku_board.SudokuBoard(board=test_board)
sb_solver = sudoku_solver.SudokuSolver(sudoku_board=sb)