import numpy as np
from copy import deepcopy

from .sudoku_board import SudokuBoard


class SudokuSolver:
    def __init__(self, sudoku_board: SudokuBoard):
        self.board_obj = deepcopy(sudoku_board)
        # self.board_obj.board[0, 0] = -100
        # print(sudoku_board.board[0, 0], self.board_obj.board[0, 0])

    def backtrack(self):
        pass

    def __backtrack__(self, in_board: SudokuBoard, fill_target_idx: int):
        """
        Recursively backtracking
        :param in_board:            Input board Object
        :param fill_target_idx:     Index of the to-fill cell,
                                        ranging in {0, 1, ..., MAX*MAX-1}
        :return:
        """
        board = deepcopy(in_board)

        fill_target_row = fill_target_idx // self.board_obj.BOX_SIZE
        fill_target_col = fill_target_idx - fill_target_row * self.board_obj.BOX_SIZE
