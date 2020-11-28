import numpy as np

from .sudoku_board import SudokuBoard


class SudokuSolver:
    def __init__(self, sudoku_board: SudokuBoard):
        self.board_obj = sudoku_board
        self.board = self.board_obj.board.copy()

    def backtrack(self):
        pass

    def __backtrack__(self, board: np.ndarray):
        pass
