from random import shuffle, seed as random_seed, randrange
import sys
import numpy as np


class SudokuBoard:
    def __init__(self, board: np.ndarray):
        self.EMPTY_LABEL = -99
        self.MAX = 9

        self.board = board.astype(np.int).copy()
        self.board_cells = board.size

        # validate board settings
        assert board.shape == (self.MAX, self.MAX), \
            "[Error] Invalid Board Shape: Expected (%d, %d), Got (%d, %d)" % (
                self.MAX, self.MAX, board.shape[0], board.shape[1])

        blanks = np.where((self.board is None) | (self.board < 1) | (self.board > self.MAX))
        board[blanks] = self.EMPTY_LABEL
        blanks_cnt = blanks[0].shape[0]
    
