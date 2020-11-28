import numpy as np


class SudokuBoard:
    def __init__(self, board: np.ndarray, invalid_tolerable: bool = False):
        self.EMPTY_LABEL = -99
        self.BOARD_SIZE = 9  # size of the board: N*N
        self.MAX_NUM = 9  # max value of the numbers in the sudoku
        self.BOX_SIZE = 3  # box size of the sudoku (non-duplicate 1-MAX_NUM in n*n)
        self.BOX_NUM = self.BOARD_SIZE // self.BOX_SIZE

        self.board = board.copy()
        self.board_cells = board.size

        # Validate Board Settings
        assert board.shape == (self.MAX_NUM, self.MAX_NUM), \
            "[Error] Invalid Board Shape: Expected (%d, %d), Got (%d, %d)" % (
                self.BOARD_SIZE, self.BOARD_SIZE, board.shape[0], board.shape[1])

        # Fill Invalid Cells by EMPTY_LABEL
        blanks = np.isnan(self.board.astype(float))  # remove None
        self.board[blanks] = self.EMPTY_LABEL
        blanks = np.where((self.board < 1) | (self.board > self.MAX_NUM))  # remove out-of-range values
        self.board[blanks] = self.EMPTY_LABEL

        # Ensures dtype=np.int
        self.board = self.board.astype(np.int)

        # Check Initialized Sudoku Board iff Invalid Boards are Tolerable
        if not invalid_tolerable:
            _res, _failure = self.__check_board_is_valid__()
            assert _res is True, "[Error] Invalid Board: Failed at %s" % _failure
            print("[INFO] Board is Valid")

    def __check_board_is_valid__(self) -> (bool, str or None):
        """
        Check whether the whole board is valid
        :return:                 True if valid, False if invalid
        """
        for row_idx in range(self.BOARD_SIZE):
            if not self.__check_row_is_valid__(row_idx=row_idx):
                return False, "Row #%d" % row_idx
        for col_idx in range(self.BOARD_SIZE):
            if not self.__check_col_is_valid__(col_idx=col_idx):
                return False, "Column #%d" % col_idx
        for box_row_idx in range(self.BOX_NUM):
            for box_col_idx in range(self.BOX_NUM):
                if not self.__check_box_is_valid__(box_idx_row=box_row_idx, box_idx_col=box_col_idx):
                    return False, "Box (row, col) = (%d, %d)" % (box_row_idx, box_col_idx)

        return True, None

    def __check_row_is_valid__(self, row_idx: int) -> bool:
        """
        Check whether a specific row is valid: no duplicates among {1, 2, ..., MAX_NUM}
        :param row_idx:         Index of the to-be-checked row (starting from 0, ending at row_cnt-1)
        :return:                True if valid, False if invalid
        """
        row = self.board[row_idx, :]  # shape (self.MAX_NUM, 1)
        row = row[np.where(row != self.EMPTY_LABEL)]  # drop empty cells

        return np.unique(row).size == row.size

    def __check_col_is_valid__(self, col_idx: int) -> bool:
        """
        Check whether a specific row is valid: no duplicates among {1, 2, ..., MAX_NUM}
        :param col_idx:         Index of the to-be-checked column (starting from 0, ending at col_cnt-1)
        :return:                True if valid, False if invalid
        """
        col = self.board[:, col_idx]  # shape (self.MAX_NUM, 1)
        col = col[np.where(col != self.EMPTY_LABEL)]  # drop empty cells

        return np.unique(col).size == col.size

    def __check_box_is_valid__(self, box_idx_row: int, box_idx_col: int) -> bool:
        """
        Check whether a specific (self.BOX_SIZE * self.BOX_SIZE) box is valid:
            no duplicates among {1, 2, ..., MAX_NUM}
        Notice:
            1. boxes start from [0, 0] to [self.BOARD_SIZE, self.BOARD_SIZE]
            2. box indices start from 0, end at self.BOARD_SIZE / self.BOX_SIZE,
        :param box_idx_row:     Index of the to-be-checked box in the row
        :param box_idx_col:     Index of the to-be-checked box in the column
        :return:                True if valid, False if invalid
        """
        row_range = (box_idx_row * self.BOX_SIZE, (box_idx_row + 1) * self.BOX_SIZE)
        col_range = (box_idx_col * self.BOX_SIZE, (box_idx_col + 1) * self.BOX_SIZE)
        box = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]]  # shape (self.BOX_SIZE, self.BOX_SIZE)
        box = box[np.where(box != self.EMPTY_LABEL)]  # drop empty cells

        return np.unique(box).size == box.size


if "__main__" == __name__:
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
    sb = SudokuBoard(board=test_board)
    # print(sb.board)
