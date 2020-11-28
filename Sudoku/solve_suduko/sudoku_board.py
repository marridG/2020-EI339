import numpy as np
from functools import reduce


class SudokuBoard:
    def __init__(self, board: np.ndarray, invalid_tolerable: bool = False):
        self.EMPTY_LABEL = -99
        self.BOARD_SIZE = 9  # size of the board: N*N
        self.MAX_NUM = 9  # max value of the numbers in the sudoku
        self.BOX_SIZE = 3  # box size of the sudoku (non-duplicate 1-MAX_NUM in n*n)
        self.BOX_NUM = self.BOARD_SIZE // self.BOX_SIZE

        self.board = board.copy()
        self.board_cells_cnt = board.size

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
        # check each row
        for row_idx in range(self.BOARD_SIZE):
            is_valid, _ = self.__check_row__is_valid_n_used_nums__(row_idx=row_idx)
            if not is_valid:
                return False, "Row #%d" % row_idx
        # check each column
        for col_idx in range(self.BOARD_SIZE):
            is_valid, _ = self.__check_col__is_valid_n_used_nums__(col_idx=col_idx)
            if not is_valid:
                return False, "Column #%d" % col_idx
        # check each box
        for box_row_idx in range(self.BOX_NUM):
            for box_col_idx in range(self.BOX_NUM):
                is_valid, _ = self.__check_box__is_valid_n_used_nums__(
                    box_idx_row=box_row_idx, box_idx_col=box_col_idx)
                if not is_valid:
                    return False, "Box (row, col) = (%d, %d)" % (box_row_idx, box_col_idx)

        return True, None

    def __check_row__is_valid_n_used_nums__(self, row_idx: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific row is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        :param row_idx:         Index of the to-be-checked row
                                    (starting from 0, ending at row_cnt-1)
        :return:                1. True if valid, False if invalid;
                                2. Used numbers in the row
        """
        row = self.board[row_idx, :]  # shape (self.MAX_NUM, 1)
        row = row[np.where(row != self.EMPTY_LABEL)]  # drop empty cells

        used_numbers = np.unique(row)
        is_valid = used_numbers.size == row.size
        return is_valid, used_numbers

    def __check_col__is_valid_n_used_nums__(self, col_idx: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific column is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        :param col_idx:         Index of the to-be-checked column
                                    (starting from 0, ending at col_cnt-1)
        :return:                1. True if valid, False if invalid;
                                2. Used numbers in the column
        """
        col = self.board[:, col_idx]  # shape (self.MAX_NUM, 1)
        col = col[np.where(col != self.EMPTY_LABEL)]  # drop empty cells

        used_numbers = np.unique(col)
        is_valid = used_numbers.size == col.size
        return is_valid, used_numbers

    def __check_box__is_valid_n_used_nums__(self, box_idx_row: int, box_idx_col: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific box is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
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

        used_numbers = np.unique(box)
        is_valid = used_numbers.size == box.size
        return is_valid, used_numbers

    def __idx_row_col_2_box__(self, row_idx: int, col_idx: int) -> (int, int):
        """
        Map indices by (row, col) to the index of the corresponding box of the cell
        :param row_idx:         Index of the cell in the row
        :param col_idx:         Index of the cell in the column
        :return:                Indices of the box, by (row, col)
        """
        box_idx_row = row_idx // self.BOX_SIZE
        box_idx_col = col_idx // self.BOX_SIZE
        return box_idx_row, box_idx_col

    def __check_box_row_col__is_valid_n_used_nums__(self, row_idx: int, col_idx: int) \
            -> (bool, np.ndarray):
        """
        [WRAPPER] "__idx_row_col_2_box__()" and "__check_box__is_valid_n_used_nums__()"
        1. Check whether a specific box is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        Notice:
            1. boxes start from [0, 0] to [self.BOARD_SIZE, self.BOARD_SIZE]
            2. box indices start from 0, end at self.BOARD_SIZE / self.BOX_SIZE,
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                True if valid, False if invalid
        """
        box_idx_row, box_idx_col = self.__idx_row_col_2_box__(
            row_idx=row_idx, col_idx=col_idx)
        box_valid, box_used_numbers = self.__check_box__is_valid_n_used_nums__(
            box_idx_row=box_idx_row, box_idx_col=box_idx_col)
        return box_valid, box_used_numbers

    def __validate_input_idx_is_valid(self, row_idx: int, col_idx: int) -> None:
        """
        Check whether a input index (row, col) is valid
        :param row_idx:         Input index of the to-be-checked cell in the row
        :param col_idx:         Input index of the to-be-checked cell in the column
        :return:                True if valid, False if invalid
        """
        assert 0 <= row_idx < self.BOARD_SIZE, \
            "[Error] Given Index out of Range:" "Expected 0 <= Row < %d, Got %d" % \
            (self.BOARD_SIZE, row_idx)
        assert 0 <= col_idx < self.BOARD_SIZE, \
            "[Error] Given Index out of Range:" "Expected 0 <= Column < %d, Got %d" % \
            (self.BOARD_SIZE, row_idx)

    def check_cell_is_valid(self, row_idx: int, col_idx: int) -> bool:
        """
        Check whether a cell is valid
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                True if valid, False if invalid
        """
        # Input Validation
        self.__validate_input_idx_is_valid(row_idx=row_idx, col_idx=col_idx)

        row_valid, _ = self.__check_row__is_valid_n_used_nums__(row_idx=row_idx)
        col_valid, _ = self.__check_col__is_valid_n_used_nums__(col_idx=col_idx)
        box_valid, _ = self.__check_box_row_col__is_valid_n_used_nums__(
            row_idx=row_idx, col_idx=col_idx)

        return row_valid and col_valid and box_valid

    def find_cell_possible_nums(self, row_idx: int, col_idx: int) -> np.ndarray:
        """
        Find the possible numbers of an empty cell
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                All possible numbers
        """
        # Input Validation
        assert self.EMPTY_LABEL == self.board[row_idx, col_idx], \
            "[Error] Target Cell (row, col) = (%d, %d) Non-Empty, Value = %d" % (
                row_idx, col_idx, self.board[row_idx, col_idx])
        self.__validate_input_idx_is_valid(row_idx=row_idx, col_idx=col_idx)

        _, row_nums = self.__check_row__is_valid_n_used_nums__(row_idx=row_idx)
        _, col_nums = self.__check_col__is_valid_n_used_nums__(col_idx=col_idx)
        _, box_nums = self.__check_box_row_col__is_valid_n_used_nums__(
            row_idx=row_idx, col_idx=col_idx)
        used_nums = reduce(np.union1d, (row_nums, col_nums, box_nums))

        all_nums = np.arange(start=1, stop=self.MAX_NUM + 1)
        possible_nums = np.setdiff1d(all_nums, used_nums)
        return possible_nums
