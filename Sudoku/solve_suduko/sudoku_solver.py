import numpy as np
from copy import deepcopy

from .sudoku_board import SudokuBoard


class SudokuSolver:
    def __init__(self):
        # Initiate with empty SudokuBoard, to get constants only
        self.board = SudokuBoard(board=np.full((9, 9), -99, dtype=np.int),
                                 invalid_tolerable=False, show_info=False)

    def backtrack(self, board: SudokuBoard, recheck: bool = True):
        board_flattened = board.board.flatten()  # <np.ndarray>, shape (cells,)

        # Find 1ST empty cell (in the flattened board) to save time
        # result shape (hit,); or using np.where()
        empty_cells_flt_idx = np.argwhere(self.board.EMPTY_LABEL == board_flattened)
        # [CASE I] all nonempty, i.e., full Sudoku
        # (actually guaranteed to be valid since checked @SudokuBoard)
        if 0 == empty_cells_flt_idx.size:
            print("[INFO] Given a Full and Valid Sudoku Board")
            return True, board
        # [CASE II] not all nonempty
        start_idx = np.min(empty_cells_flt_idx)

        to_solve_board = deepcopy(board)
        # test codes to show that deepcopy() works
        # self.board_obj.board[0, 0] = -100
        # print(sudoku_board.board[0, 0], self.board_obj.board[0, 0])

        solved, solved_board = self.__backtrack__(
            in_board=to_solve_board, to_fill_idx=start_idx, recheck=recheck)
        return solved, solved_board

    def __backtrack__(self, in_board: SudokuBoard, to_fill_idx: int, recheck: bool = True):
        """
        Recursively backtracking (ALERT: carefully handle referred <SudokuBoard>)
        Notice, the recursion indentifier is the FLATTENED index, as,
            similar e.g. [[0, 1, 2], [3, 4, 5]] => [0, 1, 2, 3, 4, 5]
        :param in_board:            Input board Object
        :param to_fill_idx:         FLATTENED Index of the to-fill cell,
                                        ranging in {0, 1, ..., MAX*MAX-1}
        :param recheck:             Whether to recheck the board in the base case
        :return:
        """
        # Map flattened index to (row, col)
        to_fill_by_row = to_fill_idx // self.board.BOARD_SIZE
        to_fill_by_col = to_fill_idx - to_fill_by_row * self.board.BOARD_SIZE

        # Recursion base case
        if to_fill_idx >= self.board.board_cells_cnt:
            solved = True
            if recheck:
                solved, _ = in_board.check_board_is_valid()
            if solved:
                return True, in_board
            else:
                return False, None

        # Continue with non-empty cells
        if self.board.EMPTY_LABEL != in_board.board[to_fill_by_row, to_fill_by_col]:
            return self.__backtrack__(
                in_board=in_board, to_fill_idx=to_fill_idx + 1, recheck=recheck)

        # Find & try each of the possible fill-in numbers of empty cells
        possible_nums = in_board.find_cell_possible_nums(
            row_idx=to_fill_by_row, col_idx=to_fill_by_col)
        for _num in possible_nums:
            filled_board = deepcopy(in_board)  # DEEP COPY REQUIRED
            filled_board.board[to_fill_by_row, to_fill_by_col] = _num
            solved, solved_board = self.__backtrack__(
                in_board=filled_board, to_fill_idx=to_fill_idx + 1, recheck=recheck)
            if solved:
                return True, solved_board

        # All cases failed, solution NOT found
        return False, None
