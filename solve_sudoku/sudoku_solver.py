import typing
import numpy as np
from copy import deepcopy
import itertools

from .sudoku_board import SudokuBoard


class SudokuSolver:
    def __init__(self, max_err: int = 5):
        # Initiate with empty SudokuBoard, to get constants only
        self.board = SudokuBoard(board=np.full((9, 9), -99, dtype=np.int),
                                 invalid_tolerable=False, show_info=False)
        self.max_err = max_err

    def __flatten_board__(self, board: SudokuBoard) \
            -> (np.ndarray,
                typing.Tuple[np.ndarray], typing.Tuple[np.ndarray, np.ndarray],
                typing.Tuple[np.ndarray], typing.Tuple[np.ndarray, np.ndarray]):
        """
        Flatten the cells of the board & return the empty/nonempty indices
        :param board:       SudokuBoard object
        :return:            1. flattened cells: shape (N,)
                            2. empty idx (board): tuple (shape (hit,), shape (hit,))
                            3. empty idx (flattened): tuple (shape (hit,))
                            4. nonempty idx (board): tuple (shape (hit,), shape (hit,))
                            5. nonempty idx (flattened): tuple (shape (hit,))
        """
        # or using np.argwhere
        empty_cells_board_idx = np.where(self.board.EMPTY_LABEL == board.board)
        nonempty_cells_board_idx = np.where(self.board.EMPTY_LABEL != board.board)

        board_flattened = board.board.flatten()  # <np.ndarray>, shape (cells,)
        empty_cells_flt_idx = np.where(self.board.EMPTY_LABEL == board_flattened)
        nonempty_cells_flt_idx = np.where(self.board.EMPTY_LABEL != board_flattened)

        return board_flattened, empty_cells_board_idx, empty_cells_flt_idx, \
               nonempty_cells_board_idx, nonempty_cells_flt_idx

    @staticmethod
    def __err__generate_combinations_by_lst_num__(lst: typing.List[int], length: int) \
            -> typing.List[typing.Tuple[int]]:
        """
        Generate combinations of given length from the given list
            e.g. [1,2,3],len 2 => [(1,2),(1,3),(2,3)]
        :param lst:             List from which combinations are generated
        :param length:          Length of the generated combinations
        :return:                List of combinations
        """
        iter_obj = itertools.combinations(lst, length)
        return list(iter_obj)

    def __solve__(self, board: SudokuBoard, method: str) \
            -> (bool, SudokuBoard) or (bool, None):
        """
        Call functions to solve (board error intolerable)
        :param board:           SudokuBoard object, board must be VALID
        :param method:          Method indicator
        :return:                <bool>solved, <SudokuBoard>solved board
        """
        methods = ["backtrack"]
        assert method in methods, \
            "[Error] Unknown Method %s. Supported: %s" % (method, ", ".join(methods))
        if "backtrack" == method:
            return self.__backtrack__(board=board)

    def solve(self, board: SudokuBoard, method: str) -> (int, bool, SudokuBoard):
        """
        Approaches wrapper (board error tolerable)
        Note: Invalid boards can always be solved, when all digits are emptied
        :param board:           SudokuBoard object, board must be VALID
        :param method:          Method indicator
        :return:                1. <int>min_emptied: -1 if valid board
                                2. <bool>solved
                                3. <SudokuBoard>solved board
        """
        # Try to Solve the Valid Board
        if board.valid is True:
            solved, solved_board = self.__solve__(board=board, method=method)
            if solved:
                return -1, True, solved_board

        # Empty some Cells and Try to Solve: Unsolved "Valid" Board / Invalid Board
        _, _, _, nonempty_idx_board, _ = self.__flatten_board__(board=board)
        to_empty_ref_idx = list(range(nonempty_idx_board[0].size))  # reference id, shape (nonempty_cnt,)
        for emptied_cnt in range(1, min(self.max_err + 1, len(to_empty_ref_idx) + 1)):
            print("[INFO] \tAttempt to Solve by Changing %2d Cells" % emptied_cnt)
            to_empty_ref_idx_comb = self.__err__generate_combinations_by_lst_num__(
                lst=to_empty_ref_idx, length=emptied_cnt)
            # iterate to-change ref-idx combination tuples (e.g. (1,), (0,3))
            for _ref_idx_tup in to_empty_ref_idx_comb:
                new_board = deepcopy(board)
                # iterate to-change ref-idx-s
                for _ref_idx in _ref_idx_tup:
                    # get to-change idx of the board by reference
                    _idx_dim1 = nonempty_idx_board[0][_ref_idx]
                    _idx_dim2 = nonempty_idx_board[1][_ref_idx]
                    new_board.board[_idx_dim1][_idx_dim2] = self.board.EMPTY_LABEL
                new_board.update_board_valid_status()
                if not new_board.valid:
                    continue
                solved, solved_board = self.__solve__(board=new_board, method=method)
                if solved:
                    return emptied_cnt, True, solved_board
        return 0, False, None

    def __backtrack__(self, board: SudokuBoard) -> (bool, SudokuBoard) or (bool, None):
        """
        Solve the Sudoku problem by backtracking
        :param board:           SudokuBoard object, board must be VALID
        :return:                <bool>solved, <SudokuBoard>solved board
        """
        assert board.valid is True, "[Error] Invalid Board"
        # Flatten board from (size,size)=(9,9) to shape (cells,)=(81,)
        # Find 1ST empty cell (in the flattened board) to save time
        board_flattened, _, empty_cells_flt_idx, _, _ = \
            self.__flatten_board__(board=board)
        # [CASE I] all nonempty, i.e., full Sudoku (guaranteed to be valid)
        if 0 == empty_cells_flt_idx[0].size:
            print("[INFO] Given a Full and Valid Sudoku Board")
            return True, board
        # [CASE II] not all nonempty
        start_idx = np.min(empty_cells_flt_idx)

        to_solve_board = deepcopy(board)
        # test codes to show that deepcopy() works
        # self.board_obj.board[0, 0] = -100
        # print(sudoku_board.board[0, 0], self.board_obj.board[0, 0])

        solved, solved_board = self.__backtrack_recursion__(
            in_board=to_solve_board, to_fill_idx=start_idx)
        return solved, solved_board

    def __backtrack_recursion__(self, in_board: SudokuBoard, to_fill_idx: int) \
            -> (bool, SudokuBoard) or (bool, None):
        """
        Recursively backtracking
        ALERT: carefully handle referred guaranteed VALID <SudokuBoard>
        Notice, the recursion identifier is the FLATTENED index, as,
            similar e.g. [[0, 1, 2], [3, 4, 5]] => [0, 1, 2, 3, 4, 5]
        :param in_board:            Input board Object
        :param to_fill_idx:         FLATTENED Index of the to-fill cell,
                                        ranging in {0, 1, ..., MAX*MAX-1}
        :return:
        """
        # Map flattened index to (row, col)
        to_fill_by_row = to_fill_idx // self.board.BOARD_SIZE
        to_fill_by_col = to_fill_idx - to_fill_by_row * self.board.BOARD_SIZE

        # Recursion base case
        if to_fill_idx >= self.board.board_cells_cnt:
            return True, in_board

        # Continue with non-empty cells
        if self.board.EMPTY_LABEL != in_board.board[to_fill_by_row, to_fill_by_col]:
            return self.__backtrack_recursion__(
                in_board=in_board, to_fill_idx=to_fill_idx + 1)

        # Find & try each of the possible fill-in numbers of empty cells
        possible_nums = in_board.find_cell_possible_nums(
            row_idx=to_fill_by_row, col_idx=to_fill_by_col)
        for _num in possible_nums:
            filled_board = deepcopy(in_board)  # DEEP COPY REQUIRED
            filled_board.board[to_fill_by_row, to_fill_by_col] = _num
            solved, solved_board = self.__backtrack_recursion__(
                in_board=filled_board, to_fill_idx=to_fill_idx + 1)
            if solved:
                return True, solved_board

        # All cases failed, solution NOT found
        return False, None
