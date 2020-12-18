import cv2
import numpy as np

from extract_sudoku import board_img_arr, opencv__train, opencv__predict
from solve_sudoku import sudoku_board, sudoku_solver

# model_path = opencv__train.train(model_path="../models/")
model_path = "../models/opencv__20201216224246.h5"

test_image = "../imgs/sudoku_puzzle.jpg"
bd_img_arr_obj = board_img_arr.BoardImgArr(
    img_path=test_image, roi_shape=(28, 28), norm_255=True)
# bd_img_arr_obj.show_board_cells()
cells_img = bd_img_arr_obj.get_cells_imgs()
model_predictor = opencv__predict.Predict(model_path=model_path)
pred_cells_num = [model_predictor.predict(roi=roi) if roi is not None else -99
                  for roi in cells_img]
pred_cells_num = np.array(pred_cells_num)
bd_img_arr_obj.update_cells_nums(numbers=pred_cells_num)
sb_obj = sudoku_board.SudokuBoard(board=bd_img_arr_obj.get_cells_nums(),
                                  invalid_tolerable=True)
# print(sb_obj.board)
print(sb_obj.output_board_as_str(line_prefix="\t"))
sb_solver_obj = sudoku_solver.SudokuSolver()
empties_cnt, solved, solved_board = \
    sb_solver_obj.solve(board=sb_obj, method="backtrack")
print("Solved = %d (Emptied %d Cell(s))" % (solved, empties_cnt))
print("Solved Board:")
print(solved_board.output_board_as_str(line_prefix="\t"))
