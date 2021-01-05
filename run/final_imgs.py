import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from imgs import load_imgs

from extract_sudoku import board_img_arr, LeNet5__train, LeNet5__predict
from solve_sudoku import sudoku_board, sudoku_solver

# test_image = "../imgs/sudoku_puzzle.jpg"
# bd_img_arr_obj = board_img_arr.BoardImgArr(
#     img_path=test_image, roi_shape=(28, 28), norm_255=True)
# # bd_img_arr_obj.show_board_cells()
# cells_img = bd_img_arr_obj.get_cells_imgs()
# model_predictor = LeNet5__predict.Predict(model_path=model_path, num_classes=20)
# pred_cells_num = [model_predictor.predict(roi=roi) if roi is not None else -99
#                   for roi in cells_img]
# pred_cells_num = np.array(pred_cells_num)
# bd_img_arr_obj.update_cells_nums(numbers=pred_cells_num)
# sb_obj = sudoku_board.SudokuBoard(board=bd_img_arr_obj.get_cells_nums(),
#                                   invalid_tolerable=True)
# # print(sb_obj.board)
# print(sb_obj.output_board_as_str(line_prefix="\t"))
# sb_solver_obj = sudoku_solver.SudokuSolver()
# empties_cnt, solved, solved_board = \
#     sb_solver_obj.solve(board=sb_obj, method="backtrack")
# print("Solved = %d (Emptied %d Cell(s))" % (solved, empties_cnt))
# print("Solved Board:")
# print(solved_board.output_board_as_str(line_prefix="\t"))


model_path = "../models/LeNet5__lr=0.00100__epoch=10__batch_size=32__epoch=10.pth"
model_predictor = LeNet5__predict.Predict(model_path=model_path, num_classes=20)
max_err = 5
sb_solver_obj = sudoku_solver.SudokuSolver(max_err=max_err)

image_path = "../imgs/test1/"
test_images = [os.path.join(image_path, fn) for fn in os.listdir(image_path)]

for test_image in test_images:
    print(test_image)
    bd_img_arr_obj = board_img_arr.BoardImgArr(
        img_path=test_image, roi_shape=(28, 28), norm_255=False,
        final_ei339=True,
        debug_digits_extraction=False  # True  # False #
    )
    compact_cell_img = bd_img_arr_obj.output_board_cells()
    cv2.imshow("out", compact_cell_img)
    cv2.waitKey(0)
    exit()
    cells_img = bd_img_arr_obj.get_cells_imgs()
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
    exit()
