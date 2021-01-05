import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from imgs import load_imgs

from extract_sudoku import board_img_arr, LeNet5__train, LeNet5__predict
from solve_sudoku import sudoku_board, sudoku_solver

# train_report = LeNet5__train.train(model_path="../models/",
#                                    dataset_settings=[("MNIST+EI339", "MNIST+EI339")],
#                                    num_classes=20,
#                                    batch_size=30, lr=1e-3, epoch=10)
# # model_path = train_report["model_fn"]
# model_path = "../models/LeNet5__lr=0.00100__epoch=10__batch_size=32__epoch=10.pth"
#
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

# train_report = LeNet5__train.train(model_path="../models/",
#                                    dataset_settings=[("MNIST+EI339", "MNIST+EI339")],
#                                    num_classes=20,
#                                    batch_size=30, lr=1e-3, epoch=10)
# model_path = train_report["model_fn"]
model_path = "../models/LeNet5__lr=0.00100__epoch=10__batch_size=32__epoch=10.pth"
model_predictor = LeNet5__predict.Predict(model_path=model_path, num_classes=20)
max_err = 5
sb_solver_obj = sudoku_solver.SudokuSolver(max_err=max_err)

test_img_group = load_imgs.get_image_fns(key="Jilin_2")
test_img_group_manual_single_target = 4  # None  #
if test_img_group_manual_single_target is not None:
    test_img_group = test_img_group[test_img_group_manual_single_target:
                                    test_img_group_manual_single_target + 4]
test_image = "../imgs/sudoku_puzzle.jpg"
out_path = "num_changed_plot"

SOLVE_SINGLE = False  # True
SOLVE_ALL = True
PLOTS_SHOW = True

# ===== solve test image =====
if SOLVE_SINGLE:
    bd_img_arr_obj = board_img_arr.BoardImgArr(
        img_path=test_image, roi_shape=(28, 28), norm_255=False)
    # bd_img_arr_obj.show_board_cells()
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

# ===== all images =====
if SOLVE_ALL:
    num_changed_list = [10, 4, 10, 4, ]
    # solve if not all solved
    if test_img_group_manual_single_target is not None or len(num_changed_list) != len(test_img_group):
        num_changed_list = []
        for img_idx, img_name in enumerate(test_img_group):
            print("[%d/%d]" % (img_idx + 1, len(test_img_group)), os.path.split(img_name)[1], "...")
            bd_img_arr_obj = board_img_arr.BoardImgArr(
                img_path=img_name, roi_shape=(28, 28), norm_255=False)
            cells_img = bd_img_arr_obj.get_cells_imgs()
            pred_cells_num = [model_predictor.predict(roi=roi) if roi is not None else -99
                              for roi in cells_img]
            pred_cells_num = np.array(pred_cells_num)
            bd_img_arr_obj.update_cells_nums(numbers=pred_cells_num)
            sb_obj = sudoku_board.SudokuBoard(board=bd_img_arr_obj.get_cells_nums(),
                                              invalid_tolerable=True)
            # print(sb_obj.output_board_as_str(line_prefix="\t"))
            empties_cnt, solved, solved_board = \
                sb_solver_obj.solve(board=sb_obj, method="backtrack")
            if solved:
                empties_cnt = 0 if -1 == empties_cnt else empties_cnt
            else:
                empties_cnt = 10
            num_changed_list.append(empties_cnt)
            if test_img_group_manual_single_target is not None:
                print()
                print(img_name, " => ", empties_cnt)
                print()

        if test_img_group_manual_single_target is None:
            num_changed_list = np.array(num_changed_list)
            np.save(os.path.join(out_path, "LeNet5__num_changed.npy"),
                    num_changed_list)
            print(num_changed_list)

    # plot
    if test_img_group_manual_single_target is None:
        plt.close("all")
        fig, ax_violin = plt.subplots()
        ax_unsolved = plt.twinx(ax_violin)
        changed_num_reshaped = np.array(num_changed_list).reshape(5, -1)
        plot_values = []
        plot_unsolved_ratio = []
        for view_idx in range(5):
            _plot_val = changed_num_reshaped[view_idx]
            plot_val = _plot_val[np.where(_plot_val < 10)]
            plot_values.append(plot_val)
            plot_unsolved_ratio.append(100 - 100. * plot_val.size / _plot_val.size)
        violin = ax_violin.violinplot(plot_values, positions=np.arange(5),
                                      showmeans=True, showextrema=True, showmedians=True)
        ratio_line = ax_unsolved.plot(plot_unsolved_ratio, marker="o", color="orange",
                                      label="Unsolved Boards Ratio")
        # handle legends
        label_flag = []
        label_str = []
        color = violin["bodies"][0].get_facecolor().flatten()
        label_flag.append(mpatches.Patch(color=color))
        label_str.append("# Cells Changed")
        label_flag.append(ratio_line[0])
        label_str.append("Unsolved Boards Ratio")
        # labels
        ax_violin.set_ylabel("# Cells Changed in Solved Boards")
        ax_unsolved.set_ylabel("Unsolved Board Ratio / %")
        ax_violin.set_xlabel("Image Views")
        fig.suptitle("Distribution of # Cells Changed in Solved Boards and Unsolved Boards Ratio\n"
                     "of Sudoku Images Taken from Different Views")
        plt.xticks(np.arange(5), ["Bird View", "Lower", "Left", "Upper", "Right"])
        plt.title(r"Using LeNet-5 + Solver, Try Solving with $\leq$ %d Cells Changed" % max_err, fontsize=9)
        plt.legend(label_flag, label_str, loc="best")
        fig.subplots_adjust(top=0.86)
        plt.savefig(s.path.join(out_path, "LeNet5_num_changed.png"), dpi=200)
        plt.show() if PLOTS_SHOW else print(end="")
