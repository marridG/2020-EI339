import os
import numpy as np
from opencv_sudoku_solver.pyimagesearch.sudoku import \
    find_puzzle, extract_digit, multi_img_view
from matplotlib import pyplot as plt
import cv2
import typing


class BoardImgArr:
    def __init__(self, img_path: str,
                 roi_shape: typing.Tuple[int, int],
                 norm_255: bool = True,
                 final_ei339: bool = False,
                 debug_board_detection: bool = False,
                 debug_digits_extraction: bool = False,
                 debug_digits_comparison: bool = False):
        """
        :param img_path:                Path of the input image of the board
        :param roi_shape:               Shape of the ROIs, dim=2 (e.g. (28,28))
        :param norm_255:                Whether to normalize by "img/=255."
                                            required True by the TF approach here
                                            required False by the Torch approach here
        :param final_ei339:   whether the final mode is on, for EI339 final test images only
        :param debug_board_detection:   Intermediate images during board detection
        :param debug_digits_extraction: Intermediate images during digits extraction
        :param debug_digits_comparison: Intermediate images of cell & extracted digit
        """
        assert os.path.exists(img_path), "[Error] Input Board Image NOT Found"
        img = cv2.imread(img_path)
        self.img = img
        self.roi_shape = roi_shape
        self.norm_255 = norm_255
        self.final_ei339 = final_ei339
        self.debug_board_detection = debug_board_detection
        self.debug_digits_extraction = debug_digits_extraction
        self.debug_digits_comparison = debug_digits_comparison

        self.board_cells_cnt = 9 * 9
        self.board_size = 9

        # ROI of cells flattened to len=81:
        #   upper-left to lower-right, count columns on one each row first
        # None or guaranteed to be of shape self.roi_shape
        self.img_cells = []
        # Numbers of cells: -99=Empty or <int>number
        self.num_cells = np.full((self.board_size, self.board_size), -99, dtype=np.int)
        self.num_cells_updated = False
        # Cell locations on the input board image: (x,y)
        self.cell_loc = []

        self.__detect_board__()

    def __detect_board__(self) -> None:
        """
        Detect board and extract cells. Update & Store:
            1. Images of cells: in self.img_cells
            2. Locations of cells on the board image: in self.cell_loc
        Based on: https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
        """
        # find the puzzle in the image and then
        puzzle_image, warped = find_puzzle(self.img)

        # a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
        # infer the location of each cell by dividing the warped image
        # into a 9x9 grid
        step_x = warped.shape[1] // 9
        step_y = warped.shape[0] // 9

        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                start_x = x * step_x
                start_y = y * step_y
                end_x = (x + 1) * step_x
                end_y = (y + 1) * step_y

                # add the (x, y)-coordinates to our cell locations list
                row.append((start_x, start_y, end_x, end_y))

                # crop the cell from the warped transform image and then
                # extract the digit from the cell
                cell_img = warped[start_y:end_y, start_x:end_x]
                digit_img = extract_digit(
                    cell_img, debug=self.debug_digits_extraction, final=self.final_ei339)

                if digit_img is None:
                    self.img_cells.append(None)
                else:
                    if self.debug_digits_comparison:
                        debug_imgs = [cell_img, digit_img]
                        debug_subtitles = ["Cell", "Digit"]
                        _ = multi_img_view.multi_img_view(
                            images=debug_imgs, subtitles=debug_subtitles, col_cnt=3, row_cnt=2,
                            title="Extract Digits", fig_size=None, close_all=True)
                        plt.show()

                    # resize the cell to 28x28 pixels and then prepare the
                    # cell for classification
                    # *** here, you can also use:
                    #   from tensorflow.keras.preprocessing.image import img_to_array
                    #   roi = img_to_array(roi)
                    roi = cv2.resize(digit_img, (28, 28))
                    if self.norm_255:
                        roi = roi.astype("float") / 255.0  # (28, 28)
                    roi = roi.reshape(self.roi_shape)

                    self.img_cells.append(roi)

            # add the row to our cell locations
            self.cell_loc.append(row)

    def show_board_cells(self) -> None:
        """
        Show how the flattened result, as describe in __init__() self.img_cells
        :return:
        """
        cell_titles = [""] * self.board_cells_cnt
        cell_imgs = [img.reshape(img.shape[0], img.shape[1])
                     if img is not None else None
                     for img in self.img_cells]
        fig = multi_img_view.multi_img_view(
            images=cell_imgs, subtitles=cell_titles, col_cnt=9, row_cnt=9,
            title="Cells ROI", fig_size=None, close_all=True)
        # fig.tight_layout()
        plt.show()

    def output_board_cells(self) -> np.ndarray:
        """
        Output the compact flattened result, as describe in __init__() self.img_cells
        :return:
        """
        (height, width) = self.roi_shape
        row_cnt = cnt_per_col = self.board_size
        col_cnt = cnt_per_row = self.board_size

        final_img = np.zeros((height * row_cnt, width * col_cnt, 1),
                             dtype=np.float32 if self.norm_255 else np.uint8)
        for col_idx in range(col_cnt):
            for row_idx in range(row_cnt):
                img_idx = col_idx + row_idx * cnt_per_col
                img = self.img_cells[img_idx]
                if img is None:
                    continue
                img = img.reshape(self.roi_shape[0], self.roi_shape[1], 1)
                # cv2.imshow("%d, %d, %d" % (img.shape[0], img.shape[1], img.shape[2]), img)
                # cv2.waitKey(0)
                dim1_start = row_idx * height
                dim1_end = dim1_start + height
                dim2_start = col_idx * width
                dim2_end = dim2_start + width
                # print("col=%d, row=%d, img=%d, dim1=%d-%d, dim2=%d-%d" %
                #       (col_idx, row_idx, img_idx, dim1_start, dim1_end, dim2_start, dim2_end))
                final_img[dim1_start:dim1_end, dim2_start:dim2_end, :] = img[:]
        # print(final_img.shape)
        return final_img

    def get_cells_imgs(self) -> typing.List[np.ndarray or None]:
        """
        Get the ROI images of the cells, flattened as describe in
            __init__() self.img_cells
        :return:
        """
        return self.img_cells

    def update_cells_nums(self, numbers: np.ndarray) -> None:
        """
        Update the numbers of cells
        :param numbers:         Numbers of the cells, flattened
                                    as describe in __init__() self.img_cells
        :return:
        """
        assert self.board_cells_cnt == numbers.size, \
            "[Error] Count of Given Numbers Mismatch. Expected %d, Got %d" \
            % (self.board_cells_cnt, numbers.size)
        self.num_cells[:, :] = numbers.reshape((self.board_size, self.board_size))
        self.num_cells_updated = True

    def get_cells_nums(self) -> np.ndarray:
        """
        Get the numbers of cells, shape (board_size, board_size)
        :return:                Numbers of cells
        """
        assert self.num_cells_updated, "[Error] Cells' Numbers NOT Updated yet."
        return self.num_cells


if "__main__" == __name__:
    test_image = "../imgs/sudoku_puzzle.jpg"
    bia_obj = BoardImgArr(img_path=test_image, roi_shape=(28, 28))
    bia_obj.show_board_cells()
