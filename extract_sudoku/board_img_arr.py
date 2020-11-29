import numpy as np
from opencv_sudoku_solver.pyimagesearch.sudoku import find_puzzle, extract_digit, multi_img_view
from matplotlib import pyplot as plt
import cv2
import typing


class BoardImgArr:
    def __init__(self, img: np.ndarray, debug: bool = False):
        """
        :param img:         Input image of the board
        :param debug:       Debug switch
        """
        self.img = img
        self.debug = debug

        self.board_cells_cnt = 9 * 9
        self.board_size = 9

        self.debug_board_detection = self.debug and True
        self.debug_digits_extraction = self.debug and False
        self.debug_digits_comparison = self.debug and False

        # ROI of cells flattened to len=81:
        #   upper-left to lower-right, count columns on one each row first
        # None or guaranteed to be of shape (1,28,28,1)
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
        """
        # find the puzzle in the image and then
        puzzle_image, warped = find_puzzle(self.img, debug=self.debug_board_detection)

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
                digit_img = extract_digit(cell_img, debug=self.debug_digits_extraction)

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
                    roi = roi.astype("float") / 255.0  # (28, 28)
                    roi = roi.reshape((roi.shape[0], roi.shape[1], 1))  # (28, 28, 1), ***
                    roi = np.expand_dims(roi, axis=0)  # (1, 28, 28, 1)

                    self.img_cells.append(roi)

            # add the row to our cell locations
            self.cell_loc.append(row)

    def show_board_cells(self) -> None:
        """
        Show how the flattened result, as describe in __init__() self.img_cells
        :return:
        """
        cell_titles = [""] * self.board_cells_cnt
        cell_imgs = [img.reshape(img.shape[1], img.shape[2])
                     if img is not None else None
                     for img in self.img_cells]
        fig = multi_img_view.multi_img_view(
            images=cell_imgs, subtitles=cell_titles, col_cnt=9, row_cnt=9,
            title="Cells ROI", fig_size=None, close_all=True)
        # fig.tight_layout()
        plt.show()

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
        self.num_cells[:, :] = numbers.reshape(shape=(self.board_size, self.board_size))
        self.num_cells_updated = True

    def get_cells_nums(self) -> np.ndarray:
        assert self.num_cells_updated, "[Error] Cells' Numbers NOT Updated yet."
        return self.num_cells


if "__main__" == __name__:
    test_image = cv2.imread("../imgs/sudoku_puzzle.jpg")
    bia_obj = BoardImgArr(img=test_image)
    bia_obj.show_board_cells()
