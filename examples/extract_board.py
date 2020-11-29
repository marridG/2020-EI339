import cv2
from extract_sudoku import board_img_arr

# using wrapped API of opencv-sudoku-solver

test_image = cv2.imread("1-2__from_lower.jpg")
bia = board_img_arr.BoardImgArr(img=test_image,
                                debug_board_detection=False,
                                debug_digits_extraction=False,
                                debug_digits_comparison=False)
bia.show_board_cells()
