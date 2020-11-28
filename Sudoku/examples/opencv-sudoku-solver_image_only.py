# import the necessary packages
import numpy as np
import cv2

# import modules
from opencv_sudoku_solver.pyimagesearch.sudoku import find_puzzle, extract_digit

test_image = cv2.imread("sudoku_puzzle.jpg")

# find the puzzle in the image
puzzleImage, warped = find_puzzle(image=test_image, debug=True)

# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []
# loop over the grid locations
for y in range(0, 9):
    # initialize the current list of cell locations
    row = []
    for x in range(0, 9):
        # compute the starting and ending (x, y)-coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY
        # add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then
        # extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        _ = extract_digit(cell, debug=True)
