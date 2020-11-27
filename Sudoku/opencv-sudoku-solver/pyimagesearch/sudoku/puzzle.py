# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

# import modules
from . import multi_img_view


def find_puzzle(image: np.ndarray, debug: bool = False) -> (np.ndarray, np.ndarray):
    """
    Find the puzzle in the input image
    :param image:   image data
    :param debug:   whether to show intermediate images
    :return:        transformed (bird view) image in (1) RGB and (2) Grayscale
    """
    debug_imgs = []  # for debug only, to-show imgs
    debug_subtitles = []  # for debug only, subtitles of the to-show imgs

    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # check to see the operations done on the original image
    if debug:
        debug_imgs.append(image.copy())
        debug_subtitles.append("Puzzle Original")
        debug_imgs.append(gray.copy())
        debug_subtitles.append("Puzzle Gray")
        debug_imgs.append(blurred.copy())
        debug_subtitles.append("Puzzle Blurred")

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        debug_imgs.append(thresh.copy())
        debug_subtitles.append("Puzzle Thresh")

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzle_cnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzle_cnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the sudoku puzzle so raise an error
    if puzzle_cnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzle_cnt], -1, (0, 255, 0), 2)
        debug_imgs.append(output)
        debug_subtitles.append("Puzzle Outline")

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down birds eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzle_cnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        debug_imgs.append(puzzle.copy())
        debug_subtitles.append("Puzzle Transform")

    # Show Debug Images
    if debug:
        _ = multi_img_view.multi_img_view(
            images=debug_imgs, subtitles=debug_subtitles, col_cnt=3, row_cnt=2,
            title="Find Puzzle", fig_size=None, close_all=True)
        plt.show()

    # return a 2-tuple of puzzle in both RGB and grayscale
    return puzzle, warped


def extract_digit(cell: np.ndarray, debug: bool = False):
    """

    :param cell:    an ROI representing an individual cell of the Sudoku puzzle
                        (it may or may not contain a digit)
    :param debug:   whether to show intermediate images
    :return:        None if no digit contours found
    """
    debug_imgs = []  # for debug only, to-show imgs
    debug_subtitles = []  # for debug only, subtitles of the to-show imgs
    if debug:
        debug_imgs.append(cell.copy())
        debug_subtitles.append("Cell Original")

    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        debug_imgs.append(thresh.copy())
        debug_subtitles.append("Threshold")
    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    if debug:
        debug_imgs.append(thresh.copy())
        debug_subtitles.append("Thresh without Border")

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        if debug:  # show debug images before return
            _ = multi_img_view.multi_img_view(
                images=debug_imgs, subtitles=debug_subtitles, col_cnt=2, row_cnt=2,
                title="Extract Digits - Returned No Contour", fig_size=None, close_all=True)
            plt.show()
        return None

    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    if debug:
        debug_imgs.append(mask.copy())
        debug_subtitles.append("Mask for the Largest Contour")

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percent_filled < 0.03:
        if debug:  # show debug images before return
            _ = multi_img_view.multi_img_view(
                images=debug_imgs, subtitles=debug_subtitles, col_cnt=2, row_cnt=2,
                title="Extract Digits - Returned <3%", fig_size=None, close_all=True)
            plt.show()
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # check to see if we should visualize the masking step
    if debug:
        debug_imgs.append(digit.copy())
        debug_subtitles.append("Digit by Mask")

    # Show Debug Images
    if debug:
        _ = multi_img_view.multi_img_view(
            images=debug_imgs, subtitles=debug_subtitles, col_cnt=3, row_cnt=2,
            title="Extract Digits", fig_size=None, close_all=True)
        plt.show()

    # return the digit to the calling function
    return digit
