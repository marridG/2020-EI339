# Sudoku Project Report
EI339 Artificial Intelligence, 2020 Fall, SJTU
**by Prof. Jun Z.**
<!-- <br> -->


<br>



<!-- MarkdownTOC -->

- [Description](#description)
- [Environment](#environment)
- [Task 1 - OpenCV Sudoku Solver and OCR](#task-1---opencv-sudoku-solver-and-ocr)
    - [Image Preprocessing](#image-preprocessing)
        - [Multi-Image View](#multi-image-view)
        - [Board Detection](#board-detection)
        - [Digits Extraction](#digits-extraction)
    - [Neural Network - `SudokuNet`](#neural-network---sudokunet)
    - [Execution](#execution)
        - [Training](#training)
        - [Puzzle Solving](#puzzle-solving)
- [Task 2 - Sudoku Solver](#task-2---sudoku-solver)
    - [Sudoku Board](#sudoku-board)

<!-- /MarkdownTOC -->




<br>

<div style="page-break-after: always;"></div>




<a id="description"></a>
## Description
<!-- 1. The project is based on Project 2 **Linux Kernel Module for Task Information** of Chapter 3 of *Operating System Concepts (10th Edition) by Abraham Silberschatz, Peter Baer Galvin, Greg Gagne*, with [source codes](https://github.com/greggagne/osc10e) provided.
2. The major tasks of the project are
    + Writing to the `/proc` File System
        * Copy the stored user input into kernel memory
        * Translate to the PID integer
    + Reading from the `/proc` File System
        * Fetch the process/task information of the assigned PID
        * Print three fields:
            - the command the task is running
            - the value of the task’s PID
            - the current state of the task -->
<br>

<a id="environment"></a>
## Environment
+ OS: `Windows 8.1 Pro`  
+ Python Interpreter: `Python 3.7.6 MSC v.1916 64bit on win32`  
+ IDE: `Pycharm 2020.1.1 (Professional Edition), Build #PY01-201.7223.92`  

<br>



<a id="task-1---opencv-sudoku-solver-and-ocr"></a>
## Task 1 - OpenCV Sudoku Solver and OCR
This part is mainly based on the provided codes ([source](https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)), the explanations of which is almost fully included in the post. For simplicity, we do not repeat details mentioned already.  
As a matter of fact, the most helpful part may be the magic-like OpenCV operations on the original image. We take advantage of its board-detection and digit-image-extraction. As for the digit recognization, we will reconstruct another network later.


<a id="image-preprocessing"></a>
### Image Preprocessing
Python OpenCV magics.

<a id="multi-image-view"></a>
#### Multi-Image View
To view and compare images more conveniently in one figure, I implement `multi_img_view(images, row_cnt, col_cnt, title, fig_size, close_all)`, the full codes of which is given by,

```Python
import typing
import matplotlib
from matplotlib import figure
from matplotlib import pyplot as plt

def multi_img_view(images: list, subtitles: list,
                   row_cnt: int, col_cnt: int,
                   title: str = "MULTI-IMG VIEW",
                   fig_size: typing.Tuple[int, int] = None,
                   close_all: bool = True) \
        -> matplotlib.figure.Figure:
    """
    Combine and Show Several Images Together
    :param images:          list of images
    :param subtitles:       list of the subtitles of the images
    :param row_cnt:         number of rows
    :param col_cnt:         number of columns
    :param title:           title of the combine image
    :param fig_size:        size of the figure, (width, height)
    :param close_all:       whether to close all figures during initialization
    :return:                plt, fig
    """
    if close_all:
        plt.close("all")
    if len(images) != len(subtitles):
        raise RuntimeError("[Error] Images Count and Subtitles Count Mismatch:"
                           "Images = %d, Subtitles = %d" % (len(images), len(subtitles)))
    if len(images) > row_cnt * col_cnt:
        raise RuntimeError("[Error] Images Count Overflow:"
                           "Got Max row*col=%d*%d, Assigned %d" % (row_cnt, col_cnt, len(images)))

    if fig_size is not None:
        fig, _ax = plt.subplots(nrows=row_cnt, ncols=col_cnt, figsize=fig_size)
    else:
        fig, _ax = plt.subplots(nrows=row_cnt, ncols=col_cnt)
    ax = _ax.flatten()

    # Subplot Styles: remove spines, x/y-ticks
    for _subplot_ax in ax:
        _subplot_ax.spines['top'].set_visible(False)
        _subplot_ax.spines['right'].set_visible(False)
        _subplot_ax.spines['bottom'].set_visible(False)
        _subplot_ax.spines['left'].set_visible(False)
        _subplot_ax.set_xticks([])
        _subplot_ax.set_yticks([])

    # Show Images & Subtitles
    for _img_idx, (_img, _img_title) in enumerate(zip(images, subtitles)):
        ax[_img_idx].imshow(_img)
        ax[_img_idx].set_title(_img_title)
        ax[_img_idx].set_xticks([])
        ax[_img_idx].set_yticks([])

    fig.suptitle(title)
    # fig.tight_layout()

    return fig
```

<a id="board-detection"></a>
#### Board Detection
This functionality is implemented in file 
`/opencv-sudoku-solver/pyimagesearch/sudoku/puzzle.py`, as,
```Python
find_puzzle(image: np.ndarray, debug: bool = False) -> (np.ndarray, np.ndarray)
```
By changing popping out debug images to adding to multi-image-view group, for the sample image, we may get the intermediate images, as,

<img src="pics/0-1.PNG" alt="drawing" width="60%; margin:auto auto;"/>


<a id="digits-extraction"></a>
#### Digits Extraction
This functionality is implemented in file 
`/opencv-sudoku-solver/pyimagesearch/sudoku/puzzle.py`, as,
```Python
extract_digit(cell: np.ndarray, debug: bool = False)
```
By changing popping out debug images to adding to multi-image-view group, for the sample image, we may get the intermediate images, in cases where there is/is not a digit, as,

<img src="pics/0-2.PNG" alt="drawing" width="90%; margin:auto auto;"/>


<a id="neural-network---sudokunet"></a>
### Neural Network - `SudokuNet`
The architecture of `SudokuNet`, implemented in `/opencv-sudoku-solver/pyimagesearch/models/sudokunet.py`, is depicted as follows, (generated using [tools](http://alexlenail.me/NN-SVG/AlexNet.html))
<img src="pics/0-3.PNG" alt="drawing" width="100%; margin:auto auto;"/>

<a id="execution"></a>
### Execution
<a id="training"></a>
#### Training
By executing `python train_digit_classifier.py --model output/digit_classifier_new.h5`, we get the following outputs,
<img src="pics/1-1.PNG" alt="drawing" width="100%; margin:auto auto;"/>

<a id="puzzle-solving"></a>
#### Puzzle Solving
By executing `python solve_sudoku_puzzle.py --model output/digit_classifier_new.h5 --image sudoku_puzzle.jpg`, we use the model trained above to solve the sample sudoku problem. The outputs (combined) are shown as follows,
<img src="pics/1-2.PNG" alt="drawing" width="100%; margin:auto auto;"/>


<br>

<div style="page-break-after: always;"></div>



<a id="task-2---sudoku-solver"></a>
## Task 2 - Sudoku Solver
<a id="sudoku-board"></a>
### Sudoku Board
