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
    - [Execution - OpenCV Sudoku Solver](#execution---opencv-sudoku-solver)
        - [Training](#training)
        - [Puzzle Solving](#puzzle-solving)
- [Task 2 - Sudoku Solver](#task-2---sudoku-solver)
    - [Sudoku Board](#sudoku-board)
    - [Sudoku Solver](#sudoku-solver)
        - [Error Tolerable Solver](#error-tolerable-solver)
    - [Execution - Sudoku Board and Solver](#execution---sudoku-board-and-solver)
- [Task 3 - LeNet-5](#task-3---lenet-5)
    - [Data Loader](#data-loader)
    - [Execution - Hyper-Parameters of Training](#execution---hyper-parameters-of-training)
- [Execution - Classifiers + Solver](#execution---classifiers--solver)
        - [SudokuNet + Solver](#sudokunet--solver)
            - [Single Test Image](#single-test-image)
            - [Multiple Test Images](#multiple-test-images)
        - [LeNet-5 + Solver](#lenet-5--solver)
- [Appendix](#appendix)
    - [Multi-Image View Implementation](#multi-image-view-implementation)
    - [Sudoku Board Implementation](#sudoku-board-implementation)
    - [Sudoku Solver Implementation](#sudoku-solver-implementation)

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


<div style="page-break-after: always;"></div>







<a id="task-1---opencv-sudoku-solver-and-ocr"></a>
## Task 1 - OpenCV Sudoku Solver and OCR
This part is mainly based on the provided codes ([source](https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)), the explanations of which is almost fully included in the post. For simplicity, we do not repeat details mentioned already.  
As a matter of fact, the most helpful part may be the magic-like OpenCV operations on the original image. We take advantage of its board-detection and digit-image-extraction. As for the digit recognization, we will reconstruct another network later.



<br>

<a id="image-preprocessing"></a>
### Image Preprocessing
Python OpenCV magics.

<a id="multi-image-view"></a>
#### Multi-Image View
To view and compare images more conveniently in one figure, I implement `multi_img_view(images, row_cnt, col_cnt, title, fig_size, close_all)`, the full codes of which is given in [Appendix: Multi-Image View Implementation](#multi-image-view-implementation)



<a id="board-detection"></a>
#### Board Detection
This functionality is implemented in file 
`/opencv-sudoku-solver/pyimagesearch/sudoku/puzzle.py`, as,
```Python
find_puzzle(image: np.ndarray, debug: bool = False) -> (np.ndarray, np.ndarray)
```
By changing popping out debug images to adding to multi-image-view group, for the sample image, we may get the intermediate images, as,

<img src="pics/0-1.PNG" alt="drawing" width="60%; margin:0 auto;"/>


<br>

<a id="digits-extraction"></a>
#### Digits Extraction
This functionality is implemented in file 
`/opencv-sudoku-solver/pyimagesearch/sudoku/puzzle.py`, as,
```Python
extract_digit(cell: np.ndarray, debug: bool = False)
```
By changing popping out debug images to adding to multi-image-view group, for the sample image, we may get the intermediate images, in cases where there is/is not a digit, as,

<img src="pics/0-2.PNG" alt="drawing" width="90%; margin:0 auto;"/>

<br>


<a id="neural-network---sudokunet"></a>
### Neural Network - `SudokuNet`
The architecture of `SudokuNet`, implemented in `/opencv-sudoku-solver/pyimagesearch/models/sudokunet.py`, is depicted as follows, (generated using [tools](http://alexlenail.me/NN-SVG/AlexNet.html))

<img src="pics/0-3.PNG" alt="drawing" width="100%; margin:0 auto;"/>



<br>


<a id="execution---opencv-sudoku-solver"></a>
### Execution - OpenCV Sudoku Solver
<a id="training"></a>
#### Training
By executing `python train_digit_classifier.py --model output/digit_classifier_new.h5`, we get the following outputs,
<img src="pics/1-1.PNG" alt="drawing" width="100%; margin:0 auto;"/>

<a id="puzzle-solving"></a>
#### Puzzle Solving
By executing `python solve_sudoku_puzzle.py --model output/digit_classifier_new.h5 --image sudoku_puzzle.jpg`, we use the model trained above to solve the sample sudoku problem. The outputs (combined) are shown as follows,
<img src="pics/1-2.PNG" alt="drawing" width="100%; margin:0 auto;"/>







<br>

<div style="page-break-after: always;"></div>



<a id="task-2---sudoku-solver"></a>
## Task 2 - Sudoku Solver
Intuitively, we may divide the task into two parts,

+ Implement a class of sudoku board, which supports actions directly related to the board such as
    * store the board
    * check the validation of the board
    * print the numbers of the board
    * get possible numbers of each cell
+ Implement a solver based on the board class

Both parts are introduced below.



<br>


<a id="sudoku-board"></a>
### Sudoku Board
The basic structure of the board class, implemented in `/solve-sudoku/sudoku_board.py`, is given in [Appendix: Sudoku Board Implementation](#sudoku-board-implementation)


<br>


<a id="sudoku-solver"></a>
### Sudoku Solver
The basic structure of the solver class, implemented in `/solve-sudoku/sudoku_solver.py`, is given in [Appendix: Sudoku Solver Implementation](#sudoku-solver-implementation)  
The solver itself is straightforward to implement, as a CSP problem. We may simply use the recursive backtracking approach as follows:

+ Iterate through all unfilled cells (for simplicity, flatten the board into an line vector)
+ For each iterated cell, try all its possible values (through the functionality supported by the `SudokuBoard` class).
    * If such values do not exist and the board is not all filled, then something goes wrong before.
    * If such values do not exist and the board is all filled, then a solution is acquired.

<a id="error-tolerable-solver"></a>
#### Error Tolerable Solver
Frankly speaking, if the board is incorrect, i.e., the predicted numbers given by the neural network is wrong, the problem itself is not related to the original one any longer. Thus, error correction is unnecessary at all.  
However, due to the requirements, here, we propose a simple but effective approach for error-tolerable sudoku boards.  
The approach is somehow straightforward, as,

+ The error correction process is entered in two cases,
    * The to-solve board itself is invalid.
    * No solutions are found for the initially seemingly valid board, i.e., actually invalid all after.
+ The error correction process is committed as follows,
    * Begin at the initial board.
    * Extract all indices of nonempty cells, i.e., indices of cells assigned with possibly wrong numbers.
    * Iterate from 1 to the number of such cells. The iteration index illustrates the number of nonempty cells to change.
    * In the current iteration upon the number of nonempty cells to change, 
        - Generate all the combinations.
            + e.g. Suppose cells indexed `[0, 1, 2, 3]` are nonempty and that we have iterated to index 2, then the combinations are `[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), ]`
        - For each combination, simply empty all the cells in the combination, since,
            + Intuitively, we ought to change the values of the cells in the combination to one of the possible values of them
            + The intuitive attempts are exactly the same as the backtracking search of a unfilled cell.
        - If a solution is found for a certain combination, the error-tolerable solution is found.
    * Notice that, a solution can always be found, since, in the worst case, we may empty all nonempty cells (resulting in "solving" a completely empty board).


<br>

<a id="execution---sudoku-board-and-solver"></a>
### Execution - Sudoku Board and Solver
As depicted in the following figure, given an invalid board as in the left, we may change one cell to get a solution as in the right.  
<img src="pics/2-1_err.PNG" alt="drawing" width="80%; margin:0 auto;"/>







<br>

<div style="page-break-after: always;"></div>



<a id="task-3---lenet-5"></a>
## Task 3 - LeNet-5
Based on `LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86, no. 11 (1998): 2278-2324`, we may implement the proposed LeNet-5 structure, as illustrated in the figure below,

<img src="pics/3-1.PNG" alt="drawing" width="100%; margin:0 auto;"/>

In modern frameworks, some tricks of `LeNet-5` (like layer $C3$, originally proposed due to the computation limits that time) are unnecessary at all. Thus, we may further simplify the network connections and add nonlinear activations (say, ReLU) for better performance.

To get familiar with `PyTorch`, LeNet-5 is implemented in the framework of `PyTorch` from scratch, instead of in `TensorFlow` based on the [Network `SudokuNet` of the post](#neural-network---sudokunet).


<br>


<a id="data-loader"></a>
### Data Loader
To feed the training data and prepare the test data, i.e., while combining datasets `MNIST` and `EI339`, where `EI339` is the dataset of handwritten Chinese numbers (1-10) created by all students in the course, notice that,

* Even if instructed, the idea of **dividing** raw `EI339` dataset images only according to student-ID is **far from reasonable**. 
    - The division is done as,
        + All the students (denote as set $S$) are divided into two non-intersect subsets, say $A, B \subset S$, $A+B=S$.
        + The training set is constructed only by all the images from students in subset $A$.
        + The test set is constructed only by all the images from students in subset $B$.
    - However, intuitively, there are apparent problems. 
        + Suppose student $b \in B$. Thus, images from $b$ are all categorized as test set.
        + As a result, no patterns of B's handwriting can be learned in the training, which makes the prediction almost meaningless.
        + After all, we are conducting the traditional task of number classifications, instead of real-time predictions, where future input samples cannot be estimated.
* For better performance, **several shuffles** are done, as,
    - shuffle the raw images of `EI339` before constructing the dataset file
    - shuffle the concatenated data from `MNIST` and `EI339` before constructing the combined dataset file
    - At the same time, the load of dataset files are remained to be flagged with shuffle enabled.


<br>

<a id="execution---hyper-parameters-of-training"></a>
### Execution - Hyper-Parameters of Training
**Firstly**, how the LeNet-5 model learns the features of the input training data.  
As is shown in the below figure (invisible elements are of values exactly 0),

+ The overall learning of any of `MNIST`, `EI339` and `MNIST+EI339` is satisfactory enough.
+ As expected, the features of digits (represented by `MNIST`) and Chinese numbers (represented by `EI339`) are quite different, resulting in extremely low test accuracies, as 0.

<img src="pics/5-1_datasets.png" alt="drawing" width="60%; margin:0 auto;"/>


**Secondly**, the impact of `BatchSize` and `Epoch` values on the test accuracies.  
As is shown in the figure below *(left: `BatchSize`; right: `Epoch`)*, there are,

+ As expected, the larger the `BatchSize`, the lower the overall test accuracies are. However, quite intuitively, the smaller the `BatchSize`, the more time the training consumes. Thus, an intermediate value should be chosen. Commonly, values of about 30 are selected for the image classification tasks.
+ As expected, the larger the `Epoch`, the higher the overall test accuracies are. Similarly, large `Epoch` values results in longer training time. Proper values depend on whether a fine-grained model is required.  
<img src="pics/5-2&3_bs&ep.png" alt="drawing" width="100%; margin:0 auto;"/>
<!-- <img src="pics/5-2_batch_size.png" alt="drawing" width="60%; margin:0 auto;"/>
<img src="pics/5-3_epoch.png" alt="drawing" width="60%; margin:0 auto;"/> -->



**Thirdly**, the impact of `LearningRate` values on the train and test results.  
As is shown in the figure below *(left: train loss; right: train accuracy tested upon training set after each training epoch)*, we have the following observations, (from which the major training babysitting is done)

+ As expected, proper `LearningRate` values (e.g. `5e-3, 1e-3`, etc.) result in a decreasing train loss.
+ `LearningRate` values (`1e-1, 5e-2`) are too large, resulting in bad loss convergence (in an increasing trend) and accuracy (in a decreasing trend).
+ Among the remaining `LearningRate` values (`1e-2, 5e-3, 1e-3, 5e-4, 1e-4`), the loss and accuracy curves are all not too steep or too shallow. Thus, with the execution time being taken into consideration, we may select any of them. Notice that,
    * A steep curve indicate a proper but still too large `LearningRate`.
    * A shallow curve indicate a proper but still too small `LearningRate`.

<img src="pics/5-4&5_lr_train_acc&loss.png" alt="drawing" width="100%; margin:0 auto;"/>
<!-- <img src="pics/5-4_lr_train_acc.png" alt="drawing" width="60%; margin:0 auto;"/>
<img src="pics/5-5_lr_train_loss.png" alt="drawing" width="60%; margin:0 auto;"/> -->


Meanwhile, we may decide `LearningRate` further based on the trends shown in the below figure *(left: train loss; right: train accuracy tested upon training set after each training epoch)*, as, 

+ (From left) too large `LearningRate` values result in low test accuracies.
+ (From left) too small `LearningRate` values may lead to too slow learning updates to acquire most features in the given epochs.
+ (From right) the trained model demonstrates big gap between validation on training and test dataset, indicating over-fitting, which is quite comprehensive and supports the [previous argument about the bad data division](#data-loader)

<img src="pics/5-6&7_lr_test_acc_&_ diff_train_test_acc.png" alt="drawing" width="100%; margin:0 auto;"/>
<!-- <img src="pics/5-6_lr_test_acc.png" alt="drawing" width="60%; margin:0 auto;"/>
<img src="pics/5-7_lr_diff_train_test_acc.png" alt="drawing" width="60%; margin:0 auto;"/> -->




<div style="page-break-after: always;"></div>



<a id="execution---classifiers--solver"></a>
## Execution - Classifiers + Solver

Here we test how either of the two classifiers works together with the solver by,

+ Run on the same test image used in the post of the OpenCV approach.
+ Run on images of Sudoku problems captured by the author, with details given as,
    * purpose: analyze the overall effectiveness of using networks to extract sudoku boards (thus, and for economical concerns, for each image, at most 5 cells are allowed to be changed to reach a solution)
    * 100 images altogether
        - from 20 boards
        - each board from five angles of views: bird-view, lower, left, upper and right
    * all problems from the book `《全民数独 2 初级篇》 马荣鸿 吉林大学出版社`
    * images preview illustrated in the following figure,

<img src="pics/6-1_img_prev.png" alt="drawing" width="100%; margin:0 auto;"/>



<br>


<a id="sudokunet--solver"></a>
#### SudokuNet + Solver
<a id="single-test-image"></a>
##### Single Test Image
By connecting the SudokuNet model with the solver, we may get the following test results, *(*left*: test Sudoku image; *middle & right*: results)*  
<img src="pics/4-1_opencv.png" alt="drawing" width="100%; margin:0 auto;"/>

From which, 

+ 4 cells are classified wrongly.
+ At least 4 cells should be emptied to get a solution.


<a id="multiple-test-images"></a>
##### Multiple Test Images
By connecting the SudokuNet model with the solver, we may get the following test results upon the 100 images describe above,  
<img src="pics/6-2_opencv_num_changed.png" alt="drawing" width="60%; margin:0 auto;"/>

From which, 

+ Unsolved ratios are high, indicating that most board images are facing the problem of incorrect digit recognition. Since the number of filled cells is big, more errors appear and less possible the board can be solved within 5 changes of cells.
+ Angles of lower (dense distribution) and right (higher probability of a solution) are more preferable.



<br>

<a id="lenet-5--solver"></a>
#### LeNet-5 + Solver
By connecting the LeNet-5 model (trained with BatchSize=32, LearningRate=0.001, Epoch=10) with the solver, we may get the following test results, *(*left*: test Sudoku image; *middle & right*: results)*  
<img src="pics/4-2_lenet.png" alt="drawing" width="100%; margin:0 auto;"/>

From which, 

+ 4 cells are classified wrongly, among which not all are the same as those by SudokuNet.
+ At least 4 cells should be emptied to get a solution.





<div style="page-break-after: always;"></div>







<a id="appendix"></a>
## Appendix

<a id="multi-image-view-implementation"></a>
### Multi-Image View Implementation
[Back to Section Multi-Image View](#multi-image-view)
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



<br>

<a id="sudoku-board-implementation"></a>
### Sudoku Board Implementation
[Back to Section Sudoku Board](#sudoku-board)
```Python
class SudokuBoard:
    def __init__(self, board: np.ndarray,
                 invalid_tolerable: bool = False, show_info: bool = True):
        """
        :param board:           Input numbers of the board, of shape (BOARD_SIZE,BOARD_SIZE)
        :param invalid_tolerable: Flag, whether to tolerate invalid input board
        :param show_info:       Flag, wehther to show hint info
        """
        pass

    def __check_row__is_valid_n_used_nums__(self, row_idx: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific row is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        :param row_idx:         Index of the to-be-checked row
                                    (starting from 0, ending at row_cnt-1)
        :return:                1. True if valid, False if invalid;
                                2. Used numbers in the row
        """
        pass

    def __check_col__is_valid_n_used_nums__(self, col_idx: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific column is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        :param col_idx:         Index of the to-be-checked column
                                    (starting from 0, ending at col_cnt-1)
        :return:                1. True if valid, False if invalid;
                                2. Used numbers in the column
        """
        pass

    def __check_box__is_valid_n_used_nums__(self, box_idx_row: int, box_idx_col: int) \
            -> (bool, np.ndarray):
        """
        1. Check whether a specific box is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        Notice:
            1. boxes start from [0, 0] to [self.BOARD_SIZE, self.BOARD_SIZE]
            2. box indices start from 0, end at self.BOARD_SIZE / self.BOX_SIZE,
        :param box_idx_row:     Index of the to-be-checked box in the row
        :param box_idx_col:     Index of the to-be-checked box in the column
        :return:                True if valid, False if invalid
        """
        pass

    def __idx_row_col_2_box__(self, row_idx: int, col_idx: int) -> (int, int):
        """
        Map indices by (row, col) to the index of the corresponding box of the cell
        :param row_idx:         Index of the cell in the row
        :param col_idx:         Index of the cell in the column
        :return:                Indices of the box, by (row, col)
        """
        pass

    def __check_box_row_col__is_valid_n_used_nums__(self, row_idx: int, col_idx: int) \
            -> (bool, np.ndarray):
        """
        [WRAPPER] "__idx_row_col_2_box__()" and "__check_box__is_valid_n_used_nums__()"
        1. Check whether a specific box is valid: no duplicates in 1~MAX_NUM
        2. Return the used numbers
        Notice:
            1. boxes start from [0, 0] to [self.BOARD_SIZE, self.BOARD_SIZE]
            2. box indices start from 0, end at self.BOARD_SIZE / self.BOX_SIZE,
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                True if valid, False if invalid
        """
        pass

    def __validate_input_idx_is_valid(self, row_idx: int, col_idx: int) -> None:
        """
        Check whether a input index (row, col) is valid
        :param row_idx:         Input index of the to-be-checked cell in the row
        :param col_idx:         Input index of the to-be-checked cell in the column
        :return:
        """
        pass

    def update_board_valid_status(self) -> None:
        """
        Update the status of validation of the board
        :return:
        """
        pass

    def check_board_is_valid(self) -> (bool, str) or (bool, None):
        """
        Check whether the whole board is valid
        :return:                If valid, (True, None);
                                If invalid, (False, Failure_Str)
        """
        pass

    def check_cell_is_valid(self, row_idx: int, col_idx: int) -> bool:
        """
        Check whether a cell is valid
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                True if valid, False if invalid
        """
        pass

    def find_cell_possible_nums(self, row_idx: int, col_idx: int) -> np.ndarray:
        """
        Find the possible numbers of an empty cell
        :param row_idx:         Index of the to-be-checked cell in the row
        :param col_idx:         Index of the to-be-checked cell in the column
        :return:                All possible numbers
        """
        pass

    def output_board_as_str(self, line_prefix: str = ""):
        """
        Output the Sudoku board as str
            Modified based on https://github.com/jeffsieu/py-sudoku
        :param line_prefix      Prefix of each line, can be \t, SPACEs, etc.
        :return:                Formatted string of the board
        """
        pass

```


<br>

<a id="sudoku-solver-implementation"></a>
### Sudoku Solver Implementation
[Back to Section Sudoku Solver](#sudoku-solver)
```Python
class SudokuSolver:
    def __init__(self):
        pass

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
        pass

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
        pass

    def __solve__(self, board: SudokuBoard, method: str) \
            -> (bool, SudokuBoard) or (bool, None):
        """
        Call functions to solve (board error intolerable)
        :param board:           SudokuBoard object, board must be VALID
        :param method:          Method indicator
        :return:                <bool>solved, <SudokuBoard>solved board
        """
        pass

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
        pass

    def __backtrack__(self, board: SudokuBoard) -> (bool, SudokuBoard) or (bool, None):
        """
        Solve the Sudoku problem by backtracking
        :param board:           SudokuBoard object, board must be VALID
        :return:                <bool>solved, <SudokuBoard>solved board
        """
        pass

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
        pass
```