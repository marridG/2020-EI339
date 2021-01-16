# 2020 EI339 - Individual Sudoku Project
Artificial Intelligence, 2020 Fall, SJTU  
**by Prof. Jun Z.**

<br>

**For more details, please refer the [report](./report/report.md).**

<br>

<a id="description"></a>
## Description

1. Understand the provided codes of `OpenCV Sudoku Solver and OCR` approach.
2. Cooperate with fellow classmates to construct the `EI339-CN` dataset of handwritten Chinese numbers `一, 二,..., 九, 十`.
3. Implement LeNet-5 and train on `MNIST + EI339-CN`.
4. Implement a CSP solver for the Sudoku problem.
4. Test the trained model and the solver upon given test images.

<br>

<a id="highlights"></a>
## Highlights

1. Fully-annotated codes: both in function prototypes and between the lines.
2. Carefully designed and wrapped functionalities: divide the board-extraction, Sudoku-board-formation, problem-solving processes into different packages and provide high-level APIs.
3. Extensive experiment upon **(1)** influence of the selection of train and test datasets; **(2)** SudokuNet VS LeNet-5; **(3)** LeNet-5 activations (with analysis), training hyper-parameters.
4. Error-tolerable CSP Sudoku solver.
5. Construct real-world Sudoku problems set based on the published book `《全民数独》` and conduct tests upon the images captured from 5 angles: bird-view, left, upper, right and lower.
6. Analysis and solid intermediate results.
7. Some other implemented useful functionalities.

<br>



<a id="project-file-tree"></a>
## Project File Tree
```
~
│  .gitignore
│  LICENSE
│  README.md
│  
├─digit_classifiers                     // codes for digit classifiers
│     networks_models.py                // network utilities (e.g. training)
│     networks_structures.py            // network structures (LeNet-5 here)
│     __init__.py
│          
├─extract_sudoku                        // [wrapper] step of extract Sudoku image
│     board_img_arr.py                  // image == extract ==> Sudoku board object
│     LeNet5__predict.py                // API of LeNet-5 prediction
│     LeNet5__train.py                  // API of LeNet-5 training
│     opencv__predict.py                // API of SudokuNet prediction
│     opencv__train.py                  // API of SudokuNet prediction
│     __init__.py
│          
│─solve_sudoku                          // codes for Sudoku solver
│      sudoku_board.py                  // implementation of Sudoku board
│      sudoku_solver.py                 // Sudoku solver based on the board object
│      __init__.py  
│
│
├─examples                              // example codes
│      extract_board.py                 // example codes to extract Sudoku boards
│      solve-sudoku.py                  // example codes to solve Sudoku problems
│      sudoku_puzzle.jpg                // example image
│
├─run                                   // final codes to execute
│  │  final_imgs.py                     // solve the given 5+5 final images
│  │  LeNet5_params.py                  // hyper-parameters test codes
│  │  LeNet5_structure.py               // structure test codes
│  │  solve_by_LeNet5.py                // LeNet-5 100-image test codes
│  │  solve_by_opencv.py                // SudokuNet 100-image test codes
│  │  
│  ├─final_out                          // results of the given 5+5 final images
│  ├─num_changed_plot                   // results of 100-image test
│  ├─params_plot                        // plots of hyper-parameters tests
│  ├─params_report                      // .json reports of hyper-parameters tests
│  ├─structure_plot                     // plots of network structure tests
│  └─structure_report                   // .json reports of structure tests
│               
│
├─data
│  │  load_local_dataset.py             // data loader codes
│  │  
│  ├─EI339-CN dataset sjtu              // EI339 dataset folder
│  │  ├─1, 2, ..., 10                   // raw data of EI339
│  │  ├─mapping                         // intermediate folder: filenames mapping of EI339
│  │  └─processed                       // data file (.pt) of EI339
│  │          
│  ├─MNIST                              // data file of MNIST
│  └─MNIST+EI339                        // combined data file (.pt) of MNIST+EI339
│          
├─imgs                                  // test images folder & loader
│  │  load_imgs.py                      // image loader
│  │  overview.py                       // generate an overview of 100-image test images
│  │  sudoku_puzzle.jpg                 // test image
│  │  __init__.py
│  │  
│  ├─JilinUnivPr                        // images captured from 《全民数独》, 吉林大学出版社
│  │  └─imgs                            // 100 images of 100-image tests
│  │          
│  └─test1                              // final 5+5 test Sudoku problems images 
│          
├─models                                // trained models
│      
├─opencv_sudoku_solver                  // given OpenCV approach proposed online
│  │  solve_sudoku_puzzle.py            // solved codes
│  │  sudoku_puzzle.jpg                 // test image
│  │  train_digit_classifier.py         // train codes
│  │  __init__.py
│  │  
│  ├─output                             // output models
│  │      
│  └─pyimagesearch                      // SudokuNet & captured board extractor
│     │  __init__.py
│     │  
│     ├─models                          // SudokuNet structure
│     │     sudokunet.py
│     │     __init__.py
│     │          
│     └─sudoku                          // captured board extractor
│           multi_img_view.py           // added functionality: combine & show multiple images together
│           puzzle.py                   // board extractor: find puzzle & extract digit
│           __init__.py
│          
└─report                                // report files
```
