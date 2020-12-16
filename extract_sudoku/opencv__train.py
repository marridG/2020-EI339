import os
import inspect
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # e.g. 20201129210907
crt_abs_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
crt_abs_father = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
script_path = os.path.join(crt_abs_father, "opencv_sudoku_solver/train_digit_classifier.py")
output_path = "../models"
model_filename_subfix = "__%s" % timestamp


def train(model_path: str = output_path, model_fn_subfix: str = model_filename_subfix):
    output_model_full_path = os.path.join(model_path, "opencv%s.h5" % model_fn_subfix)
    os.system("python \"%s\" --model \"%s\"" % (script_path, output_model_full_path))
    return output_model_full_path


if "__main__" == __name__:
    script_path = "../opencv_sudoku_solver/train_digit_classifier.py"
    train()
