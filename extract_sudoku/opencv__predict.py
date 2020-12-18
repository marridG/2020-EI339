import os
from tensorflow.keras.models import load_model
import numpy as np


class Predict:
    def __init__(self, model_path: str):
        assert os.path.exists(model_path), "[Error] Model NOT Found: %s" % model_path
        self.model = load_model(model_path)
        print("[INFO] Digits Classifier Model Loaded")

    def predict(self, roi: np.ndarray):
        """
        Predict the number in the given region
        :param roi:         Region-of-Interest, should be of shape (28,28)
        :return:            Prediction
        """
        assert (28, 28) == roi.shape, \
            "[Error] ROI Shape Mismatch. Expected (28, 28), Got %s" % (str(roi.shape))
        roi = roi.reshape((1, 28, 28, 1))  # (N,H,W,C)=(1,28,28,1)
        return self.model.predict(roi).argmax(axis=1)[0]


if "__main__" == __name__:
    pred_obj = Predict(model_path="../models/opencv__20201129212517.h5")
    pred = pred_obj.predict(roi=np.zeros((1, 28, 28, 1)))
    print(pred)
