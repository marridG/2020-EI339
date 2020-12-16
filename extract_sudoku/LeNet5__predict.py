import os
import numpy as np
import torch

from digit_classifiers import networks_structures, networks_models


class Predict:
    def __init__(self, model_path: str, num_classes: int = 20):
        """
        :param model_path:          Path of the trained model
        :param num_classes:         Number of classes in LeNet-5
        """
        assert os.path.exists(model_path), "[Error] Model NOT Found: %s" % model_path
        network = networks_structures.LeNet5(num_classes=num_classes).double()
        self.model = networks_models.NetworkModel(
            network=network, pre_trained_path=model_path)
        print("[INFO] Digits Classifier Model Loaded")

    def predict(self, roi: np.ndarray):
        """
        Predict the number in the given region
        :param roi:         Region-of-Interest, should be of shape (1,1,28,28)
        :return:            Prediction
        """
        img = torch.from_numpy(roi).double()
        return self.model.predict(img=img)


if "__main__" == __name__:
    pred_obj = Predict(model_path="../models/LeNet5__lr=0.001__epoch=1.pth")
    pred = pred_obj.predict(roi=np.random.random((1, 1, 28, 28)))
    print(pred)
