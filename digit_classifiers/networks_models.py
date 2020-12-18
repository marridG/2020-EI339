import torch
from torch.utils.data.dataloader import DataLoader
import typing
import numpy as np
import time
import os

from .networks_structures import LeNet5


class NetworkModel:
    def __init__(self, network, pre_trained_path: str = None,
                 loss_func=None, optimizer=None,
                 use_cuda: bool = False):
        """
        :param network:                 object of the target network
        :param pre_trained_path:        path to the to-load pre-trained model
        :param loss_func:               object of the loss function
        :param optimizer:               object of the optimizer
        :param use_cuda:                whether to use GPU
        """
        self.network = network
        self.loss_func = loss_func if loss_func is not None \
            else torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None \
            else torch.optim.SGD(params=network.parameters(), lr=1e-3, momentum=0.9)
        self.is_trained_cnt = 0  # number of times self.train() is called
        if pre_trained_path is not None:
            self.__load_model__(model_full_path=pre_trained_path)

        # Configure whether to use NVIDIA GPU
        self.use_cuda = (torch.cuda.is_available() and use_cuda)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.network.to(self.device)

        print("[INFO] Model Initiated")

    def train(self, train_loader: DataLoader, num_epoch: int = 10) \
            -> (np.ndarray, np.ndarray):
        """
        Train the network upon the given training set
        :param train_loader:            training dataset
        :param num_epoch:               number of epochs to train
        :return:                        1. loss each epoch; 2. accuracy each epoch
        """
        print("[INFO] Start Training")
        self.network.train()  # specify the intention to train: do learn
        loss_each_epoch = []
        accuracy_each_epoch = []

        for epoch_idx in range(num_epoch):
            if self.use_cuda:
                torch.cuda.synchronize(device=self.device)
            epoch_start = time.time()
            loss_this_epoch_accum = 0.0
            batch_cnt = len(train_loader)
            for batch_data, batch_labels in train_loader:
                # print(batch_data.shape)  # torch.Size([30, 1, 28, 28])
                # copy to GPU if required
                batch_data.to(self.device)
                batch_labels.to(self.device)
                # init gradients of trainable weights to zero
                self.optimizer.zero_grad()
                # forward
                batch_output = self.network(batch_data)
                batch_loss = self.loss_func(batch_output, batch_labels)
                # backward (BP)
                batch_loss.backward()
                loss_this_epoch_accum += batch_loss.item()
                self.optimizer.step()

            # loss of the epoch: average of accumulated loss of all batches
            loss_this_epoch = loss_this_epoch_accum / batch_cnt
            loss_each_epoch.append(loss_this_epoch)

            if self.use_cuda:
                torch.cuda.synchronize(device=self.device)
            epoch_end = time.time()

            # accuracy of the trained model of this epoch on the training set
            accuracy_tr = self.__calculate_accuracy__(data_loader=train_loader)
            accuracy_each_epoch.append(accuracy_tr)

            print("[EPOCH] %*d\t[TIME] %.2fs\t[LOSS] %.8f\t[TR ACC] %.2f%%"
                  % (len(str(num_epoch)),
                     epoch_idx, epoch_end - epoch_start, loss_this_epoch, accuracy_tr))

        self.is_trained_cnt += num_epoch
        return np.array(loss_each_epoch), np.array(accuracy_each_epoch)

    def predict(self, img: torch.Tensor) -> int:
        assert (1, 1, 28, 28) == img.shape, \
            "[Error] Image Shape Mismatch. " \
            "Expected (N,C,W,H)=(1,1,28,28), Got %s" % (str(img.shape))
        return self.network(img).argmax(axis=1).item()

    def __load_model__(self, model_full_path: str) -> None:
        """
        Load model from file
        :param model_full_path:         the full path to the model file
        :return:
        """
        assert os.path.exists(model_full_path), \
            "[Error] Unknown File \"%s\"" % model_full_path
        model_from_file = torch.load(model_full_path)
        self.network.load_state_dict(model_from_file['state_dict'])
        self.optimizer.load_state_dict(model_from_file['optimizer'])
        if "epoch" in model_from_file.keys():
            self.is_trained_cnt += model_from_file["epoch"]
        print("Model Loaded from \"%s\"" % model_full_path)

    @torch.no_grad()
    def __calculate_accuracy__(self, data_loader: DataLoader) -> float:
        """
        Calculate the accuracy of the trained network upon the given data
        :param data_loader:             the target data
        :return:                        the accuracy, by percentage
        """
        assert data_loader is not None, "[Error] Given DataLoader is Empty, as None"
        self.network.eval()  # specify the intention to evaluate: nothing to learn

        # accuracy of the trained model of this epoch upon the given data
        correct_cnt = 0
        total_cnt = 0
        for batch_data, batch_labels in data_loader:
            # copy to GPU if required
            batch_data.to(self.device)
            batch_labels.to(self.device)
            # batch_labels is of shape (batch_size,)
            # predicted values, shape (batch_size, 20)
            predict_values = self.network(batch_data)
            # predicted labels, shape (batch_size,)
            _, predict_labels = torch.max(predict_values.data, dim=1)
            total_cnt += batch_labels.size(0)
            correct_cnt += (predict_labels == batch_labels).sum().item()

        return 100. * correct_cnt / total_cnt

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the trained network upon the given data
        :param data_loader:             the target data
        :return:                        the accuracy, by percentage
        """
        assert data_loader is not None, "[Error] Given DataLoader is Empty, as None"
        assert self.is_trained_cnt > 0, "[Error] Model has NOT been Trained yet"

        # calculate accuracy of the trained model on the target set
        accuracy = self.__calculate_accuracy__(data_loader=data_loader)

        return accuracy

    def get_info(self) -> int:
        """
        Return Information of the trainer
        :return:                        1. number of epochs that the model has been trained
        """
        return self.is_trained_cnt

    def output_model(self, path: str = "./models/", filename_prefix: str = None,
                     min_loss: float = -99.99,
                     loss_each_epoch: np.ndarray = None,
                     accuracy_each_epoch: np.ndarray = None,
                     comments: str = "",
                     other_fields: dict = None):
        """
        Save the trained model
        :param path:                    path to save the model
        :param filename_prefix:         prefix of the filename
        :param min_loss:                minimum loss
        :param loss_each_epoch:         loss at each epoch
        :param accuracy_each_epoch:     accuracy upon training set at each epoch
        :param comments:                comments string
        :param other_fields:            other fields to store
        :return:                        path to the stored model file
        """
        assert self.is_trained_cnt > 0, "[Error] Model has NOT been Trained yet"
        assert loss_each_epoch is None or (self.is_trained_cnt,) == loss_each_epoch.shape, \
            "[Error] Loss each Epoch Shape Mismatch. Expected (%d,), Got (%d,)" % \
            (self.is_trained_cnt, loss_each_epoch.shape[0])
        assert accuracy_each_epoch is None or (self.is_trained_cnt,) == accuracy_each_epoch.shape, \
            "[Error] Accuracy each Epoch Shape Mismatch. Expected (%d,), Got (%d,)" % \
            (self.is_trained_cnt, accuracy_each_epoch.shape[0])
        if not os.path.exists(path=path):
            os.mkdir(path)
        assert os.path.exists(path=path), "[Error] Inaccessible Path \"%s\"" % path

        file_path = "epoch=%d.pth" % self.is_trained_cnt
        if "" != filename_prefix and not filename_prefix.endswith("__"):
            filename_prefix = filename_prefix + "__"
        file_path = os.path.join(path, filename_prefix + file_path)

        torch.save({"epoch": self.is_trained_cnt,
                    "state_dict": self.network.state_dict(),
                    "min_loss": min_loss if min_loss >= 0 else np.nan,
                    "optimizer": self.optimizer.state_dict(),
                    "loss_each_epoch": loss_each_epoch,
                    "accuracy_each_epoch": accuracy_each_epoch,
                    "comments": comments,
                    "others": other_fields},
                   file_path)

        return file_path
