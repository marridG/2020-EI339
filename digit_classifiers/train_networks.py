import torch
from torch.utils.data.dataloader import DataLoader
import typing

from .networks_structures import LeNet5


class Train:
    def __init__(self, network, num_epoch: int = 10,
                 loss_func=None, optimizer=None,
                 use_cuda: bool = False):
        """
        :param network:                 object of the target network
        :param num_epoch:               number of epochs per training
        :param loss_func:               object of the loss function
        :param optimizer:               object of the optimizer
        :param use_cuda:                whether to use GPU
        """
        self.network = network
        self.tr__num_epoch = num_epoch
        self.loss_func = loss_func if loss_func is not None \
            else torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None \
            else torch.optim.SGD(params=network.parameters(), lr=1e-3, momentum=0.9)
        # Configure whether to use NVIDIA GPU
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
        self.network.to(self.device)

        self.is_trained_cnt = 0  # number of times self.train() is called

    def train(self, train_loader: DataLoader) -> (typing.List[float], typing.List[float]):
        """
        Train the network upon the given training set
        :param train_loader:            training dataset
        :return:                        1. loss each epoch; 2. accuracy each epoch
        """
        print("[INFO] Start Training")
        self.network.train()  # specify the intention to train: do learn
        loss_each_epoch = []
        accuracy_each_epoch = []

        for epoch_idx in range(self.tr__num_epoch):
            loss_this_epoch_accum = 0.0
            batch_cnt = len(train_loader)
            for batch_data, batch_labels in train_loader:
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

            # loss of the epoch: average of accumulated loss pf all batches
            loss_this_epoch = loss_this_epoch_accum / batch_cnt
            loss_each_epoch.append(loss_this_epoch)

            # accuracy of the trained model of this epoch on the training set
            accuracy_tr = self.__calculate_accuracy__(data_loader=train_loader)
            accuracy_each_epoch.append(accuracy_tr)

            print("[EPOCH] %*d\t[LOSS] %.8f\t[TR ACC] %.2f%%"
                  % (len(str(self.tr__num_epoch)),
                     epoch_idx, loss_this_epoch, accuracy_tr))

        self.is_trained_cnt += 1
        return loss_each_epoch, accuracy_each_epoch

    @torch.no_grad()
    def __calculate_accuracy__(self, data_loader: DataLoader):
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

    def evaluate(self, data_loader: DataLoader):
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

    def get_info(self):
        """
        Return Information of the trainer
        :return:                        1. number of times self.train() has been called
                                        2. value of epoch
        """
        return self.is_trained_cnt, self.tr__num_epoch
