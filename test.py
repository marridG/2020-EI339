import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import torchvision
import numpy as np
from extract_sudoku import opencv__train
from digit_classifiers import networks_structures, network_model

# opencv__train.train()


train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=30, shuffle=True)

network = networks_structures.LeNet5(num_classes=20)
network_model = network_model.NetworkModel(network=network)
train_loss, train_acc = network_model.train(train_loader=train_loader, num_epoch=1)
output_model = network_model.output_model(path="./models/",
                                          filename_prefix="LeNet5__lr=0.001",
                                          min_loss=np.min(train_loss),
                                          loss_each_epoch=train_loss,
                                          accuracy_each_epoch=train_acc,
                                          comments="", other_fields=None)
print(output_model)
