import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import torchvision
import numpy as np
from extract_sudoku import opencv__train
from digit_classifiers import networks_structures, train_networks

# opencv__train.train()


train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=30, shuffle=True)


network = networks_structures.LeNet5(num_classes=20)
train_obj = train_networks.Train(network=network, num_epoch=10)
train_loss = train_obj.train(train_loader=train_loader)
print(train_loss)
