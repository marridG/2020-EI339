import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import numpy as np
from extract_sudoku import opencv__train, LeNet5__train
from digit_classifiers import networks_structures, networks_models
from data import load_local_dataset

# opencv__train.train()

dataset_labels = ["MNIST", "EI339", "MNIST+EI339"]
# dataset_settings = [(i, j) for i in dataset_labels for j in dataset_labels]
dataset_settings = [("MNIST+EI339", "MNIST+EI339")]
LeNet5__train.train(model_path="./models/", dataset_settings=dataset_settings,
                    batch_size=30, lr=1e-3, epoch=1)
