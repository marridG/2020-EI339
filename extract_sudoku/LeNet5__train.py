import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
import numpy as np

from data import load_local_dataset
from digit_classifiers import networks_structures, networks_models

output_path = "./models"


def train(model_path: str = output_path, dataset_settings: list = None,
          num_classes: int = 20,
          batch_size: int = 30, lr: float = 1e-3, epoch: int = 1):
    print("Hyper-parameters: batch_size=%d, lr=%.5f, epoch=%d" % (batch_size, lr, epoch))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    # train_mnist_loader = DataLoader(
    #     torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform),
    #     batch_size=batch_size, shuffle=True)
    train_mnist_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=True, transform=transform, mnist=True, ei339=False),
        batch_size=batch_size, shuffle=True)
    train_ei339_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=True, transform=transform, mnist=False, ei339=True),
        batch_size=batch_size, shuffle=True)
    train_all_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=True, transform=transform, mnist=True, ei339=True),
        batch_size=batch_size, shuffle=True)
    test_mnist_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=False, transform=transform, mnist=True, ei339=False),
        batch_size=batch_size, shuffle=True)
    test_ei339_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=False, transform=transform, mnist=False, ei339=True),
        batch_size=batch_size, shuffle=True)
    test_all_loader = DataLoader(
        dataset=load_local_dataset.TorchLocalDataLoader(
            train=False, transform=transform, mnist=True, ei339=True),
        batch_size=batch_size, shuffle=True)

    dataset_labels = ["MNIST", "EI339", "MNIST+EI339"]
    dataset_label_to_loader = {"train": {"MNIST": train_mnist_loader,
                                         "EI339": train_ei339_loader,
                                         "MNIST+EI339": train_all_loader},
                               "test": {"MNIST": test_mnist_loader,
                                        "EI339": test_ei339_loader,
                                        "MNIST+EI339": test_all_loader}, }
    report = [
        # {"train_set": "MNIST", "test_set": "MNIST",
        #  "batch_size": 30, "lr": 1e-3, "epoch": 1,
        #  "train_loss": np.array([0.3, 0.2, 0.1]), "train_acc": np.array([70.4, 80.1, 90.5]),
        #  "test_acc": 80.1, },
    ]
    
    for (train_name, test_name) in dataset_settings:
        assert train_name in dataset_labels, \
            "[Error] Invalid Train Name \"%s\". Supported are %s" \
            % (train_name, ", ".join(dataset_labels))
        assert test_name in dataset_labels, \
            "[Error] Invalid Test Name \"%s\". Supported are %s" \
            % (test_name, ", ".join(dataset_labels))

        print("Train upon %s, Test upon %s\n\tbs=%d, lr=%.5f, epoch=%d"
              % (train_name, test_name, batch_size, lr, epoch))
        train_loader = dataset_label_to_loader["train"][train_name]
        test_loader = dataset_label_to_loader["test"][test_name]
        network = networks_structures.LeNet5(num_classes=num_classes)
        model = networks_models.NetworkModel(network=network,
                                             optimizer=torch.optim.SGD(
                                                 params=network.parameters(), lr=lr, momentum=0.9))
        train_loss, train_acc = model.train(train_loader=train_loader, num_epoch=epoch)
        output_model_fn = model.output_model(path=model_path,
                                             filename_prefix="LeNet5__lr=%.5f__epoch=%d__batch_size=%d" %
                                                             (lr, epoch, batch_size),
                                             min_loss=np.min(train_loss),
                                             loss_each_epoch=train_loss,
                                             accuracy_each_epoch=train_acc,
                                             comments="", other_fields=None)
        print("\tModel Trained Upon %s, Saved at \"%s\"" % (train_name, output_model_fn))
        test_acc = model.evaluate(data_loader=test_loader)
        print("\tTest Accuracy Upon %s: %.2f%%" % (test_name, test_acc))

        report.append({"train_set": train_name, "test_set": test_name,
                       "model_fn": output_model_fn,
                       "batch_size": batch_size, "lr": lr, "epoch": epoch,
                       "train_loss": train_loss.tolist(), "train_acc": train_acc.tolist(),
                       "test_acc": test_acc, })
    return report


if "__main__" == __name__:
    print(train())
