import os
import inspect
from tqdm import tqdm
import numpy as np
import typing
import cv2
import torchvision
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# root (correct even if called)
CRT_ABS_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# keys of dataset
KEYS = ["MNIST", "EI339", "combined"]
# relative to root/
PATH_TO_DATASET = {"MNIST": "MNIST/",
                   "EI339": "EI339-CN dataset sjtu/",
                   "MNIST+EI339": "MNIST+EI339/", }
# relative to root/PATH_TO_DATASET
DATASET_MAPPING_FN = {"MNIST": None,
                      "combined": None,
                      "EI339": {"train": {"data": "mapping/train_data.npy",
                                          "label": "mapping/train_label.npy"},
                                "test": {"data": "mapping/test_data.npy",
                                         "label": "mapping/test_label.npy"}, }, }
# relative to root/PATH_TO_DATASET
DATASET_SPLITS = {"MNIST": {"raw": "raw/",
                            "train": "processed/training.pt",
                            "test": "processed/test.pt"},
                  "EI339": {"raw": "",
                            "train": "processed/training.pt",
                            "test": "processed/test.pt"},
                  "MNIST+EI339": {"raw": None,
                                  "train": "training.pt",
                                  "test": "test.pt"}, }
"""
 ~ root (CRT_ABS_PATH)
 + --- PATH_TO_DATASET
    + --- DATASET_MAPPING_FN
    + --- DATASET_SPLITS
"""


def __ei339_generate_raw_mappings__() -> \
        typing.Tuple[typing.Tuple[np.ndarray, np.ndarray],
                     typing.Tuple[np.ndarray, np.ndarray]]:
    abs_train_data_fn = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_MAPPING_FN["EI339"]["train"]["data"])
    abs_train_label_fn = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_MAPPING_FN["EI339"]["train"]["label"])
    abs_test_data_fn = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_MAPPING_FN["EI339"]["test"]["data"])
    abs_test_label_fn = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_MAPPING_FN["EI339"]["test"]["label"])
    if os.path.exists(path=abs_train_data_fn) and os.path.exists(path=abs_train_label_fn) \
            and os.path.exists(path=abs_test_data_fn) and os.path.exists(path=abs_test_label_fn):
        # print("Mappings Loaded from File")
        return (np.load(abs_train_data_fn), np.load(abs_train_label_fn)), \
               (np.load(abs_test_data_fn), np.load(abs_test_label_fn))
    __ensure_path_validation__(abs_train_data_fn)
    __ensure_path_validation__(abs_train_label_fn)
    __ensure_path_validation__(abs_test_data_fn)
    __ensure_path_validation__(abs_test_label_fn)

    train_data_map, train_label_map = [], []
    test_data_map, test_label_map = [], []
    for label_num in tqdm(range(1, 10 + 1)):
        # print("Mapping Images of Label %d" % label_num)
        abs_path_to_file_folder = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["EI339"],
            DATASET_SPLITS["EI339"]["raw"], str(label_num))
        abs_path_to_tr_files = os.path.join(abs_path_to_file_folder, "training/")
        path_to_test_files = os.path.join(abs_path_to_file_folder, "testing/")

        save_label_num = 0 if 10 == label_num else label_num
        save_label_num += 10
        # Training Data
        for file in os.listdir(abs_path_to_tr_files):
            abs_path_to_tr_file = os.path.join(abs_path_to_tr_files, file)
            train_data_map.append(abs_path_to_tr_file)
            train_label_map.append(save_label_num)
        # Test Data
        for file in os.listdir(path_to_test_files):
            abs_path_to_test_file = os.path.join(path_to_test_files, file)
            test_data_map.append(abs_path_to_test_file)
            test_label_map.append(save_label_num)

    train_data_map = np.array(train_data_map)  # (cnt,) <str> as <U129>
    train_label_map = np.array(train_label_map)  # (cnt,) <np.int32>
    train_idx = np.arange(train_label_map.size)
    np.random.shuffle(train_idx)
    train_data_map = train_data_map[train_idx]
    train_label_map = train_label_map[train_idx]
    print("EI339: Train Data Mapping Shuffled")
    test_data_map = np.array(test_data_map)  # (cnt,) <str> as <U129>
    test_label_map = np.array(test_label_map)  # (cnt,) <int>
    test_idx = np.arange(test_label_map.size)
    np.random.shuffle(test_idx)
    test_data_map = test_data_map[test_idx]
    test_label_map = test_label_map[test_idx]
    print("EI339: Test Data Mapping Shuffled")
    np.save(arr=train_data_map, file=abs_train_data_fn)
    np.save(arr=train_label_map, file=abs_train_label_fn)
    np.save(arr=test_data_map, file=abs_test_data_fn)
    np.save(arr=test_label_map, file=abs_test_label_fn)

    return (train_data_map, train_label_map), (test_data_map, test_label_map)


def __ei339_load_raw_image__(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(28, 28))
    # _, img = cv2.threshold(img, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
    img = 255 - img
    return img


def __ensure_path_validation__(filename_with_path: str) -> None:
    path = os.path.split(filename_with_path)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    assert os.path.exists(path), "[Error] Access to Directory \"%s\" is Denied" % path


def __ei339_process_raw_data__() -> None:
    abs_train_dataset_path = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_SPLITS["EI339"]["train"])
    abs_test_dataset_path = os.path.join(
        CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_SPLITS["EI339"]["test"])
    if os.path.exists(abs_train_dataset_path) and os.path.exists(abs_test_dataset_path):
        return
    __ensure_path_validation__(abs_train_dataset_path)
    __ensure_path_validation__(abs_test_dataset_path)

    (train_data_fn, train_label), (test_data_fn, test_label) = \
        __ei339_generate_raw_mappings__()

    # train data
    train_data = []
    for file in tqdm(train_data_fn):
        train_data.append(__ei339_load_raw_image__(path=file))
    train_data = np.array(train_data)
    train_data = torch.from_numpy(train_data)  # torch.Size([7385, 28, 28])
    train_label = torch.from_numpy(train_label).long()  # torch.Size([7385])
    # print(train_data.shape, train_label.shape)

    # test data
    test_data = []
    for file in tqdm(test_data_fn):
        test_data.append(__ei339_load_raw_image__(path=file))
    test_data = np.array(test_data)
    test_data = torch.from_numpy(test_data)  # torch.Size([2034, 28, 28])
    test_label = torch.from_numpy(test_label).long()  # torch.Size([2034])
    # print(test_data.shape, test_label.shape)

    torch.save((train_data, train_label), f=abs_train_dataset_path)
    torch.save((test_data, test_label), f=abs_test_dataset_path)
    print("EI339: Train & Test Data Saved")


def __combine_dataset__(data_fn_list: list, output_filename: str) -> None:
    assert len(data_fn_list) > 1, "[Error] Given to-Combine List if of Length 1"
    if os.path.exists(output_filename):
        return
    __ensure_path_validation__(output_filename)
    for file in data_fn_list:
        if not os.path.exists(file):
            raise RuntimeError("[Error] File \"%s\" NOT Exist" % file)

    data_list, targets_list = [], []
    for file in data_fn_list:
        _data, _target = torch.load(file)
        data_list.append(_data)
        targets_list.append(_target)
    data = torch.cat(data_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    torch.save((data, targets), f=output_filename)
    print("Dataset Combined")
    for file in data_fn_list:
        print("\tFrom \"%s\"" % file)
    print("\tTo \"%s\"" % output_filename)


class TorchLocalDataLoader(Dataset):
    def __init__(self, train: bool = True,
                 transform: torchvision.transforms.transforms.Compose = None,
                 mnist: bool = False, ei339: bool = False):
        assert (mnist or ei339) is True, "[Error] No Dataset is Selected"
        self.transform = transform

        self.mnist_train_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["MNIST"], DATASET_SPLITS["MNIST"]["train"])
        self.mnist_test_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["MNIST"], DATASET_SPLITS["MNIST"]["test"])
        self.ei339_train_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_SPLITS["EI339"]["train"])
        self.ei339_test_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["EI339"], DATASET_SPLITS["EI339"]["test"])
        self.combined_train_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["MNIST+EI339"], DATASET_SPLITS["MNIST+EI339"]["train"])
        self.combined_test_path = os.path.join(
            CRT_ABS_PATH, PATH_TO_DATASET["MNIST+EI339"], DATASET_SPLITS["MNIST+EI339"]["test"])

        # initialize dataset: MNIST, EI339, combined
        torchvision.datasets.MNIST(CRT_ABS_PATH, train=True, download=True)
        torchvision.datasets.MNIST(CRT_ABS_PATH, train=False, download=True)
        __ei339_process_raw_data__()
        __combine_dataset__([self.mnist_train_path, self.ei339_train_path],
                            self.combined_train_path)
        __combine_dataset__([self.mnist_test_path, self.ei339_test_path],
                            self.combined_test_path)

        # get data from file, save to self.data, self.targets (type Tensor)
        if mnist is True and ei339 is True:
            data_file = self.combined_train_path if train else self.combined_test_path
            self.data, self.targets = torch.load(data_file)
        elif mnist is True:
            data_file = self.mnist_train_path if train else self.mnist_test_path
            self.data, self.targets = torch.load(data_file)
        else:  # ei339 is True
            data_file = self.ei339_train_path if train else self.ei339_test_path
            self.data, self.targets = torch.load(data_file)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.data[idx], int(self.targets[idx])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


if "__main__" == __name__:
    # # see MNIST processed file data structure
    # #   Tuple[Tensor(Size([60000, 28, 28])), Tensor(Size([60000]))]
    # a = torch.load(os.path.join(PATH_TO_DATASET["MNIST"], DATASET_SPLITS["MNIST"]["train"]))
    # print(type(a))
    # print(a[0].shape)
    # print(type(a[0][0]))
    # print(a[1].shape)
    # print(type(a[1][0]))

    # __ei339_process_raw_data__()

    loader = TorchLocalDataLoader(
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)), ]),
        mnist=True,
        ei339=True
    )
    train_loader = DataLoader(dataset=loader, batch_size=30, shuffle=True)
