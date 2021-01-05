import os
import json
import numpy as np
from matplotlib import pyplot as plt

from extract_sudoku import LeNet5__train

DATASET_LABELS = ["MNIST", "EI339", "MNIST+EI339"]
dataset_settings = [("MNIST+EI339", "MNIST+EI339")]  # comparison group datasets
ACTIVATIONS_NAMES = {0: "relu", 1: "tanh", 2: "sigmoid",
                     3: "leaky_relu_negslope=0.01", 4: "elu_alpha=1.0"}
FINAL_OUT_NAMES = {0: "no", 1: "softmax"}
structure_settings = [(i, j) for i in ACTIVATIONS_NAMES.keys()
                      for j in FINAL_OUT_NAMES.keys()]
BS, LR, EP = 32, 1e-3, 10

REPORTS_PATH = "structure_report"
REPORTS_FN_TPL = os.path.join(REPORTS_PATH,
                              "[Report] LeNet5__activation=#%d_%s__final=%d_%s.json")
PLOTS_PATH = "structure_plot"
PLOTS_SHOW = False  # True  # True=show, False=NOT show
PLOTS_SAVE_DPI = 200
if not os.path.exists(REPORTS_PATH):
    os.mkdir(REPORTS_PATH)
assert os.path.exists(REPORTS_PATH), "[Error] Access to Report Path \"%s\" Denied"

# ========== TRAIN ==========
for (actv, finl) in structure_settings:
    report_fn = REPORTS_FN_TPL % (actv, ACTIVATIONS_NAMES[actv], finl, FINAL_OUT_NAMES[finl])
    if os.path.exists(report_fn):
        continue
    train_report = LeNet5__train.train(
        model_path="../models/",
        structure_settings=(actv, finl),
        dataset_settings=dataset_settings,
        num_classes=20, batch_size=BS, lr=LR, epoch=EP)
    json.dump(train_report, fp=open(report_fn, "w"), indent=2)
print("All %d Setting(s) Trained" % len(structure_settings))

# # ========== PLOTS ==========
# # I. Train & Test Dataset
# plt.close("all")
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ((bs, lr, ep), _) = settings_dataset[0]  # [((30, 1e-3, 10), dataset_settings_all), ][0]
# left_values = np.arange(len(DATASET_LABELS))  # training sets
# loc_values = np.arange(len(DATASET_LABELS))  # test sets
# report = json.load(open(REPORTS_FN_TPL % (bs, lr, ep), "r"))
# for _loc in loc_values:
#     heights = []
#     for _tr in left_values:
#         for _rep in report:
#             if DATASET_LABELS[_tr] == _rep["train_set"] \
#                     and DATASET_LABELS[_loc] == _rep["test_set"]:
#                 heights.append(_rep["test_acc"])
#                 break
#     ax.bar(left_values, heights, zs=_loc, zdir='y', alpha=0.8)
# ax.set_xlabel("Train Sets"), ax.set_ylabel("Test Sets")
# ax.set_zlabel("Test Accuracy / %")
# ax.set_xticks(left_values), ax.set_yticks(loc_values)
# ax.set_xticklabels(DATASET_LABELS, rotation=20)
# ax.set_yticklabels(DATASET_LABELS, rotation=20)
# ax.view_init(elev=22, azim=132)
# fig.suptitle("Test Accuracies of Different Train and Test Datasets")
# fig.subplots_adjust(top=0.95, bottom=0.05)
# plt.title("BatchSize=%d, LearningRate=%.5f, Epoch=%d" % (bs, lr, ep),
#           fontsize=9)
# plt.savefig(os.path.join(PLOTS_PATH, "datasets.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
#
# # II. Params: Batch Sizes
# plt.close("all")
# fig = plt.figure()
# [(_, lr, ep)], [d_set] = params_single, dataset_settings_single
# d_set = d_set[0]  # from [(dataset, dataset)] to (dataset,dataset)
# test_acc_bs = []
# for _bs_val in BS_VALUES:
#     report = json.load(open(REPORTS_FN_TPL % (_bs_val, lr, ep), "r"))[0]
#     test_acc_bs.append(report["test_acc"])
# plt.plot(BS_VALUES, test_acc_bs, marker="o", linewidth=2)
# plt.xlabel("BatchSize Values"), plt.ylabel("Test Accuracy / %")
# plt.title("LearningRate=%.5f, Epoch=%d, Train&Test=%s\nBatchSize Values=%r"
#           % (lr, ep, d_set, BS_VALUES),
#           fontsize=9)
# fig.suptitle("Test Accuracies of Different BatchSize Values")
# plt.savefig(os.path.join(PLOTS_PATH, "batch_size.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
#
# # III. Params: Epochs
# plt.close("all")
# fig = plt.figure()
# [(bs, lr, _)], [d_set] = params_single, dataset_settings_single
# d_set = d_set[0]  # from [(dataset, dataset)] to (dataset,dataset)
# test_acc_ep = []
# for _ep_val in EPOCH_VALUES:
#     report = json.load(open(REPORTS_FN_TPL % (bs, lr, _ep_val), "r"))[0]
#     test_acc_ep.append(report["test_acc"])
# plt.plot(EPOCH_VALUES, test_acc_ep, marker="o", linewidth=2)
# plt.xlabel("Epoch Values"), plt.ylabel("Test Accuracy / %")
# # plt.title("BatchSize=%d, LearningRate=%.5f, Epoch=%d, Train&Test=%s" % (bs, lr, ep, d_set[0]),
# #           fontsize=9)
# plt.title("BatchSize=%d, LearningRate=%.5f, Train&Test=%s\nEpoch Values=%r"
#           % (bs, lr, d_set, EPOCH_VALUES),
#           fontsize=9)
# fig.suptitle("Test Accuracies of Different Epoch Values")
# plt.savefig(os.path.join(PLOTS_PATH, "epoch.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
#
# # IV. Params: Learning Rates
# [(bs, _, ep)], [d_set] = params_single, dataset_settings_single
# d_set = d_set[0]  # from [(dataset, dataset)] to (dataset,dataset)
# test_acc_lr = []
# train_loss_ep__lrs = []
# train_acc_ep__lrs = []
# for _lr_val in LR_VALUES:
#     report = json.load(open(REPORTS_FN_TPL % (bs, _lr_val, ep), "r"))[0]
#     test_acc_lr.append(report["test_acc"])
#     train_loss_ep__lrs.append(report["train_loss"])
#     train_acc_ep__lrs.append(report["train_acc"])
# # IV-1: Test Acc <-> LR
# plt.close("all")
# fig = plt.figure()
# plt.plot(LR_VALUES, test_acc_lr, marker="o", linewidth=2)
# plt.axhline(y=100, linestyle="--", color="grey")
# plt.xlabel("LearningRate Values"), plt.ylabel("Test Accuracy / %")
# plt.title("BatchSize=%d, Epoch=%d, Train&Test=%s\nLearningRate Values=%r"
#           % (bs, ep, d_set, LR_VALUES),
#           fontsize=9)
# fig.suptitle("Test Accuracies of Different LearningRate Values")
# plt.savefig(os.path.join(PLOTS_PATH, "lr_test_acc.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
# # IV-2: Train Loss <-> Epoch w.r.t. LR
# plt.close("all")
# fig = plt.figure()
# for _lr_val, _train_loss in zip(LR_VALUES, train_loss_ep__lrs):
#     plt.plot(np.arange(ep), _train_loss, linewidth=2, label="lr=%.5f" % _lr_val)
# plt.xlabel("Epoch"), plt.ylabel("Train Loss")
# plt.title("BatchSize=%d, Epoch=%d, Train&Test=%s\nLearningRate Values=%r"
#           % (bs, ep, d_set, LR_VALUES),
#           fontsize=9)
# fig.suptitle("Train Loss each Epoch of Different LearningRate Values")
# plt.legend()
# plt.savefig(os.path.join(PLOTS_PATH, "lr_train_loss.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
# # IV-3: Train Acc <-> Epoch w.r.t LR
# plt.close("all")
# fig = plt.figure()
# for _lr_val, _train_acc in zip(LR_VALUES, train_acc_ep__lrs):
#     plt.plot(np.arange(ep), _train_acc, linewidth=2, label="lr=%.5f" % _lr_val)
# plt.axhline(y=100, linestyle="--", color="grey")
# plt.xlabel("Epoch"), plt.ylabel("Train Accuracy / %")
# plt.title("BatchSize=%d, Epoch=%d, Train&Test=%s\nLearningRate Values=%r"
#           % (bs, ep, d_set, LR_VALUES),
#           fontsize=9)
# fig.suptitle("Train Accuracy each Epoch of Different LearningRate Values")
# plt.legend()
# plt.savefig(os.path.join(PLOTS_PATH, "lr_train_acc.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")
# # IV-3: Last Train Acc - Test Acc <-> LR
# plt.close("all")
# fig = plt.figure()
# diff_tr_test_acc = []
# for _train_acc, _test_acc in zip(train_acc_ep__lrs, test_acc_lr):
#     diff_tr_test_acc.append(_train_acc[-1] - _test_acc)
# plt.plot(LR_VALUES, diff_tr_test_acc, color="grey", marker="o", linewidth=2)
# # plt.axhline(y=0, linestyle="--", color="grey")
# plt.xlabel("LearningRate Values"), plt.ylabel("Difference of Train and Test Accuracy / %")
# plt.title("BatchSize=%d, Epoch=%d, Train&Test=%s\nLearningRate Values=%r"
#           % (bs, ep, d_set, LR_VALUES),
#           fontsize=9)
# fig.suptitle("Difference of Train and Test Accuracy of Different LearningRate Values")
# plt.savefig(os.path.join(PLOTS_PATH, "lr_diff_train_test_acc.png"), dpi=PLOTS_SAVE_DPI)
# plt.show() if PLOTS_SHOW else print(end="")

print("All Plots Saved")
