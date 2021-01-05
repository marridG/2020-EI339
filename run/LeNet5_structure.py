import os
import json
import numpy as np
from matplotlib import pyplot as plt

from extract_sudoku import LeNet5__train

DATASET_LABELS = ["MNIST", "EI339", "MNIST+EI339"]
dataset_settings = [("MNIST+EI339", "MNIST+EI339")]  # comparison group datasets
ACTIVATIONS_NAMES = {0: "ReLU", 1: "Tanh", 2: "Sigmoid",
                     3: "Leaky_ReLU",  # negative slope=0.01
                     4: "ELU"  # alpha=1.0
                     }
FINAL_OUT_NAMES = {0: "no"}  # , 1: "softmax"}
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
        print("[INFO] Exists:", report_fn)
        continue
    train_report = LeNet5__train.train(
        model_path="../models/",
        structure_settings=(actv, finl),
        dataset_settings=dataset_settings,
        num_classes=20, batch_size=BS, lr=LR, epoch=EP)
    json.dump(train_report, fp=open(report_fn, "w"), indent=2)
print("All %d Setting(s) Trained" % len(structure_settings))

# # ========== PLOTS ==========
# II. Params: Activation: Train Acc <-> Epoch w.r.t. Activation
plt.close("all")
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
ax2.set_ylim(8, 12)  # outliers only
ax.set_ylim(80, 101)  # most of the data
# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()
d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

d_set = dataset_settings[0][0]
for _act_val in ACTIVATIONS_NAMES.keys():
    report = json.load(open(REPORTS_FN_TPL % (_act_val, ACTIVATIONS_NAMES[_act_val],
                                              0, FINAL_OUT_NAMES[0]), "r"))[0]
    ax.plot(np.arange(EP), report["train_acc"], linewidth=2,
            label="%s" % ACTIVATIONS_NAMES[_act_val])
    ax2.plot(np.arange(EP), report["train_acc"], linewidth=2,
             label="%s" % ACTIVATIONS_NAMES[_act_val])
ax.axhline(y=100, linestyle="--", color="grey")
ax2.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy / %"), ax2.set_ylabel("Test Accuracy / %")
ax.set_title("BatchSize=%d, LearningRate=%.5f, Epoch=%d, Train&Test=%s\nActivations=%r"
             % (BS, LR, EP, d_set, list(ACTIVATIONS_NAMES.values())),
             fontsize=9)
fig.suptitle("Train Accuracies of Different Avtivation Functions")
ax.legend()
plt.savefig(os.path.join(PLOTS_PATH, "activation_test_acc.png"), dpi=PLOTS_SAVE_DPI)
plt.show() if PLOTS_SHOW else print(end="")

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
