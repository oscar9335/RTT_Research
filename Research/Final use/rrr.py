# import matplotlib.pyplot as plt
# import numpy as np

# # Data for Best case
# labels = ['4 mcAP', '3 mcAP', '2 mcAP', '1 mcAP']
# dnn_mde_best = [0.0082, 0.0112, 0.0130, 0.0338]
# regdnn_mde_best = [None, 0.0085, 0.0107, 0.0354]  # None for 4 mcAP, RegDNN無意義

# # Data for Worst case
# dnn_mde_worst = [0.0082, 0.0131, 0.0217, 0.0743]
# regdnn_mde_worst = [None, 0.0075, 0.0160, 0.0685]

# # Colors: DNN灰(4mcAP), DNN藍, RegDNN橘
# colors_dnn = ['#888888', '#70b8c4', '#70b8c4', '#70b8c4']
# colors_regdnn = ['#ffffff', '#c47b70', '#c47b70', '#c47b70']  # 4 mcAP RegDNN留白

# def plot_grouped_bar(labels, dnn, regdnn, title=None, ylim=None):
#     x = np.arange(len(labels))
#     width = 0.35

#     plt.figure(figsize=(8, 6))
#     # DNN
#     bars1 = plt.bar(x - width/2, dnn, width, label='DNN', color=colors_dnn, edgecolor='k')
#     # RegDNN
#     bars2 = plt.bar(x + width/2, 
#                     [v if v is not None else 0 for v in regdnn], 
#                     width, 
#                     label='RegDNN', color=colors_regdnn, edgecolor='k')

#     # 標數字
#     for bar, val in zip(bars1, dnn):
#         plt.text(bar.get_x() + bar.get_width()/2, val+0.0005, f'{val:.4f}', ha='center', va='bottom', fontsize=12)
#     for bar, val in zip(bars2, regdnn):
#         if val is not None:
#             plt.text(bar.get_x() + bar.get_width()/2, val+0.0005, f'{val:.4f}', ha='center', va='bottom', fontsize=12)

#     plt.xticks(x, labels, fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.ylabel('MDE (m)', fontsize=14)
#     plt.legend(fontsize=14)
#     if ylim:
#         plt.ylim(*ylim)
#     plt.tight_layout()
#     if title:
#         plt.title(title, fontsize=16)
#     plt.show()

# # 畫 Best 場景
# plot_grouped_bar(labels, dnn_mde_best, regdnn_mde_best)

# # 畫 Worst 場景
# plot_grouped_bar(labels, dnn_mde_worst, regdnn_mde_worst)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data for Best case
labels = ['4 mcAP', '3 mcAP', '2 mcAP', '1 mcAP']
dnn_mde_best = [0.0082, 0.0112, 0.0130, 0.0338]
regdnn_mde_best = [None, 0.0085, 0.0107, 0.0354]

# Data for Worst case
dnn_mde_worst = [0.0082, 0.0131, 0.0217, 0.0743]
regdnn_mde_worst = [None, 0.0075, 0.0160, 0.0685]

# Colors
color_dnn_4mc = '#888888'
color_dnn = '#70b8c4'
color_regdnn = '#c47b70'
colors_dnn = [color_dnn_4mc, color_dnn, color_dnn, color_dnn]
colors_regdnn = ['white', color_regdnn, color_regdnn, color_regdnn]  # 4 mcAP RegDNN留白

def plot_grouped_bar(labels, dnn, regdnn, show_regdnn_for_4mc=False, ylim=None):
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    # DNN
    bars1 = plt.bar(x - width/2, dnn, width, label='DNN', color=colors_dnn, zorder=2)
    # RegDNN
    regdnn_values = [v if v is not None else 0 for v in regdnn]
    bars2 = plt.bar(x + width/2, regdnn_values, width, 
                    label='RegDNN', color=colors_regdnn, zorder=2)

    # Bar上的字（12號字體，稍微往右偏，垂直置底）
    for bar, val in zip(bars1, dnn):
        plt.text(bar.get_x() + bar.get_width() * 0.015, val + 0.0005, f'{val:.4f}',
                 ha='left', va='bottom', fontsize=12)
    for bar, val in zip(bars2, regdnn):
        if val is not None:
            plt.text(bar.get_x() + bar.get_width() * 0.025, val + 0.0005, f'{val:.4f}',
                     ha='left', va='bottom', fontsize=12)

    # Legend（顏色對應bar）
    legend_elements = [
        Patch(facecolor=color_dnn_4mc, label='4 mcAP (DNN)'),
        Patch(facecolor=color_dnn, label='DNN'),
        Patch(facecolor=color_regdnn, label='RegDNN')
    ]
    plt.legend(handles=legend_elements, fontsize=14)

    plt.xticks(x, labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('MDE (m)', fontsize=14)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.show()

# 畫 Best 場景
plot_grouped_bar(labels, dnn_mde_best, regdnn_mde_best)

# 畫 Worst 場景
plot_grouped_bar(labels, dnn_mde_worst, regdnn_mde_worst)
