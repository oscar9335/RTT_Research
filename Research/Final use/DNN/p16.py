import matplotlib.pyplot as plt
import numpy as np

# week 座標點（確保 4~8 間距加大）
weeks_numeric = [0, 1, 2, 3, 4, 8]
weeks_labels = ['0', '1', '2', '3', '4', '8']
weeks_labels = ['Basemodel', '2024/12/21', '2024/12/27', '2025/01/03', '2025/01/10', '2025/02/07']

# MDE 數據
KNN_MDE = [
    [0.2264, 1.9980, 2.2101, 2.0307, 2.1273, 2.1813],  # Baseline
    [0.0374, 1.2798, 1.5996, 1.4844, 1.2489, 1.6507],  # 1 mcAP Best
    [0.0190, 0.9005, 1.0337, 1.1223, 0.9931, 1.3914],  # 2 mcAP Best
    [0.0147, 0.7874, 1.0508, 1.0166, 1.0841, 1.3046],  # 3 mcAP Best
    [0.0143, 0.7581, 0.9453, 1.0562, 1.0645, 1.2158],  # 4 mcAP
]
DNN_MDE = [
    [0.2676, 1.9955, 2.2472, 2.0834, 2.1314, 2.1328],  # Baseline
    [0.0338, 1.2851, 1.5454, 1.4230, 1.2349, 1.6114],  # 1 mcAP Best
    [0.0130, 0.8194, 1.1574, 1.0963, 0.9749, 1.3483],  # 2 mcAP Best
    [0.0112, 0.7496, 1.0673, 1.0387, 1.0538, 1.3441],  # 3 mcAP Best
    [0.0082, 0.7713, 0.9342, 1.0082, 1.0504, 1.2559],  # 4 mcAP
]

labels = ['Baseline', '1 mcAP', '2 mcAP', '3 mcAP', '4 mcAP']
markers = ['o', 's', '^', 'D', '*']
# colors_knn = ['#2c7bb6', '#abd9e9', '#74add1', '#4575b4', '#313695']
colors_knn = ['#4C72B0', '#DD8452', '#55A868', 'tab:red', 'tab:purple']
# colors_dnn = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
colors_dnn = ['#4C72B0', '#DD8452', '#55A868', 'tab:red', 'tab:purple']

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))

# 畫 KNN
for i, (label, mde, marker, color) in enumerate(zip(labels, KNN_MDE, markers, colors_knn)):
    plt.plot(weeks_numeric, mde, label=f'KNN {label}', marker=marker, linestyle='--', color=color, linewidth=2, alpha=0.7,markersize=5,)

# 畫 DNN
for i, (label, mde, marker, color) in enumerate(zip(labels, DNN_MDE, markers, colors_dnn)):
    plt.plot(weeks_numeric, mde, label=f'DNN {label}', marker=marker, linestyle='-', color=color, linewidth=2, alpha=0.8, markersize=5,)

plt.xlabel("Week", fontsize=12)
plt.ylabel("MDE (m)", fontsize=12)
plt.xticks(weeks_numeric, weeks_labels)
plt.xticks(rotation=20) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, ncol=2)
plt.tight_layout()
plt.show()
