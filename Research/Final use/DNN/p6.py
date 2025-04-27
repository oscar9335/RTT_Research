import matplotlib.pyplot as plt

# 時間點
weeks = ["0 week", "1 week", "2 week", "3 week", "4 week", "→ 8 week"]

# DNN MDE 數據
dnn_mde_data = {
    "Baseline": [0.2640 ,2.0009,2.2474,2.073,2.1398,2.1392],
    "1 mcAP": [0.0391 ,1.2771 ,1.54 ,1.4053 ,1.2301 ,1.6215],
    "2 mcAPs": [0.0139,0.8137,1.142,1.0922,0.9513,1.3775],
    "3 mcAPs": [0.0086 ,0.7346,1.06,1.0031,1.0473,1.3411],
    "4 mcAPs": [0.0075,0.7864,0.9804,1.0507,1.1053,1.3065]
}

# KNN MDE 數據
knn_mde_data = {
    "Baseline": [0.2294, 1.9656, 2.1988, 2.0213, 2.1361, 2.2110],
    "1 mcAP": [0.0415, 1.2489, 1.5928, 1.5432, 1.2946, 1.5846],
    "2 mcAPs": [0.0211, 0.7555, 1.0158, 1.0268, 0.9319, 1.2847],
    "3 mcAPs": [0.0137, 0.6344, 0.8767, 0.9467, 0.8414, 1.0719],
    "4 mcAPs": [0.01, 0.5789, 0.7360, 0.7576, 0.7626, 0.9740]
}

# 不同 marker 標示
markers = ['o', 's', '^', 'D', 'X']

# 顏色設定
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# 繪製 MDE 圖
plt.figure(figsize=(12, 7))

# 畫 DNN
for (label, values), marker, color in zip(dnn_mde_data.items(), markers, colors):
    plt.plot(weeks, values, marker=marker, label=f"DNN {label}", linewidth=2, color=color)

# 畫 KNN
for (label, values), marker, color in zip(knn_mde_data.items(), markers, colors):
    plt.plot(weeks, values, marker=marker, label=f"KNN {label}", linewidth=2, linestyle='--', color=color, alpha=0.5)

plt.xlabel("Time Passed", fontsize=16)
plt.ylabel("Mean Distance Error (m)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# DNN Accuracy 數據
dnn_accuracy_data = {
    "Baseline": [0.9046,0.1907,0.142,0.1306,0.1127,0.082],
    "1 mcAP": [0.9806,0.288,0.197,0.2627,0.2658,0.1302],
    "2 mcAPs": [0.9902,0.4161,0.2578,0.3225,0.3046,0.1192],
    "3 mcAPs": [0.9942,0.4632,0.3087,0.3472,0.2388,0.1357],
    "4 mcAPs": [0.9947,0.4622,0.3199,0.3393,0.264,0.157]
}

# KNN Accuracy 數據
knn_accuracy_data = {
    "Baseline": [0.9146, 0.2178, 0.1507, 0.1164, 0.1183, 0.0924],
    "1 mcAP": [0.9755, 0.3141, 0.2080, 0.1854, 0.2173, 0.1689],
    "2 mcAPs": [0.9873, 0.4589, 0.3001, 0.2921, 0.2641, 0.1474],
    "3 mcAPs": [0.9923, 0.4893, 0.3437, 0.3253, 0.2912, 0.1942],
    "4 mcAPs": [0.9934, 0.4987, 0.3878, 0.3702, 0.3332, 0.2080]
}

# 繪製 Accuracy 圖
plt.figure(figsize=(12, 7))

# 畫 DNN
for (label, values), marker, color in zip(dnn_accuracy_data.items(), markers, colors):
    plt.plot(weeks, values, marker=marker, label=f"DNN {label}", linewidth=2, color=color)

# 畫 KNN
for (label, values), marker, color in zip(knn_accuracy_data.items(), markers, colors):
    plt.plot(weeks, values, marker=marker, label=f"KNN {label}", linewidth=2, linestyle='--', color=color, alpha=0.5)

plt.xlabel("Time Passed", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
