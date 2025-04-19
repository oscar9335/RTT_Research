import matplotlib.pyplot as plt

# 時間點
weeks = ["0 week", "1 week", "2 week", "3 week", "4 week", "→ 8 week"]

# MDE 數據
mde_data = {
    "Baseline": [0.2294, 1.9656, 2.1988, 2.0213, 2.1361, 2.2110],
    "1 mcAP (AP1)": [0.0415, 1.2489, 1.5928, 1.5432, 1.2946, 1.5846],
    "2 mcAPs (AP1 & AP4)": [0.0211, 0.7555, 1.0158, 1.0268, 0.9319, 1.2847],
    "3 mcAPs": [0.0137, 0.6344, 0.8767, 0.9467, 0.8414, 1.0719],
    "4 mcAPs": [0.01, 0.5789, 0.7360, 0.7576, 0.7626, 0.9740]
}

# Accuracy 數據
accuracy_data = {
    "Baseline": [0.9146, 0.2178, 0.1507, 0.1164, 0.1183, 0.0924],
    "1 mcAP (AP1)": [0.9755, 0.3141, 0.2080, 0.1854, 0.2173, 0.1689],
    "2 mcAPs (AP1 & AP4)": [0.9873, 0.4589, 0.3001, 0.2921, 0.2641, 0.1474],
    "3 mcAPs": [0.9923, 0.4893, 0.3437, 0.3253, 0.2912, 0.1942],
    "4 mcAPs": [0.9934, 0.4987, 0.3878, 0.3702, 0.3332, 0.2080]
}


# 不同 marker 標示
markers = ['o', 's', '^', 'D', 'X']

# 繪製 MDE 圖
plt.figure(figsize=(10, 6))
for (label, values), marker in zip(mde_data.items(), markers):
    plt.plot(weeks, values, marker=marker, label=label, linewidth=2)
plt.xlabel("Time Passed", fontsize=14)
plt.ylabel("Mean Distance Error (m)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 繪製 Accuracy 圖
plt.figure(figsize=(10, 6))
for (label, values), marker in zip(accuracy_data.items(), markers):
    values = [v if v is not None else float('nan') for v in values]
    plt.plot(weeks, values, marker=marker, label=label, linewidth=2)
plt.xlabel("Time Passed", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
