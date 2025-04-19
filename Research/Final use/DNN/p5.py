import matplotlib.pyplot as plt

# 時間點
weeks = ["0 week", "1 week", "2 week", "3 week", "4 week", "→ 8 week"]

# MDE 數據
mde_data = {
    "Baseline": [0.2640 ,2.0009,2.2474,2.073,2.1398,2.1392],
    "1 mcAP (AP1)": [0.0391 ,1.2771 ,1.54 ,1.4053 ,1.2301 ,1.6215],
    "2 mcAPs (AP1 & AP4)": [0.0139,0.8137,1.142,1.0922,0.9513,1.3775],
    "3 mcAPs": [0.0086 ,0.7346,1.06,1.0031,1.0473,1.3411],
    "4 mcAPs": [0.0075,0.7864,0.9804,1.0507,1.1053,1.3065]
}

# Accuracy 數據
accuracy_data = {
    "Baseline": [0.9046,0.1907,0.142,0.1306,0.1127,0.082],
    "1 mcAP (AP1)": [0.9806,0.288,0.197,0.2627,0.2658,0.1302],
    "2 mcAPs (AP1 & AP4)": [0.9902,0.4161,0.2578,0.3225,0.3046,0.1192],
    "3 mcAPs": [0.9942,0.4632,0.3087,0.3472,0.2388,0.1357],
    "4 mcAPs": [0.9947,0.4622,0.3199,0.3393,0.264,0.157]
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
