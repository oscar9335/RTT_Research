import matplotlib.pyplot as plt

# 時間點
weeks = [1, 2, 3, 4, 8]
percentages = ["1.25%", "2.5%", "5%", "10%"]
markers = ['o', 's', '^', 'D']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

# DNN @ 2 mcAP worst MDE 每週數據
dnn_mde_worst = {
    "1.25%": [0.1359, 0.0894, 0.0792, 0.0983, 0.0649],
    "2.5%": [0.0666, 0.0655, 0.0502, 0.0727, 0.0524],
    "5%": [0.0533, 0.0525, 0.0399, 0.0474, 0.0377],
    "10%": [0.0422, 0.0331, 0.0322, 0.0423, 0.0292]
}

# RegDNN @ 2 mcAP worst MDE 每週數據
regdnn_mde_worst = {
    "1.25%": [0.1318, 0.1016, 0.0662, 0.1079, 0.0646],
    "2.5%": [0.0769, 0.0671, 0.0546, 0.0658, 0.0445],
    "5%": [0.0504, 0.0482, 0.0435, 0.0535, 0.0395],
    "10%": [0.0426, 0.0400, 0.0374, 0.0457, 0.0326]
}

# 畫圖
fig, ax = plt.subplots(figsize=(10, 6))

for i, p in enumerate(percentages):
    ax.plot(weeks, dnn_mde_worst[p], marker=markers[i], linestyle='--', color=colors[i], label=f'DNN {p}')
    ax.plot(weeks, regdnn_mde_worst[p], marker=markers[i], linestyle='-', color=colors[i], label=f'RegDNN {p}')

ax.set_xlabel('Week', fontsize=14)
ax.set_ylabel('MDE (m)', fontsize=14)
ax.set_title('', fontsize=16)
ax.set_xticks(weeks)
ax.grid(True)
ax.legend(fontsize=11, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
