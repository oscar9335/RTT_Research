import matplotlib.pyplot as plt

# 時間點
weeks = [1, 2, 3, 4, 8]
percentages = ["1.25%", "2.5%", "5%", "10%"]
markers = ['o', 's', '^', 'D']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

# DNN 的 MDE 資料（Third）
third_mde_data = {
    "1.25%": [0.0754, 0.0937, 0.0601, 0.0585, 0.0360],
    "2.5%": [0.0443, 0.0530, 0.0411, 0.0405, 0.0221],
    "5%": [0.0340, 0.0449, 0.0309, 0.0385, 0.0340],
    "10%": [0.0301, 0.0287, 0.0220, 0.0271, 0.0162]
}

# RegDNN 的 MDE 資料
regdnn_mde_data = {
    "1.25%": [0.0697, 0.0784, 0.0581, 0.0568, 0.0494],
    "2.5%": [0.0405, 0.0500, 0.0419, 0.0405, 0.0328],
    "5%": [0.0320, 0.0364, 0.0290, 0.0333, 0.0216],
    "10%": [0.0287, 0.0287, 0.0216, 0.0266, 0.0180]
}

# 繪圖
fig, ax = plt.subplots(figsize=(10, 6))

for i, p in enumerate(percentages):
    ax.plot(weeks, third_mde_data[p], marker=markers[i], linestyle='--', color=colors[i], label=f'DNN {p}')
    ax.plot(weeks, regdnn_mde_data[p], marker=markers[i], linestyle='-', color=colors[i], label=f'RegDNN {p}')

ax.set_xlabel('Week', fontsize=14)
ax.set_ylabel('MDE (m)', fontsize=14)
ax.set_title('', fontsize=16)
ax.set_xticks(weeks)
ax.grid(True)
ax.legend(fontsize=11, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
