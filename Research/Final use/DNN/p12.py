import matplotlib.pyplot as plt

# 時間點
weeks = [1, 2, 3, 4, 8]
percentages = ["1.25%", "2.5%", "5%", "10%"]

# Best 組（Third）
third_mde_data = {
    "1.25%": [0.0754, 0.0937, 0.0601, 0.0585, 0.0360],
    "2.5%": [0.0443, 0.0530, 0.0411, 0.0405, 0.0221],
    "5%": [0.0340, 0.0449, 0.0309, 0.0385, 0.0340],
    "10%": [0.0301, 0.0287, 0.0220, 0.0271, 0.0162]
}

# Worst 組（Sixth）
sixth_mde_data = {
    "1.25%": [0.1359, 0.0894, 0.0792, 0.0983, 0.0649],
    "2.5%": [0.0666, 0.0655, 0.0502, 0.0727, 0.0524],
    "5%": [0.0533, 0.0525, 0.0399, 0.0474, 0.0377],
    "10%": [0.0422, 0.0331, 0.0322, 0.0423, 0.0292]
}

# 標記與顏色設置
markers = ['o', 's', '^', 'D']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

# 畫圖
fig, ax = plt.subplots(figsize=(10, 6))

for i, p in enumerate(percentages):
    ax.plot(weeks, third_mde_data[p], marker=markers[i], linestyle='-', color=colors[i], label=f'Best {p}')
    ax.plot(weeks, sixth_mde_data[p], marker=markers[i], linestyle='--', color=colors[i], label=f'Worst {p}')

ax.set_xlabel('Week', fontsize=14)
ax.set_ylabel('MDE (m)', fontsize=14)
ax.set_title('', fontsize=16)
ax.set_xticks(weeks)
ax.grid(True)
ax.legend(fontsize=11, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
