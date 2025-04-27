import matplotlib.pyplot as plt

# 資料
data_sizes = [0.25, 1.25, 2.5, 5, 10]
loss = [1.2717, 0.1290, 0.0764, 0.0606, 0.0550]

# 畫圖
fig, ax = plt.subplots(figsize=(8, 6))

# 畫 Loss 曲線
ax.plot(data_sizes, loss, marker='o', color='tab:blue')
ax.set_xlabel('Fine-tune Data Size (%)', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(0, 1.4)
ax.grid(True)

# 標註每個點的數值
for x, y in zip(data_sizes, loss):
    ax.text(x, y + 0.03, f'{y:.4f}', ha='center', fontsize=14, color='tab:blue')

# 在 2.5% 的地方標示收斂點
converge_x = 2.5
converge_y = 0.0764
ax.annotate('Convergence starts here',
            xy=(converge_x, converge_y),
            xytext=(5, 0.5),
            fontsize=14,
            arrowprops=dict(arrowstyle="->", lw=2, color='red'))

plt.tight_layout()
plt.show()
