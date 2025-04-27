import matplotlib.pyplot as plt

# 資料
data_sizes = [0.25, 1.25, 2.5, 5, 10]  # 百分比
accuracy = [0.6634, 0.9688, 0.9837, 0.9871, 0.9878]
mde = [0.4757, 0.0450, 0.0242, 0.0190, 0.0202]

# 畫圖
fig, ax1 = plt.subplots(figsize=(8, 6))

# Accuracy 左 y 軸
color = 'tab:blue'
ax1.set_xlabel('Fine-tune Data Size (%)', fontsize=14)
ax1.set_ylabel('Accuracy', color=color, fontsize=14)
acc_line = ax1.plot(data_sizes, accuracy, marker='o', color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_ylim(0.6, 1.0)

# # 標註 Accuracy 數值
# for x, y in zip(data_sizes, accuracy):
#     ax1.text(x, y + 0.01, f'{y:.4f}', color=color, fontsize=14, ha='center')

# MDE 右 y 軸
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('MDE (m)', color=color, fontsize=14)
mde_line = ax2.plot(data_sizes, mde, marker='s', linestyle='--', color=color, label='MDE (m)')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
ax2.set_ylim(0, 0.5)

# # 標註 MDE 數值
# for x, y in zip(data_sizes, mde):
#     ax2.text(x, y + 0.01, f'{y:.4f}', color=color, fontsize=14, ha='center')

# 合併圖例放到右邊中間
lines = acc_line + mde_line
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=14)

plt.grid(True)
plt.tight_layout()
plt.show()
