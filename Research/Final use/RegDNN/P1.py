import matplotlib.pyplot as plt
import numpy as np

# MDE data
mde_data = [0.0075, 0.0391, 0.0330, 0.0752, 0.0615]



# 顏色與位置
x = np.array([0, 1, 1.3, 2, 2.3])
colors = ['gray', '#70b8c4', '#c47b70', '#70b8c4', '#c47b70']
groups = ['4 mcAP', '1 mcAP (Best)', '1 mcAP (Worst)']

# 畫 MDE 圖
plt.figure(figsize=(8, 6))
bars = plt.bar(x, mde_data, color=colors, width=0.25)

# 數值標籤
for xi, val in zip(x, mde_data):
    plt.text(xi, val + 0.001, f'{val:.4f}', ha='center', fontsize=14)

# 座標設定
plt.xticks([0, 1.15, 2.15], groups, fontsize=15)
plt.ylabel('MDE (m)', fontsize=15)
plt.ylim(0, 0.1)

# 圖例
custom_lines = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['gray', '#70b8c4', '#c47b70']]
plt.legend(custom_lines, ['4 mcAP (DNN)', '1 mcAP DNN', '1 mcAP RegDNN'], loc='upper left', fontsize=15)

plt.tight_layout()
plt.show()
