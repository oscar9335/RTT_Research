import matplotlib.pyplot as plt
import numpy as np

# Accuracy 資料
accuracy_data = [99.47, 99.02, 99.27, 98.77, 99.14]

# 顏色與位置設定
x = np.array([0, 1, 1.3, 2, 2.3])
colors = ['gray', '#70b8c4', '#c47b70', '#70b8c4', '#c47b70']
groups = ['4 mcAP', '2 mcAP (Best)', '2 mcAP (Worst)']

# 畫 Accuracy 圖
plt.figure(figsize=(8, 6))
bars = plt.bar(x, accuracy_data, color=colors, width=0.25)

# 數值標示
for xi, val in zip(x, accuracy_data):
    plt.text(xi, val + 0.1, f'{val:.2f}%', ha='center', fontsize=14)

# 座標設定
plt.xticks([0, 1.15, 2.15], groups, fontsize=15)
plt.ylabel('Accuracy (%)', fontsize=15)
plt.ylim(97, 100)

# 圖例設定
custom_lines = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['gray', '#70b8c4', '#c47b70']]
plt.legend(custom_lines, ['4 mcAP (DNN)', '2 mcAP DNN', '2 mcAP RegDNN'], loc='lower left', fontsize=15)

plt.tight_layout()
plt.show()
