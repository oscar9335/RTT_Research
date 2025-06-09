import matplotlib.pyplot as plt
import numpy as np

# 設定字型大小
plt.rcParams.update({'font.size': 14})

# Best AP scenario data (4, 3, 2, 1 mcAP)
labels = ['4 mcAP', '3 mcAP', '2 mcAP', '1 mcAP']
dnn_best = [0.0082, 0.0112, 0.0130, 0.0338]
regdnn_best = [None, 0.0085, 0.0107, 0.0354]  # 4 mcAP 無RegDNN

# Worst AP scenario data
dnn_worst = [0.0082, 0.0131, 0.0217, 0.0743]
regdnn_worst = [None, 0.0075, 0.0160, 0.0685]

# bar寬
width = 0.35
x = np.arange(len(labels))

# 顏色參考論文/期刊配色，DNN灰藍, RegDNN粉磚
color_dnn = "#72a2c8"
color_regdnn = "#c88272"
color_4mcAP = "#888888"

# ========== Best ==========
fig, ax = plt.subplots(figsize=(8,6)
# 4 mcAP DNN
ax.bar(x[0], dnn_best[0], width=width, color=color_4mcAP, label='4 mcAP DNN')
# 3,2,1 mcAP DNN
ax.bar(x[1:], dnn_best[1:], width=width, color=color_dnn, label='DNN')
# 3,2,1 mcAP RegDNN
ax.bar(x[1:]+width, [v for v in regdnn_best[1:]], width=width, color=color_regdnn, label='RegDNN')

# 數值標註
for i in range(len(labels)):
    # DNN
    if dnn_best[i] is not None:
        ax.text(x[i], dnn_best[i]+0.001, f"{dnn_best[i]:.4f}", ha='center', va='bottom', fontsize=13)
    # RegDNN (跳過4mcAP)
    if i > 0 and regdnn_best[i] is not None:
        ax.text(x[i]+width, regdnn_best[i]+0.001, f"{regdnn_best[i]:.4f}", ha='center', va='bottom', fontsize=13)

ax.set_xticks(x + width/2)
ax.set_xticklabels(labels)
ax.set_ylabel('MDE (m)')
ax.set_ylim(0, 0.04)
ax.legend()
plt.tight_layout()
plt.show()


# ========== Worst ==========
fig, ax = plt.subplots(figsize=(7,5))
# 4 mcAP DNN
ax.bar(x[0], dnn_worst[0], width=width, color=color_4mcAP, label='4 mcAP DNN')
# 3,2,1 mcAP DNN
ax.bar(x[1:], dnn_worst[1:], width=width, color=color_dnn, label='DNN')
# 3,2,1 mcAP RegDNN
ax.bar(x[1:]+width, [v for v in regdnn_worst[1:]], width=width, color=color_regdnn, label='RegDNN')

for i in range(len(labels)):
    # DNN
    if dnn_worst[i] is not None:
        ax.text(x[i], dnn_worst[i]+0.001, f"{dnn_worst[i]:.4f}", ha='center', va='bottom', fontsize=13)
    # RegDNN (跳過4mcAP)
    if i > 0 and regdnn_best[i] is not None:
        ax.text(x[i]+width, regdnn_worst[i]+0.001, f"{regdnn_worst[i]:.4f}", ha='center', va='bottom', fontsize=13)

ax.set_xticks(x + width/2)
ax.set_xticklabels(labels)
ax.set_ylabel('MDE (m)')
ax.set_ylim(0, 0.04)
ax.legend()
plt.tight_layout()
plt.show()