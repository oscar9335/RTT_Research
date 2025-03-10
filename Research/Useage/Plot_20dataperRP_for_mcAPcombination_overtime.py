import matplotlib.pyplot as plt
import numpy as np

# 定義 Fine-tuning 週數
weeks = [1, 2, 3, 4, 8]

# Accuracy 數據
accuracy = {
    "0 mcAP": [0.8208, 0.8117, 0.8257, 0.8213, 0.8624],
    "1 mcAP": [0.9088, 0.9553, 0.9452, 0.9000, 0.9509],
    "2 mcAP": [0.9584, 0.9662, 0.9697, 0.9536, 0.9751],
    "3 mcAP": [0.9754, 0.9759, 0.9731, 0.9716, 0.9744],
    "4 mcAP": [0.9710, 0.9757, 0.9662, 0.9696, 0.9725],
}

# MDE 數據
mde = {
    "0 mcAP": [0.4871, 0.5049, 0.3974, 0.4471, 0.2667],
    "1 mcAP": [0.1911, 0.1004, 0.1011, 0.1551, 0.1017],
    "2 mcAP": [0.0612, 0.0546, 0.0427, 0.0696, 0.0402],
    "3 mcAP": [0.0376, 0.0339, 0.0396, 0.0438, 0.0377],
    "4 mcAP": [0.0412, 0.0330, 0.0465, 0.0397, 0.0382],
}

# 重新繪製 Accuracy 和 MDE 圖表，使用英文標題

# 重新繪製 Accuracy 和 MDE 圖表，將圖例 (Legend) 放在 figure 的右上角

# 設定字體大小
font_size = 14

# 畫 Accuracy 圖 (Legend at lower right)
plt.figure(figsize=(8, 5))
for label, values in accuracy.items():
    plt.plot(weeks, values, marker='o', linestyle='-', label=label)

plt.xlabel("Fine-tuning Weeks", fontsize=font_size)
plt.ylabel("Accuracy", fontsize=font_size)
plt.title("Impact of Different mcAP Numbers on Accuracy Over Fine-tuning Weeks", fontsize=font_size)
plt.legend(title="mcAP Count", fontsize=font_size-2, loc='lower right', bbox_to_anchor=(1.15, 0))
plt.grid(True)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
plt.savefig("accuracy_vs_finetuning_weeks_final.png", dpi=300, bbox_inches='tight')
plt.show()

# 畫 MDE 圖
plt.figure(figsize=(8, 5))
for label, values in mde.items():
    plt.plot(weeks, values, marker='s', linestyle='-', label=label)

plt.xlabel("Fine-tuning Weeks", fontsize=font_size)
plt.ylabel("Mean Distance Error (MDE)", fontsize=font_size)
plt.title("Impact of Different mcAP Numbers on MDE Over Fine-tuning Weeks", fontsize=font_size)
plt.legend(title="mcAP Count", fontsize=font_size-2, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
plt.savefig("mde_vs_finetuning_weeks_updated.png", dpi=300, bbox_inches='tight')
plt.show()

# 回傳新儲存的圖片路徑
["/mnt/data/accuracy_vs_finetuning_weeks_updated.png", "/mnt/data/mde_vs_finetuning_weeks_updated.png"]
