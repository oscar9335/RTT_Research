import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 建立資料
data = {
    "Model": ["mcAP1", "mcAP2", "mcAP3", "mcAP4"],
    "MDE (m)": [0.0374, 0.0381, 0.0759, 0.0544],
    "Accuracy (%)": [97.94, 98.33, 96.54, 97.63]
}
df = pd.DataFrame(data)

# 手動指定不同顏色（每根柱子一色）
color_list = ['#70b8c4', '#70c4a1', '#c4b570', '#c47b70']  # 可以根據喜好調整

# Accuracy 圖
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="Accuracy (%)", data=df, palette=color_list)
plt.ylim(96, 99)
plt.title("", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for index, value in enumerate(df["Accuracy (%)"]):
    plt.text(index, value + 0.1, f"{value:.2f}%", ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.show()

# MDE 圖
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="MDE (m)", data=df, palette=color_list)
plt.ylim(0, max(df["MDE (m)"]) + 0.02)
plt.title("", fontsize=14)
plt.ylabel("MDE (m)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for index, value in enumerate(df["MDE (m)"]):
    plt.text(index, value + 0.001, f"{value:.4f}", ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.show()