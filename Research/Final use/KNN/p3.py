import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 更新後的資料
data = {
    "Model": [
        "mcAP1+2", "mcAP1+3", "mcAP1+4",
        "mcAP2+3", "mcAP2+4", "mcAP3+4"
    ],
    "MDE (m)": [
        0.0191, 0.0323, 0.0190,
        0.0277, 0.0209, 0.0294
    ],
    "Accuracy (%)": [
        98.74, 98.29, 98.66,
        98.43, 98.76, 98.27
    ]
}

df = pd.DataFrame(data)



# 自訂顏色（每根柱子不同色）
color_list = ['#70b8c4', '#70c4a1', '#c4b570', '#c47b70', '#8e70c4', '#70c48e']

# Accuracy 圖
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="Accuracy (%)", data=df, palette=color_list)
plt.ylim(95, 100)
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
plt.ylim(0, max(df["MDE (m)"]) + 0.01)
plt.title("", fontsize=14)
plt.ylabel("MDE (m)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for index, value in enumerate(df["MDE (m)"]):
    plt.text(index, value + 0.001, f"{value:.4f}", ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.show()