
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 建立三個 AP 升級情境下的資料
data = {
    "Model": [
        "mcAP1+2+3", "mcAP1+2+4", "mcAP1+3+4", "mcAP2+3+4"
    ],
    "MDE (m)": [
        0.0176, 0.0147, 0.0159, 0.0160
    ],
    "Accuracy (%)": [
        98.72, 98.99, 98.91, 98.95
    ]
}

df = pd.DataFrame(data)


# 顏色配置
color_list = ['#70b8c4', '#70c4a1', '#c4b570', '#c47b70']

# Accuracy 圖
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="Accuracy (%)", data=df, palette=color_list)
plt.ylim(98.2, 99.2)
plt.title("", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.xticks(rotation=30, fontsize=14)
plt.yticks(fontsize=14)
for index, value in enumerate(df["Accuracy (%)"]):
    plt.text(index, value + 0.02, f"{value:.2f}%", ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.show()

# MDE 圖
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="MDE (m)", data=df, palette=color_list)
plt.ylim(0, max(df["MDE (m)"]) + 0.005)
plt.title("", fontsize=14)
plt.ylabel("MDE (m)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.xticks(rotation=30, fontsize=14)
plt.yticks(fontsize=14)
for index, value in enumerate(df["MDE (m)"]):
    plt.text(index, value + 0.0005, f"{value:.4f}", ha='center', va='bottom', fontsize=14)
plt.tight_layout()
plt.show()
