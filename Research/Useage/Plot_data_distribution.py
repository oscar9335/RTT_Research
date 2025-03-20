import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
file_path = "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"
df = pd.read_csv(file_path)

# 確保 Label 為字串，以便分組
df["Label"] = df["Label"].astype(str)

# 選取要繪製分布圖的 features
features = [
    "AP1_Distance (mm)", "AP2_Distance (mm)", "AP3_Distance (mm)", "AP4_Distance (mm)",
    "AP1_Rssi", "AP2_Rssi", "AP3_Rssi", "AP4_Rssi",
    "AP1_StdDev (mm)", "AP2_StdDev (mm)", "AP3_StdDev (mm)", "AP4_StdDev (mm)"
]

# 繪製每個 feature 在不同 Label 下的分布圖
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Label", y=feature, data=df)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {feature} by Label")
    plt.xlabel("Label")
    plt.ylabel(feature)
    plt.show()
