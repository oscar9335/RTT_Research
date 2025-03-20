import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
file_path = "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"
df = pd.read_csv(file_path)

# 確保 Label 為字串
df["Label"] = df["Label"].astype(str)

# 選取要繪製分布圖的 features
features = [
    "AP1_Distance (mm)", "AP2_Distance (mm)", "AP3_Distance (mm)", "AP4_Distance (mm)"
]

# ,
#     "AP1_Rssi", "AP2_Rssi", "AP3_Rssi", "AP4_Rssi",
#     "AP1_StdDev (mm)", "AP2_StdDev (mm)", "AP3_StdDev (mm)", "AP4_StdDev (mm)"

# 讓使用者輸入要分析的 RP (Label)
selected_label = input("請輸入要分析的 RP (Label): ")
filtered_df = df[df["Label"] == selected_label]

if filtered_df.empty:
    print(f"沒有找到 Label 為 {selected_label} 的數據！")
else:
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df[feature], bins=30, kde=True)
        plt.title(f"Frequency Distribution of {feature} for Label {selected_label}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
