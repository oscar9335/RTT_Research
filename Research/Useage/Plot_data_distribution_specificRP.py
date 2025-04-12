import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 讀取 CSV 檔案
file_path = "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"
df = pd.read_csv(file_path)

# 確保 Label 為字串
df["Label"] = df["Label"].astype(str)
# df["label"] = df["label"].astype(str)

# 選取要繪製分布圖的 features
features = [
    "AP1_Distance (mm)", "AP1_StdDev (mm)","AP1_Rssi",
    "AP2_Distance (mm)", "AP2_StdDev (mm)","AP2_Rssi",
    "AP3_Distance (mm)", "AP3_StdDev (mm)","AP3_Rssi",
    "AP4_Distance (mm)", "AP4_StdDev (mm)","AP4_Rssi"
]

# 建立輸出資料夾
output_dir = "output_distribution_plots_test"
os.makedirs(output_dir, exist_ok=True)

while True:
    # 讓使用者輸入要分析的 RP (Label)
    selected_label = input("請輸入要分析的 RP (Label): ")
    filtered_df = df[df["Label"] == selected_label]
    # filtered_df = df[df["label"] == selected_label]

    if filtered_df.empty:
        print(f"沒有找到 Label 為 {selected_label} 的數據！")
    else:
        for feature in features:
            data = filtered_df[feature].dropna()
            if data.empty or data.std() == 0:
                print(f"{feature} 的資料不足以繪圖（空值或無變化）")
                continue

            mean = data.mean()
            std = data.std()
            lower_bound = mean - std
            upper_bound = mean + std

            plt.figure(figsize=(10, 6))
            sns.histplot(data, kde=True)

            # 畫出平均線與 ±1σ 線
            plt.axvline(mean, color='red', linestyle='--', label='Mean')
            plt.axvline(lower_bound, color='orange', linestyle='--', label='-1σ')
            plt.axvline(upper_bound, color='orange', linestyle='--', label='+1σ')
            plt.axvspan(lower_bound, upper_bound, color='orange', alpha=0.2)

            title = f"Distribution of {feature} for Label {selected_label}"
            plt.title(title)
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)

            # 將標題轉換成檔名：移除空格與括號
            safe_filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-") + ".png"
            save_path = os.path.join(output_dir, safe_filename)
            plt.savefig(save_path)
            plt.close()

            print(f"圖已儲存：{save_path}")
