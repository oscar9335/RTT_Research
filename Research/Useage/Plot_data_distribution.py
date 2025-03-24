import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 讀取 CSV 檔案
file_path = "split_dataset_csv\\test_data_test.csv"
df = pd.read_csv(file_path)

print("欄位名稱：", df.columns.tolist())

# 確保 Label 為字串，以便分組
# df["Label"] = df["Label"].astype(str)
df["label"] = df["label"].astype(str)

# 選取要繪製分布圖的 features
features = [
    "AP1_Distance (mm)", "AP1_StdDev (mm)", "AP1_Rssi",
    "AP2_Distance (mm)", "AP2_StdDev (mm)", "AP2_Rssi",
    "AP3_Distance (mm)", "AP3_StdDev (mm)", "AP3_Rssi",
    "AP4_Distance (mm)", "AP4_StdDev (mm)", "AP4_Rssi"
]

# 建立輸出資料夾
output_dir = "output_distribution_plots_test_ALL"
os.makedirs(output_dir, exist_ok=True)

# 繪製每個 feature 在不同 Label 下的分布圖
for feature in features:
    plt.figure(figsize=(10, 6))
    # sns.boxplot(x="Label", y=feature, data=df)
    sns.boxplot(x="label", y=feature, data=df)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {feature} by Label")
    title = f"Distribution of {feature} by Label"
    plt.xlabel("Label")
    plt.ylabel(feature)

    # 將標題轉換成檔名：移除空格與括號
    safe_filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-") + ".png"
    save_path = os.path.join(output_dir, safe_filename)
    plt.savefig(save_path)
    plt.close()

    print(f"圖已儲存：{save_path}")
