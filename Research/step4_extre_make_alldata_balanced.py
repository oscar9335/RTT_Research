import pandas as pd
import numpy as np

# 讀取 Excel 檔案
file_path = 'standalized_timestamp_allignment_2025_01_10_rtt_logs.csv'
df = pd.read_csv(file_path)

print(df.head())

# 假設 label 的欄位名稱為 'label'
# 計算每個 label 的資料筆數
label_counts = df['Label'].value_counts()

# 找出最少的資料筆數
min_count = label_counts.min()

# 隨機抽取每個 label 的資料，使其數量等於 min_count
df_balanced = df.groupby('Label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

# 儲存處理後的資料
output_path = 'standalized_timestamp_allignment_Balanced_2025_01_10_rtt_logs.csv'
df_balanced.to_csv(output_path, index=False)

print(f"處理後的資料已儲存至 {output_path}")
