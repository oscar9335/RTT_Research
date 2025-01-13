import pandas as pd

# 讀取兩份 CSV 檔案
file1_path = 'new feature add\\timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv'
file2_path = 'new feature add\\standalized_timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv'

# 載入資料
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)



# 依據 timestamp 和 label 進行合併
merged_df = pd.merge(df1, df2, on=['timeStemp', 'Label'], how='inner')

# 將合併後的資料另存為新的 CSV 檔案
merged_file_path = 'new feature add\\timestamp_allignment_Balanced_2024_12_14_rtt_logs_withcorrected_distance.csv'
merged_df.to_csv(merged_file_path, index=False)

print(f"合併完成，已將結果儲存至：{merged_file_path}")
