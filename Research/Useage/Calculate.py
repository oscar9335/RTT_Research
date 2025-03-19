import pandas as pd

# 讀取 CSV 檔案
file_path = "merged_rtt_logs.csv"
df = pd.read_csv(file_path)

# 計算每個 RP (Label) 在每個 AP (1~4) 收到的標準差 (Std_Dev) 平均值
std_dev_columns = ['AP1_StdDev (mm)', 'AP2_StdDev (mm)', 'AP3_StdDev (mm)', 'AP4_StdDev (mm)']
grouped_std_dev = df.groupby('Label')[std_dev_columns].mean().reset_index()

print(grouped_std_dev)

# 轉換為逗號分隔的單行輸出
ap1_std_dev_values = ','.join(map(str, grouped_std_dev['AP4_StdDev (mm)']))
print(ap1_std_dev_values)
