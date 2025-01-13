import pandas as pd
import os

# 指定資料夾路徑
# input_folder = '2025_01_10\\timestamp allign data'  # 替換為你的資料夾路徑
# output_file = '2025_01_10\\timestamp_allignment_2025_01_10_rtt_logs.csv'  # 合併後的輸出檔案名稱

input_folder = '2025_01_10\\standalize timestamp allign data'  # 替換為你的資料夾路徑
output_file = '2025_01_10\\standalized_timestamp_allignment_2025_01_10_rtt_logs.csv'  # 合併後的輸出檔案名稱

# input_folder = 'new feature add'
# output_file = 'new feature add\\2024_12_14_corrected_distance'

# 獲取資料夾內所有 CSV 檔案的路徑
file_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.csv')]

# 合併所有 CSV 檔案
combined_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths], ignore_index=True)

# 儲存合併後的檔案
combined_data.to_csv(output_file, index=False)

print(f"所有檔案已合併並儲存為: {output_file}")
