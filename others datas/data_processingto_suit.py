import pandas as pd

# 讀取新版 ESP32C3 測試資料
esp_dataset = pd.read_csv('TRAIN_DATASET.csv')

# 轉換 FTM 單位：nanoseconds → millimeters
ftm_cols = [col for col in esp_dataset.columns if 'FTM' in col]
for col in ftm_cols:
    esp_dataset[col] = esp_dataset[col] * 0.15

# 根據 (X, Y) 群組成 label，從0開始
unique_positions = esp_dataset[['X Position (meters)', 'Y Position (meters)']].drop_duplicates().reset_index(drop=True)
position_to_label = { (row['X Position (meters)'], row['Y Position (meters)']) : idx for idx, row in unique_positions.iterrows() }

# 新增 Label 欄位
esp_dataset['Label'] = esp_dataset.apply(lambda row: position_to_label[(row['X Position (meters)'], row['Y Position (meters)'])], axis=1)

# 整理成輸出格式：Label + X + Y + FTM(mm) + RSSI
final_columns = ['Label', 'X Position (meters)', 'Y Position (meters)'] + ftm_cols + [col for col in esp_dataset.columns if 'RSSI' in col]
final_esp_dataset = esp_dataset[final_columns]

# 輸出成新的CSV
final_esp_dataset.to_csv('ESP32C3_processed_for_trainging.csv', index=False)

print("轉換完成！新資料儲存在 ESP32C3_processed_for_testing.csv")
