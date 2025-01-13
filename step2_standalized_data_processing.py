import pandas as pd
import os

# 設定來源資料夾和目標資料夾
input_folder = '2025_01_10\\2025_01_10_standalize'  # 資料來源資料夾路徑
output_folder = '2025_01_10\\standalize timestamp allign data'  # 處理後檔案的存放資料夾路徑

# 確保目標資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 讀取資料夾中的所有檔案
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # 檢查是否為 CSV 檔案
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)

        # Step 1: 修改 BSSID 對應的 AP SSID
        data.loc[data['BSSID'] == '24:29:34:e2:4c:36', 'AP SSID'] = 'AP1'
        data.loc[data['BSSID'] == '24:29:34:e1:ef:d4', 'AP SSID'] = 'AP2'
        data.loc[data['BSSID'] == 'e4:5e:1b:a0:5e:85', 'AP SSID'] = 'AP3'
        data.loc[data['BSSID'] == 'b0:e4:d5:88:16:86', 'AP SSID'] = 'AP4'

        # Step 2: 忽略 timeStemp 欄位的最後一位數
        data['timeStemp'] = data['timeStemp'].astype(str).str[:-1]  # 刪除最後一位數
        data['timeStemp'] = data['timeStemp'].astype(int)  # 轉回數字型別（如果需要）

        # Step 3: Group by Timestamp 和 AP SSID，計算平均值
        grouped_data = (
            data.groupby(['timeStemp','Label' ,'AP SSID'])
            .agg({
                # 'Label': 'first',
                'Corrected Distance (mm)': 'mean',
                'Rssi': 'mean',
            })
            .reset_index()
        )

        # Step 3: 將資料轉換成每個 Timestamp 一 row
        pivoted_data = grouped_data.pivot(
            
            index=['timeStemp','Label'],
            columns='AP SSID',
            values=['Corrected Distance (mm)', 'Rssi']
        )

        # 展平多層欄位名稱
        pivoted_data.columns = [f"{ap}_{metric}" for metric, ap in pivoted_data.columns]
        pivoted_data.reset_index(inplace=True)

        # 將處理後的結果存成新的 CSV 檔案
        output_file_path = os.path.join(output_folder, f"processed_{file_name}")
        pivoted_data.to_csv(output_file_path, index=False)

        print(f"Processed and saved: {output_file_path}")

print("所有檔案處理完成！")
