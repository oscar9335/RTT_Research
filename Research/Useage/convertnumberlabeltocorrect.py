import pandas as pd

def update_csv_labels(input_file, output_file):
    # 讀取 CSV 檔案
    df = pd.read_csv(input_file)
    
    # 定義 Label 對應關係
    label_mapping = {
        '11': '1-1','10': '1-2','9': '1-3','8': '1-4','7': '1-5','6': '1-6','5': '1-7','4': '1-8','3': '1-9','2': '1-10','1': '1-11',
        '12': '2-1','30': '2-11',
        '13': '3-1','29': '3-11',
        '14': '4-1','28': '4-11',
        '15': '5-1','27': '5-11',
        '16': '6-1','17': '6-2','18': '6-3','19': '6-4','20': '6-5','21': '6-6','22': '6-7','23': '6-8','24': '6-9','25': '6-10','26': '6-11',
        '49': '7-1','31': '7-11',
        '48': '8-1','32': '8-11',
        '47': '9-1','33': '9-11',
        '46': '10-1','34': '10-11',
        '45': '11-1','44': '11-2','43': '11-3','42': '11-4','41': '11-5','40': '11-6','39': '11-7','38': '11-8','37': '11-9','36': '11-10','35': '11-11'
    }
    
    # 確保 Label 欄位是字串類型，並執行對應替換
    # df['Label'] = df['Label'].astype(str).map(label_mapping).fillna(df['Label'])
    df['label'] = df['label'].astype(str).map(label_mapping).fillna(df['label'])
    
    # 儲存更新後的 CSV 檔案
    df.to_csv(output_file, index=False)
    print(f"更新後的檔案已儲存至 {output_file}")

# 使用範例
input_file = "split_dataset_csv\\val_data_test.csv"  # 原始檔案
output_file = "val_data_test.csv"  # 更新後的檔案
update_csv_labels(input_file, output_file)
