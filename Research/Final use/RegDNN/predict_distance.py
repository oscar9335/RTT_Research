import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 載入 CSV 資料，請根據實際檔案路徑進行修改
df = pd.read_csv("timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv")

# 提取 AP1 與 AP4 有支援 FTM 的部分，僅選取有有效 distance 的資料
ap1_data = df[['AP1_Rssi', 'AP1_Distance (mm)']].dropna().rename(
    columns={'AP1_Rssi': 'Rssi', 'AP1_Distance (mm)': 'Distance'}
)
ap4_data = df[['AP4_Rssi', 'AP4_Distance (mm)']].dropna().rename(
    columns={'AP4_Rssi': 'Rssi', 'AP4_Distance (mm)': 'Distance'}
)

# 合併 AP1 與 AP4 的資料，形成訓練資料集
train_data = pd.concat([ap1_data, ap4_data], ignore_index=True)

# 定義特徵與目標值
X_train = train_data[['Rssi']]
y_train = train_data['Distance']

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

print("訓練完畢！模型係數:", model.coef_, "截距:", model.intercept_)

# 預測 AP2 的距離，先將欄位名稱轉換為 'Rssi'
X_ap2 = df[['AP2_Rssi']].rename(columns={'AP2_Rssi': 'Rssi'})
# 找出 AP2_Rssi 非缺失值的 index
mask_ap2 = X_ap2['Rssi'].notna()
df.loc[mask_ap2, 'AP2_Distance_predicted'] = model.predict(X_ap2[mask_ap2])

# 同理，預測 AP3 的距離
X_ap3 = df[['AP3_Rssi']].rename(columns={'AP3_Rssi': 'Rssi'})
mask_ap3 = X_ap3['Rssi'].notna()
df.loc[mask_ap3, 'AP3_Distance_predicted'] = model.predict(X_ap3[mask_ap3])

# 若原始資料中有提供 AP2/3 的實際 distance 資料，計算預測誤差
if 'AP2_Distance (mm)' in df.columns:
    valid_ap2 = df[['AP2_Distance (mm)', 'AP2_Distance_predicted']].dropna()
    mae_ap2 = mean_absolute_error(valid_ap2['AP2_Distance (mm)'], valid_ap2['AP2_Distance_predicted'])
    mse_ap2 = mean_squared_error(valid_ap2['AP2_Distance (mm)'], valid_ap2['AP2_Distance_predicted'])
    print("AP2 MAE:", mae_ap2, "MSE:", mse_ap2)

if 'AP3_Distance (mm)' in df.columns:
    valid_ap3 = df[['AP3_Distance (mm)', 'AP3_Distance_predicted']].dropna()
    mae_ap3 = mean_absolute_error(valid_ap3['AP3_Distance (mm)'], valid_ap3['AP3_Distance_predicted'])
    mse_ap3 = mean_squared_error(valid_ap3['AP3_Distance (mm)'], valid_ap3['AP3_Distance_predicted'])
    print("AP3 MAE:", mae_ap3, "MSE:", mse_ap3)

# 顯示前幾筆預測結果與原始距離（如果存在）的比較
cols_to_show = ['AP2_Rssi', 'AP2_Distance_predicted']
if 'AP2_Distance (mm)' in df.columns:
    cols_to_show.append('AP2_Distance (mm)')

cols_to_show += ['AP3_Rssi', 'AP3_Distance_predicted']
if 'AP3_Distance (mm)' in df.columns:
    cols_to_show.append('AP3_Distance (mm)')

print(df[cols_to_show].head())
