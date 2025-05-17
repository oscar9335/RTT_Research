import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 載入資料
df = pd.read_csv("timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv")

# 提取 AP1、AP4 有支援 FTM 的資料
ap1_data = df[['AP1_Rssi', 'AP1_Distance (mm)']].dropna().rename(columns={'AP1_Rssi': 'Rssi', 'AP1_Distance (mm)': 'Distance'})
ap4_data = df[['AP4_Rssi', 'AP4_Distance (mm)']].dropna().rename(columns={'AP4_Rssi': 'Rssi', 'AP4_Distance (mm)': 'Distance'})

# 合併 AP1 與 AP4 資料
train_data = pd.concat([ap1_data, ap4_data], ignore_index=True)

# 特徵與目標值
X_train = train_data[['Rssi']]
y_train = train_data['Distance']

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 建立並訓練 Regressor
model = SGDRegressor(tol=1e-4, penalty="l2", alpha=0.0001, eta0=0.1, learning_rate='optimal', loss='huber', max_iter=5000)
model.fit(X_train_scaled, y_train)

print("訓練完畢！模型係數:", model.coef_, "截距:", model.intercept_)

# 預測 AP2
X_ap2 = df[['AP2_Rssi']].rename(columns={'AP2_Rssi': 'Rssi'})
mask_ap2 = X_ap2['Rssi'].notna()
X_ap2_scaled = scaler.transform(X_ap2[mask_ap2])
df.loc[mask_ap2, 'AP2_Distance_predicted'] = model.predict(X_ap2_scaled)

# 預測 AP3
X_ap3 = df[['AP3_Rssi']].rename(columns={'AP3_Rssi': 'Rssi'})
mask_ap3 = X_ap3['Rssi'].notna()
X_ap3_scaled = scaler.transform(X_ap3[mask_ap3])
df.loc[mask_ap3, 'AP3_Distance_predicted'] = model.predict(X_ap3_scaled)

# 計算預測誤差
if 'AP2_Distance (mm)' in df.columns:
    valid_ap2 = df[['AP2_Distance (mm)', 'AP2_Distance_predicted']].dropna()
    mae_ap2 = mean_absolute_error(valid_ap2['AP2_Distance (mm)'], valid_ap2['AP2_Distance_predicted'])
    mse_ap2 = mean_squared_error(valid_ap2['AP2_Distance (mm)'], valid_ap2['AP2_Distance_predicted'])
    r2_ap2 = r2_score(valid_ap2['AP2_Distance (mm)'], valid_ap2['AP2_Distance_predicted'])
    print(f"AP2 MAE: {mae_ap2:.2f} mm, MSE: {mse_ap2:.2f} mm², R²: {r2_ap2:.4f}")

if 'AP3_Distance (mm)' in df.columns:
    valid_ap3 = df[['AP3_Distance (mm)', 'AP3_Distance_predicted']].dropna()
    mae_ap3 = mean_absolute_error(valid_ap3['AP3_Distance (mm)'], valid_ap3['AP3_Distance_predicted'])
    mse_ap3 = mean_squared_error(valid_ap3['AP3_Distance (mm)'], valid_ap3['AP3_Distance_predicted'])
    r2_ap3 = r2_score(valid_ap3['AP3_Distance (mm)'], valid_ap3['AP3_Distance_predicted'])
    print(f"AP3 MAE: {mae_ap3:.2f} mm, MSE: {mse_ap3:.2f} mm², R²: {r2_ap3:.4f}")

# 整理並顯示部分預測結果
if 'AP2_Distance (mm)' in df.columns:
    df['AP2_Error'] = df['AP2_Distance (mm)'] - df['AP2_Distance_predicted']
if 'AP3_Distance (mm)' in df.columns:
    df['AP3_Error'] = df['AP3_Distance (mm)'] - df['AP3_Distance_predicted']

cols_to_show = ['AP2_Rssi', 'AP2_Distance_predicted']
if 'AP2_Distance (mm)' in df.columns:
    cols_to_show.append('AP2_Distance (mm)')
    cols_to_show.append('AP2_Error')

cols_to_show += ['AP3_Rssi', 'AP3_Distance_predicted']
if 'AP3_Distance (mm)' in df.columns:
    cols_to_show.append('AP3_Distance (mm)')
    cols_to_show.append('AP3_Error')

print("\n預測結果（部分樣本）:")
print(df[cols_to_show].head())
