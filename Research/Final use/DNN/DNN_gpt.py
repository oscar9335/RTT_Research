from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import time
import seaborn as sns

# -------------------
# 資料讀取與前處理部分
# -------------------

label_to_coordinates = {
    "1-1": (0, 0), "1-2": (0.6, 0), "1-3": (1.2, 0), "1-4": (1.8, 0), "1-5": (2.4, 0), "1-6": (3.0, 0),
    "1-7": (3.6, 0), "1-8": (4.2, 0), "1-9": (4.8, 0), "1-10": (5.4, 0), "1-11": (6.0, 0),
    "2-1": (0, 0.6), "2-11": (6.0, 0.6),
    "3-1": (0, 1.2), "3-11": (6.0, 1.2),
    "4-1": (0, 1.8), "4-11": (6.0, 1.8),
    "5-1": (0, 2.4), "5-11": (6.0, 2.4),
    "6-1": (0, 3.0), "6-2": (0.6, 3.0), "6-3": (1.2, 3.0), "6-4": (1.8, 3.0), "6-5": (2.4, 3.0),
    "6-6": (3.0, 3.0), "6-7": (3.6, 3.0), "6-8": (4.2, 3.0), "6-9": (4.8, 3.0), "6-10": (5.4, 3.0), "6-11": (6.0, 3.0),
    "7-1": (0, 3.6), "7-11": (6.0, 3.6),
    "8-1": (0, 4.2), "8-11": (6.0, 4.2),
    "9-1": (0, 4.8), "9-11": (6.0, 4.8),
    "10-1": (0, 5.4), "10-11": (6.0, 5.4),
    "11-1": (0, 6.0), "11-2": (0.6, 6.0), "11-3": (1.2, 6.0), "11-4": (1.8, 6.0), "11-5": (2.4, 6.0),
    "11-6": (3.0, 6.0), "11-7": (3.6, 6.0), "11-8": (4.2, 6.0), "11-9": (4.8, 6.0), "11-10": (5.4, 6.0), "11-11": (6.0, 6.0)
}
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
selected_columns = ['Label',
                    'AP1_Distance (mm)','AP4_Distance (mm)',
                    'AP1_StdDev (mm)','AP4_StdDev (mm)',
                    'AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi'
                   ]  

what_data = "test"  # 可用來指定模型名稱
file_path = "DNN\\timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"
date = "2024_12_14"

# 讀取指定欄位資料
data = pd.read_csv(file_path, usecols=selected_columns)
print(data.head())

target_column = 'Label'
label_column = 'Label'
data = data.sort_values(by=label_column).reset_index(drop=True)

# (此處若需要前後n筆處理，可加入，否則用原始資料)
processed_data = data

# 缺失值處理
data_imputed = processed_data.groupby(label_column).apply(
    lambda group: group.fillna(group.mean())
).reset_index()

print("Number of data per RP : " + str(len(data_imputed)/49))

# 建立 Label 映射：將原始 Label 映射為數值 (注意減1是為了讓標籤從 0 開始)
y = data_imputed[target_column]
reverse_label_mapping = {v: int(k) - 1 for k, v in label_mapping.items()}
y_numeric = y.map(reverse_label_mapping)

print("Final reverse_label_mapping in DNN:", reverse_label_mapping)
print("y_numeric unique values in DNN:", y_numeric.unique())

# 移除多餘欄位並標準化
X = data_imputed.drop(columns=['level_1','Label'])
scaler = StandardScaler()
columns_to_scale = selected_columns.copy()
columns_to_scale.remove('Label')
X_scaled = scaler.fit_transform(X[columns_to_scale])

# 保存標準化器 (可用於之後的推論)
joblib.dump(scaler, f'scaler_{what_data}.pkl')
print(X_scaled)

# 分割資料：使用 StratifiedShuffleSplit 保持各類均勻分佈
from sklearn.model_selection import StratifiedShuffleSplit
data_amount = 320
N_val = 20
N_train = data_amount  # 每類訓練資料數

data_df = pd.DataFrame(X_scaled)
data_df['label'] = y_numeric

train_data_full = data_df.groupby('label', group_keys=False).sample(n=N_train, replace=False)
sss = StratifiedShuffleSplit(n_splits=1, test_size=N_val / N_train, random_state=42)
train_index, val_index = next(sss.split(train_data_full.drop(columns=['label']), train_data_full['label']))
train_data = train_data_full.iloc[train_index]
val_data = train_data_full.iloc[val_index]
remaining_data = data_df.drop(train_data_full.index)

X_train, y_train = train_data.drop(columns=['label']).values, train_data['label'].values
X_val, y_val = val_data.drop(columns=['label']).values, val_data['label'].values
X_test, y_test = remaining_data.drop(columns=['label']).values, remaining_data['label'].values

# 查看各資料集 Label 數量分佈
train_label_counts = pd.Series(y_train).value_counts().sort_index()
val_label_counts = pd.Series(y_val).value_counts().sort_index()
test_label_counts = pd.Series(y_test).value_counts().sort_index()

all_labels = sorted(set(train_label_counts.index) | set(val_label_counts.index) | set(test_label_counts.index))
label_distribution = pd.DataFrame(index=all_labels)
label_distribution["Training Set"] = train_label_counts
label_distribution["Validation Set"] = val_label_counts
label_distribution["Test Set"] = test_label_counts
label_distribution = label_distribution.fillna(0).astype(int)
print(label_distribution)

# -------------------
# 優化後的 DNN 模型定義
# -------------------

# 記錄開始時間
start_time = time.time()

# 建立 DNN 模型 (包含 BatchNormalization 與 Dropout)
model = keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(label_mapping), activation='softmax')
])

# 可顯式設定學習率
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 設定 EarlyStopping 以在驗證集不再改善時提前停止
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 如有需要也可以加入 ModelCheckpoint 保存最佳模型權重：
# checkpoint = ModelCheckpoint('best_dnn_model.h5', monitor='val_loss', save_best_only=True)

# 訓練模型 (設置較大 epoch 數並依 EarlyStopping 停止)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10000,
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stop])

end_time = time.time()
print("Training time: {:.2f} seconds".format(end_time - start_time))

# -------------------
# 模型評估與結果呈現
# -------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

# 預測結果
y_test_pred = np.argmax(model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# 若需要混淆矩陣
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 若需要儲存模型，可使用：
model.save('optimized_dnn_model.h5')
