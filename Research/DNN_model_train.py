import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import json
import os 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
import joblib  # 用於保存模型

label_to_coordinates = {
    "1-1": (0, 0), "1-2": (0.6, 0), "1-3": (1.2, 0), "1-4": (1.8, 0), "1-5": (2.4, 0), "1-6": (3.0, 0),"1-7": (3.6, 0), "1-8": (4.2, 0), "1-9": (4.8, 0), "1-10": (5.4, 0), "1-11": (6.0, 0),
    "2-1": (0, 0.6), "2-11": (6.0, 0.6),
    "3-1": (0, 1.2), "3-11": (6.0, 1.2),
    "4-1": (0, 1.8), "4-11": (6.0, 1.8),
    "5-1": (0, 2.4), "5-11": (6.0, 2.4),
    "6-1": (0, 3.0), "6-2": (0.6, 3.0), "6-3": (1.2, 3.0), "6-4": (1.8, 3.0), "6-5": (2.4, 3.0),"6-6": (3.0, 3.0), "6-7": (3.6, 3.0), "6-8": (4.2, 3.0), "6-9": (4.8, 3.0), "6-10": (5.4, 3.0), "6-11": (6.0, 3.0),
    "7-1": (0, 3.6), "7-11": (6.0, 3.6),
    "8-1": (0, 4.2), "8-11": (6.0, 4.2),
    "9-1": (0, 4.8), "9-11": (6.0, 4.8),
    "10-1": (0, 5.4), "10-11": (6.0, 5.4),
    "11-1": (0, 6.0), "11-2": (0.6, 6.0), "11-3": (1.2, 6.0), "11-4": (1.8, 6.0), "11-5": (2.4, 6.0),"11-6": (3.0, 6.0), "11-7": (3.6, 6.0), "11-8": (4.2, 6.0), "11-9": (4.8, 6.0), "11-10": (5.4, 6.0), "11-11": (6.0, 6.0)
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
                        'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)',
                        'AP1_StdDev (mm)','AP2_StdDev (mm)','AP3_StdDev (mm)','AP4_StdDev (mm)',
                                'AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi'
                                ]  

# 'AP1_StdDev (mm)','AP2_StdDev (mm)',
what_data = "test"
# what_data = "DNN_3Layer_1AP_Best_combine_F_S_R_20241214_usingearlystopepchoc10000"

# 'AP1_Distance (mm)',
# 'AP1_StdDev (mm)',
# 1. 讀取資料
file_path = "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"

date = "2024_12_14"

data = pd.read_csv(file_path, usecols=selected_columns)

print(data.head())

target_column = 'Label'  # 替換成目標欄位名稱
label_column = 'Label'  # 替換成目標欄位名稱
# 資料前處理 (一): 刪除前後n筆資料
n = 1
# 確保依據Label排序
data = data.sort_values(by=label_column).reset_index(drop=True)

# 建立一個空的 DataFrame 用於存放處理後的資料
processed_data = pd.DataFrame(columns=data.columns)


# # 針對每個Label群組進行處理
# for label, group in data.groupby(label_column):
#     group = group.iloc[n:-n]
#     # 將處理後的群組資料加入
#     processed_data = pd.concat([processed_data, group], ignore_index=True)

processed_data = data


data_imputed = processed_data.groupby(label_column).apply(
    lambda group: group.fillna(group.mean())
).reset_index()
data_imputed

print("Number of data per RP : " + str(len(data_imputed)/49))
# 建立 Label 映射
y = data_imputed[target_column]
# label_mapping = {str(i): label for i, label in enumerate(y.unique())}
reverse_label_mapping = {v: int(k) - 1 for k, v in label_mapping.items()}  # 讓數字標籤 -1
y_numeric = y.map(reverse_label_mapping)

print("Final reverse_label_mapping in DNN:", reverse_label_mapping)
print("y_numeric unique values in DNN:", y_numeric.unique())
# 把label部分拿掉
X = data_imputed.drop(columns=['level_1','Label'])
print(X.head())
scaler = StandardScaler()
columns_to_scale = selected_columns.copy()  # 建立副本，避免影響原始變數
columns_to_scale.remove('Label')  # 在副本上移除 'Label'
X_scaled = scaler.fit_transform(X[columns_to_scale])

# 保存標準化器
joblib.dump(scaler, f'scaler_{what_data}.pkl')

print(X_scaled)

from sklearn.model_selection import StratifiedShuffleSplit

all_mde = []
all_accuracy = []

best_mde = float('inf')  # 初始化最佳 MDE

modelname = "2mcAPbestbset"

for i in range(1):

    ap = 'test'
    root = 'test'

    dataamount = 320
    N_val = 20

    N_train = dataamount # 訓練集每個類別至少要有 N_train 筆資料
    test_val_ratio = 1  # 剩餘資料中，50% 作為驗證集，50% 作為測試集
    # 轉為 DataFrame 方便操作
    data = pd.DataFrame(X_scaled)
    data['label'] = y_numeric  # 加入 label 欄位

    train_data_full = data.groupby('label', group_keys=False).sample(n=N_train, replace=False,random_state=42) 

    train_data_full

    sss = StratifiedShuffleSplit(n_splits=1, test_size=N_val / N_train,random_state=42) 
    train_index, val_index = next(sss.split(train_data_full.drop(columns=['label']), train_data_full['label']))
    train_data = train_data_full.iloc[train_index]
    val_data = train_data_full.iloc[val_index]

    remaining_data = data.drop(train_data_full.index)
    X_train, y_train = train_data.drop(columns=['label']).values, train_data['label'].values
    X_val, y_val = val_data.drop(columns=['label']).values, val_data['label'].values
    X_test, y_test = remaining_data.drop(columns=['label']).values, remaining_data['label'].values

    # **計算每個 Set 內各 Label 的資料數量**
    train_label_counts = pd.Series(y_train).value_counts().sort_index()
    val_label_counts = pd.Series(y_val).value_counts().sort_index()
    test_label_counts = pd.Series(y_test).value_counts().sort_index()

    # **確保所有 Labels 都有出現在三個 Set 裡**
    all_labels = sorted(set(train_label_counts.index) | set(val_label_counts.index) | set(test_label_counts.index))
    label_distribution = pd.DataFrame(index=all_labels)

    label_distribution["Training Set"] = train_label_counts
    label_distribution["Validation Set"] = val_label_counts
    label_distribution["Test Set"] = test_label_counts

    # **用 0 填補缺失值（表示該 Label 在該 Set 中沒有數據）**
    label_distribution = label_distribution.fillna(0).astype(int)

    from IPython.display import display
    display(label_distribution)

    import time
    # 記錄開始時間
    start_time = time.time()
    # X_train, y_train 
    # X_val, y_val
    # X_test, y_test 
    # 建立 DNN 模型

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(label_mapping), activation='softmax')
    ])

    # 編譯模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 設定 EarlyStopping 回呼函數
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    early_stop_loss = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # 訓練模型，包含驗證集
    model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10000, batch_size=32, verbose=1, callbacks=[early_stop])
    
    # # 訓練模型，包含驗證集
    # model.fit(X_train, y_train,
    #         epochs=10000, batch_size=32, verbose=1, callbacks=[early_stop_loss])

    # 記錄結束時間
    end_time = time.time()


    # Needsave
    # 計算訓練時間（秒）
    training_time = end_time - start_time
    print(f"訓練時間：{training_time:.2f} 秒")
    mde_report_per_fold = {}

    # 進行預測
    y_test_pred_numeric = model.predict(X_test)
    y_pred_classes = np.argmax(y_test_pred_numeric, axis=1)
    y_test_pred_labels = [label_mapping[str(num + 1)] for num in y_pred_classes]  # 補回 +1


    # 計算 MDE
    y_test_pred_coordinates = np.array([label_to_coordinates[label] for label in y_test_pred_labels])
    y_test_coordinates = np.array([label_to_coordinates[label_mapping[str(label + 1)]] for label in y_test])


    distances = np.linalg.norm(y_test_pred_coordinates - y_test_coordinates, axis=1)
    avg_mde = np.mean(distances)

    # if best_mde > avg_mde:
    #     best_mde = avg_mde
    #     model.save(f'{modelname}.h5')

    model.save(f'{modelname}.h5')

    print(f"MDE: {avg_mde:.4f}")

    all_mde.append(avg_mde)

    # 記錄每個 RP 在當前 fold 的 MDE
    for true_label, distance in zip(y_test, distances):
        if true_label not in mde_report_per_fold:
            mde_report_per_fold[true_label] = []
        mde_report_per_fold[true_label].append(distance)  # 存所有 fold 的 MDE

    # 計算 5-Fold 平均 MDE
    # mde_report_avg = {label: {"mde": np.mean(distances), "count": len(distances)} for label, distances in mde_report_per_fold.items()}

    mde_report_avg = {int(label): {"mde": np.mean(distances), "count": len(distances)} 
                    for label, distances in mde_report_per_fold.items()}
                    

    # # 儲存到 JSON 檔案
    # file_path = f"Testing_mde_using_loss_Bestcomb_{i}"
    # with open(file_path, "w") as f:
    #     json.dump(mde_report_avg, f, indent=4)

    # print(f"MDE report saved to: {file_path}")
    

## mde every RP

    # 記錄每個 RP 在當前 fold 的 MDE
    for true_label, distance in zip(y_test, distances):
        true_label = int(true_label)  # 確保鍵是 int
        if true_label not in mde_report_per_fold:
            mde_report_per_fold[true_label] = []
        mde_report_per_fold[true_label].append(distance)  # 存所有 fold 的 MDE

    # 計算 5-Fold 平均 MDE，並記錄 MDE 不為 0 的 error 值
    mde_report_avg = {}
    
    for label, dist_list in mde_report_per_fold.items():
        mean_dist = np.mean(dist_list)
        count = len(dist_list)

        # 如果 mean_dist > 0，則記錄個別的 error 值
        error_dict = {str(idx + 1): float(d) for idx, d in enumerate(dist_list) if d > 0}

        # 建構最終輸出格式
        mde_report_avg[int(label)] = {
            "mde": mean_dist,
            "count": count
        }

        # 只有當 MDE 大於 0 時才存儲 error 值
        if error_dict:
            mde_report_avg[int(label)]["error"] = error_dict

    # # 儲存到 JSON 檔案
    # file_path = f"Testing_mde_detailed_using_loss_Bestcomb_{i}.json"
    # with open(file_path, "w") as f:
    #     json.dump(mde_report_avg, f, indent=4)

    # print(f"MDE report saved to: {file_path}")


## accuracy
    # **計算 Accuracy**
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    all_accuracy.append(test_accuracy)

    # **記錄每個 RP 的 Accuracy**
    accuracy_report_per_fold = {}
    for true_label, pred_label in zip(y_test, y_pred_classes):
        if true_label not in accuracy_report_per_fold:
            accuracy_report_per_fold[true_label] = {"correct": 0, "total": 0}
        accuracy_report_per_fold[true_label]["total"] += 1
        if true_label == pred_label:
            accuracy_report_per_fold[true_label]["correct"] += 1

    # 計算每個 RP 的 Accuracy
    accuracy_report_avg = {int(label): {"accuracy": correct_info["correct"] / correct_info["total"], 
                                        "count": correct_info["total"]}
                        for label, correct_info in accuracy_report_per_fold.items()}
    
    
    # # 儲存到 JSON 檔案
    # file_path = f"Testing_accuracy_using_loss_Bestcomb_{i}"
    # with open(file_path, "w") as f:
    #     json.dump(accuracy_report_avg, f, indent=4)

    # print(f"Accuracy report saved to: {file_path}")

print([round(float(mde), 4) for mde in all_mde])
print([round(float(acc), 4) for acc in all_accuracy])