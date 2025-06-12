import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import joblib
import json
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import GridSearchCV


all_mde = []
all_accuracy = []

best_mde = float('inf')  # 初始化最佳 MDE

all_errors = []

for loop in range(20):

### 1. 訓練 regressor 
    df_reg = pd.read_csv("timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv")

    # 篩選有有效 distance 的資料
    ap1_data = df_reg[['AP1_Rssi', 'AP1_Distance (mm)']].dropna().rename(
        columns={'AP1_Rssi': 'Rssi', 'AP1_Distance (mm)': 'Distance'}
    )

    # ap2_data = df_reg[['AP2_Rssi', 'AP2_Distance (mm)']].dropna().rename(
    #     columns={'AP2_Rssi': 'Rssi', 'AP2_Distance (mm)': 'Distance'}
    # )

    # ap3_data = df_reg[['AP3_Rssi', 'AP3_Distance (mm)']].dropna().rename(
    #     columns={'AP3_Rssi': 'Rssi', 'AP3_Distance (mm)': 'Distance'}
    # )

    # ap4_data = df_reg[['AP4_Rssi', 'AP4_Distance (mm)']].dropna().rename(
    #     columns={'AP4_Rssi': 'Rssi', 'AP4_Distance (mm)': 'Distance'}
    # )

    train_data_reg = pd.concat([ap1_data], ignore_index=True)
    X_train_reg = train_data_reg[['Rssi']]
    y_train_reg = train_data_reg['Distance']

# 標準化 regressor 輸入（單一特徵）
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    
### 訓練 regressor

    # # 定義要搜尋的超參數範圍
    # param_grid = {
    #     'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    #     'loss': ['huber', 'squared_loss', 'epsilon_insensitive'],
    #     'learning_rate': ['constant', 'optimal', 'invscaling'],
    #     'eta0': [0.001, 0.01, 0.1]
    # }

    # # 建立基本的 SGDRegressor（其餘參數可依需求調整）
    # sgd_reg = SGDRegressor(tol=1e-4, penalty="l2", max_iter=5000)

    # # 利用 GridSearchCV 進行 5-fold 交叉驗證，評分依據為負均方誤差 (越大越好)
    # grid_search = GridSearchCV(sgd_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1,verbose=1)
    # grid_search.fit(X_train_reg_scaled, y_train_reg)

    # print("Best Parameters for regressor:", grid_search.best_params_)
    # print("Best Score (neg MSE):", grid_search.best_score_)

    # # 取出最佳參數後建立 regressor 並訓練
    # best_params = grid_search.best_params_
    model_reg = SGDRegressor(tol=1e-4, penalty="l2",alpha= 0.0001, eta0= 0.1, learning_rate= 'optimal', loss= 'huber',max_iter=5000)
    model_reg.fit(X_train_reg_scaled, y_train_reg)
    print("Regressor trained. Coefficient:", model_reg.coef_, "Intercept:", model_reg.intercept_)

### 1-2 存下 regressor
    joblib.dump(model_reg, f'regressor_model_AP1&AP4_{loop}.pkl')
    joblib.dump(scaler_reg, f'scaler_regressor_input_{loop}.pkl')



### 2. 讀取 DNN 需要的資料
    # DNN 使用的原始欄位
    selected_columns = ['Label',
                        'AP1_Distance (mm)',
                        'AP1_StdDev (mm)',
                        'AP1_Rssi', 'AP2_Rssi', 'AP3_Rssi', 'AP4_Rssi']

    file_path = "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv"
    data = pd.read_csv(file_path, usecols=selected_columns)
    print("原始資料預覽：")
    print(data.head())

    label_column = 'Label'
    # 這邊簡單用全部資料當作 processed_data，你也可以根據需要刪除前後 n 筆資料等
    processed_data = data.copy()

    # 資料填補：以每個 Label 群組內的平均值填補缺失值
    data_imputed = processed_data.groupby(label_column).apply(
        lambda group: group.fillna(group.mean())
    ).reset_index()

### 3. 利用 regressor 擴充 AP 2 3 的 Distance 預測值
    # 建立新欄位，初值設定為 NaN
    # data_imputed['AP1_Distance_predicted'] = np.nan
    data_imputed['AP2_Distance_predicted'] = np.nan
    data_imputed['AP3_Distance_predicted'] = np.nan
    data_imputed['AP4_Distance_predicted'] = np.nan

    # # 利用 AP1_Rssi 預測 AP1_Distance_predicted
    # mask_ap1 = data_imputed['AP1_Rssi'].notna()
    # if mask_ap1.any():
    #     # 先將 AP2_Rssi 轉換成 regressor 所需的格式並標準化
    #     AP1_Rssi_scaled = scaler_reg.transform(data_imputed.loc[mask_ap1, ['AP1_Rssi']].rename(columns={'AP1_Rssi': 'Rssi'}))
    #     data_imputed.loc[mask_ap1, 'AP1_Distance_predicted'] = model_reg.predict(AP1_Rssi_scaled)


    # 利用 AP2_Rssi 預測 AP2_Distance_predicted
    mask_ap2 = data_imputed['AP2_Rssi'].notna()
    if mask_ap2.any():
        # 先將 AP2_Rssi 轉換成 regressor 所需的格式並標準化
        AP2_Rssi_scaled = scaler_reg.transform(data_imputed.loc[mask_ap2, ['AP2_Rssi']].rename(columns={'AP2_Rssi': 'Rssi'}))
        data_imputed.loc[mask_ap2, 'AP2_Distance_predicted'] = model_reg.predict(AP2_Rssi_scaled)
    
    # # 利用 AP3_Rssi 預測 AP3_Distance_predicted
    mask_ap3 = data_imputed['AP3_Rssi'].notna()
    if mask_ap3.any():
        AP3_Rssi_scaled = scaler_reg.transform(data_imputed.loc[mask_ap3, ['AP3_Rssi']].rename(columns={'AP3_Rssi': 'Rssi'}))
        data_imputed.loc[mask_ap3, 'AP3_Distance_predicted'] = model_reg.predict(AP3_Rssi_scaled)

    # 利用 AP4_Rssi 預測 AP4_Distance_predicted
    mask_ap4 = data_imputed['AP4_Rssi'].notna()
    if mask_ap4.any():
        AP4_Rssi_scaled = scaler_reg.transform(data_imputed.loc[mask_ap4, ['AP4_Rssi']].rename(columns={'AP4_Rssi': 'Rssi'}))
        data_imputed.loc[mask_ap4, 'AP4_Distance_predicted'] = model_reg.predict(AP4_Rssi_scaled)

    # # 利用 AP4_Rssi 預測 AP4_Distance_predicted
    # mask_ap4 = data_imputed['AP4_Rssi'].notna()
    # data_imputed.loc[mask_ap4, 'AP4_Distance_predicted'] = model_reg.predict(
    #     data_imputed.loc[mask_ap4, ['AP4_Rssi']].rename(columns={'AP4_Rssi': 'Rssi'})
    #     )

    # 更新 DNN 模型用的特徵欄位，將 regressor 預測值加入
    selected_columns_dnn = selected_columns + ['AP2_Distance_predicted','AP3_Distance_predicted','AP4_Distance_predicted']

### 4. 後續 DNN 資料準備與訓練
    print("每個 RP 的資料筆數: " + str(len(data_imputed)/49))

    # 以下部分依照原有 DNN code 執行
    target_column = 'Label'
    y = data_imputed[target_column]

    # 建立 label mapping (依你的需求調整)
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

    # 這裡的 mapping 會將 Label 轉換為數值（數值減 1）
    reverse_label_mapping = {v: int(k) - 1 for k, v in label_mapping.items()}
    y_numeric = y.map(reverse_label_mapping)
    print("DNN 最終的 reverse_label_mapping:", reverse_label_mapping)
    print("y_numeric 的 unique 值:", y_numeric.unique())

    # 取出作為模型輸入的特徵，這邊不含 Label 欄位
    X = data_imputed[selected_columns_dnn].drop(columns=['Label'])
    print("擴充後的特徵資料預覽：")
    print(X.head())

    print("*******************所有特徵欄位*******************：")
    print(list(X.columns))

    # 統一對合併後的所有 DNN 特徵進行標準化，避免重複標準化
    scaler_dnn = StandardScaler()
    X_scaled = scaler_dnn.fit_transform(X)
    joblib.dump(scaler_dnn, f'scaler_regressor_dnn_AP1&AP4_{loop}.pkl')
    print("標準化後的特徵陣列：")
    print(X_scaled)

    # 後續的資料切分、DNN 模型定義與訓練步驟請沿用你原有的流程
    # 以下僅示範建立 DNN 模型與訓練（請根據你的需求補上完整流程）

    from sklearn.model_selection import StratifiedShuffleSplit
    import os

    # # 設定輸出資料夾
    # output_dir = 'split_dataset_csv'
    # os.makedirs(output_dir, exist_ok=True)


    ap = 'test'
    root = 'test'

    dataamount = 320
    N_val = 20

    N_train = dataamount # 訓練集每個類別至少要有 N_train 筆資料
    test_val_ratio = 1  # 剩餘資料中，50% 作為驗證集，50% 作為測試集

    # 轉為 DataFrame 方便操作
    data = pd.DataFrame(X_scaled)
    data['label'] = y_numeric  # 加入 label 欄位

    print(data)

    train_data_full = data.groupby('label', group_keys=False).sample(n=N_train, replace=False,random_state=42) 

    train_data_full

    sss = StratifiedShuffleSplit(n_splits=1, test_size=N_val / N_train,random_state=42) 
    train_index, val_index = next(sss.split(train_data_full.drop(columns=['label']), train_data_full['label']))
    train_data = train_data_full.iloc[train_index]
    val_data = train_data_full.iloc[val_index]
    remaining_data = data.drop(train_data_full.index)

    # 將特徵欄位加入欄名（selected_columns 去除 Label）
    feature_names = selected_columns_dnn.copy()
    feature_names.remove('Label')

    print("Feature names")
    print(feature_names)

    # 重新建立有欄名的 DataFrame，並補上 label
    train_data_named = pd.DataFrame(train_data.drop(columns=['label']).values, columns=feature_names)
    train_data_named['label'] = train_data['label'].values

    val_data_named = pd.DataFrame(val_data.drop(columns=['label']).values, columns=feature_names)
    val_data_named['label'] = val_data['label'].values

    test_data_named = pd.DataFrame(remaining_data.drop(columns=['label']).values, columns=feature_names)
    test_data_named['label'] = remaining_data['label'].values



    X_train, y_train = train_data.drop(columns=['label']).values, train_data['label'].values
    X_val, y_val = val_data.drop(columns=['label']).values, val_data['label'].values
    X_test, y_test = remaining_data.drop(columns=['label']).values, remaining_data['label'].values

    print("train data")
    print(X_train)

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

    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping

    # 假設 X_train, y_train, X_val, y_val, X_test, y_test 已經切分好
    # 這裡示範一個簡單模型
    # model_dnn = keras.Sequential([
    #     keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.Dense(len(label_mapping), activation='softmax')
    # ])
    # model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # # 訓練模型，包含驗證集
    # model_dnn.fit(X_train, y_train,
    #         validation_data=(X_val, y_val),
    #         epochs=10000, batch_size=32, verbose=1, callbacks=[early_stop])

    # 建立 DNN 模型 (包含 BatchNormalization 與 Dropout)
    model_dnn = keras.Sequential([
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
    model_dnn.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 設定 EarlyStopping 以在驗證集不再改善時提前停止
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 如有需要也可以加入 ModelCheckpoint 保存最佳模型權重：
    # checkpoint = ModelCheckpoint('best_dnn_model.h5', monitor='val_loss', save_best_only=True)

    # 訓練模型 (設置較大 epoch 數並依 EarlyStopping 停止)
    history = model_dnn.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10000,
                        batch_size=32,
                        verbose=1,
                        callbacks=[early_stop])

    # 記錄結束時間
    end_time = time.time()



    training_time = end_time - start_time
    print(f"訓練時間：{training_time:.2f} 秒")
    mde_report_per_fold = {}

    # 進行預測
    y_test_pred_numeric = model_dnn.predict(X_test)
    y_pred_classes = np.argmax(y_test_pred_numeric, axis=1)
    y_test_pred_labels = [label_mapping[str(num + 1)] for num in y_pred_classes]  # 補回 +1


    # 計算 MDE
    y_test_pred_coordinates = np.array([label_to_coordinates[label] for label in y_test_pred_labels])
    y_test_coordinates = np.array([label_to_coordinates[label_mapping[str(label + 1)]] for label in y_test])


    distances = np.linalg.norm(y_test_pred_coordinates - y_test_coordinates, axis=1)

    all_errors.extend(distances)   # ★★★★★ 新增這行，累積10次所有error

    avg_mde = np.mean(distances)
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
                    

    # 儲存到 JSON 檔案
    file_path = f"Testing_mde_using_loss_Bestcomb_{loop}"
    with open(file_path, "w") as f:
        json.dump(mde_report_avg, f, indent=4)

    print(f"MDE report saved to: {file_path}")


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

    # 儲存到 JSON 檔案
    file_path = f"Testing_mde_detailed_using_loss_Bestcomb_{loop}.json"
    with open(file_path, "w") as f:
        json.dump(mde_report_avg, f, indent=4)

    print(f"MDE report saved to: {file_path}")


    ## accuracy
        # **計算 Accuracy**
    _, test_accuracy = model_dnn.evaluate(X_test, y_test, verbose=0)
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


    # 儲存到 JSON 檔案
    file_path = f"Testing_accuracy_using_loss_Bestcomb_{loop}"
    with open(file_path, "w") as f:
        json.dump(accuracy_report_avg, f, indent=4)

    print(f"Accuracy report saved to: {file_path}")

    # save DNN model
    model_dnn.save(f'regressor_dnn_AP1&AP4_{loop}.h5')



print([round(float(mde), 4) for mde in all_mde])
print("平均 MDE:", round(sum(map(float, all_mde)) / len(all_mde), 4))
print([round(float(acc), 4) for acc in all_accuracy])
print("平均 Accuracy:", round(sum(map(float, all_accuracy)) / len(all_accuracy), 4))



import matplotlib.pyplot as plt
import numpy as np

errors = np.array(all_errors)
errors = errors[~np.isnan(errors)]  # 若有nan則去除

# 計算CDF
sorted_errors = np.sort(errors)

np.savetxt('RegDNN all_mde_errors_20runs_1mcAP AP1.txt', errors, fmt='%.6f')

cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)

plt.figure(figsize=(8, 6))
plt.plot(sorted_errors, cdf, marker='.', linestyle='-')
plt.xlabel('Mean Distance Error (m)')
plt.ylabel('CDF')
plt.title('CDF of Prediction Errors (all 10 runs)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

