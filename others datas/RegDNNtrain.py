import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import joblib
import json
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


all_mde = []
all_accuracy = []

best_mde = float('inf')  # 初始化最佳 MDE

df_reg = pd.read_csv("ESP32C3_processed_for_trainging.csv")

print(df_reg.head())

for loop in range(10):

### 1. 訓練 regressor 
    df_reg = pd.read_csv("ESP32C3_processed_for_trainging.csv")

    # 篩選有有效 distance 的資料
    ap1_data = df_reg[['AP1_Rssi', 'AP1_Distance (mm)']].dropna().rename(
        columns={'AP1_Rssi': 'Rssi', 'AP1_Distance (mm)': 'Distance'}
    )

    ap2_data = df_reg[['AP2_Rssi', 'AP2_Distance (mm)']].dropna().rename(
        columns={'AP2_Rssi': 'Rssi', 'AP2_Distance (mm)': 'Distance'}
    )

    ap3_data = df_reg[['AP3_Rssi', 'AP3_Distance (mm)']].dropna().rename(
        columns={'AP3_Rssi': 'Rssi', 'AP3_Distance (mm)': 'Distance'}
    )

    ap4_data = df_reg[['AP4_Rssi', 'AP4_Distance (mm)']].dropna().rename(
        columns={'AP4_Rssi': 'Rssi', 'AP4_Distance (mm)': 'Distance'}
    )

    train_data_reg = pd.concat([ap1_data,ap2_data,ap3_data,ap4_data], ignore_index=True)
    X_train_reg = train_data_reg[['Rssi']]
    y_train_reg = train_data_reg['Distance']
    
### 訓練 regressor
    # PassiveAggressiveRegressor 
    model_reg = SGDRegressor(tol=1e-4, penalty="l2", max_iter=5000,alpha= 0.0001, eta0= 0.1, learning_rate= 'optimal', loss= 'huber')
    model_reg.fit(X_train_reg, y_train_reg)
    print("Regressor trained. Coefficient:", model_reg.coef_, "Intercept:", model_reg.intercept_)



### 2. 讀取 DNN 需要的資料
    # DNN 使用的原始欄位
    selected_columns = ['Label',
                        'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)',
                        'AP1_Rssi', 'AP2_Rssi', 'AP3_Rssi', 'AP4_Rssi','AP5_Rssi', 'AP6_Rssi', 'AP7_Rssi', 'AP8_Rssi']

    file_path = "ESP32C3_processed_for_trainging.csv"
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
    data_imputed['AP5_Distance_predicted'] = np.nan
    data_imputed['AP6_Distance_predicted'] = np.nan
    data_imputed['AP7_Distance_predicted'] = np.nan
    data_imputed['AP8_Distance_predicted'] = np.nan

    # # 利用 AP1_Rssi 預測 AP1_Distance_predicted
    mask_ap5 = data_imputed['AP5_Rssi'].notna()
    data_imputed.loc[mask_ap5, 'AP5_Distance_predicted'] = model_reg.predict(
        data_imputed.loc[mask_ap5, ['AP5_Rssi']].rename(columns={'AP5_Rssi': 'Rssi'})
    )

    # 利用 AP2_Rssi 預測 AP2_Distance_predicted
    mask_ap6 = data_imputed['AP6_Rssi'].notna()
    data_imputed.loc[mask_ap6, 'AP6_Distance_predicted'] = model_reg.predict(
        data_imputed.loc[mask_ap6, ['AP6_Rssi']].rename(columns={'AP6_Rssi': 'Rssi'})
    )

    # 利用 AP3_Rssi 預測 AP3_Distance_predicted
    mask_ap7 = data_imputed['AP7_Rssi'].notna()
    data_imputed.loc[mask_ap7, 'AP7_Distance_predicted'] = model_reg.predict(
        data_imputed.loc[mask_ap7, ['AP7_Rssi']].rename(columns={'AP7_Rssi': 'Rssi'})
    )

    # # 利用 AP4_Rssi 預測 AP4_Distance_predicted
    mask_ap8 = data_imputed['AP8_Rssi'].notna()
    data_imputed.loc[mask_ap8, 'AP8_Distance_predicted'] = model_reg.predict(
        data_imputed.loc[mask_ap8, ['AP8_Rssi']].rename(columns={'AP8_Rssi': 'Rssi'})
    )

    # 更新 DNN 模型用的特徵欄位，將 regressor 預測值加入
    selected_columns_dnn = selected_columns + ['AP5_Distance_predicted', 'AP6_Distance_predicted' ,'AP7_Distance_predicted','AP8_Distance_predicted']

### 4. 後續 DNN 資料準備與訓練
    print("每個 RP 的資料筆數: " + str(len(data_imputed)/49))

    # 以下部分依照原有 DNN code 執行
    target_column = 'Label'
    y = data_imputed[target_column]

    
    label_to_coordinates = {
        # 按照我的 excel
    }



    print("DNN 最終的 reverse_label_mapping:", reverse_label_mapping)
    print("y_numeric 的 unique 值:", y_numeric.unique())

    # 取出作為模型輸入的特徵，這邊不含 Label 欄位
    X = data_imputed[selected_columns_dnn].drop(columns=['Label'])
    print("擴充後的特徵資料預覽：")
    print(X.head())

    print("*******************所有特徵欄位*******************：")
    print(list(X.columns))

    # 標準化 and Save
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
                    

    # # 儲存到 JSON 檔案
    # file_path = f"Testing_mde_using_loss_Bestcomb_{loop}"
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


    # # 儲存到 JSON 檔案
    # file_path = f"Testing_accuracy_using_loss_Bestcomb_{loop}"
    # with open(file_path, "w") as f:
    #     json.dump(accuracy_report_avg, f, indent=4)

    # print(f"Accuracy report saved to: {file_path}")

    
    joblib.dump(scaler, f'scaler_regressor_dnn_AP1&AP4_{loop}.pkl')
    joblib.dump(model_reg, f'regressor_model_AP1&AP4_{loop}.pkl')    
    model_dnn.save(f'regressor_dnn_AP1&AP4_{loop}.h5')




print([round(float(mde), 4) for mde in all_mde])
print([round(float(acc), 4) for acc in all_accuracy])


