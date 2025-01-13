import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import json



# Step 2: 分割資料集
def split_data(data, target_column, test_size=0.2, val_size=0.1):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 先分割出測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 再從訓練集中分割出驗證集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

# Step 3: 訓練KNN分類器並評估
def train_knn_classifier(X_train, y_train, X_val, y_val, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 驗證模型
    y_val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return knn

# Step 3: 修改為支持 K-fold 的 KNN 評估並計算平均混淆矩陣
def k_fold_knn_evaluation_with_avg_cm(data, target_column, k=5, n_neighbors=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    overall_accuracy = []
    fold_reports = []
    cumulative_cm = None  # 用於累積混淆矩陣
    total_samples = 0     # 用於記錄累積的樣本數
    
    # 把label部分拿掉
    X = data.drop(columns=[target_column])
    y = data[target_column]

    #開始k fold
    fold_index = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 建立 KNN 模型
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        
        # 預測
        y_test_pred = knn.predict(X_test)
        
        # 計算整體準確率
        accuracy = accuracy_score(y_test, y_test_pred)
        overall_accuracy.append(accuracy)
        
        # 計算分類報告
        report = classification_report(
            y_test, y_test_pred,
            target_names=[str(label) for label in np.unique(y)],
            zero_division=0,
            output_dict=True  # 以字典形式輸出，方便進一步分析
        )
        fold_reports.append(report)
        
        # # 計算混淆矩陣
        # cm = confusion_matrix(y_test, y_test_pred)
        # if cumulative_cm is None:
        #     cumulative_cm = cm * len(y_test)  # 累積時乘以該 fold 的樣本數
        # else:
        #     cumulative_cm += cm * len(y_test)
        
        # total_samples += len(y_test)  # 累積樣本數'

        # 計算混淆矩陣
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(20, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(str(fold_index) + "CM.png")

        print(f"Fold {fold_index} - Accuracy: {accuracy:.4f}")
        fold_index += 1
        # # 平均混淆矩陣
        # avg_cm = cumulative_cm / total_samples 


        # # 繪製平均混淆矩陣
        # plt.figure(figsize=(20, 15))  # 增大圖形尺寸
        # sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues", 
        #             xticklabels=np.unique(y), yticklabels=np.unique(y))
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title(f"Average Confusion Matrix across {k} folds")
        # plt.xticks(rotation=45)  # 適當旋轉 X 軸標籤
        # plt.yticks(rotation=0)   # Y 軸標籤保持水平
        # plt.show()

    # 平均整體準確率
    avg_accuracy = np.mean(overall_accuracy)
    print(f"\nAverage Accuracy across {k} folds: {avg_accuracy:.4f}")

    # 計算每個類別的平均性能
    avg_report = {}
    for label in np.unique(y):
        label = str(label)
        avg_report[label] = {
            "precision": np.mean([report[label]["precision"] for report in fold_reports if label in report]),
            "recall": np.mean([report[label]["recall"] for report in fold_reports if label in report]),
            "f1-score": np.mean([report[label]["f1-score"] for report in fold_reports if label in report])
        }
    
    with open('knn_report.txt', "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {'knn_report.txt'}")
    
    print("\nAverage Classification Report per Label:")
    for label, metrics in avg_report.items():
        print(f"Label {label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}")

    return avg_accuracy, avg_report

# 修改 evaluate_model 函數以繪製混淆矩陣
def evaluate_model(knn, X_test, y_test):
    y_test_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Overall Test Accuracy: {test_accuracy:.4f}")

    # 計算每個類別的準確性
    report = classification_report(y_test, y_test_pred, target_names=[str(label) for label in np.unique(y_test)], zero_division=0)
    print("Detailed Classification Report:")
    print(report)

    # 生成 classification report 並保存
    report = classification_report(y_test, y_test_pred, output_dict=True)
    with open('knn_report.txt', "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {'knn_report.txt'}")
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




# 資料前處理 (一): 刪除前後n筆資料
def remove_first_last_n(data, label_column, n=20):
    # 確保依據Label排序
    data = data.sort_values(by=label_column).reset_index(drop=True)
    
    # 建立一個空的 DataFrame 用於存放處理後的資料
    processed_data = pd.DataFrame(columns=data.columns)
    
    # 針對每個Label群組進行處理
    for label, group in data.groupby(label_column):
        # 刪除前n筆和後n筆資料
        if len(group) > 2 * n:  # 確保群組資料足夠
            group = group.iloc[n:-n]
        else:
            group = pd.DataFrame()  # 若資料不足，刪除整個群組
        # 將處理後的群組資料加入
        processed_data = pd.concat([processed_data, group], ignore_index=True)
    
    return processed_data

# 資料前處理 (二-1): 用KNNImputer填補缺失值
def KNN_inputer_fill_nan(data,n=5):
    imputer = KNNImputer(n_neighbors=n)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed


# main function here

path = '2024_11_29\\timestamp_allignment_2024_11_29_rtt_logs.csv'

selected_columns = ['Label', 'AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi']
# selected_columns = ['Label', 'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)'] 
# selected_columns = ['Label', 'AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi','AP1_StdDev (mm)','AP2_StdDev (mm)','AP3_StdDev (mm)','AP4_StdDev (mm)'] 
# selected_columns = ['Label', 'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)','AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi']
# selected_columns = ['Label', 'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)','AP1_StdDev (mm)','AP2_StdDev (mm)','AP3_StdDev (mm)','AP4_StdDev (mm)']   
# selected_columns = ['Label', 'AP1_Distance (mm)','AP2_Distance (mm)','AP3_Distance (mm)','AP4_Distance (mm)','AP1_StdDev (mm)','AP2_StdDev (mm)','AP3_StdDev (mm)','AP4_StdDev (mm)','AP1_Rssi','AP2_Rssi','AP3_Rssi','AP4_Rssi']  

data = pd.read_csv(path, usecols=selected_columns)

# 使用函數刪除前10筆和後10筆
removeF_T_10_data = remove_first_last_n(data, label_column='Label', n=10)

target_column = 'Label'  # 替換成目標欄位名稱

# 法一   最好
clear_data = KNN_inputer_fill_nan(removeF_T_10_data)


# 執行 K-fold 評估並顯示平均混淆矩陣
k = 5  # 設置為 5 折交叉驗證
n_neighbors = 5  # 設定 KNN 的鄰居數量
average_accuracy, average_report = k_fold_knn_evaluation_with_avg_cm(clear_data, target_column='Label', k=k, n_neighbors=n_neighbors)