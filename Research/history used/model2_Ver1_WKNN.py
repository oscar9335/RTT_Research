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

# Step 3: 訓練WKNN分類器並評估
def train_wknn_classifier(X_train, y_train, X_val, y_val, k=3):
    wknn = KNeighborsClassifier(n_neighbors=k, weights='distance')  # 使用距離作為權重
    wknn.fit(X_train, y_train)

    # 驗證模型
    y_val_pred = wknn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return wknn

# Step 3: K-fold WKNN 評估並計算平均混淆矩陣
def k_fold_wknn_evaluation_with_avg_cm(data, target_column, k=5, n_neighbors=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    overall_accuracy = []
    fold_reports = []
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    fold_index = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 建立 WKNN 模型
        wknn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        wknn.fit(X_train, y_train)
        
        # 預測
        y_test_pred = wknn.predict(X_test)
        
        # 計算整體準確率
        accuracy = accuracy_score(y_test, y_test_pred)
        overall_accuracy.append(accuracy)
        
        # 計算分類報告
        report = classification_report(
            y_test, y_test_pred,
            target_names=[str(label) for label in np.unique(y)],
            zero_division=0,
            output_dict=True
        )
        fold_reports.append(report)
        
        # 計算混淆矩陣
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(20, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - Fold {fold_index}")
        plt.savefig(f"Fold_{fold_index}_CM_WKNN.png")

        print(f"Fold {fold_index} - Accuracy: {accuracy:.4f}")
        fold_index += 1

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
    
    with open('wknn_report.txt', "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {'wknn_report.txt'}")

    print("\nAverage Classification Report per Label:")
    for label, metrics in avg_report.items():
        print(f"Label {label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}")

    return avg_accuracy, avg_report

# 資料前處理 (一): 刪除前後n筆資料
def remove_first_last_n(data, label_column, n=20):
    data = data.sort_values(by=label_column).reset_index(drop=True)
    processed_data = pd.DataFrame(columns=data.columns)
    for label, group in data.groupby(label_column):
        if len(group) > 2 * n:
            group = group.iloc[n:-n]
        else:
            group = pd.DataFrame()
        processed_data = pd.concat([processed_data, group], ignore_index=True)
    return processed_data

# 資料前處理 (二-1): 用KNNImputer填補缺失值
def KNN_inputer_fill_nan(data, n=5):
    imputer = KNNImputer(n_neighbors=n)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

# Main function
path = 'timestamp_allignment_2024_12_21_rtt_logs.csv'

selected_columns = ['Label', 'AP1_Distance (mm)', 'AP2_Distance (mm)', 'AP3_Distance (mm)', 'AP4_Distance (mm)',
                    'AP1_StdDev (mm)', 'AP2_StdDev (mm)', 'AP3_StdDev (mm)', 'AP4_StdDev (mm)',
                    'AP1_Rssi', 'AP2_Rssi', 'AP3_Rssi', 'AP4_Rssi']

data = pd.read_csv(path, usecols=selected_columns)

# 移除前後資料
removeF_T_10_data = remove_first_last_n(data, label_column='Label', n=10)

# 填補缺失值
clear_data = KNN_inputer_fill_nan(removeF_T_10_data)

# 執行 K-fold 評估並顯示平均混淆矩陣
k = 5  # 設置為 5 折交叉驗證
n_neighbors = 5  # 設定 WKNN 的鄰居數量
average_accuracy, average_report = k_fold_wknn_evaluation_with_avg_cm(clear_data, target_column='Label', k=k, n_neighbors=n_neighbors)
