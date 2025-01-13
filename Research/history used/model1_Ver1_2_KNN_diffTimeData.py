import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 資料前處理 (一): 刪除每個 Label 的前後 n 筆資料
def remove_first_last_n(data, label_column, n=20):
    data = data.sort_values(by=label_column).reset_index(drop=True)
    processed_data = pd.DataFrame(columns=data.columns)
    for label, group in data.groupby(label_column):
        if len(group) > 2 * n:
            group = group.iloc[n:-n]
        else:
            group = pd.DataFrame()  # 若資料不足，刪除整個群組
        processed_data = pd.concat([processed_data, group], ignore_index=True)
    return processed_data

# 資料前處理 (二): 用 KNNImputer 填補缺失值
def KNN_inputer_fill_nan(data, n=5):
    imputer = KNNImputer(n_neighbors=n)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

# 訓練 KNN 模型
def train_knn_model(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_random_sample(knn, test_data, label_column, sample_fraction=0.1):
    """
    隨機選取測試資料的 10%，進行預測和評估。

    :param knn: 已訓練好的 KNN 模型
    :param test_data: 測試資料的 DataFrame
    :param label_column: 標籤列名稱
    :param sample_fraction: 隨機取樣比例（默認 10%）
    """
    # 隨機抽取測試資料的 10%
    sample_data = test_data.sample(frac=sample_fraction, random_state=42)

    # 分離特徵與標籤
    X_test = sample_data.drop(columns=[label_column])
    y_test = sample_data[label_column]

    # 預測
    y_pred = knn.predict(X_test)

    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (on 10% of test data): {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in np.unique(y_test)]))

    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on 10% of Test Data")
    plt.show()

# 使用測試資料進行預測與評估
def evaluate_model(knn, X_test, y_test,report_save_path):
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in np.unique(y_test)]))

    # 生成 classification report 並保存
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(report_save_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {report_save_path}")

    # 計算混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    num_classes = cm.shape[0]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test),
                cbar=False)

    # 手動修改對角線文字顏色為紅色
    for i in range(num_classes):
        plt.text(i + 0.5, i + 0.5, cm[i, i], 
                 color="red", ha="center", va="center", fontsize=12, fontweight="bold")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix with Highlighted Diagonal")
    plt.show()

# 主流程函數
def train_and_predict(train_path, test_path, selected_columns, label_column, n_neighbors=5, n_remove=20, report_save_path='figure.txt'):
    # 讀取資料
    train_data = pd.read_csv(train_path, usecols=selected_columns)
    test_data = pd.read_csv(test_path, usecols=selected_columns)
    
    # 資料清理
    train_data = remove_first_last_n(train_data, label_column=label_column, n=n_remove)
    test_data = remove_first_last_n(test_data, label_column=label_column, n=n_remove)
    train_data = KNN_inputer_fill_nan(train_data)
    test_data = KNN_inputer_fill_nan(test_data)
    
    # 分離特徵與標籤
    X_train = train_data.drop(columns=[label_column])
    y_train = train_data[label_column]
    X_test = test_data.drop(columns=[label_column])
    y_test = test_data[label_column]


    
    # 訓練資料分佈
    plot_class_distribution(train_data, label_column='Label', title="Training Data Distribution")

    # 測試資料分佈
    plot_class_distribution(test_data, label_column='Label', title="Testing Data Distribution")
    
    # 訓練模型
    knn = train_knn_model(X_train, y_train, n_neighbors=n_neighbors)
    
    # 評估模型
    evaluate_model(knn, X_test, y_test,report_save_path)


def plot_class_distribution(data, label_column, title):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=label_column, data=data, order=data[label_column].value_counts().index)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()




# 主程式執行
if __name__ == "__main__":
    train_path = 'timestamp_allignment_2024_12_14_rtt_logs.csv'
    test_path = 'timestamp_allignment_2024_12_21_rtt_logs.csv'
    
    selected_columns = ['Label', 'AP1_Distance (mm)', 'AP2_Distance (mm)', 'AP3_Distance (mm)', 'AP4_Distance (mm)', 
                        'AP1_StdDev (mm)', 'AP2_StdDev (mm)', 'AP3_StdDev (mm)', 'AP4_StdDev (mm)', 
                        ]
    
    # 'AP1_Rssi', 'AP2_Rssi', 'AP3_Rssi', 'AP4_Rssi'
    
    label_column = 'Label'


    
    # 訓練與預測
    train_and_predict(train_path, test_path, selected_columns, label_column, n_neighbors=5, n_remove=10,report_save_path = 'figure.txt')


# 使用主流程測試 用 evaluate_random_sample function 做
# if __name__ == "__main__":
#     train_path = 'timestamp_allignment_2024_11_29_rtt_logs.csv'
#     test_path = 'timestamp_allignment_2024_12_14_rtt_logs.csv'
    
#     selected_columns = ['Label', 'AP1_Distance (mm)', 'AP2_Distance (mm)', 'AP3_Distance (mm)', 'AP4_Distance (mm)', 
#                         'AP1_StdDev (mm)', 'AP2_StdDev (mm)', 'AP3_StdDev (mm)', 'AP4_StdDev (mm)', 
#                         'AP1_Rssi', 'AP2_Rssi', 'AP3_Rssi', 'AP4_Rssi']
    
#     label_column = 'Label'
    
#     # 讀取資料
#     train_data = pd.read_csv(train_path, usecols=selected_columns)
#     test_data = pd.read_csv(test_path, usecols=selected_columns)

#     # 資料前處理
#     train_data = remove_first_last_n(train_data, label_column=label_column, n=10)
#     test_data = remove_first_last_n(test_data, label_column=label_column, n=10)
#     train_data = KNN_inputer_fill_nan(train_data)
#     test_data = KNN_inputer_fill_nan(test_data)

#     # 分離訓練特徵與標籤
#     X_train = train_data.drop(columns=[label_column])
#     y_train = train_data[label_column]

#     # 訓練模型
#     knn = train_knn_model(X_train, y_train, n_neighbors=5)

#     # 隨機選取 10% 測試資料進行預測與評估
#     evaluate_random_sample(knn, test_data, label_column=label_column, sample_fraction=0.1)