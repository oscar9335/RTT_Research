import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. 載入原始資料與真實座標
df = pd.read_csv("ESP32C3_processed_for_trainging.csv")
coords_df = df[["Label", "X Position (meters)", "Y Position (meters)"]]

# 2. 選擇要用於 DNN 的特徵（不含任何 regressor 預測欄位），並丟掉有缺值的列
selected_columns = [
    "Label",
    # "AP1_Distance (mm)", "AP2_Distance (mm)",
    # "AP3_Distance (mm)", "AP4_Distance (mm)",
    "AP1_Rssi", "AP2_Rssi", "AP3_Rssi", "AP4_Rssi",
    "AP5_Rssi", "AP6_Rssi", "AP7_Rssi", "AP8_Rssi"
]
data = df[selected_columns].dropna().copy()

# 3. 標準化特徵
feature_names = [c for c in selected_columns if c != "Label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[feature_names].values)
y = data["Label"].astype(int).values

# 4. 把特徵、標籤、真實座標組到同一張表裡，保留原始 index
df_all = pd.DataFrame(X_scaled, columns=feature_names, index=data.index)
df_all["label"]   = y
df_all["Coord_X"] = coords_df.loc[data.index, "X Position (meters)"]
df_all["Coord_Y"] = coords_df.loc[data.index, "Y Position (meters)"]

if "Coord_X" in feature_names: feature_names.remove("Coord_X")
if "Coord_Y" in feature_names: feature_names.remove("Coord_Y")
if "label" in feature_names: feature_names.remove("label")
if "Label" in feature_names: feature_names.remove("Label")

print(feature_names)

# 5. 針對每個 label 做 7:1:2 切分
train_parts, val_parts, test_parts = [], [], []
for lbl, grp in df_all.groupby("label"):
    tv, tst = train_test_split(grp, test_size=0.8,  random_state=42)
    tr, vl  = train_test_split(tv,  test_size=0.125, random_state=42)
    train_parts.append(tr)
    val_parts.append(vl)
    test_parts.append(tst)

train_df = pd.concat(train_parts)
val_df   = pd.concat(val_parts)
test_df  = pd.concat(test_parts)

X_train = train_df[feature_names].values
y_train = train_df["label"].values
X_val   = val_df[feature_names].values
y_val   = val_df["label"].values
X_test  = test_df[feature_names].values
y_test  = test_df["label"].values
coords_test = test_df[["Coord_X", "Coord_Y"]].values

# 6. 定義並訓練 DNN
num_classes = len(np.unique(y_train))
model_dnn = keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(256, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(num_classes, activation="softmax")
])
model_dnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
start_time = time.time()
model_dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10000, batch_size=32, verbose=1,
    callbacks=[early_stop]
)
print(f"Training time: {time.time() - start_time:.2f} seconds")

# 7. 評估整體 Accuracy
loss, test_acc = model_dnn.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 8. 計算 MDE（使用每個 label 的中心座標）
label_center = df_all.groupby("label")[["Coord_X","Coord_Y"]].first().to_dict("index")
y_pred_classes = np.argmax(model_dnn.predict(X_test), axis=1)
y_pred_coords = np.array([
    (label_center[l]["Coord_X"], label_center[l]["Coord_Y"])
    for l in y_pred_classes
])
distances = np.linalg.norm(y_pred_coords - coords_test, axis=1)
print(f"MDE: {distances.mean():.4f} m")

import pandas as pd

train_counts = train_df['label'].value_counts().sort_index()
val_counts   = val_df  ['label'].value_counts().sort_index()
test_counts  = test_df ['label'].value_counts().sort_index()

label_distribution = pd.DataFrame({
    'Train': train_counts,
    'Val':   val_counts,
    'Test':  test_counts
}).fillna(0).astype(int)

print("每個 RP 的資料分布 (train / val / test)：")
print(label_distribution)