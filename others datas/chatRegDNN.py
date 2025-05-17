import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. Load raw data and extract true coordinates
df = pd.read_csv("ESP32C3_processed_for_trainging.csv")
coords_df = df[["Label", "X Position (meters)", "Y Position (meters)"]]

# 2. Train regressor on AP1–AP4 RSSI→Distance
ap_data_list = []
for i in range(1, 5):
    subset = df[[f"AP{i}_Rssi", f"AP{i}_Distance (mm)"]].dropna()
    subset.columns = ["Rssi", "Distance"]
    ap_data_list.append(subset)
train_data_reg = pd.concat(ap_data_list, ignore_index=True)
X_train_reg = train_data_reg[["Rssi"]]
y_train_reg = train_data_reg["Distance"]

model_reg = SGDRegressor(
    tol=1e-4, penalty="l2", max_iter=5000,
    alpha=0.0001, eta0=0.1, learning_rate="optimal", loss="huber"
)
model_reg.fit(X_train_reg, y_train_reg)
print("Regressor trained.")

# 3. Prepare DNN features and impute AP5–AP8 distances
selected_columns = [
    "Label",
    "AP1_Distance (mm)", "AP2_Distance (mm)",
    "AP3_Distance (mm)", "AP4_Distance (mm)",
    "AP1_Rssi", "AP2_Rssi", "AP3_Rssi", "AP4_Rssi",
    "AP5_Rssi", "AP6_Rssi", "AP7_Rssi", "AP8_Rssi"
]
data_imputed = df[selected_columns].copy()

for i in range(5, 9):
    data_imputed[f"AP{i}_Distance_predicted"] = np.nan
    mask = data_imputed[f"AP{i}_Rssi"].notna()
    data_imputed.loc[mask, f"AP{i}_Distance_predicted"] = model_reg.predict(
        data_imputed.loc[mask, [f"AP{i}_Rssi"]].rename(columns={f"AP{i}_Rssi": "Rssi"})
    )

# 4. Standardize features
selected_columns_dnn = selected_columns + [
    "AP5_Distance_predicted", "AP6_Distance_predicted",
    "AP7_Distance_predicted", "AP8_Distance_predicted"
]
feature_names = selected_columns_dnn.copy()
feature_names.remove("Label")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_imputed[feature_names])

# 5. Build unified DataFrame with features, label, and true coords
df_all = pd.DataFrame(
    X_scaled,
    columns=feature_names,
    index=data_imputed.index
)
df_all["label"]   = data_imputed["Label"].astype(int)
df_all["Coord_X"] = coords_df.loc[data_imputed.index, "X Position (meters)"]
df_all["Coord_Y"] = coords_df.loc[data_imputed.index, "Y Position (meters)"]

if "Coord_X" in feature_names: feature_names.remove("Coord_X")
if "Coord_Y" in feature_names: feature_names.remove("Coord_Y")
if "label" in feature_names: feature_names.remove("label")
if "Label" in feature_names: feature_names.remove("Label")

print(feature_names)

# 6. Split 7:1:2 per label
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

# 7. Define and train DNN
model_dnn = keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(256, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(128, activation="relu"), BatchNormalization(), Dropout(0.3),
    Dense(len(np.unique(y_train)), activation="softmax")
])
model_dnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
start_time = time.time()
history = model_dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10000,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)
print(f"Training time: {time.time()-start_time:.2f} sec")

# 8. Evaluate accuracy
loss, test_acc = model_dnn.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 9. Compute MDE
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