import matplotlib.pyplot as plt

# 原始資料定義
weeks_full_labels = ["0", "1", "2", "3", "4", "8"]
weeks_partial_labels = ["1", "2", "3", "4", "8"]

# 使用數字表示實際 week 值，這樣 x 軸間距才符合比例
weeks_full_numeric = [0, 1, 2, 3, 4, 8]
weeks_partial_numeric = [1, 2, 3, 4, 8]

mde_data = {
    "baseline":         [0.2640, 0.4704, 0.4065, 0.3414, 0.3437, 0.2264],
    "1 mcAP best":      [0.0391, 0.1154, 0.1116, 0.1024, 0.0853, 0.0666],
    "2 mcAP best":      [0.0139, 0.0443, 0.0530, 0.0411, 0.0405, 0.0221],
    "3 mcAP best":      [0.0086, 0.0290, 0.0344, 0.0252, 0.0339, 0.0234],
    "4 mcAP":           [0.0075, 0.0242, 0.0254, 0.0257, 0.0305, 0.0206],
}

accuracy_data = {
    "baseline":         [0.9046, 0.8323, 0.8444, 0.8508, 0.8540, 0.8851],
    "1 mcAP best":      [0.9806, 0.9416, 0.9470, 0.9426, 0.9498, 0.9628],
    "2 mcAP best":      [0.9902, 0.9706, 0.9642, 0.9707, 0.9693, 0.9853],
    "3 mcAP best":      [0.9942, 0.9799, 0.9742, 0.9795, 0.9770, 0.9852],
    "4 mcAP":           [0.9947, 0.9837, 0.9802, 0.9797, 0.9800, 0.9855],
}

markers = ['o', 's', '^', 'D', 'X']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
plt.rcParams.update({'font.size': 14})

# 畫圖函數，依照是否排除 0 week 選取不同的 x 軸數字及 label
def plot_metrics(data, ylabel, numeric_weeks, label_weeks, exclude_zero=False):
    plt.figure(figsize=(8, 5))
    for idx, (label, values) in enumerate(data.items()):
        # 若 exclude_zero 為 True，則使用從 1 week 開始的資料
        plot_values = values[1:] if exclude_zero else values
        plt.plot(numeric_weeks, plot_values, label=label, marker=markers[idx], color=colors[idx])
    plt.xlabel("Week")
    plt.ylabel(ylabel)
    plt.xticks(numeric_weeks, label_weeks)  # 設定自訂 x 軸 tick
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 畫圖：含 0 week 的數據
plot_metrics(mde_data, "MDE (m)", weeks_full_numeric, weeks_full_labels, exclude_zero=False)
plot_metrics(accuracy_data, "Accuracy", weeks_full_numeric, weeks_full_labels, exclude_zero=False)

# 畫圖：不含 0 week 的數據
plot_metrics(mde_data, "MDE (m)", weeks_partial_numeric, weeks_partial_labels, exclude_zero=True)
plot_metrics(accuracy_data, "Accuracy", weeks_partial_numeric, weeks_partial_labels, exclude_zero=True)
