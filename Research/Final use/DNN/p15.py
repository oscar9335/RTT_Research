import matplotlib.pyplot as plt

# 資料設定
weeks_all = [0, 1, 2, 3, 4, 8]

# MDE 數據
mde_data = {
    "4 mcAP": [0.0075, 0.0242, 0.0254, 0.0257, 0.0305, 0.0206],
    "3 mcAP": [0.0086, 0.0290, 0.0344, 0.0252, 0.0339, 0.0234],
    "2 mcAP": [0.0139, 0.0443, 0.0530, 0.0411, 0.0405, 0.0221],
    "1 mcAP": [0.0391, 0.1154, 0.1116, 0.1024, 0.0853, 0.0666],
    "0 mcAP": [0.2640, 0.4704, 0.4065, 0.3414, 0.3437, 0.2264]
}

# 畫圖設定
markers = ['o', 's', '^', 'D', 'X']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
labels_order = ["1 mcAP", "2 mcAP", "3 mcAP", "4 mcAP", "0 mcAP"]

# 繪圖函數
def plot_dual_axis_style(data_dict, ylabel, x_weeks):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 畫一般 mcAP 線 (非 0 mcAP)
    for idx, label in enumerate(labels_order[:-1]):
        data = data_dict[label]
        ax1.plot(x_weeks, data, marker=markers[idx], color=colors[idx],
                 label=f'{label}', linewidth=2)

    ax1.set_xlabel('Week', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True)
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_weeks)
    ax1.tick_params(labelsize=14)

    # 畫 0 mcAP 為虛線並使用不同 Y 軸
    idx = labels_order.index("0 mcAP")
    ax2 = ax1.twinx()
    ax2.plot(x_weeks, data_dict["0 mcAP"], marker=markers[idx], color=colors[idx],
             linestyle='--', label='0 mcAP', linewidth=2)
    ax2.set_ylabel(f'{ylabel} (0 mcAP)', fontsize=14)
    ax2.tick_params(axis='y', labelcolor=colors[idx], labelsize=14)

    # 圖例合併
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

# 執行繪圖
plot_dual_axis_style(mde_data, 'MDE (m)', weeks_all)
