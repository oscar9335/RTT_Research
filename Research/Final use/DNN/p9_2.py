import matplotlib.pyplot as plt

# 資料設定
weeks_all = [0, 1, 2, 3, 4, 8]
weeks_loss = [1, 2, 3, 4, 8]  # Loss 沒有 week 0

# 只有 1,2,3,4,8
loss_data = {
    "0.25%": [1.2717, 1.4304, 1.1099, 0.9868, 1.5670],
    "1.25%": [0.1290, 0.1513, 0.1462, 0.1218, 0.1038],
    "2.5%": [0.0764, 0.0780, 0.0822, 0.0820, 0.0767],
    "5%": [0.0606, 0.0549, 0.0547, 0.0643, 0.0826],
    "10%": [0.0550, 0.0389, 0.0445, 0.0425, 0.0381]
}

# 週數為 0,1,2,3,4,8
accuracy_data = {
    "0.25%": [0.9877,0.5583, 0.5035, 0.6321, 0.5156, 0.5975],
    "1.25%": [0.9877,0.9422, 0.9496, 0.9471, 0.9437, 0.9575],
    "2.5%": [0.9877,0.9692, 0.9624, 0.9676, 0.9581, 0.9679],
    "5%": [0.9877,0.9751, 0.9703, 0.9755, 0.9725, 0.9786],
    "10%": [0.9877,0.9792, 0.9809, 0.9801, 0.9757, 0.9824]
}

mde_data = {
    "0.25%": [0.0201, 0.8112, 0.8097, 0.6423, 0.7165, 0.5947],
    "1.25%": [0.0201,0.1359, 0.0894, 0.0792, 0.0983, 0.0649],
    "2.5%": [0.0201,0.0666, 0.0655, 0.0502, 0.0727, 0.0524],
    "5%": [0.0201,0.0533, 0.0525, 0.0399, 0.0474, 0.0377],
    "10%": [0.0201,0.0422, 0.0331, 0.0322, 0.0423, 0.0292]
}

# 指定畫圖的 marker、顏色、順序
markers = ['o', 's', '^', 'D', 'X']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
labels_order = ["1.25%", "2.5%", "5%", "10%", "0.25%"]

def plot_with_custom_style(data_dict, ylabel, title, x_weeks):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    for idx, label in enumerate(labels_order[:-1]):
        data = data_dict[label]
        ax1.plot(x_weeks, data, marker=markers[idx], color=colors[idx], label=f'{label} data')

    ax1.set_xlabel('Week', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True)
    ax1.tick_params(axis='y')
    ax1.set_xticks(x_weeks)

    idx = labels_order.index("0.25%")
    ax2 = ax1.twinx()
    ax2.plot(x_weeks, data_dict["0.25%"], marker=markers[idx], color=colors[idx],
             linestyle='--', label='0.25% data')
    ax2.set_ylabel(f'{ylabel} (0.25%)', fontsize=14)
    ax2.tick_params(axis='y', colors=colors[idx])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 繪圖
plot_with_custom_style(loss_data, 'Loss', '', weeks_loss)
plot_with_custom_style(accuracy_data, 'Accuracy', '', weeks_all)
plot_with_custom_style(mde_data, 'MDE', '', weeks_all)
