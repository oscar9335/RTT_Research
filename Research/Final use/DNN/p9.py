import matplotlib.pyplot as plt

# 資料設定
weeks = [1, 2, 3, 4, 8]

loss_data = {
    "0.25%": [2.7556, 2.7417, 2.0890, 2.6189, 2.6350 ],
    "1.25%": [0.7406, 0.7253, 0.6516, 0.6302, 0.5031],
    "2.5%": [0.5931, 0.5609, 0.5128, 0.5143, 0.4067 ],
    "5%":[0.4738, 0.4575, 0.4146, 0.4650, 0.3418 ],
    "10%": [0.4021, 0.3948, 0.3843, 0.4031, 0.3163]
}

accuracy_data = {
    "0.25%": [0.4364, 0.3689, 0.4637, 0.3519, 0.3186],
    "1.25%": [0.7908, 0.7974, 0.8075, 0.8251, 0.8466 ],
    "2.5%": [0.8323, 0.8444, 0.8508, 0.8540, 0.8851 ],
    "5%": [0.8668, 0.8708, 0.8777, 0.8665, 0.9050 ],
    "10%": [0.8851, 0.8883, 0.8840, 0.8863, 0.9137]
}

mde_data = {
    "0.25%": [1.3818, 1.5815, 1.2262, 1.4643, 1.3996 ],
    "1.25%":[0.5679, 0.4919, 0.4165, 0.4063, 0.3059 ],
    "2.5%": [0.4704, 0.4065, 0.3414, 0.3437, 0.2264 ],
    "5%":[0.3591, 0.3395, 0.2730, 0.3071, 0.1796 ],
    "10%": [0.3182, 0.2946, 0.2664, 0.2620, 0.1652]
}

# 指定畫圖的 marker、顏色、順序
markers = ['o', 's', '^', 'D', 'X']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
labels_order = ["1.25%", "2.5%", "5%", "10%", "0.25%"]

def plot_with_custom_style(data_dict, ylabel, title):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 主y軸 (正常資料)
    for idx, label in enumerate(labels_order[:-1]):  # 最後一個0.25%不畫在這邊
        data = data_dict[label]
        ax1.plot(weeks, data, marker=markers[idx], color=colors[idx], label=f'{label} data')
    
    ax1.set_xlabel('Week', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True)
    ax1.tick_params(axis='y')
    ax1.set_xticks(weeks)

    # 副y軸 (畫0.25%資料)
    idx = labels_order.index("0.25%")
    ax2 = ax1.twinx()
    ax2.plot(weeks, data_dict["0.25%"], marker=markers[idx], color=colors[idx],
             linestyle='--', label='0.25% data')
    ax2.set_ylabel(f'{ylabel} (0.25%)', fontsize=14)
    ax2.tick_params(axis='y', colors=colors[idx])

    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 依序畫出 Loss, Accuracy, MDE 三張圖
plot_with_custom_style(loss_data, 'Loss', '')
plot_with_custom_style(accuracy_data, 'Accuracy', '')
plot_with_custom_style(mde_data, 'MDE', '')
