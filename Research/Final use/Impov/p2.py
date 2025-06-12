import matplotlib.pyplot as plt

# x 軸對應的實際數值間隔與標籤
x_values = [0, 1, 2, 3, 4, 8]
x_labels = ['Basemodel', '2024/12/21', '2024/12/27', '2025/01/03', '2025/01/10', '2025/02/07']

best_dnn = {
    '0.25%': [0.0130, 0.4499, 0.6236, 0.4716, 0.3746, 0.4744],
    '2.5%':  [0.0130, 0.0454, 0.0514, 0.0351, 0.0570, 0.0295],
    '10%':   [0.0130, 0.0301, 0.0290, 0.0222, 0.0294, 0.0178]
}
best_regdnn = {
    '0.25%': [0.0107, 0.5521, 0.7863, 0.5720, 0.4832, 0.7080],
    '2.5%':  [0.0107, 0.0388, 0.0569, 0.0437, 0.0386, 0.0323],
    '10%':   [0.0107, 0.0206, 0.0248, 0.0201, 0.0215, 0.0186]
}

worst_dnn = {
    '0.25%': [0.0217, 0.7309, 0.7259, 0.5904, 0.6536, 0.5801],
    '2.5%':  [0.0217, 0.0717, 0.0538, 0.0499, 0.0984, 0.0511],
    '10%':   [0.0217, 0.0382, 0.0342, 0.0317, 0.0452, 0.0277]
}
worst_regdnn = {
    '0.25%': [0.0160, 0.9333, 0.8691, 0.7400, 0.6966, 0.5699],
    '2.5%':  [0.0160, 0.0759, 0.0579, 0.0531, 0.0564, 0.0372],
    '10%':   [0.0160, 0.0277, 0.0266, 0.0297, 0.0346, 0.0258]
}

colors = ['#4C72B0', '#DD8452', '#55A868']
markers = ['o', 's', 'D']
plt.rcParams.update({'font.size': 10})

def plot_finetune(ax, dnn, regdnn):
    keys = ['0.25%', '2.5%', '10%']
    for i, key in enumerate(keys):
        ax.plot(x_values, dnn[key], color=colors[i], marker=markers[i], linestyle='-', linewidth=2, markersize=5, label=f'DNN {key}')
        ax.plot(x_values, regdnn[key], color=colors[i], marker=markers[i], linestyle='--', linewidth=2, markersize=5, label=f'RegDNNLoc {key}')
    ax.set_ylabel('MDE (m)')
    ax.set_xlabel('Date')
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(frameon=False, fontsize=13, ncol=2, loc='upper right')

# Best Case
plt.figure(figsize=(10,6))
plot_finetune(plt.gca(), best_dnn, best_regdnn)
plt.tight_layout()
plt.show()

# Worst Case
plt.figure(figsize=(10,6))
plot_finetune(plt.gca(), worst_dnn, worst_regdnn)
plt.tight_layout()
plt.show()
