import matplotlib.pyplot as plt

# x軸
dates = ['Basemodel', '2024/12/21', '2024/12/27', '2025/01/03', '2025/01/10', '2025/02/07']

# 2 mcAP Best 數據
best_dnn = {
    'No fine-tune':   [0.0130, 0.8194, 1.1574, 1.0963, 0.9749, 1.3483],
    '0.25%':          [0.0130, 0.4499, 0.6236, 0.4716, 0.3746, 0.4744],
    '2.5%':           [0.0130, 0.0454, 0.0514, 0.0351, 0.0570, 0.0295],
    '10%':            [0.0130, 0.0301, 0.0290, 0.0222, 0.0294, 0.0178]
}
best_regdnn = {
    'No fine-tune':   [0.0107, 0.7820, 1.1562, 1.0427, 0.9790, 1.3313],
    '0.25%':          [0.0107, 0.5521, 0.7863, 0.5720, 0.4832, 0.7080],
    '2.5%':           [0.0107, 0.0388, 0.0569, 0.0437, 0.0386, 0.0323],
    '10%':            [0.0107, 0.0206, 0.0248, 0.0201, 0.0215, 0.0186]
}

# 2 mcAP Worst 數據
worst_dnn = {
    'No fine-tune':   [0.0217, 1.2751, 1.3779, 1.2660, 1.5051, 1.3345],
    '0.25%':          [0.0217, 0.7309, 0.7259, 0.5904, 0.6536, 0.5801],
    '2.5%':           [0.0217, 0.0717, 0.0538, 0.0499, 0.0984, 0.0511],
    '10%':            [0.0217, 0.0382, 0.0342, 0.0317, 0.0452, 0.0277]
}
worst_regdnn = {
    'No fine-tune':   [0.0160, 1.2670, 1.3780, 1.3332, 1.4490, 1.3330],
    '0.25%':          [0.0160, 0.9333, 0.8691, 0.7400, 0.6966, 0.5699],
    '2.5%':           [0.0160, 0.0759, 0.0579, 0.0531, 0.0564, 0.0372],
    '10%':            [0.0160, 0.0277, 0.0266, 0.0297, 0.0346, 0.0258]
}

def plot_ft_result(ax, dnn, regdnn, title):
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    keys = ['No fine-tune', '0.25%', '2.5%', '10%']
    for i, k in enumerate(keys):
        ax.plot(dates, dnn[k], label=f'DNN {k}', color=colors[i], linestyle='-')
        ax.plot(dates, regdnn[k], label=f'RegDNN {k}', color=colors[i], linestyle='--')
    ax.set_ylabel('MDE (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plot_ft_result(plt.gca(), best_dnn, best_regdnn, 'Fine-tuning (MDE) - 2 mcAP Best (AP1, AP4)')

plt.subplot(2, 1, 2)
plot_ft_result(plt.gca(), worst_dnn, worst_regdnn, 'Fine-tuning (MDE) - 2 mcAP Worst (AP2, AP3)')

plt.tight_layout()
plt.show()
