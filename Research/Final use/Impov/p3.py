import matplotlib.pyplot as plt

data_sizes = ['0.25%', '2.5%', '10%']
regdnn_win_ratio = [13.3, 76.7, 76.7]

plt.figure(figsize=(6, 5))
bars = plt.bar(data_sizes, regdnn_win_ratio, color=['#4C72B0', '#DD8452', '#55A868'])
plt.ylim(0, 100)
plt.ylabel('RegDNN Better Ratio (%)', fontsize=12)
plt.xlabel('Fine-tune Data Size', fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.7)

# 標註百分比數值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=13)

plt.tight_layout()
plt.show()
