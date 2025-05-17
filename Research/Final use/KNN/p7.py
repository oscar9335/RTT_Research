import matplotlib.pyplot as plt

# Data
models = ['RSSI-only', 'FTM-only', 'RSSI + FTM']
mde_values = [0.2294, 0.0147, 0.0081]
colors = ['#999999', '#70b8c4', '#70c4a1']

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(models, mde_values, color=colors)

# Annotate values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005,
             f'{yval:.4f}', ha='center', va='bottom', fontsize=14)

# Style
plt.ylabel('MDE (m)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 0.25)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
