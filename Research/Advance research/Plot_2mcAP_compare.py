import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Original 1mcAP(worst)', 'Regressor_DNN 1mcAP(worst)', 'Original 1mcAP(best)', 'Regressor_DNN 1mcAP(best)', '4 mcAP (optimal)']
accuracies = [96.28,96.63,97.88,98.02,99.18]
mdes = [0.0887,0.0719 ,0.0411 ,0.0346 ,0.0117]

x = np.arange(len(models))

# Assign colors: same for original, same for regressor, unique for 4 mcAP
colors_acc = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Blue, Orange, Green

# Plot Accuracy Bar Chart
plt.figure(figsize=(8, 6))
bars = plt.bar(x, accuracies, color=colors_acc, width=0.5)
plt.title('Accuracy Comparison', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(x, models, fontsize=12, rotation=12)
plt.yticks(fontsize=12)
plt.ylim(96.0, 99.5)
plt.grid(axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}%', 
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Plot MDE Bar Chart
plt.figure(figsize=(8, 6))
bars = plt.bar(x, mdes, color=colors_acc, width=0.5)  # Use same color scheme
plt.title('MDE Comparison', fontsize=18)
plt.ylabel('Mean Distance Error (m)', fontsize=14)
plt.xticks(x, models, fontsize=12, rotation=12)
plt.yticks(fontsize=12)
plt.ylim(0.010, 0.095)
plt.grid(axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.0006, f'{height:.4f}', 
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
