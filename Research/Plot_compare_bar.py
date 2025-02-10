import matplotlib.pyplot as plt
import numpy as np

# Re-plot the bar chart with English labels

# Model names
models = ['Baseline', '4 mcAP', '3 mcAP', '2 mcAP', '1 mcAP']

# KNN and DNN MDE data
knn_mde = [0.2399, 0.0100, 0.0137, 0.0211, 0.0486]
dnn_mde = [0.2474, 0.0073, 0.0105, 0.0170, 0.0314]

# Bar chart parameters
x = np.arange(len(models))  # Model indices
width = 0.35                # Bar width



# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, knn_mde, width, label='KNN', alpha=0.8, color='skyblue')
bar2 = ax.bar(x + width/2, dnn_mde, width, label='DNN', alpha=0.8, color='gold')

# Chart title and labels
ax.set_xlabel('Model Type')
ax.set_ylabel('MDE (m)')
ax.set_title('Comparison of MDE between DNN and KNN Models (0 Week Results)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Display values on top of the bars
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
