import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['1 mcAP (best)', '2 mcAP (best)', '3 mcAP (best)', '4 mcAP']
mde_knn = [0.0465, 0.0178, 0.0090, 0.0081]
mde_dnn = [0.0391, 0.0139, 0.0086, 0.0075]

x = np.arange(len(models))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, mde_knn, width, label='KNN', color='gray')
bars2 = ax.bar(x + width/2, mde_dnn, width, label='DNN', color='skyblue')

# Labels
ax.set_ylabel('MDE (m)', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(fontsize=15)

plt.tight_layout()
plt.show()
