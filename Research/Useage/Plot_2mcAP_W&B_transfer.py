
import matplotlib.pyplot as plt
import numpy as np

# New data for accuracy (excluding week 0)
weeks_acc = np.array([1, 2, 3, 4, 8])
regressor_DNN = np.array([0.9650, 0.9684, 0.9717, 0.9573, 0.9726])
original = np.array([0.9642, 0.9615, 0.9660, 0.9605, 0.9709])
optimal = np.array([0.9737, 0.9738, 0.9779, 0.9646, 0.9694])

# Plot settings
plt.figure(figsize=(8, 6))
plt.plot(weeks_acc, regressor_DNN, marker='o', linestyle='-', label="regressor_DNN", color='blue')
plt.plot(weeks_acc, original, marker='s', linestyle='--', label="original", color='red')
plt.plot(weeks_acc, optimal, marker='s', linestyle='--', label="4 mcAP optimal", color='orange')

# Labels and title
plt.xlabel("Time Pass (Weeks)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy over time", fontsize=14)
plt.legend(fontsize=12)

# Grid and show
plt.grid(True)
plt.xticks(weeks_acc, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

