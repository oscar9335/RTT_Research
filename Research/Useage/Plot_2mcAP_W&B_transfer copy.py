
import matplotlib.pyplot as plt
import numpy as np

# New data for accuracy (excluding week 0)
weeks_acc = np.array([1, 2, 3, 4, 8])
regressor_DNN = np.array([0.0528, 0.0589, 0.0452, 0.0581, 0.0462])
original = np.array([0.0465, 0.0648, 0.0521, 0.0584, 0.0553])
optimal = np.array([0.0335, 0.0361, 0.0391, 0.0511, 0.0537])

# Plot settings
plt.figure(figsize=(8, 6))
plt.plot(weeks_acc, regressor_DNN, marker='o', linestyle='-', label="regressor_DNN", color='blue')
plt.plot(weeks_acc, original, marker='s', linestyle='--', label="original", color='red')
plt.plot(weeks_acc, optimal, marker='s', linestyle='--', label="4 mcAP optimal", color='orange')

# Labels and title
plt.xlabel("Time Pass (Weeks)", fontsize=14)
plt.ylabel("MDE", fontsize=14)
plt.title("MDE over time", fontsize=14)
plt.legend(fontsize=12)

# Grid and show
plt.grid(True)
plt.xticks(weeks_acc, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

