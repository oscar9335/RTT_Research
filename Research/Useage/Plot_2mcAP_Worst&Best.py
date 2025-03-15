import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
weeks = np.array([0, 1, 2, 3, 4, 8])
mde_best = np.array([0.0101, 0.8750, 1.0663, 1.1828, 0.9304, 1.2968])
mde_worst = np.array([0.0173, 1.2326, 1.3465, 1.3552, 1.5949, 1.4073])

# Plot settings
plt.figure(figsize=(8, 6))
plt.plot(weeks, mde_best, marker='o', linestyle='-', label="AP1 & AP3 (Best)")
plt.plot(weeks, mde_worst, marker='s', linestyle='--', label="AP2 & AP4 (Worst)")

# Labels and title
plt.xlabel("Time Pass (Weeks)", fontsize=14)
plt.ylabel("MDE (m)", fontsize=14)
plt.title("Best and Worst combination of 2mcAP MDE decay over time", fontsize=14)
plt.legend(fontsize=12)

# Annotate 0 week values without overlapping
plt.text(0, mde_best[0] - 0.05, f"{mde_best[0]:.4f} m", fontsize=12, ha='center', color='blue')
plt.text(0, mde_worst[0] + 0.05, f"{mde_worst[0]:.4f} m", fontsize=12, ha='center', color='orange')

# Grid and show
plt.grid(True)
plt.xticks(weeks, fontsize=12)
plt.yticks(fontsize=12)
plt.show()




