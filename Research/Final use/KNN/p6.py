import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['baseline', '1 mcAP', '2 mcAP', '3 mcAP', '4 mcAP']
MDE = [0.2294, 0.061, 0.0227, 0.0112, 0.0081]
Accuracy = [91.46, 97.05, 98.52, 99.11, 99.34]

# Colors
colors = ['#999999', '#70b8c4', '#70c4a1', '#c4b570', '#c47b70']

# Set font size
plt.rcParams.update({'font.size': 14})

# Plot MDE
plt.figure(figsize=(8, 6))
plt.bar(labels, MDE, color=colors)
plt.ylabel("MDE (m)")
plt.tight_layout()
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.bar(labels, Accuracy, color=colors)
plt.ylabel("Accuracy (%)")
plt.ylim(88, 100)
plt.tight_layout()
plt.show()
