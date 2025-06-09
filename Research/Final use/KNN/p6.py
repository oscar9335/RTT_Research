import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['baseline', '1 mcAP', '2 mcAP', '3 mcAP', '4 mcAP']
MDE = [0.2264, 0.0515, 0.0247, 0.0161, 0.0143]
Accuracy = [91.61, 97.61, 98.53, 98.89, 99.03]

# Colors
colors = ['#999999', '#70b8c4', '#70c4a1', '#c4b570', '#c47b70']

# Set font size
plt.rcParams.update({'font.size': 14})

# Plot MDE
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, MDE, color=colors)
plt.ylabel("MDE (m)")
# Add value labels on top
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002, 
            f'{height:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Plot Accuracy
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, Accuracy, color=colors)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(88, 100)

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, 
            f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
