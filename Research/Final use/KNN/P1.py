import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Baseline', 'AP1', 'AP1,AP2', 'AP1,AP2,AP3']
mde_baseline = [0.2264, 0, 0, 0]
mde_ftm = [0, 0.6374, 0.1040, 0.0440]
mde_rssi_ftm = [0, 0.2982, 0.0693, 0.0279]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Shift adjustment
offsets = [-width/2, 0, width/2]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars with adjusted alignment for better centering
bars1 = ax.bar(x[0] + offsets[1], mde_baseline[0], width, label='RSSI only', color='#999999')  # Baseline
bars2 = ax.bar(x[1] + offsets[0], mde_ftm[1], width, label='FTM only', color='#70b8c4')
bars3 = ax.bar(x[1] + offsets[2], mde_rssi_ftm[1], width, label='RSSI + FTM', color='#c47b70')
bars4 = ax.bar(x[2] + offsets[0], mde_ftm[2], width, color='#70b8c4')
bars5 = ax.bar(x[2] + offsets[2], mde_rssi_ftm[2], width, color='#c47b70')
bars6 = ax.bar(x[3] + offsets[0], mde_ftm[3], width, color='#70b8c4')
bars7 = ax.bar(x[3] + offsets[2], mde_rssi_ftm[3], width, color='#c47b70')

# Labels and styling
ax.set_ylabel('MDE (m)', fontsize=15)
ax.set_title('', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(fontsize=15)

plt.tight_layout()
plt.show()


# Accuracy values for each configuration
accuracy_baseline = [91.61, 0, 0, 0]
accuracy_ftm = [0, 78.60, 96.00, 97.34]
accuracy_rssi_ftm = [0, 89.12, 97.53, 98.23]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars with adjusted alignment for centering
bars1 = ax.bar(x[0] + offsets[1], accuracy_baseline[0], width, label='RSSI only', color='#999999')
bars2 = ax.bar(x[1] + offsets[0], accuracy_ftm[1], width, label='FTM only', color='#70b8c4')
bars3 = ax.bar(x[1] + offsets[2], accuracy_rssi_ftm[1], width, label='RSSI + FTM', color='#c47b70')
bars4 = ax.bar(x[2] + offsets[0], accuracy_ftm[2], width, color='#70b8c4')
bars5 = ax.bar(x[2] + offsets[2], accuracy_rssi_ftm[2], width, color='#c47b70')
bars6 = ax.bar(x[3] + offsets[0], accuracy_ftm[3], width, color='#70b8c4')
bars7 = ax.bar(x[3] + offsets[2], accuracy_rssi_ftm[3], width, color='#c47b70')

# Labels and styling
ax.set_ylabel('Accuracy (%)', fontsize=15)
ax.set_title('', fontsize=15)
ax.set_ylim(70, 100)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(loc = "lower left",fontsize=15)

plt.tight_layout()
plt.show()