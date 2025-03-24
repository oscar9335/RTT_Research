import matplotlib.pyplot as plt

# Model names and corresponding MDE values
# model_names = ['Baseline', 'Single-AP (AP1 Only)', 'Dual-AP (AP1+AP2)', 'Triple-AP (AP1+AP2+AP3)']
model_names = ['Baseline', '4 Distance data']
# mde_values = [0.2399, 1.8845, 0.2788, 0.0272]
mde_values = [0.2399, 0.0139] 

# Set color scheme (optional)
colors = ['lightgray', 'powderblue', 'lightblue', 'skyblue']

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, mde_values, color=colors)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.4f}', ha='center', va='bottom')

# Titles and labels
plt.title('Model Accuracy Using Different AP Input Configurations', fontsize=14)
plt.xlabel('Model')
plt.ylabel('MDE (m)')
plt.ylim(0, max(mde_values) + 0.5)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
