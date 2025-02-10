import matplotlib.pyplot as plt

# Data for the models
weeks = [0, 1, 2, 3, 4]

# Set 1
baseline_mde = [0.2399, 1.9656, 2.1987, 2.0213, 2.1360]
mc4_mde = [0.01, 0.5789, 0.7360, 0.7576, 0.7626]
mc1_best_mde = [0.0486, 1.2489, 1.5928, 1.5432, 1.2946]
mc2_best_mde = [0.0211, 0.7555, 1.0158, 1.0268, 0.9319]
mc3_best_mde = [0.0137, 0.6344, 0.8767, 0.9467, 0.8414]

# Set 2
baseline_mde = [0.2474, 1.9747, 2.2374, 2.1519, 2.2165]
mc4_mde = [0.0073, 0.7721, 0.9848, 0.9972, 1.0129]
mc1_best_mde = [0.0314, 1.1767, 1.6230, 1.4766, 1.3354]
mc2_best_mde = [0.0170, 0.7847, 1.1468, 1.1167, 0.9904]
mc3_best_mde = [0.0105, 0.8555, 1.0618, 1.0475, 1.0811]

# Create a figure
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(weeks, baseline_mde, label="DNN Baseline Model", marker='o')
plt.plot(weeks, mc4_mde, label="DNN 4 mcAP Model", marker='o')
plt.plot(weeks, mc1_best_mde, label="DNN 1 mcAP (Best) Model", marker='o')
plt.plot(weeks, mc2_best_mde, label="DNN 2 mcAP (Best) Model", marker='o')
plt.plot(weeks, mc3_best_mde, label="DNN 3 mcAP (Best) Model", marker='o')

# Add labels, title, and legend
plt.title("MDE Comparison Over Time", fontsize=14)
plt.xlabel("Time (Weeks)", fontsize=12)
plt.ylabel("Mean Distance Error (MDE) (m)", fontsize=12)
plt.xticks(weeks)
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
