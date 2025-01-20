import matplotlib.pyplot as plt

# Data for the models
weeks = [0, 1, 2, 3, 4]

baseline_mde = [0.2399, 1.9656, 2.1987, 2.0213, 2.1360]
mc4_mde = [0.01, 0.5789, 0.7360, 0.7576, 0.7626]
mc1_best_mde = [0.0486, 1.2489, 1.5928, 1.5432, 1.2946]
mc2_best_mde = [0.0211, 0.7555, 1.0158, 1.0268, 0.9319]
mc3_best_mde = [0.0137, 0.6344, 0.8767, 0.9467, 0.8414]

# Create a figure
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(weeks, baseline_mde, label="Baseline Model", marker='o')
plt.plot(weeks, mc4_mde, label="4 mcAP Model", marker='o')
plt.plot(weeks, mc1_best_mde, label="1 mcAP (Best) Model", marker='o')
plt.plot(weeks, mc2_best_mde, label="2 mcAP (Best) Model", marker='o')
plt.plot(weeks, mc3_best_mde, label="3 mcAP (Best) Model", marker='o')

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
