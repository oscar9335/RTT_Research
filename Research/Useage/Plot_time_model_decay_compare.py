import matplotlib.pyplot as plt

# Data for the models
weeks = [0, 1, 2, 3, 4]

# Set 1
baseline_mde_set1 = [0.2399, 1.9656, 2.1987, 2.0213, 2.1360]
mc4_mde_set1 = [0.01, 0.5789, 0.7360, 0.7576, 0.7626]
mc1_best_mde_set1 = [0.0486, 1.2489, 1.5928, 1.5432, 1.2946]
mc2_best_mde_set1 = [0.0211, 0.7555, 1.0158, 1.0268, 0.9319]
mc3_best_mde_set1 = [0.0137, 0.6344, 0.8767, 0.9467, 0.8414]

# Set 2
baseline_mde_set2 = [0.2474, 1.9747, 2.2374, 2.1519, 2.2165]
mc4_mde_set2 = [0.0073, 0.7721, 0.9848, 0.9972, 1.0129]
mc1_best_mde_set2 = [0.0314, 1.1767, 1.6230, 1.4766, 1.3354]
mc2_best_mde_set2 = [0.0170, 0.7847, 1.1468, 1.1167, 0.9904]
mc3_best_mde_set2 = [0.0105, 0.8555, 1.0618, 1.0475, 1.0811]

# Define consistent colors for each model
colors = {
    "baseline": 'blue',
    "mc4": 'orange',
    "mc1_best": 'green',
    "mc2_best": 'red',
    "mc3_best": 'purple'
}

# First Plot: Set 1 highlighted, Set 2 semi-transparent
plt.figure(figsize=(10, 6))
# Set 1 (highlighted)
plt.plot(weeks, baseline_mde_set1, label="KNN Baseline Model", marker='o', color=colors["baseline"])
plt.plot(weeks, mc4_mde_set1, label="KNN 4 mcAP Model", marker='o', color=colors["mc4"])
plt.plot(weeks, mc1_best_mde_set1, label="KNN 1 mcAP (Best) Model", marker='o', color=colors["mc1_best"])
plt.plot(weeks, mc2_best_mde_set1, label="KNN 2 mcAP (Best) Model", marker='o', color=colors["mc2_best"])
plt.plot(weeks, mc3_best_mde_set1, label="KNN 3 mcAP (Best) Model", marker='o', color=colors["mc3_best"])

# Set 2 (semi-transparent)
plt.plot(weeks, baseline_mde_set2, label="DNN Baseline Model", marker='o', color=colors["baseline"], alpha=0.3)
plt.plot(weeks, mc4_mde_set2, label="DNN 4 mcAP Model", marker='o', color=colors["mc4"], alpha=0.3)
plt.plot(weeks, mc1_best_mde_set2, label="DNN 1 mcAP (Best) Model", marker='o', color=colors["mc1_best"], alpha=0.3)
plt.plot(weeks, mc2_best_mde_set2, label="DNN 2 mcAP (Best) Model", marker='o', color=colors["mc2_best"], alpha=0.3)
plt.plot(weeks, mc3_best_mde_set2, label="DNN 3 mcAP (Best) Model", marker='o', color=colors["mc3_best"], alpha=0.3)

plt.title("MDE Comparison Over Time (KNN result Highlighted)", fontsize=14)
plt.xlabel("Time (Weeks)", fontsize=12)
plt.ylabel("Mean Distance Error (MDE) (m)", fontsize=12)
plt.xticks(weeks)
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# Second Plot: Set 2 highlighted, Set 1 semi-transparent
plt.figure(figsize=(10, 6))
# Set 2 (highlighted)
plt.plot(weeks, baseline_mde_set2, label="DNN Baseline Model", marker='o', color=colors["baseline"])
plt.plot(weeks, mc4_mde_set2, label="DNN 4 mcAP Model", marker='o', color=colors["mc4"])
plt.plot(weeks, mc1_best_mde_set2, label="DNN 1 mcAP (Best) Model", marker='o', color=colors["mc1_best"])
plt.plot(weeks, mc2_best_mde_set2, label="DNN 2 mcAP (Best) Model", marker='o', color=colors["mc2_best"])
plt.plot(weeks, mc3_best_mde_set2, label="DNN 3 mcAP (Best) Model", marker='o', color=colors["mc3_best"])

# Set 1 (semi-transparent)
plt.plot(weeks, baseline_mde_set1, label="KNN Baseline Model", marker='o', color=colors["baseline"], alpha=0.3)
plt.plot(weeks, mc4_mde_set1, label="KNN 4 mcAP Model", marker='o', color=colors["mc4"], alpha=0.3)
plt.plot(weeks, mc1_best_mde_set1, label="KNN 1 mcAP (Best) Model", marker='o', color=colors["mc1_best"], alpha=0.3)
plt.plot(weeks, mc2_best_mde_set1, label="KNN 2 mcAP (Best) Model", marker='o', color=colors["mc2_best"], alpha=0.3)
plt.plot(weeks, mc3_best_mde_set1, label="KNN 3 mcAP (Best) Model", marker='o', color=colors["mc3_best"], alpha=0.3)

plt.title("MDE Comparison Over Time (KNN result Highlighted)", fontsize=14)
plt.xlabel("Time (Weeks)", fontsize=12)
plt.ylabel("Mean Distance Error (MDE) (m)", fontsize=12)
plt.xticks(weeks)
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
