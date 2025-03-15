
import matplotlib.pyplot as plt
import numpy as np

# New data for accuracy (excluding week 0)
weeks_acc = np.array([1, 2, 3, 4, 8])
acc_worst_finetune = np.array([0.9530, 0.9550, 0.9569, 0.9382, 0.9637])
acc_best_finetune = np.array([0.9584, 0.9662, 0.9697, 0.9536, 0.9751])

# Plot settings
plt.figure(figsize=(8, 6))
plt.plot(weeks_acc, acc_best_finetune, marker='o', linestyle='-', label="AP1 & AP3 (Best)", color='blue')
plt.plot(weeks_acc, acc_worst_finetune, marker='s', linestyle='--', label="AP2 & AP4 (Worst)", color='red')

# Labels and title
plt.xlabel("Time Pass (Weeks)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Fine-Tuned Best and Worst combination of 2mcAP Accuracy over time", fontsize=14)
plt.legend(fontsize=12)

# Grid and show
plt.grid(True)
plt.xticks(weeks_acc, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

