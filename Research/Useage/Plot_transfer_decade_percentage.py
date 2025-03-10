import matplotlib.pyplot as plt

# Data for plotting
weeks = [1, 2, 3, 4, 8]

accuracy_data = {
    "0.25%": [0.8523, 0.7100, 0.8495, 0.6877, 0.6110],
    "1.25%": [0.8830, 0.8426, 0.8622, 0.9148, 0.8622],
    "2.5%": [0.9452, 0.9440, 0.9371, 0.9579, 0.9404],
    "5%": [0.9754, 0.9759, 0.9731, 0.9716, 0.9744],
    "10%": [0.9744, 0.9845, 0.9818, 0.9729, 0.9739],
}

mde_data = {
    "0.25%": [0.2368, 0.4455, 0.2042, 0.4358, 0.5349],
    "1.25%": [0.1858, 0.2216, 0.1837, 0.1187, 0.1885],
    "2.5%": [0.0819, 0.0721, 0.0839, 0.0573, 0.0902],
    "5%": [0.0376, 0.0339, 0.0396, 0.0438, 0.0377],
    "10%": [0.0352, 0.0207, 0.0232, 0.0342, 0.0374],
}

# Plot Accuracy trends with larger font and save
fig, ax = plt.subplots(figsize=(8,6))
for label, values in accuracy_data.items():
    ax.plot(weeks, values, marker="o", label=label)
    # for i, txt in enumerate(values):
    #     ax.annotate(f"{txt:.4f}", (weeks[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=12)

ax.set_xlabel("Week", fontsize=14)
ax.set_xticks(weeks)
ax.set_xticklabels(weeks, fontsize=12)
ax.set_ylabel("Accuracy", fontsize=14)
y_min = min(min(accuracy_data.values())) * 0.9
y_max = 1.0
ax.set_ylim(y_min, y_max)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=12)
ax.set_title("Accuracy Decay Over Time (Fine-Tuning Each Week) for 3mcAP best combination", fontsize=16)
ax.legend(title="Training Data per RP", fontsize=12, loc='upper right')  # 圖例放在圖內右上角
ax.grid(True)
fig.savefig("accuracy_decay_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot MDE trends with y-axis starting from 0 and save
fig, ax = plt.subplots(figsize=(8,6))
for label, values in mde_data.items():
    ax.plot(weeks, values, marker="s", linestyle="dashed", label=label)
    # for i, txt in enumerate(values):
    #     ax.annotate(f"{txt:.4f}", (weeks[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=12)

ax.set_xlabel("Week", fontsize=14)
ax.set_xticks(weeks)
ax.set_xticklabels(weeks, fontsize=12)
ax.set_ylabel("MDE (m)", fontsize=14)
ax.set_ylim(0, max(max(mde_data.values())) * 1.1)  # 設定 y 軸從 0 開始，並稍微增加上界
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=12)
ax.set_title("MDE Increase Over Time (Fine-Tuning Each Week) for 3mcAP best combination", fontsize=16)
ax.legend(title="Training Data per RP", fontsize=12, loc='upper right')  # 圖例放在圖內右上角
ax.grid(True)
fig.savefig("mde_increase_plot.png", dpi=300, bbox_inches='tight')
plt.show()
