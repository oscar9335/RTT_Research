import matplotlib.pyplot as plt

# Data
training_data_per_rp = [0.25, 1.25, 2.5, 5, 10]

training_time = [34.65,5.98,7.30,9.52,12.42]
accuracy = [0.5100,0.6408,0.7211,0.8208,0.8502]
mde = [1.3131, 0.8998, 0.7257,0.4871,0.4092]


# Convert x-axis labels to percentage strings
x_labels = ["0.25%", "1.25%", "2.5%", "5%", "10%"]

# Re-plot accuracy and MDE with larger font sizes
fig, ax1 = plt.subplots(figsize=(8,6))

ax1.set_xlabel("Training Data per RP (% of Pretrained Model)", fontsize=14)
ax1.set_ylabel("Accuracy", color="tab:blue", fontsize=14)
ax1.plot(training_data_per_rp, accuracy, marker="o", color="tab:blue", label="Accuracy")
ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=12)
ax1.set_xticks(training_data_per_rp)
ax1.set_xticklabels(x_labels, fontsize=12)

# Annotate Accuracy values
for i, txt in enumerate(accuracy):
    ax1.annotate(f"{txt:.4f}", (training_data_per_rp[i], accuracy[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:blue", fontsize=12)

ax2 = ax1.twinx()
ax2.set_ylabel("MDE (m)", color="tab:red", fontsize=14)
ax2.plot(training_data_per_rp, mde, marker="s", linestyle="dashed", color="tab:red", label="MDE")
ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=12)

# Annotate MDE values
for i, txt in enumerate(mde):
    ax2.annotate(f"{txt:.4f}", (training_data_per_rp[i], mde[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:red", fontsize=12)

fig.tight_layout()
plt.title("Effect of Training Data per RP on Accuracy and MDE Baseline(no mc AP)", fontsize=16)
plt.savefig("accuracy_mde_plot.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()

# Re-plot training time with larger font sizes
plt.figure(figsize=(8,6))
plt.plot(training_data_per_rp, training_time, marker="^", color="tab:green", label="Training Time (s)")

# Annotate Training Time values
for i, txt in enumerate(training_time):
    plt.annotate(f"{txt:.2f}s", (training_data_per_rp[i], training_time[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:green", fontsize=12)

plt.xlabel("Training Data per RP (% of Pretrained Model)", fontsize=14)
plt.ylabel("Training Time (s)", fontsize=14)
plt.xticks(training_data_per_rp, x_labels, fontsize=12)
plt.yticks(fontsize=12)
plt.title("Training Time vs. Training Data per RP (%) Baseline(no mc AP)", fontsize=16)
plt.legend(fontsize=12)
plt.savefig("training_time_plot.png", dpi=300, bbox_inches='tight')  # Save plot
plt.show()
