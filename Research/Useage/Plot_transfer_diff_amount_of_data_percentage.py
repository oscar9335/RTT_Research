import matplotlib.pyplot as plt

# Data
training_data_per_rp = [0.25, 1.25, 2.5, 5, 10]
accuracy = [0.8690, 0.8877, 0.9460, 0.9690, 0.9806]
mde = [0.2046, 0.1620, 0.0709, 0.0455 , 0.0270]
training_time = [13.13, 4.52, 8.20, 7.80, 9.88]

# Convert x-axis labels to percentage strings
x_labels = ["0.25%", "1.25%", "2.5%", "5%", "10%"]

# Re-plot accuracy and MDE with annotations
fig, ax1 = plt.subplots()

ax1.set_xlabel("Training Data per RP (% of Pretrained Model)")
ax1.set_ylabel("Accuracy", color="tab:blue")
ax1.plot(training_data_per_rp, accuracy, marker="o", color="tab:blue", label="Accuracy")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(training_data_per_rp)
ax1.set_xticklabels(x_labels)

# Annotate Accuracy values
for i, txt in enumerate(accuracy):
    ax1.annotate(f"{txt:.4f}", (training_data_per_rp[i], accuracy[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("MDE (m)", color="tab:red")
ax2.plot(training_data_per_rp, mde, marker="s", linestyle="dashed", color="tab:red", label="MDE")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Annotate MDE values
for i, txt in enumerate(mde):
    ax2.annotate(f"{txt:.4f}", (training_data_per_rp[i], mde[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:red")

fig.tight_layout()
plt.title("Effect of Training Data per RP on Accuracy and MDE")
plt.show()

# Re-plot training time with annotations
plt.figure()
plt.plot(training_data_per_rp, training_time, marker="^", color="tab:green", label="Training Time (s)")

# Annotate Training Time values
for i, txt in enumerate(training_time):
    plt.annotate(f"{txt:.2f}s", (training_data_per_rp[i], training_time[i]), textcoords="offset points", xytext=(0,5), ha='center', color="tab:green")

plt.xlabel("Training Data per RP (% of Pretrained Model)")
plt.ylabel("Training Time (s)")
plt.xticks(training_data_per_rp, x_labels)
plt.title("Training Time vs. Training Data per RP (%)")
plt.legend()
plt.show()
