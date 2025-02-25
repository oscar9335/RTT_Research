# Re-import necessary libraries and redefine data as execution state was reset
import matplotlib.pyplot as plt

# Data
training_data_per_rp = [1, 5, 10, 20, 40]
accuracy = [0.8824, 0.9410, 0.9606, 0.9722, 0.9851]
mde = [0.1850, 0.0795, 0.0541, 0.0426, 0.0210]
training_time = [668.25, 162.04, 62.31, 34.78, 24.58]

# Re-plot accuracy and MDE with annotations
fig, ax1 = plt.subplots()

ax1.set_xlabel("Training Data per RP")
ax1.set_ylabel("Accuracy", color="tab:blue")
ax1.plot(training_data_per_rp, accuracy, marker="o", color="tab:blue", label="Accuracy")
ax1.tick_params(axis="y", labelcolor="tab:blue")

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

plt.xlabel("Training Data per RP")
plt.ylabel("Training Time (s)")
plt.title("Training Time vs. Training Data per RP")
plt.legend()
plt.show()
