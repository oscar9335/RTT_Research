import matplotlib.pyplot as plt

# Data for plotting
weeks = [1, 2, 3, 4, 5]

accuracy_data = {
    "0.25%": [0.8690, 0.7589, 0.7576, 0.8300, 0.8056],
    "1.25%": [0.8877, 0.8904, 0.8591, 0.8987, 0.8777],
    "2.5%": [0.9460, 0.9550, 0.9348, 0.9571, 0.9480],
    "5%": [0.9690, 0.9676, 0.9758, 0.9407, 0.9687],
    "10%": [0.9806, 0.9781, 0.9839, 0.9740, 0.9810],
}

mde_data = {
    "0.25%": [0.2046, 0.3485, 0.3898, 0.2932, 0.3140],
    "1.25%": [0.1620, 0.1372, 0.1988, 0.1393, 0.1807],
    "2.5%": [0.0709, 0.0588, 0.0878, 0.0673, 0.0804],
    "5%": [0.0455, 0.0434, 0.0373, 0.0793, 0.0510],
    "10%": [0.0270, 0.0300, 0.0259, 0.0331, 0.0275],
}

# Plot Accuracy trends
plt.figure()
for label, values in accuracy_data.items():
    plt.plot(weeks, values, marker="o", label=label)
    for i, txt in enumerate(values):
        plt.annotate(f"{txt:.4f}", (weeks[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.xlabel("Week")
plt.ylabel("Accuracy")
plt.title("Accuracy Decay Over Time (Fine-Tuning Each Week)")
plt.legend(title="Training Data per RP")
plt.grid(True)
plt.show()

# Plot MDE trends
plt.figure()
for label, values in mde_data.items():
    plt.plot(weeks, values, marker="s", linestyle="dashed", label=label)
    for i, txt in enumerate(values):
        plt.annotate(f"{txt:.4f}", (weeks[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.xlabel("Week")
plt.ylabel("MDE (m)")
plt.title("MDE Increase Over Time (Fine-Tuning Each Week)")
plt.legend(title="Training Data per RP")
plt.grid(True)
plt.show()
