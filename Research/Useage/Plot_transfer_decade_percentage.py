import matplotlib.pyplot as plt

# Data for plotting
weeks = [1, 2, 3, 4, 5]

accuracy_data = {
    "0.25%": [0.8881, 0.7055, 0.7574, 0.8581, 0.8163],
    "1.25%": [0.9017, 0.9074, 0.9133, 0.9101, 0.9123],
    "2.5%": [0.9493, 0.9586, 0.9442, 0.9514, 0.9529],
    "5%": [0.9710, 0.9757, 0.9662, 0.9696, 0.9725 ],
    "10%": [0.9814, 0.9822, 0.9805, 0.9799, 0.9830 ],
}

mde_data = {
    "0.25%": [0.1719, 0.3777, 0.3574, 0.1834, 0.2605],
    "1.25%": [0.1311, 0.1259, 0.1334, 0.1288, 0.1413 ],
    "2.5%": [0.0718, 0.0533, 0.0763, 0.0673, 0.0734 ],
    "5%": [0.0412, 0.0330, 0.0465, 0.0397, 0.0382 ],
    "10%": [0.0243, 0.0241, 0.0285, 0.0257, 0.0235],
}

# Plot Accuracy trends
plt.figure()
for label, values in accuracy_data.items():
    plt.plot(weeks, values, marker="o", label=label)
    for i, txt in enumerate(values):
        plt.annotate(f"{txt:.4f}", (weeks[i], values[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.xlabel("Week")
plt.xticks(weeks, weeks)
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
plt.xticks(weeks, weeks)
plt.ylabel("MDE (m)")
plt.title("MDE Increase Over Time (Fine-Tuning Each Week)")
plt.legend(title="Training Data per RP")
plt.grid(True)
plt.show()
