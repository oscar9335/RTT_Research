import matplotlib.pyplot as plt


# Data for plotting
weeks = [1, 2, 3, 4]

accuracy_data = {
    "1 data per RP": [0.8824, 0.8549, 0.8412, 0.7646],
    "5 data per RP": [0.9410, 0.9581, 0.9458, 0.9414],
    "10 data per RP": [0.9606, 0.9680, 0.9691, 0.9592],
    "20 data per RP": [0.9722, 0.9787, 0.9789, 0.9754],
    "40 data per RP": [0.9851, 0.9859, 0.9877, 0.9815],
}

mde_data = {
    "1 data per RP": [0.1850, 0.2135, 0.2728, 0.3522],
    "5 data per RP": [0.0795, 0.0584, 0.0760, 0.0839],
    "10 data per RP": [0.0541, 0.0391, 0.0448, 0.0668],
    "20 data per RP": [0.0426, 0.0316, 0.0324, 0.0320],
    "40 data per RP": [0.0210, 0.0201, 0.0218, 0.0204],
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
plt.legend()
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
plt.legend()
plt.grid(True)
plt.show()
