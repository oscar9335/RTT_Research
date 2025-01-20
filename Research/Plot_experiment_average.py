import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["0 AP (Baseline)", "1 AP", "2 APs", "3 APs", "4 APs"]
mde_values = [0.2399, 0.0610, 0.0261, 0.0147, 0.0100]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, mde_values, color='skyblue', alpha=0.9)

# Annotate the bar heights
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f'{height:.4f} m', ha='center', fontsize=10)

# Add labels and title
plt.title("MDE for Different Numbers of mcAPs (Including Baseline)", fontsize=14)
plt.xlabel("Number of mcAPs in 4 APs", fontsize=12)  # Modify X-axis label here
plt.ylabel("MDE (m)", fontsize=12)

# Add grid lines
plt.grid(axis='y', alpha=0.4, linestyle='--')

# Adjust layout for better display
plt.tight_layout()

# Show the plot
plt.show()
