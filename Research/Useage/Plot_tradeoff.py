import matplotlib.pyplot as plt
import numpy as np

# Data 
methods_en = ['Train New Model', 'Transfer Learning using 10% new data', 'No Retraining']
time_cost_en = [50, 1, 0]  # Training cost (assuming N = 0)
accuracy_en = [0.007, 0.06, 0.77]  # MDE (m)

# Plot with annotations
fig, ax1 = plt.subplots()


# Adjusting data to reflect large N for data collection cost
large_N = 1000  # Representing a very large data collection time cost
data_collection_cost_large = [large_N, large_N / 10, 0]  # For new model, transfer learning, no retraining

# Total time cost including large N
total_time_cost_large = [time_cost_en[i] + data_collection_cost_large[i] for i in range(3)]

# Plot with large N representation
fig, ax1 = plt.subplots(figsize=(10, 6))

# Time cost bar chart (stacked)
bars_data_large = ax1.bar(methods_en, data_collection_cost_large, color='lightgreen', label='Data Collection Cost (N)')
bars_training_large = ax1.bar(methods_en, time_cost_en, bottom=data_collection_cost_large, color='skyblue', label='Training Cost (s)')

ax1.set_xlabel('Method')
ax1.set_ylabel('Time Cost (s)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add labels to the stacked bars
for i in range(len(methods_en)):
    total_height = total_time_cost_large[i]
    ax1.annotate(f'{total_height}s',
                 xy=(bars_data_large[i].get_x() + bars_data_large[i].get_width() / 2, total_height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')

# Accuracy line plot
ax2 = ax1.twinx()
line, = ax2.plot(methods_en, accuracy_en, color='red', marker='o', label='MDE (m)')
ax2.set_ylabel('MDE (m)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add labels to the line plot points
for i, txt in enumerate(accuracy_en):
    ax2.annotate(f'{txt}m',
                 (methods_en[i], accuracy_en[i]),
                 textcoords="offset points",
                 xytext=(0, 5),
                 ha='center', va='bottom')

# Title, legends, and layout
plt.title('Trade-off Between Time Cost and Accuracy')
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
fig.tight_layout()

plt.show()


# # Time cost bar chart
# bars = ax1.bar(methods_en, time_cost_en, color='skyblue', label='Time Cost (s)')
# ax1.set_xlabel('Method')
# ax1.set_ylabel('Time Cost (s)', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Add labels to the bars
# for bar in bars:
#     height = bar.get_height()
#     ax1.annotate(f'{height}s',
#                  xy=(bar.get_x() + bar.get_width() / 2, height),
#                  xytext=(0, 3),  # Offset text slightly above the bar
#                  textcoords="offset points",
#                  ha='center', va='bottom')

# # Accuracy line plot
# ax2 = ax1.twinx()
# line, = ax2.plot(methods_en, accuracy_en, color='red', marker='o', label='MDE (m)')
# ax2.set_ylabel('MDE (m)', color='red')
# ax2.tick_params(axis='y', labelcolor='red')

# # Add labels to the line plot points
# for i, txt in enumerate(accuracy_en):
#     ax2.annotate(f'{txt}m',
#                  (methods_en[i], accuracy_en[i]),
#                  textcoords="offset points",
#                  xytext=(0, 5),
#                  ha='center', va='bottom')

# # Title and layout
# plt.title('Trade-off Between Time Cost and Accuracy')
# fig.tight_layout()

# plt.show()
