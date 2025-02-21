import os
import pandas as pd
import numpy as np

# Function to correct distances based on standard deviation
def correct_distance(file_path):
    data = pd.read_csv(file_path)
    
    # Apply corrected distance calculation with normalization
    data['Corrected Distance (mm)'] = data.apply(
        lambda row: np.sign(row['Distance (mm)']) * abs(row['Distance (mm)']) / (1 + (row['StdDev (mm)'] / abs(row['Distance (mm)'])))
        if row['Distance (mm)'] != 0 else 0,  # Handle zero distances gracefully
        axis=1
    )

    
    return data

# Process all CSV files in a given folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            
            # Process and save corrected file
            corrected_data = correct_distance(input_file_path)
            corrected_data.to_csv(output_file_path, index=False)

# Define input and output directories
input_folder = '2025_01_10\\raw data'  # Replace with your actual input folder path
output_folder = '2025_01_10\\2025_01_10_standalize'

# Process all files in the input folder
process_folder(input_folder, output_folder)

print(f"Processing complete. Corrected files are saved in {output_folder}.")
