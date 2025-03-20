import re
import csv
import os

# Path for input directory
input_directory = "txt_files_RAW_data"  # Replace with your directory path
output_directory = os.path.join(input_directory, "2024_12_14\\raw data")  # Output directory path

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Regular expression pattern to match each line of data
pattern = re.compile(
    r"Label: (.*?), Timestamp: (.*?), AP SSID: (.*?), BSSID: (.*?), Rssi: (-?\d+), Distance: (-?\d+) mm, StdDev: (\d+) mm, timeStemp: (\d+), mcOn: (true|false)"
)


# Iterate through all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):  # Process only .txt files
        input_file_path = os.path.join(input_directory, filename)
        
        # Generate output file name based on input file name (replace .txt with .csv)
        output_file_name = filename.replace(".txt", ".csv")
        output_file_path = os.path.join(output_directory, output_file_name)
        
        # Open the input text file and corresponding output CSV file
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
            # Define the CSV writer
            writer = csv.writer(outfile)

            # Write the CSV header
            writer.writerow(["Label", "Timestamp", "AP SSID", "BSSID", "Rssi", "Distance (mm)", "StdDev (mm)", "timeStemp", "mcOn"])

            # Process each line
            for line in infile:
                match = pattern.match(line)
                if match:
                    # Extract data using groups
                    writer.writerow(match.groups())

            #  # Process each line
            # for line in infile:
            #     match = pattern.match(line)
            #     if match:
            #         # Extract data using groups
            #         row = list(match.groups())

            #         # Ensure Label is written with quotes
            #         row[0] = f'"{row[0]}"'

            #         # Write the row to the CSV file
            #         writer.writerow(row)

        # Output the path to the saved CSV file
        print(f"File saved at: {output_file_path}")
