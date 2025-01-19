import pandas as pd

# File paths for your local system
file_paths = {
    "2024_12_14": "timestamp_allignment_Balanced_2024_12_14_rtt_logs.csv",
    "2024_12_21": "timestamp_allignment_Balanced_2024_12_21_rtt_logs.csv",
    "2024_12_27": "timestamp_allignment_Balanced_2024_12_27_rtt_logs.csv",
    "2025_01_03": "timestamp_allignment_Balanced_2025_01_03_rtt_logs.csv",
    "2025_01_10": "timestamp_allignment_Balanced_2025_01_10_rtt_logs.csv"
}

# Check for 'Label' column in each file
missing_label_files = []
for date, path in file_paths.items():
    try:
        data = pd.read_csv(path)
        if 'Label' not in data.columns:
            missing_label_files.append(date)
        print(f"{date} columns: {data.columns.tolist()}")
    except Exception as e:
        print(f"Error processing file for {date}: {e}")

print("Files missing 'Label' column:", missing_label_files)
