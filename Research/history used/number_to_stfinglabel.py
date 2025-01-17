import pandas as pd

# Load the uploaded CSV file
file_path = 'timestamp_allignment_Balanced_2024_12_21_rtt_logs.csv'
data = pd.read_csv(file_path)

# Define the label mapping
label_mapping = {
    '11': '1-1','10': '1-2','9': '1-3','8': '1-4','7': '1-5','6': '1-6','5': '1-7','4': '1-8','3': '1-9','2': '1-10','1': '1-11',
    '12': '2-1','30': '2-11',
    '13': '3-1','29': '3-11',
    '14': '4-1','28': '4-11',
    '15': '5-1','27': '5-11',
    '16': '6-1','17': '6-2','18': '6-3','19': '6-4','20': '6-5','21': '6-6','22': '6-7','23': '6-8','24': '6-9','25': '6-10','26': '6-11',
    '49': '7-1','31': '7-11',
    '48': '8-1','32': '8-11',
    '47': '9-1','33': '9-11',
    '46': '10-1','34': '10-11',
    '45': '11-1','44': '11-2','43': '11-3','42': '11-4','41': '11-5','40': '11-6','39': '11-7','38': '11-8','37': '11-9','36': '11-10','35': '11-11'
}

# Apply the label mapping to the dataset
# Assuming there's a column named "label" in the dataset
if 'Label' in data.columns:
    data['Label'] = data['Label'].astype(str).map(label_mapping)

# Save the updated dataset
output_path = 'history used//timestamp_allignment_Balanced_2024_12_21_rtt_logs.csv'
data.to_csv(output_path, index=False)

output_path
