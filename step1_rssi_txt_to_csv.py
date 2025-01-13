import pandas as pd
import re

# Define a function to parse the log file
def parse_rtt_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        match = re.match(r"Timestamp: (.*), AP SSID: (.*?), BSSID: (.*?), RSSI: (.*? dBm).*timestamp: (\d+)", line)
        if match:
            timestamp, ssid, bssid, rssi, tstamp = match.groups()
            data.append({
                "Timestamp": timestamp,
                "AP SSID": ssid,
                "BSSID": bssid,
                "RSSI": rssi,
                "RTT Timestamp": tstamp
            })
    
    return pd.DataFrame(data)

# File path to the log file
file_path = 'testall_rtt_log.txt'  # Update this path to your file location

# Parse the log file
rtt_data = parse_rtt_log(file_path)

# Save the parsed data to a CSV file
output_path = 'testall_rtt_log.csv'  # Specify your desired output path
rtt_data.to_csv(output_path, index=False)

print(f"RTT data has been saved to {output_path}")
