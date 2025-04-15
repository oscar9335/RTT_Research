import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定義實驗資料
data = [
    {"Setting": "Baseline (RSSI only)", "APs Used": "-", "Data Type": "RSSI", "MDE (m)": 0.2294, "Accuracy (%)": 91.46, "Worst Case (m)": 1.2281},
    {"Setting": "AP1FTMonly", "APs Used": "AP1", "Data Type": "Distance", "MDE (m)": 1.9492, "Accuracy (%)": 39.49, "Worst Case (m)": 6.2892},
    {"Setting": "AP1,AP2FTMonly", "APs Used": "AP1, AP2", "Data Type": "Distance", "MDE (m)": 0.2762, "Accuracy (%)": 88.85, "Worst Case (m)": 1.3520},
    {"Setting": "AP1,AP2,AP3FTMonly", "APs Used": "AP1, AP2, AP3", "Data Type": "Distance", "MDE (m)": 0.0435, "Accuracy (%)": 97.08, "Worst Case (m)": 0.1447},
    {"Setting": "AP1", "APs Used": "AP1", "Data Type": "Distance + RSSI", "MDE (m)": 0.7770, "Accuracy (%)": 72.80, "Worst Case (m)": 4.4036},
    {"Setting": "AP1,AP2", "APs Used": "AP1, AP2", "Data Type": "Distance + RSSI", "MDE (m)": 0.0696, "Accuracy (%)": 96.88, "Worst Case (m)": 0.3652},
    {"Setting": "AP1,AP2,AP3", "APs Used": "AP1, AP2, AP3", "Data Type": "Distance + RSSI", "MDE (m)": 0.0196, "Accuracy (%)": 98.59, "Worst Case (m)": 0.0606},
]

df = pd.DataFrame(data)

# 顯示表格
print("\n=== RTT Fingerprint Experiment Summary ===")
print(df.to_string(index=False))

# 畫出 MDE 直方圖
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Setting", y="MDE (m)", palette="Blues_d")
plt.xticks(rotation=45, ha='right')
plt.title("Comparison of Mean Distance Error (MDE)")
plt.ylabel("MDE (meters)")
plt.tight_layout()
plt.show()
