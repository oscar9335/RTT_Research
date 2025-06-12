import numpy as np
import pandas as pd

# 原始 MDE 資料
data = {
    "Data Size": ["No fine-tune", "0.25%", "2.5%", "10%"],
    "Basemodel": [0.0082, 0.0082, 0.0082, 0.0082],
    "2024/12/21": [0.7713, 0.4878, 0.0255, 0.0167],
    "2024/12/27": [0.9342, 0.4543, 0.0260, 0.0138],
    "2025/01/03": [1.0082, 0.3845, 0.0213, 0.0121],
    "2025/01/10": [1.0504, 0.3466, 0.0247, 0.0151],
    "2025/02/28": [1.2559, 0.5510, 0.0248, 0.0117],
}

df = pd.DataFrame(data)
df.set_index("Data Size", inplace=True)

# 計算 FT Impv 百分比
ft_impv = {}
for date in ["2024/12/21", "2024/12/27", "2025/01/03", "2025/01/10", "2025/02/28"]:
    no_ft = df.loc["No fine-tune", date]
    ft_impv[date] = []
    for idx in df.index:
        mde_ft = df.loc[idx, date]
        if idx == "No fine-tune":
            ft_impv[date].append(0.0)
        else:
            impv = (no_ft - mde_ft) / no_ft * 100
            ft_impv[date].append(round(impv, 2))

ft_impv_df = pd.DataFrame(ft_impv, index=df.index)

print("FT Impv. (%) Table:")
print(ft_impv_df)
