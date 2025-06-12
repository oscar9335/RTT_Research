import pandas as pd

# 輸入你的原始表格數據
data = {
    "Data Size": ["No fine-tune", "0.25%", "2.5%", "10%"],
    "Basemodel": [0.2676, 0.2676, 0.2676, 0.2676],
    "2024/12/21": [1.9955, 1.3289, 0.4643, 0.3261],
    "2024/12/27": [2.2472, 1.4534, 0.3791, 0.3092],
    "2025/01/03": [2.0834, 1.0411, 0.3310, 0.2556],
    "2025/01/10": [2.1314, 1.1584, 0.3395, 0.2624],
    "2025/02/07": [2.1328, 0.9667, 0.2203, 0.1585],
}

df = pd.DataFrame(data)
df.set_index("Data Size", inplace=True)

# 計算 FT Impv 百分比
ft_impv = {}
for date in ["2024/12/21", "2024/12/27", "2025/01/03", "2025/01/10", "2025/02/07"]:
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

# 只顯示 0.25%、2.5%、10% 三行
ft_impv_df = ft_impv_df.loc[["0.25%", "2.5%", "10%"]]

print("FT Impv. (%) Table:")
print(ft_impv_df)
