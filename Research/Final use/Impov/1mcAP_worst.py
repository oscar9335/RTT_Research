import pandas as pd

# 原始資料
data = {
    ("No fine-tune", "DNN"):    [1.5126, 1.7644, 1.5284, 1.5319, 1.7350],
    ("No fine-tune", "RegDNN"): [1.5386, 1.7226, 1.5260, 1.5234, 1.6889],
    ("0.25%", "DNN"):           [1.0614, 1.1027, 0.7446, 0.7472, 0.6675],
    ("0.25%", "RegDNN"):        [1.2149, 1.1913, 0.8465, 0.9464, 0.7611],
    ("2.5%", "DNN"):            [0.2073, 0.1643, 0.1515, 0.1897, 0.0830],
    ("2.5%", "RegDNN"):         [0.1787, 0.1856, 0.1748, 0.1643, 0.0718],
    ("10%", "DNN"):             [0.1167, 0.1057, 0.1105, 0.1184, 0.0563],
    ("10%", "RegDNN"):          [0.1045, 0.1012, 0.1014, 0.0959, 0.0461],
}
dates = ["2024/12/21", "2024/12/27", "2025/01/03", "2025/01/10", "2025/02/07"]
index = pd.MultiIndex.from_tuples(data.keys(), names=["Data Size", "Model"])
df = pd.DataFrame(list(data.values()), index=index, columns=dates)

def calc_ft_impv(df, model):
    noft = df.loc[("No fine-tune", model)]
    out = []
    for sz in ["0.25%", "2.5%", "10%"]:
        mde = df.loc[(sz, model)]
        impv = (noft - mde) / noft * 100
        out.append([round(x, 2) for x in impv])
    return pd.DataFrame(out, index=["0.25%", "2.5%", "10%"], columns=dates)

ftimpv_dnn = calc_ft_impv(df, "DNN")
ftimpv_regdnn = calc_ft_impv(df, "RegDNN")

print("DNN:")
print(ftimpv_dnn)
print("\nRegDNN:")
print(ftimpv_regdnn)
