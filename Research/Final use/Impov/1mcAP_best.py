import pandas as pd

# 原始資料
data = {
    ("No fine-tune", "DNN"):    [1.2851, 1.5454, 1.4230, 1.2349, 1.6114],
    ("No fine-tune", "RegDNN"): [1.2416, 1.5333, 1.4821, 1.2269, 1.6083],
    ("0.25%", "DNN"):           [0.9156, 0.8540, 0.5900, 0.6059, 0.7363],
    ("0.25%", "RegDNN"):        [0.8972, 1.0704, 0.8556, 0.6398, 0.9858],
    ("2.5%", "DNN"):            [0.1189, 0.1199, 0.0893, 0.1090, 0.0765],
    ("2.5%", "RegDNN"):         [0.1056, 0.1046, 0.0867, 0.0803, 0.0677],
    ("10%", "DNN"):             [0.0764, 0.0648, 0.0603, 0.0672, 0.0563],
    ("10%", "RegDNN"):          [0.0547, 0.0469, 0.0510, 0.0502, 0.0536],
}
dates = ["2024/12/21", "2024/12/27", "2025/01/03", "2025/01/10", "2025/02/07"]
index = pd.MultiIndex.from_tuples(data.keys(), names=["Data Size", "Model"])
df = pd.DataFrame(list(data.values()), index=index, columns=dates)

# 分 DNN/RegDNN 計算 FT Impv.
def calc_ft_impv(df, model):
    # 取得對應的 No fine-tune MDE
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
