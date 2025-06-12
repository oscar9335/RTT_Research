import pandas as pd

data = {
    ("No fine-tune", "DNN"):    [0.8194, 1.1574, 1.0963, 0.9749, 1.3483],
    ("No fine-tune", "RegDNN"): [0.7820, 1.1562, 1.0427, 0.9790, 1.3313],
    ("0.25%", "DNN"):           [0.4499, 0.6236, 0.4716, 0.3746, 0.4744],
    ("0.25%", "RegDNN"):        [0.5521, 0.7863, 0.5720, 0.4832, 0.7080],
    ("2.5%", "DNN"):            [0.0454, 0.0514, 0.0351, 0.0570, 0.0295],
    ("2.5%", "RegDNN"):         [0.0388, 0.0569, 0.0437, 0.0386, 0.0323],
    ("10%", "DNN"):             [0.0301, 0.0290, 0.0222, 0.0294, 0.0178],
    ("10%", "RegDNN"):          [0.0206, 0.0248, 0.0201, 0.0215, 0.0186],
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

print("DNN:\n", ftimpv_dnn)
print("\nRegDNN:\n", ftimpv_regdnn)
