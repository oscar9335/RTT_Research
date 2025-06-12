import pandas as pd

data = {
    ("No fine-tune", "DNN"):    [0.7496, 1.0673, 1.0387, 1.0538, 1.3441],
    ("No fine-tune", "RegDNN"): [0.7174, 1.0122, 0.9838, 0.9971, 1.2665],
    ("0.25%", "DNN"):           [0.4883, 0.4256, 0.3454, 0.3710, 0.4330],
    ("0.25%", "RegDNN"):        [0.4646, 0.6343, 0.4468, 0.4287, 0.7138],
    ("2.5%", "DNN"):            [0.0278, 0.0328, 0.0193, 0.0371, 0.0285],
    ("2.5%", "RegDNN"):         [0.0262, 0.0308, 0.0254, 0.0254, 0.0325],
    ("10%", "DNN"):             [0.0210, 0.0169, 0.0125, 0.0194, 0.0144],
    ("10%", "RegDNN"):          [0.0125, 0.0148, 0.0104, 0.0123, 0.0161],
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
