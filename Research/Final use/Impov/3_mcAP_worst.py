import pandas as pd

data = {
    ("No fine-tune", "DNN"):    [0.9647, 1.1851, 1.0758, 1.2215, 1.3345],
    ("No fine-tune", "RegDNN"): [1.0080, 1.2041, 1.0999, 1.1815, 1.3627],
    ("0.25%", "DNN"):           [0.6040, 0.5429, 0.4194, 0.4456, 0.4427],
    ("0.25%", "RegDNN"):        [0.6994, 0.6940, 0.5214, 0.4218, 0.5551],
    ("2.5%", "DNN"):            [0.0395, 0.0318, 0.0301, 0.0543, 0.0314],
    ("2.5%", "RegDNN"):         [0.0362, 0.0327, 0.0305, 0.0346, 0.0301],
    ("10%", "DNN"):             [0.0213, 0.0188, 0.0204, 0.0298, 0.0165],
    ("10%", "RegDNN"):          [0.0203, 0.0152, 0.0163, 0.0169, 0.0183],
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
