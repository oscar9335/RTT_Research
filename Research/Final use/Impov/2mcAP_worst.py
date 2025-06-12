import pandas as pd

# 數據
data = {
    ("No fine-tune", "DNN"):    [1.2751, 1.3779, 1.2660, 1.5051, 1.3345],
    ("No fine-tune", "RegDNN"): [1.2670, 1.3780, 1.3332, 1.4490, 1.3330],
    ("0.25%", "DNN"):           [0.7309, 0.7259, 0.5904, 0.6536, 0.5801],
    ("0.25%", "RegDNN"):        [0.9333, 0.8691, 0.7400, 0.6966, 0.5699],
    ("2.5%", "DNN"):            [0.0717, 0.0538, 0.0499, 0.0984, 0.0511],
    ("2.5%", "RegDNN"):         [0.0759, 0.0579, 0.0531, 0.0564, 0.0372],
    ("10%", "DNN"):             [0.0382, 0.0342, 0.0317, 0.0452, 0.0277],
    ("10%", "RegDNN"):          [0.0277, 0.0266, 0.0297, 0.0346, 0.0258],
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
