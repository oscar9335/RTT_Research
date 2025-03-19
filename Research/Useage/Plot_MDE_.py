import json
import numpy as np
import matplotlib.pyplot as plt

label_mapping = {
    '11': '1-1','10': '1-2','9': '1-3','8': '1-4','7': '1-5','6': '1-6','5': '1-7','4': '1-8','3': '1-9','2': '1-10','1': '1-11',
    '12': '2-1','30': '2-11',
    '13': '3-1','29': '3-11',
    '14': '4-1','28': '4-11',
    '15': '5-1','27': '5-11',
    '16': '6-1','17': '6-2','18': '6-3','19': '6-4','20': '6-5','21': '6-6','22': '6-7','23': '6-8','24': '6-9','25': '6-10','26': '6-11',
    '49': '7-1','31': '7-11',
    '48': '8-1','32': '8-11',
    '47': '9-1','33': '9-11',
    '46': '10-1','34': '10-11',
    '45': '11-1','44': '11-2','43': '11-3','42': '11-4','41': '11-5','40': '11-6','39': '11-7','38': '11-8','37': '11-9','36': '11-10','35': '11-11'
}

# 讀取 JSON 數據
file_path = "2mcAP_FTMonly_WORST_2024_12_14_mde"
with open(file_path) as file:
    mde_data = json.load(file)

# 初始化網格大小
rows, cols = 11, 11
grid = np.full((rows, cols), np.nan)  # 初始化網格
labels = np.empty((rows, cols), dtype=object)  # 初始化標籤

# **修正 key 值，+1 後再映射**
for key, value in mde_data.items():
    new_key = str(int(key) + 1)  # key +1
    if new_key in label_mapping:  # 確保 key 存在於 mapping
        r, c = map(int, label_mapping[new_key].split('-'))
        grid[rows - r, c - 1] = value["mde"]  # 反轉行索引以正確對應圖表
        labels[rows - r, c - 1] = label_mapping[new_key]

# **逆時針旋轉 90 度**
rotated_grid = np.fliplr(grid.T)  # 先轉置，再左右翻轉
rotated_labels = np.fliplr(labels.T)

# 繪製旋轉後的圖表
fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.cm.Reds  # 顏色映射

# 畫出每個網格
for i in range(cols):  # 旋轉後的行數變為原來的列數
    for j in range(rows):  # 旋轉後的列數變為原來的行數
        value = rotated_grid[i, j]
        label = rotated_labels[i, j]
        if not np.isnan(value):
            ax.text(j, i + 0.2, f'{label}', ha='center', va='center', color='black', fontsize=10)
            ax.text(j, i - 0.2, f'{value:.4f}', ha='center', va='center', color='blue', fontsize=12)
        rect_color = cmap(value / np.nanmax(rotated_grid)) if not np.isnan(value) else 'white'
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=rect_color, alpha=0.5))

# 格式化圖表
ax.set_xlim(-0.5, rows - 0.5)
ax.set_ylim(-0.5, cols - 0.5)
ax.set_xticks(np.arange(-0.5, rows, 1), minor=True)
ax.set_yticks(np.arange(-0.5, cols, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.title("2mcAP worst( 2 FTM only )_2024_12_14_mde")
plt.show()