import matplotlib.pyplot as plt
import numpy as np

# 資料設定：從第三組與第六組取出 week=8 (index=4) 的 MDE 值
third_mde_data = {
    "1.25%": [0.0754, 0.0937, 0.0601, 0.0585, 0.0360],
    "2.5%": [0.0443, 0.0530, 0.0411, 0.0405, 0.0221],
    "5%": [0.0340, 0.0449, 0.0309, 0.0385, 0.0340],
    "10%": [0.0301, 0.0287, 0.0220, 0.0271, 0.0162]
}

sixth_mde_data = {
    "1.25%": [0.1359, 0.0894, 0.0792, 0.0983, 0.0649],
    "2.5%": [0.0666, 0.0655, 0.0502, 0.0727, 0.0524],
    "5%": [0.0533, 0.0525, 0.0399, 0.0474, 0.0377],
    "10%": [0.0422, 0.0331, 0.0322, 0.0423, 0.0292]
}

week_index = 4
percentages = ["1.25%", "2.5%", "5%", "10%"]
third_final_mde = [third_mde_data[p][week_index] for p in percentages]
sixth_final_mde = [sixth_mde_data[p][week_index] for p in percentages]

# 畫圖
x = np.arange(len(percentages))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, third_final_mde, width, label='2 mcAP Best', color='tab:blue')
ax.bar(x + width/2, sixth_final_mde, width, label='2 mcAP Worst', color='tab:red')

ax.set_ylabel('MDE (m)', fontsize=14)
ax.set_xlabel('Fine-tune Data Volume per RP', fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_title('', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(percentages, fontsize=14)
ax.legend(fontsize=14)
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()
