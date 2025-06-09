import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    # 讀取 txt 檔案，每一行一個數字，去除空行
    with open(filepath, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip() != ""]
    return np.array(data)

# 換成你的檔名
file1 = 'all_mde_errors_20runs_2mcAP worst AP13 DNN.txt'
file2 = 'all_mde_errors_20runs_2mcAP worst AP13 RegDNN.txt'

data1 = load_data(file1)
data2 = load_data(file2)

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

x1, y1 = compute_cdf(data1)
x2, y2 = compute_cdf(data2)

plt.figure(figsize=(8,6))
plt.plot(x1, y1, label='2 mcAP worst DNN', linewidth=2)
plt.plot(x2, y2, label='2 mcAP worst RegDNN', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('CDF Comparison')
plt.ylim(0.97, 1.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
