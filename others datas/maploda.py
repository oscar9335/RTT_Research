import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 地圖參數（從 map.yaml 來）
resolution = 0.05  # 每像素幾米
origin = [-3.78, -12.7]  # 地圖左下角的實際座標 (meter)

# 讀取地圖圖檔
map_img = mpimg.imread('map.pgm')

# 顯示地圖
height, width = map_img.shape
extent = [origin[0], origin[0] + width * resolution,
          origin[1], origin[1] + height * resolution]

plt.imshow(map_img, cmap='gray', origin='lower', extent=extent)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Indoor Map")
plt.grid(True)
plt.show()
