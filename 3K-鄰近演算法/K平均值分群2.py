import numpy as np
import cv2
from matplotlib import pyplot as plt

# 隨機產生兩組數值
# xiaomi組，長和寬都在[0,20]內
xiaomi = np.random.randint(0, 20, (30, 2))
# dami組，長和寬的大小都在[40,60]
dami = np.random.randint(40, 60, (30, 2))
# 組合資料
MI = np.vstack((xiaomi, dami))
# 轉為 float32 類型
MI = np.float32(MI)

# 呼叫 kmeans 模組
# 設定參數 criteria 值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 呼叫 kmeans 函數
ret, label, center = cv2.kmeans(MI, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 列印傳回值
print(ret)
print(label)
print(center)

# 根據 kmeans 的處理結果，將資料分類，分為 XM 和 DM 兩大類
XM = MI[label.ravel() == 0]
DM = MI[label.ravel() == 1]

# 繪製分類結果資料及中心點
plt.scatter(XM[:, 0], XM[:, 1], c='g', marker='s')
plt.scatter(DM[:, 0], DM[:, 1], c='r', marker='o')
plt.scatter(center[0, 0], center[0, 1], s=200, c='b', marker='o')
plt.scatter(center[1, 0], center[1, 1], s=200, c='b', marker='s')
plt.xlabel('Height')
plt.ylabel('Width')
plt.show()
