import numpy as np
import cv2
from matplotlib import pyplot as plt

# 隨機產生兩組陣列
# 產生60個值在[0,50]內的小直徑資料
xiaoMI = np.random.randint(0, 50, 60)
# 產生60個值在[200,250]內的大直徑資料
daMI = np.random.randint(200, 250, 60)

# 將xiaoMI和daMI組合為MIMI
MI = np.hstack((xiaoMI, daMI))
# 使用reshape函數將其轉為(120, 1)
MI = MI.reshape((120, 1))
# 將MI轉為float32類型
MI = np.float32(MI)

# 呼叫kmeans模組
# 設定參數criteria的值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 設定參數flags的值
flags = cv2.KMEANS_RANDOM_CENTERS

# 呼叫函數kmeans
retval, bestLabels, centers = cv2.kmeans(MI, 2, None, criteria, 10, flags)

# 列印傳回值
print(retval)
print(bestLabels)
print(centers)

# 取得分類結果
XM = MI[bestLabels == 0]
DM = MI[bestLabels == 1]

# 繪製分類結果
# 繪製原始資料
plt.plot(XM, 'ro')
plt.plot(DM, 'bo')
# 繪製中心點
plt.plot(centers[0], 'rx')
plt.plot(centers[1], 'bx')
plt.show()
