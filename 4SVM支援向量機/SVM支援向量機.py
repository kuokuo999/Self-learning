import cv2
import numpy as np
import matplotlib.pyplot as plt

# 第1步準備資料
# 表現為A級的員工的筆試、面試成績
a = np.random.randint(95, 100, (20, 2)).astype(np.float32)
# 表現為B級的員工的筆試、面試成績
b = np.random.randint(90, 95, (20, 2)).astype(np.float32)

# 合併資料
data = np.vstack((a, b)).astype(np.float32)

# 第2步建立分組標籤，0代表A級，1代表B級
# aLabel對應著a的標籤，為類型0-等級A
aLabel = np.zeros((20, 1))
# bLabel對應著b的標籤，為類型1-等級B
bLabel = np.ones((20, 1))

# 合併標籤
label = np.vstack((aLabel, bLabel)).astype(np.int32)

# 第3步訓練
# 用機器學習模組 SVM_create()建立 svm
svm = cv2.ml.SVM_create()
# 屬性設定，直接採用預設值即可
# svm.setType(cv2.ml.SVM_C_SVC) # svm type
# svm.setKernel(cv2.ml.SVM_LINEAR) # kernel type
# svm.setC(0.01)
# 訓練
result = svm.train(data, cv2.ml.ROW_SAMPLE, label)

# 第4步預測
# 產生兩個隨機的筆試成績和面試成績資料對
test = np.array([[98, 90], [90, 99]], dtype=np.float32)
# 預測
(p1, p2) = svm.predict(test)

# 第5步觀察結果
# 視覺化
plt.scatter(a[:, 0], a[:, 1], 80, 'g', 'o')
plt.scatter(b[:, 0], b[:, 1], 80, 'b', 's')
plt.scatter(test[:, 0], test[:, 1], 80, 'r', '*')
plt.show()

# 列印原始測試資料test，預測結果
print("Test Data:\n", test)
print("Predictions:\n", p1)
