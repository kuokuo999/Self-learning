import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model

# 載入 CIFAR-10 資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# 選擇一個測試圖片
i = 8
img = X_test[i]

# 將圖片重塑為 4D 張量並進行正規化
X_test_img = img.reshape(1, 32, 32, 3).astype("float32") / 255

# 載入預先訓練好的模型
model = load_model("cifar10.h5")

# 進行預測
probs = model.predict(X_test_img, batch_size=1)

# 繪製原始圖片和預測的概率
plt.figure()
plt.subplot(1, 2, 1)
plt.title("圖片範例：" + str(Y_test[i]))
plt.imshow(img, cmap="binary")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Probabilities for Each Image Class")
plt.bar(np.arange(10), probs.reshape(10), align="center")
plt.xticks(np.arange(10), np.arange(10).astype(str))
plt.show()
