import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model

# 指定亂數種子
seed = 10
np.random.seed(seed)
# 載入資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test.astype("float32") / 255
# 載入預先訓練好的模型
model = load_model("cifar10.h5")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 測試資料集的分類和機率的預測值
print("Predicting ...")
Y_probs = model.predict(X_test)   # 機率
Y_pred = np.argmax(Y_probs, axis=1)  # 分類
# 建立分類錯誤的 DataFrame 物件
Y_test = Y_test.flatten()
df = pd.DataFrame({"label": Y_test, "predict": Y_pred})
df = df[Y_test != Y_pred]  # 篩選出分類錯誤的資料
print(df.head())
df.head().to_html("Ch9_1_3b.html")
# 隨機選 1 個錯誤分類的數字索引
i = df.sample(n=1).index.values.astype(int)[0]
print("Index: ", i)
img = X_test[i]
# 繪出圖表的預測結果
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Example of Image:" + str(Y_test[i]))
plt.imshow(img, cmap="binary")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Probabilities for Each Image Class")
plt.bar(np.arange(10), Y_probs[i], align="center")
plt.xticks(np.arange(10), np.arange(10).astype(str))
plt.show()
