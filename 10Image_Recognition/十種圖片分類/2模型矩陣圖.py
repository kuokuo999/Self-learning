import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical

# 指定亂數種子
seed = 10
np.random.seed(seed)
# 載入資料集
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# 因為是固定範圍, 所以執行正規化, 從 0-255 至 0-1
X_test = X_test.astype("float32") / 255
# One-hot編碼
Y_test_bk = Y_test.copy()   # 備份 Y_test 資料集
Y_test = to_categorical(Y_test)
# 建立Keras的Sequential模型
model = Sequential()
model = load_model("cifar10.h5")
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 評估模型
print("Testing ...")
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 計算分類的預測值
print("\nPredicting ...")
Y_probs = model.predict(X_test)
Y_pred = np.argmax(Y_probs, axis=1)
# 顯示混淆矩陣
tb = pd.crosstab(Y_test_bk.astype(int).flatten(), 
                 Y_pred.astype(int),
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("Ch9_1_3.html")
