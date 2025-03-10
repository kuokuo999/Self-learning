import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model


# 載入資料集
df = pd.read_csv("./iris_data.csv")
# 顯示資料集的形狀
print(df.shape)
# 查看前5筆記錄
print(df.head())
df.head().to_html("Ch6_1_1a_01.html")
# 顯示資料集的描述資料
print(df.describe())
df.describe().to_html("Ch6_1_1a_02.html")


target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
Y = df["target"].map(target_mapping)
# 使用Matplotlib顯示視覺化圖表
colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = .5)
plt.scatter(df["sepal_length"], df["sepal_width"], color=colmap[Y])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.subplot(1, 2, 2)
plt.scatter(df["petal_length"], df["petal_width"], color=colmap[Y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# 使用Seaborn顯示視覺化圖表
sns.pairplot(df, hue="target")

#資料預處理
np.random.seed(7)  # 指定亂數種子
# 載入資料集
df = pd.read_csv("./iris_data.csv")
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
df["target"] = df["target"].map(target_mapping)
dataset = df.values
np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,0:4].astype(float)
Y = to_categorical(dataset[:,4])
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割成訓練和測試資料集
X_train, Y_train = X[:120], Y[:120]     # 訓練資料前120筆
X_test, Y_test = X[120:], Y[120:]       # 測試資料後30筆
# 建立Keras的Sequential模型
model = Sequential()
model.add(Dense(6, input_shape=(4,), activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 訓練模型
print("Training ...")
model.fit(X_train, Y_train, epochs=100, batch_size=5)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test)
print("準確度 = {:.2f}".format(accuracy))
# 儲存Keras模型
print("Saving Model: iris.h5 ...")
model.save("iris.h5")


# 載入模型
model = load_model("iris.h5")

# 編譯模型（此步驟可省略，因為模型已經編譯過）
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

# 預測
print("\nPredicting ...")
Y_pred_prob = model.predict(X_test)  # 預測概率
Y_pred = np.argmax(Y_pred_prob, axis=1)  # 根據最大概率取得預測類別

# 顯示預測結果
print(Y_pred)
Y_target = dataset[:,4][120:].astype(int)
print(Y_target)

# 顯示混淆矩陣
tb = pd.crosstab(Y_target, Y_pred, rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("Ch6_1_3.html")



