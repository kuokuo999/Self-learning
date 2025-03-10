# 以MNIST資料集訓練VAE模型，並生成影像
# 載入相關套件
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from scipy.stats import norm
## 取得 MNIST 訓練資料
# 取得 MNIST 訓練資料
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
x_tr, x_te = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)
print(x_tr.shape, x_te.shape)
## 定義編碼器模型
# 超參數設定
batch_size, n_epoch = 100, 100  # 訓練執行批量、週期
n_hidden, z_dim = 256, 2        # 編碼器隱藏層神經元個數、輸出層神經元個數
# encoder
x = Input(shape=(x_tr.shape[1:]))
x_encoded = Dense(n_hidden, activation='relu')(x)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)

# encoder 後接 Dense，估算平均數 mu
mu = Dense(z_dim)(x_encoded)

# encoder 後接 Dense，估算 log 變異數 log_var
log_var = Dense(z_dim)(x_encoded)
## 定義抽樣函數
# 定義抽樣函數
def sampling(args):
    # 根據 mu, log_var 取隨機亂數
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

# 定義匿名函數，進行抽樣
z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
## 定義解碼器模型
# decoder
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_tr.shape[1], activation='sigmoid')

# 解碼的輸入為匿名函數
z_decoded = z_decoder1(z)
z_decoded = z_decoder2(z_decoded)
y = y_decoder(z_decoded)
## 定義特殊的損失函數(loss)
# 定義特殊的損失函數(loss)
reconstruction_loss = tf.keras.losses.binary_crossentropy(x, y) * x_tr.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

vae = Model(x, y)   # x:MNIST圖像， y:解碼器的輸出
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

# 顯示模型彙總資訊
vae.summary()
## 訓練模型
# 訓練模型
vae.fit(x_tr,
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=(x_te, None), verbose=1)
## 取得編碼器的輸出，以測試資料預測，以編碼器的輸出繪圖
# 取得編碼器的輸出 mu
encoder = Model(x, mu)
encoder.summary()
# 以測試資料預測，以編碼器的輸出繪圖
x_te_latent = encoder.predict(x_te, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te)
plt.colorbar()
plt.show()
## 取得解碼器的輸出，以測試資料預測，以解碼器的輸出圖像
# 取得解碼器的輸出
decoder_input = Input(shape=(z_dim,))
_z_decoded = z_decoder1(decoder_input)
_z_decoded = z_decoder2(_z_decoded)
_y = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()
# 顯示 2D manifold 
n = 15           # 顯示 15x15 視窗
digit_size = 28  # 圖像尺寸
figure = np.zeros((digit_size * n, digit_size * n))

# 
grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# 取得各種機率下的生成的樣本
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()