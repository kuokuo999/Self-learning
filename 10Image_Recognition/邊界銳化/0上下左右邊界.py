import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

img = np.load("C:/Users/User/Desktop/self_learning_code/邊界銳化/digit3.npy")
filters = [[
    [-1, -1, -1],
    [ 1,  1,  1],
    [ 0,  0,  0]],
   [[-1,  1,  0],
    [-1,  1,  0],
    [-1,  1,  0]],
   [[ 0,  0,  0],
    [ 1,  1,  1],
    [-1, -1, -1]],
   [[ 0,  1, -1],
    [ 0,  1, -1],
    [ 0,  1, -1]]]

plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("original")

for i in range(2, 6):
    plt.subplot(1, 5, i)
    c = signal.convolve2d(img, filters[i-2], boundary="symm", mode="same")
    plt.imshow(c, cmap="gray")
    plt.axis("off")
    plt.title("filter"+str(i-1))   

plt.show()





