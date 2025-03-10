import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

img = np.load("digit8.npy")
sharpen = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
    ]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("original image")

plt.subplot(1, 2, 2)
c_digit = signal.convolve2d(img, sharpen, boundary="symm", mode="same")
plt.imshow(c_digit, cmap="gray")
plt.axis("off")
plt.title("sharpen image")
plt.show()     