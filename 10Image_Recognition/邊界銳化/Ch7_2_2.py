import numpy as np
import matplotlib.pyplot as plt

img = np.load("digit8.npy")

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
   