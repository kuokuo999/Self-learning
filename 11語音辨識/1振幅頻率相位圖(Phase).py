# 振幅(Amplitude)、頻率(Frequency)及相位(Phase)
## 載入相關套件
# 載入相關套件
import numpy as np
import matplotlib.pyplot as plt
## 振幅(Amplitude)及頻率(Frequency)
sample_rate = 100
frequency = 1
audio_length = 3

t = np.linspace(0, audio_length, sample_rate * audio_length)
y = np.sin( 2 * np.pi *frequency * t)
plt.plot(t, y, 'g')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

plt.axhline(0, color = 'blue')

plt.arrow(0.25, 0 , 0, 1, color = 'r', width=0.02, length_includes_head = True)
plt.text(0.3, 0.5, '振幅', fontsize = 14)

plt.arrow(1, 0.1 , 1, 0, color = 'r', width=0.02, length_includes_head = True)
text = plt.text(1.1, 0.25, '波長(Wave length)', fontsize = 14)
