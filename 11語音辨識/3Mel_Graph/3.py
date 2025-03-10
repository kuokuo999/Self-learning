# 載入相關套件
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 載入音訊檔案
wav_file = 'C:/Users/User/Desktop/語音辨識/3Mel_Graph/dog.wav'
data, sr = librosa.load(wav_file)  # 預設取樣頻率 22.05K

# 顯示音訊資訊
print(f'取樣頻率={sr}, 總樣本數={data.shape}')

# 高頻強調（Pre-emphasis）處理
data_preemph = librosa.effects.preemphasis(data)

# 計算梅爾頻譜圖（Mel-Spectrogram）
mel_spec = librosa.feature.melspectrogram(y=data_preemph, sr=sr)

# 將梅爾頻譜圖轉換為對數刻度
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# 繪製梅爾頻譜圖
plt.figure(figsize=(10, 6))
librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram with Pre-emphasis')
plt.tight_layout()
plt.show()
#Log: 使用對數刻度展示能量大小。
#Mel Spectrogram: 基於梅爾刻度的頻譜表示。
#Pre-emphasis: 在計算頻譜前應用了高頻強調處理。