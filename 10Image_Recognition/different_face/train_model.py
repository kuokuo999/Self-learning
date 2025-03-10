import cv2
import numpy as np

# 載入人臉追蹤模型
detector = cv2.CascadeClassifier('C:/Users/User/Downloads/different_face/haarcascade_frontalface_default.xml')

# 啟用訓練人臉模型方法
recog = cv2.face.LBPHFaceRecognizer_create()

faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

# 讀取和處理賴清德的照片
for i in range(1, 20):
    img_path = f'face01/{i}.jpg'
    img = cv2.imread(img_path)  # 依序開啟每一張賴清德的照片

    if img is None:
        print(f"Cannot read image {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray, 'uint8')               # 轉換成指定編碼的 numpy 陣列
    faces_detected = detector.detectMultiScale(gray)  # 擷取人臉區域

    if len(faces_detected) == 0:
        print(f"No faces detected in image {img_path}")
    
    for (x, y, w, h) in faces_detected:
        faces.append(img_np[y:y+h, x:x+w])         # 記錄人臉的位置和大小內像素的數值
        ids.append(1)                             # 記錄賴清德人臉對應的 id

# 讀取和處理川普的照片
for i in range(1, 20):
    img_path = f'face02/{i}.jpg'
    img = cv2.imread(img_path)  # 依序開啟每一張川普的照片

    if img is None:
        print(f"Cannot read image {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray, 'uint8')               # 轉換成指定編碼的 numpy 陣列
    faces_detected = detector.detectMultiScale(gray)  # 擷取人臉區域

    if len(faces_detected) == 0:
        print(f"No faces detected in image {img_path}")

    for (x, y, w, h) in faces_detected:
        faces.append(img_np[y:y+h, x:x+w])         # 記錄川普人臉的位置和大小內像素的數值
        ids.append(2)                             # 記錄川普人臉對應的 id

print('Training...')
recog.train(faces, np.array(ids))  # 開始訓練
recog.save('C:/Users/User/Downloads/different_face/face.yml')  # 訓練完成儲存為 face.yml
print('Training complete!')
