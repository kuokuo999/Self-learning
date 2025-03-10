import cv2

# 啟用訓練人臉模型方法
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/User/Downloads/different_face/face.yml')  # 讀取人臉模型檔

# 載入人臉追蹤模型
cascade_path = 'C:/Users/User/Downloads/different_face/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)  # 啟用人臉追蹤

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(img, (540, 300))  # 縮小尺寸，加快辨識效率
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉 (目的在於標記出外框)

    # 建立姓名和 id 的對照表
    name = {
        '1': '歐巴馬',
        '2': '川普',
        '3': '未知'
    }

    # 依序判斷每張臉屬於哪個 id
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 標記人臉外框
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])  # 取出 id 號碼以及信心指數 confidence
        if confidence < 80:  # confidence 值越小表示識別越準確
            text = name.get(str(idnum), '未知')  # 如果信心指數小於 80，取得對應的名字
        else:
            text = '未知'  # 否則名字顯示未知
        # 在人臉外框旁加上名字
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(5) == ord('q'):
        break  # 按下 q 鍵停止

cap.release()
cv2.destroyAllWindows()
