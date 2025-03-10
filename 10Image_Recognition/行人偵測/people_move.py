import cv2

# 創建攝像頭對象 (0 代表默認的攝像頭)
cap = cv2.VideoCapture(0)

# 讀取行人偵測的 Haar 特徵分類器
car_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

while True:
    # 從攝像頭讀取一幀
    ret, frame = cap.read()
    
    if not ret:
        print("無法讀取鏡頭影像")
        break
    
    # 將幀轉換為灰階影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # 模糊化以去除雜訊
    
    # 偵測行人
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    
    # 繪製偵測到的行人外框
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 顯示結果
    cv2.imshow('oxxostudio', frame)
    
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉所有窗口
cap.release()
cv2.destroyAllWindows()
