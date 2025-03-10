import cv2

# 開啟攝像頭
cap = cv2.VideoCapture(0)  # 0 代表第一個攝像頭，如果有多個攝像頭可以使用 1, 2 等等來選擇

# 讀取汽車模型
car = cv2.CascadeClassifier("cars.xml")

while True:
    # 從攝像頭讀取每一幀
    ret, img = cap.read()
    
    if not ret:
        print("未能讀取影像，請檢查攝像頭連接")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白影像
    gray = cv2.medianBlur(gray, 5)  # 模糊化去除雜訊
    
    # 偵測汽車
    cars = car.detectMultiScale(gray, 1.1, 3)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 繪製外框
    
    # 顯示結果
    cv2.imshow('oxxostudio', img)
    
    # 按下 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
