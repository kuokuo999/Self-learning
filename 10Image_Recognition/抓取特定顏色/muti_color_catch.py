import cv2
import numpy as np

# 設定紅色的HSV範圍
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 設定藍色的HSV範圍
blue_lower = np.array([90, 100, 100])
blue_upper = np.array([140, 255, 255])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    img = cv2.resize(img, (640, 360))

    # 轉換圖像為HSV色彩空間
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 紅色範圍檢測
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_output = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    red_output = cv2.dilate(red_output, kernel)
    red_output = cv2.erode(red_output, kernel)

    contours, hierarchy = cv2.findContours(red_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        color = (0, 0, 255)  # 紅色邊框
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    # 藍色範圍檢測
    blue_output = cv2.inRange(hsv_img, blue_lower, blue_upper)
    blue_output = cv2.dilate(blue_output, kernel)
    blue_output = cv2.erode(blue_output, kernel)

    contours, hierarchy = cv2.findContours(blue_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        color = (255, 255, 0)  # 藍色邊框
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
