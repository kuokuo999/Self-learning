import cv2

# 讀取圖像
o = cv2.imread(r"C:/Users/User/Desktop/1.png")

# 檢查圖像是否讀取成功
if o is None:
    print("無法讀取文件")
else:
    # 顯示原始圖像
    cv2.imshow("original", o)
    
    # 將圖像轉換為灰度圖像
    gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    
    # 二值化處理
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 查找輪廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原始圖像上繪製輪廓
    o = cv2.drawContours(o, contours, -1, (0, 0, 255), 5)
    
    # 顯示結果圖像
    cv2.imshow("result", o)
    
    # 等待按鍵按下
    cv2.waitKey(0)
    
    # 銷毀所有窗口
    cv2.destroyAllWindows()
