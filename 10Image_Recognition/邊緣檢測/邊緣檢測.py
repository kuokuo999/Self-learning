import cv2
import os

# 將文件路徑更改為您複製的路徑
image_path = r"C:/Users/User/Desktop/1.jpg"

# 打印文件路徑以確保路徑格式正確
print(f"文件路徑: {image_path}")

# 檢查文件是否存在
if not os.path.exists(image_path):
    print(f"文件路徑錯誤或文件不存在: {image_path}")
else:
    # 讀取灰度圖像
    o = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 檢查圖像是否讀取成功
    if o is None:
        print(f"無法讀取文件: {image_path}")
    else:
        # 邊緣檢測
        r1 = cv2.Canny(o, 128, 200)
        r2 = cv2.Canny(o, 32, 128)

        # 顯示圖像
        cv2.imshow("original", o)
        cv2.imshow("result_r1", r1)
        cv2.imshow("result_r2", r2)

        # 等待按鍵按下
        cv2.waitKey(0)

        # 銷毀所有窗口
        cv2.destroyAllWindows()
