import cv2

# 创建一个空的追踪器列表
trackers = []
tracking = False
colors = [(0, 0, 255), (0, 255, 255)]  # 创建框框颜色列表

cap = cv2.VideoCapture(0)  # 读取摄像头
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    frame = cv2.resize(frame, (400, 230))  # 缩小尺寸加快速度
    keyName = cv2.waitKey(50)

    if keyName == ord('q'):
        break

    # 按下 a 键开始标记物体框
    if keyName == ord('a'):
        trackers = []  # 清空之前的追踪器
        for i in range(2):
            area = cv2.selectROI('oxxostudio', frame, showCrosshair=False, fromCenter=False)
            tracker = cv2.TrackerKCF_create()  # 使用 TrackerKCF 创建追踪器
            trackers.append((tracker, area))  # 保存追踪器及其区域
        tracking = True

    if tracking:
        for tracker, area in trackers:
            if not tracker.init(frame, area):  # 初始化追踪器
                print("Failed to initialize tracker.")
            success, box = tracker.update(frame)
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                color = colors[trackers.index((tracker, area)) % len(colors)]
                cv2.rectangle(frame, p1, p2, color, 3)

    cv2.imshow('oxxostudio', frame)

cap.release()
cv2.destroyAllWindows()
