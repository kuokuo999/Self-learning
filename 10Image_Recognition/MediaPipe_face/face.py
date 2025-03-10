import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='C:\\Users\\User\\Downloads\\MediaPipe\\blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()             # 讀取影片的每一幀
        w = frame.shape[1]
        h = frame.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        colorArr = [(255,255,255),(0,0,255)]   # 定義標記人臉的顏色
        faceNum = 0   # 人臉串列位置從 0 開始
        for detection in detection_result.detections:
            if faceNum > 1:
                color = (0,255,255)   # 如果人臉數量大於 2，就都用黃色標記
            else:
                color = colorArr[faceNum]  # 如果人臉數量小於等於 2，使用串列定義的顏色
            bbox = detection.bounding_box
            lx = bbox.origin_x
            ly = bbox.origin_y
            width = bbox.width
            height = bbox.height
            cv2.rectangle(frame,(lx,ly),(lx+width,ly+height),color,5)
            for keyPoint in detection.keypoints:
                cx = int(keyPoint.x*w)
                cy = int(keyPoint.y*h)
                cv2.circle(frame,(cx,cy),10,(0,0,255),-1) 
            faceNum = faceNum + 1      # 人臉數量增加 1
        if not ret:
            print("Cannot receive frame")
            break
        cv2.imshow('oxxostudio', frame)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()