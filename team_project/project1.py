import cv2
import numpy as np
from roboflow import Roboflow

# Roboflow 초기화
rf = Roboflow(api_key="tEFOW2xP2wtvNyZmpXnL")
project = rf.workspace("jyk-ucnhk").project("jyk")
version = project.version(11)
model = version.model

# 마우스 이벤트 관련 변수
drawing = False
roi_points = []
roi_rect = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, roi_points, roi_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_rect = (roi_points[0][0], roi_points[0][1], x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_rect = (roi_points[0][0], roi_points[0][1], x, y)
        roi_points = []

# 영상 캡처
cap = cv2.VideoCapture(0)
cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 영역 표시
    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        roi_points = roi_points.reshape((-1, 1, 2))

    # YOLO 객체 감지
    predictions = model.predict(frame, confidence=40, overlap=30).json()
    
    # 감지된 객체 표시
    for prediction in predictions['predictions']:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        class_name = prediction['class']
        confidence = prediction['confidence']

        # 객체가 ROI 내에 있는지 확인
        if roi_rect is not None and cv2.pointPolygonTest(roi_points, (x, y), False) >= 0:
            # ROI 내의 객체만 표시
            cv2.rectangle(frame, (x - width//2, y - height//2), 
                         (x + width//2, y + height//2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                       (x - width//2, y - height//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 결과 표시
    cv2.imshow('Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
