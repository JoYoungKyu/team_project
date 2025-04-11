import cv2
import numpy as np
import torch
from ultralytics import YOLO

# YOLO11n 모델 로드
model = YOLO('yolo11n.pt')  # YOLO11n 모델 로드

# YOLO11n 모델 설정
CONFIDENCE_THRESHOLD = 0.25  # 신뢰도 임계값
IOU_THRESHOLD = 0.45        # IOU 임계값
MAX_DETECTIONS = 100        # 최대 감지 객체 수

# YOLO11n 클래스 정의
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'truck', 'traffic light']

# 클래스별 색상 정의
COLORS = {
    'person': (255, 0, 0),      # 파란색
    'vehicle': (0, 255, 0),     # 녹색 (차량 관련)
    'traffic light': (0, 255, 255)  # 청록색
}

# 클래스 카테고리 정의
VEHICLE_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'truck']

def get_class_color(class_name):
    if class_name == 'person':
        return COLORS['person']
    elif class_name in VEHICLE_CLASSES:
        return COLORS['vehicle']
    elif class_name == 'traffic light':
        return COLORS['traffic light']
    else:
        return (0, 0, 255)  # 빨간색 (기본)

# 마우스 이벤트 관련 변수
drawing = False
roi_points = []
roi_rect = None
roi_list = []  # 여러 ROI를 저장할 리스트
current_roi = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, roi_points, roi_rect, current_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_rect = (roi_points[0][0], roi_points[0][1], x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_rect = (roi_points[0][0], roi_points[0][1], x, y)
        if roi_rect:
            roi_list.append(roi_rect)
        roi_points = []
        roi_rect = None
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 마우스 오른쪽 버튼으로 ROI 삭제
        for i, roi in enumerate(roi_list):
            x1, y1, x2, y2 = roi
            if (x1 <= x <= x2 and y1 <= y <= y2) or (x2 <= x <= x1 and y2 <= y <= y1):
                roi_list.pop(i)
                break

# 영상 캡처
cap = cv2.VideoCapture("video\KakaoTalk_20250411_084730901.mp4")
cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', draw_rectangle)

# 객체 카운트를 위한 딕셔너리
object_counts = {class_name: 0 for class_name in CLASSES}

# FPS 계산을 위한 변수
fps = 0
frame_count = 0
start_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS 계산
    frame_count += 1
    if frame_count % 30 == 0:
        end_time = cv2.getTickCount()
        fps = 30 / ((end_time - start_time) / cv2.getTickFrequency())
        start_time = end_time

    # ROI 영역 표시
    for roi in roi_list:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        roi_points = roi_points.reshape((-1, 1, 2))

    # 현재 그리는 ROI 표시
    if roi_rect is not None:
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # YOLO11n 객체 감지
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS)
    
    # 객체 카운트 초기화
    object_counts = {class_name: 0 for class_name in CLASSES}
    
    # 감지된 객체 표시
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else None

            # 유효한 클래스인 경우에만 처리
            if class_name is None or class_name not in CLASSES:
                continue

            # 각 ROI에 대해 객체가 포함되어 있는지 확인
            for roi in roi_list:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                roi_points = np.array([[roi_x1, roi_y1], [roi_x2, roi_y1], 
                                     [roi_x2, roi_y2], [roi_x1, roi_y2]], np.int32)
                roi_points = roi_points.reshape((-1, 1, 2))
                
                # 객체의 중심점이 ROI 내에 있는지 확인
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                    # ROI 내의 객체만 표시
                    color = get_class_color(class_name)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 객체 카운트 업데이트
                    object_counts[class_name] += 1

    # ROI별 객체 카운트 표시
    y_offset = 30
    for roi_idx, roi in enumerate(roi_list):
        cv2.putText(frame, f"ROI {roi_idx + 1} Statistics:", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        
        # 카테고리별 통계 표시
        categories = {
            'People': ['person'],
            'Vehicles': VEHICLE_CLASSES,
            'Traffic Lights': ['traffic light']
        }
        
        for category, classes in categories.items():
            count = sum(object_counts[class_name] for class_name in classes)
            if count > 0:
                color = get_class_color(classes[0])
                cv2.putText(frame, f"{category}: {count}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30

    # FPS 표시
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면 크기를 640x320으로 조정
    frame = cv2.resize(frame, (640, 640))

    # 결과 표시
    cv2.imshow('Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
