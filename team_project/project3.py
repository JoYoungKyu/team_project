import cv2
import numpy as np
import os
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import timeit

# Roboflow API 설정
try:
    print("JYK.v1i.yolov11-1 모델 로드를 시작합니다...")
    rf = Roboflow(api_key="KLlcHdVtvytxtpDiXA0W")
    project = rf.workspace("joyk").project("jyk-jipji")
    version = project.version(1)
    model = version.model
    print("JYK.v1i.yolov11-1 모델 로드 완료!")
    
    # 모델 설정 확인
    print(f"모델 클래스: {model.classes}")
    print(f"모델 버전: {version.version}")
    print(f"프로젝트 이름: {project.name}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    print("프로그램을 종료합니다.")
    exit(1)

# 모델 설정
CONFIDENCE_THRESHOLD = 0.25  # 신뢰도 임계값
IOU_THRESHOLD = 0.45        # IOU 임계값
MAX_DETECTIONS = 100        # 최대 감지 객체 수

# 클래스 정의
CLASSES = [
    # 차량 종류
    'car', 'bus', 'truck', 'motorcycle', 'bicycle',
    # 사람
    'person',
    # 신호등
    'red_light', 'green_light', 'yellow_light',
    # 횡단보도
    'crosswalk'
]

# 클래스별 색상 정의
COLORS = {
    # 차량 관련
    'car': (0, 255, 0),         # 녹색
    'bus': (0, 200, 0),         # 진한 녹색
    'truck': (0, 150, 0),       # 더 진한 녹색
    'motorcycle': (0, 100, 0),  # 가장 진한 녹색
    'bicycle': (0, 50, 0),      # 가장 진한 녹색
    # 사람
    'person': (255, 0, 255),    # 자홍색
    # 신호등
    'red_light': (0, 0, 255),   # 빨간색
    'green_light': (0, 255, 0), # 녹색
    'yellow_light': (0, 255, 255), # 노란색
    # 횡단보도
    'crosswalk': (255, 255, 255) # 흰색
}

# 클래스 카테고리 정의
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
TRAFFIC_LIGHT_CLASSES = ['red_light', 'green_light', 'yellow_light']

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))  # 기본값: 흰색

# 마우스 이벤트 관련 변수
drawing = False
points = []  # 다각형의 꼭지점을 저장할 리스트
roi_list = []  # 여러 ROI를 저장할 리스트
current_roi = None

def draw_polygon(event, x, y, flags, param):
    global drawing, points, current_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 왼쪽 클릭: 꼭지점 추가
        points.append((x, y))
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 오른쪽 클릭: 현재 다각형 완성
        if len(points) >= 3:  # 최소 3개의 점이 필요
            roi_list.append(points.copy())
        points = []  # 다음 다각형을 위해 초기화
        
    elif event == cv2.EVENT_MBUTTONDOWN:
        # 가운데 버튼 클릭: 마지막 ROI 삭제
        if roi_list:
            roi_list.pop()

def is_point_in_polygon(point, polygon):
    # 점이 다각형 내부에 있는지 확인
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# 영상 캡처
video_path = "video\KakaoTalk_20250414_114947758.mp4"
if not os.path.exists(video_path):
    print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오 캡처를 시작할 수 없습니다.")
    exit(1)

cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', draw_polygon)

# 객체 카운트를 위한 딕셔너리 초기화
object_counts = {class_name: 0 for class_name in CLASSES}

# 객체의 이전 위치와 방향을 저장할 딕셔너리
previous_positions = {}
movement_directions = {}
movement_status = {}

print("프로그램을 시작합니다...")
print("사용 방법:")
print("- 왼쪽 클릭: ROI 꼭지점 추가")
print("- 오른쪽 클릭: ROI 완성")
print("- 가운데 버튼: 마지막 ROI 삭제")
print("- 'q' 키: 프로그램 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("비디오 종료")
        break

    # 객체 카운트 초기화
    object_counts = {class_name: 0 for class_name in CLASSES}
    
    # 현재 프레임의 객체 위치 저장
    current_positions = {}
    
    # JYK.v1i.yolov11-1 객체 감지
    try:
        start_time = timeit.default_timer()
        
        # 모델로 프레임 예측
        results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
        
        # 예측 결과 확인
        if not results:
            print("감지된 객체가 없습니다.")
            continue
            
        # 예측 결과 파싱
        predictions = results.json()['predictions']
        detected_count = len(predictions)
        print(f"감지된 객체 수: {detected_count}")
        
        # 각 예측 결과 처리
        for pred in predictions:
            # 바운딩 박스 좌표 계산
            x1 = int(pred['x'] - pred['width']/2)
            y1 = int(pred['y'] - pred['height']/2)
            x2 = int(pred['x'] + pred['width']/2)
            y2 = int(pred['y'] + pred['height']/2)
            
            # 객체 중심점 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 객체 정보 추출
            confidence = pred['confidence']
            class_name = pred['class']
            
            print(f"감지된 객체: {class_name}, 신뢰도: {confidence:.2f}")
            
            # 유효한 클래스인 경우에만 처리
            if class_name not in CLASSES:
                print(f"알 수 없는 클래스: {class_name}")
                continue
                
            # 객체의 현재 위치 저장
            current_positions[(x1, y1, x2, y2)] = (center_x, center_y)
            
            # 방향 계산
            direction = 0
            box_angle = 0
            status = "normal_entry"
            
            if (x1, y1, x2, y2) in previous_positions:
                direction, box_angle = calculate_direction(
                    (center_x, center_y), 
                    previous_positions[(x1, y1, x2, y2)],
                    (x1, y1, x2, y2)
                )
                
                # 이전 방향과 상태 가져오기
                previous_direction = movement_directions.get((x1, y1, x2, y2))
                previous_status = movement_status.get((x1, y1, x2, y2), "normal_entry")
                
                # 상황 감지
                if class_name.lower() in VEHICLE_CLASSES:
                    status = detect_situation(
                        direction, 
                        box_angle, 
                        previous_direction, 
                        previous_status,
                        class_name,
                        (center_x, center_y)
                    )
                
                # 현재 방향과 상태 저장
                movement_directions[(x1, y1, x2, y2)] = direction
                movement_status[(x1, y1, x2, y2)] = status
            
            # 객체와 상태 표시
            color = get_class_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if class_name.lower() in VEHICLE_CLASSES:
                cv2.putText(frame, f"Status: {status}", 
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 방향 표시 (화살표)
            if direction != 0:
                arrow_length = 30
                end_x = int(center_x + arrow_length * np.cos(direction))
                end_y = int(center_y + arrow_length * np.sin(direction))
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 2)
            
            # 객체 카운트 업데이트
            object_counts[class_name] += 1

        end_time = timeit.default_timer()
        FPS = int(1./(end_time - start_time))

    except Exception as e:
        print(f"객체 감지 중 오류 발생: {str(e)}")
        continue

    # 이전 위치 업데이트
    previous_positions = current_positions

    # 현재 그리는 다각형 표시
    if points:
        for i, point in enumerate(points):
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(frame, points[i-1], point, (255, 0, 0), 2)
        if len(points) > 1:
            cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)

    # 저장된 ROI 표시
    for roi in roi_list:
        if len(roi) >= 3:
            pts = np.array(roi, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 객체 카운트 표시
    count_text_y = 30
    for class_name, count in object_counts.items():
        if count > 0:  # 0개 이상인 클래스만 표시
            color = get_class_color(class_name)
            cv2.putText(frame, f"{class_name}: {count}", 
                        (10, count_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            count_text_y += 30

    # FPS 및 감지된 객체 수 표시
    cv2.putText(frame, f"FPS: {FPS}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Detected: {detected_count}", (frame.shape[1] - 150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('Object Detection', frame)

    # 프레임 간 지연 시간 설정
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")
