import cv2
import numpy as np
import os
from roboflow import Roboflow

# Roboflow API 설정
rf = Roboflow(api_key="KLlcHdVtvytxtpDiXA0W")
project = rf.workspace("joyk").project("jyk-jipji")
version = project.version(1)
dataset = version.download("yolov11")

# 모델 설정
CONFIDENCE_THRESHOLD = 0.25  # 신뢰도 임계값
IOU_THRESHOLD = 0.45        # IOU 임계값

# 클래스 정의
CLASSES = [
    # 차량 종류
    'car', 'bus', 'truck', 'motorcycle',
    # 사람
    'person',
    # 신호등
    'red_light', 'green_light', 'yellow_light',
    # 횡단보도
    'crosswalk',
    # 행위/상황
    'violation_redlight', 'wrong_way_entry', 'entering_sidewalk',
    'illegal_u_turn', 'blocking_intersection', 'conflict_pedestrian',
    'normal_entry'
]

# 클래스별 색상 정의
COLORS = {
    # 차량 관련
    'car': (0, 255, 0),         # 녹색
    'bus': (0, 200, 0),         # 진한 녹색
    'truck': (0, 150, 0),       # 더 진한 녹색
    'motorcycle': (0, 100, 0),  # 가장 진한 녹색
    # 사람
    'person': (255, 0, 255),    # 자홍색
    # 신호등
    'red_light': (0, 0, 255),   # 빨간색
    'green_light': (0, 255, 0), # 녹색
    'yellow_light': (0, 255, 255), # 노란색
    # 횡단보도
    'crosswalk': (255, 255, 255), # 흰색
    # 행위/상황
    'violation_redlight': (255, 0, 0),    # 빨간색
    'wrong_way_entry': (255, 0, 0),       # 빨간색
    'entering_sidewalk': (255, 0, 0),     # 빨간색
    'illegal_u_turn': (255, 0, 0),        # 빨간색
    'blocking_intersection': (255, 0, 0), # 빨간색
    'conflict_pedestrian': (255, 0, 0),   # 빨간색
    'normal_entry': (0, 255, 0)           # 녹색
}

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))  # 기본값: 흰색

# 영상 캡처
video_path = "video\KakaoTalk_20250414_114935256.mp4"
if not os.path.exists(video_path):
    print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오 캡처를 시작할 수 없습니다.")
    exit(1)

cv2.namedWindow('Object Detection')

# 객체 카운트를 위한 딕셔너리 초기화
object_counts = {class_name: 0 for class_name in CLASSES}

# 이전 프레임의 객체 위치 저장
previous_positions = {}

# 프레임 카운터 초기화
frame_count = 0

def calculate_direction(prev_pos, curr_pos):
    """객체의 이동 방향을 계산"""
    if prev_pos is None:
        return None
    
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    
    # 방향 계산 (라디안)
    angle = np.arctan2(dy, dx)
    return angle

def analyze_behavior(class_name, direction, position, frame_shape):
    """객체의 행동을 분석"""
    behaviors = []
    
    # 차량 관련 행동 분석
    if class_name in ['car', 'bus', 'truck', 'motorcycle']:
        # 역주행 감지
        if direction is not None and abs(direction) > np.pi/2:
            behaviors.append('wrong_way_entry')
        
        # 인도 침입 감지 (프레임 상단 20% 영역)
        if position[1] < frame_shape[0] * 0.2:
            behaviors.append('entering_sidewalk')
    
    # 보행자 관련 행동 분석
    elif class_name == 'person':
        # 보차 충돌 감지
        if any(obj_class in ['car', 'bus', 'truck', 'motorcycle'] 
               for obj_class in previous_positions.keys()):
            behaviors.append('conflict_pedestrian')
    
    return behaviors

print("프로그램을 시작합니다...")
print("'q' 키를 누르면 프로그램이 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("비디오 종료")
        break

    # 프레임 카운트 증가
    frame_count += 1
    
    # 객체 카운트 초기화
    object_counts = {class_name: 0 for class_name in CLASSES}
    
    # 행위/상황 카운트 초기화
    behavior_counts = {behavior: 0 for behavior in CLASSES if behavior in [
        'violation_redlight', 'wrong_way_entry', 'entering_sidewalk',
        'illegal_u_turn', 'blocking_intersection', 'conflict_pedestrian',
        'normal_entry'
    ]}
    
    try:
        # 객체 감지
        results = version.model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
        
        if results:
            predictions = results.json()['predictions']
            
            # 객체 표시
            for pred in predictions:
                x = int(pred['x'] - pred['width']/2)
                y = int(pred['y'] - pred['height']/2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                class_name = pred['class'].lower()  # 클래스 이름을 소문자로 변환
                
                # bus, truck 클래스는 90% 이상의 신뢰도가 있는 경우에만 표시
                if class_name in ['bus', 'truck'] and confidence < 0.9:
                    continue
                
                # person, motorcycle, bicycle 클래스는 50% 이상의 신뢰도가 있는 경우에만 표시
                if class_name in ['person', 'motorcycle', 'bicycle'] and confidence < 0.5:
                    continue
                
                if class_name in CLASSES:
                    # 객체의 현재 위치
                    current_pos = (x + w/2, y + h/2)
                    
                    # 방향 계산
                    prev_pos = previous_positions.get(class_name)
                    direction = calculate_direction(prev_pos, current_pos)
                    
                    # 행동 분석
                    behaviors = analyze_behavior(class_name, direction, current_pos, frame.shape)
                    
                    # 행위/상황 카운트 업데이트
                    for behavior in behaviors:
                        if behavior in behavior_counts:
                            behavior_counts[behavior] += 1
                    
                    # 객체 표시
                    color = get_class_color(class_name)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 방향 화살표 그리기
                    if direction is not None:
                        arrow_length = 30
                        end_x = int(current_pos[0] + arrow_length * np.cos(direction))
                        end_y = int(current_pos[1] + arrow_length * np.sin(direction))
                        cv2.arrowedLine(frame, (int(current_pos[0]), int(current_pos[1])),
                                      (end_x, end_y), color, 2)
                    
                    # 클래스 이름, 신뢰도, 행동 표시
                    label = f"{class_name}: {confidence:.2f}"
                    if behaviors:
                        label += f" ({', '.join(behaviors)})"
                    
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # 라벨 배경 그리기
                    cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), color, -1)
                    
                    # 라벨 텍스트 그리기
                    cv2.putText(frame, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 객체 카운트 업데이트
                    object_counts[class_name] += 1
                    
                    # 현재 위치 저장
                    previous_positions[class_name] = current_pos

    except Exception as e:
        print(f"객체 감지 중 오류 발생: {str(e)}")
        continue

    # 화면 상단에 정보 표시
    info_y = 30
    line_height = 30
    
    # 프레임 수 표시
    cv2.putText(frame, f"Frame: {frame_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    
    # 객체 수 표시
    total_objects = sum(object_counts.values())
    cv2.putText(frame, f"Objects: {total_objects}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    
    # 행위/상황 수 표시
    total_behaviors = sum(behavior_counts.values())
    cv2.putText(frame, f"Behaviors: {total_behaviors}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    
    # 각 행위/상황별 수 표시
    for behavior, count in behavior_counts.items():
        if count > 0:
            cv2.putText(frame, f"{behavior}: {count}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            info_y += line_height

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