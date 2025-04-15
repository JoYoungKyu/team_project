import cv2
import numpy as np
import os
import winsound
from roboflow import Roboflow
from collections import deque

# Roboflow API 연결
rf = Roboflow(api_key="KLlcHdVtvytxtpDiXA0W")
project = rf.workspace("joyk").project("jyk-jipji")
version = project.version(2)
model = version.model

# 감지 설정
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DISAPPEARED = 30  # 추적이 사라졌다고 간주할 프레임 수

# 클래스 정의
CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'person', 'red_light', 'green_light', 'yellow_light', 'crosswalk',
           'violation_redlight', 'wrong_way_entry', 'entering_sidewalk', 'illegal_u_turn', 'blocking_intersection',
           'conflict_pedestrian', 'normal_entry']

# 방향 표시 제외 클래스
NO_VECTOR_CLASSES = ['person', 'red_light', 'green_light', 'yellow_light', 'crosswalk']

# 클래스별 색상 지정
COLORS = {
    'car': (0, 255, 0), 'bus': (0, 200, 0), 'truck': (0, 150, 0), 'motorcycle': (0, 100, 0),
    'person': (255, 0, 255),
    'red_light': (0, 0, 255), 'green_light': (0, 255, 0), 'yellow_light': (0, 255, 255),
    'crosswalk': (255, 255, 255),
    'violation_redlight': (0, 0, 255), 'wrong_way_entry': (0, 100, 255),
    'entering_sidewalk': (255, 100, 0), 'illegal_u_turn': (255, 0, 100),
    'blocking_intersection': (200, 0, 100), 'conflict_pedestrian': (150, 0, 255),
    'normal_entry': (0, 255, 0)
}

# 클래스 이름에 맞는 색상 반환
def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))

# 객체 중심점 추적기 정의
class CentroidTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_object_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.tracks = {}  # 각 객체의 중심점 경로 저장

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = deque(maxlen=10)
        self.tracks[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.tracks[object_id]

    def update(self, input_centroids):
        # 감지된 중심점이 없는 경우: 사라졌다고 판단
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects, self.tracks

        # 처음 등록
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # 기존 객체들과 새 객체들 거리 비교
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(obj_centroids)[:, None] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.tracks[obj_id].append(input_centroids[col])
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # 새로 등장한 객체 등록
            unused_cols = set(range(0, len(input_centroids))).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

            # 사라진 객체 체크
            unused_rows = set(range(0, len(obj_centroids))).difference(used_rows)
            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        return self.objects, self.tracks

# 방향 계산 (벡터 각도)
def compute_direction(track):
    if len(track) < 2:
        return None
    dx = track[-1][0] - track[0][0]
    dy = track[-1][1] - track[0][1]
    return np.arctan2(dy, dx)

# 두 점 사이 거리 계산
def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 비디오 불러오기
video_path = "video/KakaoTalk_20250415_123136238.mp4"
if not os.path.exists(video_path):
    print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Violation Detection with Alert')

# 마우스 클릭으로 기준 벡터 설정 (초기화 기능)
click_points = []
base_vectors = []
selecting_base = True

# 마우스 이벤트 콜백
def click_event(event, x, y, flags, param):
    global click_points, base_vectors, selecting_base
    if selecting_base and event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        if len(click_points) % 2 == 0:
            pt1, pt2 = click_points[-2], click_points[-1]
            base_vectors.append(np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]]))
    elif event == cv2.EVENT_RBUTTONDOWN:
        selecting_base = False

cv2.setMouseCallback('Violation Detection with Alert', click_event)

# 클래스별 객체 추적기 생성
trackers = {
    cls: CentroidTracker() for cls in ['car', 'bus', 'truck', 'motorcycle', 'person',
                                       'violation_redlight', 'wrong_way_entry', 'entering_sidewalk', 
                                       'illegal_u_turn', 'blocking_intersection', 'conflict_pedestrian', 
                                       'normal_entry']
}

# 프레임 반복 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    alarm_triggered = False  # 경고 여부

    try:
        # 객체 감지 수행
        results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
        predictions = results.json()['predictions']

        # 빨간 신호등 감지 여부
        red_light_on = any(pred['class'].lower() == 'red_light' for pred in predictions)
        detections = {cls: [] for cls in trackers}

        # 감지된 객체 정보 파싱
        for pred in predictions:
            class_name = pred['class'].lower()
            confidence = pred['confidence']

            # 버스, 트럭은 더 높은 신뢰도 요구
            if class_name in ['bus', 'truck'] and confidence < 0.9:
                continue
            if confidence < 0.2:
                continue

            # 바운딩 박스 좌표 계산
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            w = int(pred['width'])
            h = int(pred['height'])
            center = (x + w // 2, y + h // 2)

            color = get_class_color(class_name)
            label = f"{class_name} {confidence:.2f}"

            # 사각형 및 클래스 라벨 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if class_name in trackers:
                detections[class_name].append(center)

        # 추적기 업데이트 및 라벨링
        obj_positions = {}
        for cls, tracker in trackers.items():
            objects, tracks = tracker.update(detections[cls])
            obj_positions[cls] = objects

            for object_id, center in objects.items():
                track = tracks[object_id]
                direction = compute_direction(track)
                color = get_class_color(cls)

                # 이동 방향 화살표 그리기
                if cls != 'person':
                    cv2.circle(frame, center, 4, color, -1)
                    if direction:
                        dx = int(30 * np.cos(direction))
                        dy = int(30 * np.sin(direction))
                        cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy), color, 2)

                # 라벨(ID 포함) 표시
                label = f"{cls} ID:{object_id}"
                cv2.putText(frame, label, (center[0] - 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 보행자와 차량 간 거리 계산 → 충돌 판단
        for p in obj_positions['person'].values():
            for vehicle_cls in ['car', 'bus', 'truck', 'motorcycle']:
                for v in obj_positions[vehicle_cls].values():
                    if compute_distance(p, v) < 80:
                        cv2.line(frame, p, v, (0, 0, 255), 2)
                        cv2.putText(frame, "conflict_pedestrian", (p[0], p[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        alarm_triggered = True

    except Exception as e:
        print(f"오류: {e}")

    # 경고음 재생
    if alarm_triggered:
        winsound.Beep(1000, 200)

    # ✅ 출력 크기 조정 (640x320)
    resized_frame = cv2.resize(frame, (640, 320))
    cv2.imshow('Violation Detection with Alert', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
