## 🎁우리인재개발원(우리컴퓨터아카데미)
```
교차로 교통 장애물 및 이벤트 감지 시스템 개발
```

## 🎁우리인재 팀
깃허브 링크 : https://github.com/joyoungkyu

```
팀장 : 조영규 
팀원 : 김우혁, 유혜정, 정날빛
```


## 🎁프로젝트 주제
```
교차로 돌발진입 포착
```

## 🎁타임테이블
```
타임테이블 . . . 
```

## 🎁프로젝트 진행
![Image](https://github.com/user-attachments/assets/7f57d601-8528-42d8-817a-8e7887fecec1)

### 데이터수집
![C-221008_13_CR06_01_A0645](https://github.com/user-attachments/assets/9b2732e5-5a2d-4626-98f8-9c20a711d37f)

야간 : ![캡처](https://github.com/user-attachments/assets/ed4e4bd4-4650-441f-aa07-3d064f06821f)

참고자료 : https://docs.ultralytics.com/

https://www.its.go.kr/opendata/

https://www.roboflow.com/

### 데이터검증
자동라벨링 버튼사진 : ![2025-04-16 16 46 31](https://github.com/user-attachments/assets/5f673aea-1d1d-449d-a24b-0018c3b4b3b9)

자동라벨링 과정사진 : ![class](https://github.com/user-attachments/assets/1bc6035e-bef5-4d1a-b889-32c17b62fc46)

결과사진 : ![캡처](https://github.com/user-attachments/assets/b8f0ec96-9fdd-4b30-9ad4-dbdd1d307afe)

### 데이터 전처리
데이터셋 사진 : ![KakaoTalk_20250415_175443746](https://github.com/user-attachments/assets/e68cfec2-9211-486d-a619-8cf8492f3923)

비전 제작 : ![캡처](https://github.com/user-attachments/assets/b34e5fd7-24b2-466e-8935-fb2a32ad3bbe)
```
Train : Vaild : Test =  8 : 1 : 1
```
### 모델 학습 및 튜닝
모델 코드 사진 : ![KakaoTalk_20250416_094056361](https://github.com/user-attachments/assets/3b5d50cd-fd9e-4e86-913c-4204ae693cd1)

훈련 결과 그래프 : ![KakaoTalk_20250416_091435673](https://github.com/user-attachments/assets/6fcd1797-41c6-4d50-bf3a-d82547ccd852)

![KakaoTalk_20250416_091511983](https://github.com/user-attachments/assets/d26edcd1-71a6-4efe-817c-870b27691a18)

![KakaoTalk_20250416_092031604](https://github.com/user-attachments/assets/1879875e-ba69-43c7-af0c-360f1e3377bf)

![KakaoTalk_20250416_092042011](https://github.com/user-attachments/assets/f8b4c1e1-c346-4a53-8afe-bf1621dff422)

### 모델 분석 및 검증

YOLO11모델 : ![캡처](https://github.com/user-attachments/assets/8609560a-b575-440c-a570-bb6bfdc36c3f)

실제 재작 모듈 : ![image](https://github.com/user-attachments/assets/561e2eea-0ecb-4c46-9024-fb199d85adcc)

```
둘다 실행 결과를 캡처한 것으로 둘의 차이점을 비교본으로 개시
```
### 모델 배포
실시간 영상 : PPT링크 참고
데시보드 연동 이미지 (직접 만든 모델) : ![dashboard](https://github.com/user-attachments/assets/1e7f5e09-b6b0-4e69-9c42-d73d9764799c) ![dashboard_result](https://github.com/user-attachments/assets/f0438397-0f04-4343-a0b4-5900cd7afeab)

코드 (직접 만든 모델) : https://github.com/JoYoungKyu/team_project/blob/main/dashboard/violation_dashboard.py

데시보드 연동 이미지 (YOLO11n.pt모델) : ![yolo11](https://github.com/user-attachments/assets/ebf5d643-d533-44b6-a161-d71573c572fe)

코드 (YOLO11n.pt모델) : https://github.com/JoYoungKyu/team_project/blob/main/dashboard/app.py

텐서보드 (직접 만든 모델) : ![KakaoTalk_20250416_173344730](https://github.com/user-attachments/assets/5cec6e6c-2832-4641-8c6a-2b82747badc1)

### 피드백
```
컴퓨전매트릭스 시도(목요일에 예정),초기모델보다 더 정확하게 훈련시킨 모델사용, 영역지정은 제외, 벡터값을 기준으로 불법유턴같은 돌발상황 포착 사람과 차량의 거리측정해서 가까우면 위험 알리기 (경고음) 이용, 인도영역을 지적하는 문제 해결
```
### 향후계획
앞으로 더 보강해서 위기상황시 CCTV상황실에 직접적으로 알림이 가는 기능도 추가

## 🎁PPT 자료
```
PPT 링크 및 파일 . . .
```
