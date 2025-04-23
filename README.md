## 우리인재개발원(우리컴퓨터아카데미)
```
교차로 교통 장애물 및 이벤트 감지 시스템 개발
```
## 우리인재 팀
깃허브 링크 : https://github.com/joyoungkyu
```
팀장 : 조영규 
팀원 : 김우혁, 유혜정, 정날빛
```
## 프로젝트 주제
```
교차로 돌발진입 포착
```
## 타임테이블
```
타임테이블 . . . 
```
## 프로젝트 진행
![Image](https://github.com/user-attachments/assets/7f57d601-8528-42d8-817a-8e7887fecec1)
### 데이터수집
![이미지](https://github.com/user-attachments/assets/4258d17c-e76c-4dfa-8044-170fd6362205)

야간 :
![야간 이미지](https://github.com/user-attachments/assets/8d9de2d7-6b4b-400b-b1e2-a745b9241d55)
```
참고자료 : https://docs.ultralytics.com/
https://www.its.go.kr/opendata/
https://www.roboflow.com/
```
### 데이터검증
자동라벨링 버튼사진 :
![4](https://github.com/user-attachments/assets/0c3a0d0a-7643-4566-ad50-7c0d7afb04a6)

자동라벨링 과정사진 :
![2](https://github.com/user-attachments/assets/d5c0e756-5b7b-4f9f-a70a-0fc25fb6ad71)

결과사진 :
![결과이미지](https://github.com/user-attachments/assets/81a2b7d4-60f2-4842-ad8c-a3c1fd8d2de8)
### 데이터 전처리
데이터셋 사진 :

![4](https://github.com/user-attachments/assets/401e62ab-7730-4512-8b93-384dc9fc5a9f)

비전 제작 :

![캡처](https://github.com/user-attachments/assets/b34e5fd7-24b2-466e-8935-fb2a32ad3bbe)

```
Train : Vaild : Test =  8 : 1 : 1
```
### 모델 학습 및 튜닝
모델 코드 사진 :

![코드](https://github.com/user-attachments/assets/3f495da8-2da4-4a2d-b114-0d2b4d42c5ab)

훈련 결과 그래프 :

![그래프](https://github.com/user-attachments/assets/7e8cdeff-bbd8-47e9-90dc-6d29592f7c65)

![그래프(상세)](https://github.com/user-attachments/assets/deb2a836-683c-4be6-8c74-af84123567cb)

![클래스](https://github.com/user-attachments/assets/36b09dd6-dbda-42c4-925c-30ecbbcad6b1)

![KakaoTalk_20250416_092042011](https://github.com/user-attachments/assets/f8b4c1e1-c346-4a53-8afe-bf1621dff422)

컨피던스 메트릭스 결과(제작한 모델) :

![Figure_1-3](https://github.com/user-attachments/assets/36ff73e0-2c2d-4bb3-9066-35795ce7ad0e)

![Figure_2-3](https://github.com/user-attachments/assets/fa9a1c13-a384-4717-890e-462df77d6b69)

### 모델 분석 및 검증
YOLO11모델 :

![캡처](https://github.com/user-attachments/assets/8609560a-b575-440c-a570-bb6bfdc36c3f)

실제 재작 모듈 :

![image](https://github.com/user-attachments/assets/561e2eea-0ecb-4c46-9024-fb199d85adcc)

```
둘다 실행 결과를 캡처한 것으로 둘의 차이점을 비교본으로 개시
```
### 모델 배포
실시간 영상 : PPT링크 참고

데시보드 연동 이미지 (직접 만든 모델) :

![dashboard](https://github.com/user-attachments/assets/1e7f5e09-b6b0-4e69-9c42-d73d9764799c) ![dashboard_result](https://github.com/user-attachments/assets/f0438397-0f04-4343-a0b4-5900cd7afeab) ![제작모델 보드이미지](https://github.com/user-attachments/assets/7fc6b2bf-a70b-4581-aeba-2d40a212d0bb) ![제작모델 보드이미지1](https://github.com/user-attachments/assets/dbd629dd-504d-4a71-bfd0-de7d61e21a2b) ![제작모델 보드이미지2](https://github.com/user-attachments/assets/4c9fe4fb-5446-4830-8a1a-11bbb9ee9881) ![제작모델 보드이미지3](https://github.com/user-attachments/assets/eeac66d1-339a-4d87-b53b-2833c41931fc) ![제작모델 보드이미지4](https://github.com/user-attachments/assets/ae463b59-f0b2-4663-983d-de005abc9370) ![제작모델 보드이미지5](https://github.com/user-attachments/assets/63680ce9-9f82-4afc-837c-4118740d56e2)

코드 (직접 만든 모델) :

https://github.com/JoYoungKyu/team_project/blob/main/dashboard/violation_dashboard2.py

코드 (YOLO11n.pt모델) :

https://github.com/JoYoungKyu/team_project/blob/main/dashboard/app.py

데시보드 연동 이미지 (YOLO11n.pt모델) :

![대시보드](https://github.com/user-attachments/assets/0992c478-e752-47b0-9a0b-c97fc2441b88)

텐서보드 (직접 만든 모델) :

![2025-04-23 14 03 09](https://github.com/user-attachments/assets/22aff234-3694-44ec-bf76-9141b3318260)

![제작모델 텐서보드이미지](https://github.com/user-attachments/assets/49211271-305a-4263-a07f-f33c4fcd7a23)

![제작모델 텐서보드이미지1](https://github.com/user-attachments/assets/75a0c8d2-d189-4d77-8afe-49866f9de6b1)

### 피드백
```
컴퓨전매트릭스 시도(목요일에 예정),초기모델보다 더 정확하게 훈련시킨 모델사용, 영역지정은 제외, 벡터값을 기준으로 불법유턴같은 돌발상황 포착 사람과 차량의 거리측정해서 가까우면 위험 알리기 (경고음) 이용
```
알림 화면 :

![2025-04-23 13 59 55](https://github.com/user-attachments/assets/d13d90aa-52dc-4374-a794-946c061a9724)

### 향후계획
1. 인도 및 차도 구분
2. 차량과 차량 사이의 거리 계산 및 알림 서비스
3. 여러 종류의 차량 데이터 수집 및 가공
4. 여러 종류의 동물 데이터 수집 및 가공
5. 라벨링 신뢰도 조정

## PPT 자료
JoyK_교차로 돌발 진입 감지.pdf
