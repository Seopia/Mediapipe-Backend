# Flask Keypoint Receiver API

이 서버는 클라이언트(예: React 앱)에서 전달되는 MediaPipe 키포인트 좌표 데이터를 수신합니다.  
`/ai/keypoints` 엔드포인트로 JSON 데이터를 POST 방식으로 받고, 서버 콘솔에 좌표를 출력합니다.

---

## 📦 주요 기능

- CORS 허용 (React 프론트엔드와 통신 가능)
- `/ai/keypoints` 라우트에서 좌표 JSON 수신
- Flask 서버에서 간단한 확인용 로그 출력

---

## 🖥️ 실행 방법

```bash
# 가상환경 추천
python -m venv venv
source venv/bin/activate  # 또는 Windows: venv\\Scripts\\activate

# 패키지 설치
pip install flask flask-cors

# 서버 실행
python app.py
```

## POST 예시

### 요청 URL
```bash
POST http://localhost:5000/ai/keypoints
```
### 예시 데이터
```json
{
  "coords": {
    "face_keypoints_2d": [123, 456],
    "pose_keypoints_2d": [789, 321],
    "hand_left_keypoints_2d": [100, 200],
    "hand_right_keypoints_2d": [300, 400]
  }
}
```
### 응답
```json
{
  "status": "received"
}
```

## 참고
- 클라이언트와 연동 시 CORS 설정이 필요하므로 flask-cors를 반드시 설치해야 합니다.
```bash
pip install flask-cors
```

