# Neuro Sync

**AI 기반 실시간 졸음운전 감지 안전장치**

Jetson Orin Nano 기반 시스템으로 MUSE2 EEG 뇌파를 분석하여 운전 중 졸음운전을 실시간으로 감지하고 즉시 경고함.

---

## 성능 사양

| 항목 | 값 |
|------|-----|
| 정확도 | 53% (프로토타입 단계) |
| 반응 속도 | 약 5초 (윈도우 크기 기준) |
| 갱신 주기 | 1초마다 |
| 메모리 사용량 | 약 800MB |
| 전력 소비 | 약 15W (Jetson Orin Nano) |

---

## 시스템 구조

```
MUSE2 헤드셋 (256Hz EEG)
         |
         | AF7, AF8 채널
         v
   FastAPI 서버
   - Bandpass 필터 (0.5~40Hz)
   - Median 제거
   - Z-score 정규화
   - 신경망 추론
   - 확률값 반환
         |
         v
   점수화 엔진
   - 즉각 점수 계산
   - 평활화 (이동평균)
   - 누적 졸음시간
   - 위험도 판정
         |
         v
   Jetson Orin Nano UI
   - OpenCV 대시보드
   - 경고 시스템
   - 원격 통신
```

---

## 빠른 시작

### 1. 설치

```bash
pip install fastapi uvicorn tensorflow scipy pandas numpy \
            opencv-python pyyaml websockets requests
```

### 2. FastAPI 서버 실행 (Terminal 1)

```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
python muse_inference_api.py
```

상태 확인: `curl http://localhost:8000/health`

### 3. 실시간 감지 실행 (Terminal 2)

```bash
python realtime_detector.py
```

### 4. 검증

```bash
python test_api.py
python drowsiness_scorer.py
python validate_eeg.py --source file --file test_data.csv
```

---

## 설정 방법

`config.yaml` 파일에서 다음 항목을 조정합니다.

```yaml
# EEG 데이터 소스
eeg_source:
  type: file              # muse2 | tcp | file
  kwargs:
    filepath: test_data.csv

# FastAPI 서버
api:
  url: "http://localhost:8000"
  timeout: 5.0
  apply_minmax: false

# 점수화 설정
scorer:
  window_size: 30                    # 평활화 윈도우 크기
  drowsy_threshold: 0.55             # 졸음 판정 확률 기준
  instant_alert_threshold: 0.80      # 즉시 경고 확률 기준
  accumulated_time_limit: 20.0       # 30초 중 누적 시간 기준

# 경고 설정
alert:
  enabled: true
  alarm_file: /path/to/alarm.wav
  gpio_pin: 17                       # GPIO 핀번호 (진동 모터)
  send_to_server: false
  server_url: null

# UI 설정
ui:
  display_type: opencv              # opencv | web | headless
  fps: 10
  show_stats: true

# 디버그 모드
debug: false
```

---

## 프로젝트 구조

```
nrsc/
├── 핵심 엔진
│   ├── muse_inference_api.py       신경망 서버 (FastAPI)
│   ├── drowsiness_scorer.py        점수화 로직
│   ├── realtime_detector.py        UI 및 경고 제어
│   └── test_api.py                 API 테스트
│
├── 데이터 및 설정
│   ├── eeg_data_source.py          EEG 입출력 추상화 계층
│   ├── config.py                   설정 관리 (Python)
│   ├── config.yaml                 배포 설정 (YAML)
│   └── validate_eeg.py             데이터 검증 도구
│
└── 문서
    ├── README.md                   이 파일
    ├── PROTOCOL.md                 팀 간 통신 규약
    ├── CHECKLIST.md                단계별 작업 가이드
    └── INTEGRATION.md              시스템 통합 설명
```

---

## API 엔드포인트

### GET /health

모델 로드 상태 및 입출력 형태 확인합니다.

```bash
curl http://localhost:8000/health
```

### POST /predict/batch

요청:
```json
{
  "af7": [12.3, 11.9, ...],
  "af8": [10.1, 10.5, ...],
  "apply_minmax": true
}
```

응답:
```json
{
  "n_samples": 2560,
  "n_windows": 6,
  "results": [
    {
      "t_start_sec": 0.0,
      "t_end_sec": 5.0,
      "prob_raw": 0.65,
      "prob_scaled": 0.72,
      "state": 1
    }
  ]
}
```

### POST /session/start → POST /session/{sid}/append → POST /session/{sid}/end

실시간 스트리밍. 1초씩 청크를 전송하면 새 윈도우 생성 시점마다 결과를 반환합니다.

### WebSocket /ws/stream

지연시간 최소화가 필요한 경우 WebSocket을 사용합니다.

---

## 점수화 알고리즘

API 응답의 확률값을 위험도(0~100 점)로 변환합니다.

```
입력: prob_raw (0~1)
   |
   +-- 즉각 점수 = prob_raw × 100
   |
   +-- 평활화 점수 = 최근 30개 윈도우의 평균
   |
   +-- 누적 졸음시간 = 30초 동안의 졸음 지속 시간
   |
   v
위험도 판정:
  80점 이상    : 위험 (즉시 경고)
  60~80점      : 주의
  60점 미만    : 안전
```

예시 코드:

```python
from drowsiness_scorer import DrowsinessScorer

scorer = DrowsinessScorer(
    drowsy_threshold=0.55,
    instant_alert_threshold=0.80,
    accumulated_time_limit=20.0,
    window_size=30
)

score = scorer.score(prob_raw=0.7, prob_scaled=0.75)
print(score.risk_level, score.smoothed_score)
```

---

## 사용 예제

### 기본 사용

```python
import requests

# 세션 시작
sid = requests.post("http://localhost:8000/session/start").json()["session_id"]

# 1초 단위 청크 전송
for i in range(7):
    af7_chunk = [...]  # 256개 샘플
    af8_chunk = [...]  # 256개 샘플
    
    resp = requests.post(
        f"http://localhost:8000/session/{sid}/append",
        json={
            "af7": af7_chunk,
            "af8": af8_chunk,
            "apply_minmax": False
        }
    )
    
    for result in resp.json()["new_results"]:
        print(f"확률: {result['prob_raw']:.3f}")

# 세션 종료
requests.post(f"http://localhost:8000/session/{sid}/end")
```

### Jetson 통합

```python
from realtime_detector import RealtimeDrowsinessDetector
from eeg_data_source import create_reader

# 감지기 생성
detector = RealtimeDrowsinessDetector(
    alert_callback=lambda s: print(f"경고: {s.risk_level}")
)
detector.start_session()

# EEG 리더 선택
reader = create_reader("file", filepath="test_data.csv")
reader.start_stream(detector.process_eeg_chunk)

# 통계 조회
stats = detector.get_stats()
print(f"처리 프레임: {stats['frame_count']}")
print(f"경고 발생: {stats['alert_count']}")
```

---

## 고급 설정

### 경고 콜백 커스터마이징

```python
import os
import requests

def alert_handler(score):
    if score.risk_level == "위험":
        # 경고음 재생
        os.system("aplay /path/to/alarm.wav")
        
        # GPIO를 통한 진동 모터 제어
        os.system("echo 1 > /sys/class/gpio/gpio17/value")
        
        # 원격 서버에 알림 전송
        requests.post("https://alert.server.com/incident", json={
            "level": score.risk_level,
            "score": score.smoothed_score,
            "timestamp": score.timestamp
        })

detector = RealtimeDrowsinessDetector(alert_callback=alert_handler)
```

### 성능 최적화

GPU 활성화:
```bash
export CUDA_VISIBLE_DEVICES=0
```

메모리 부족 시 배치 크기 감소:
```python
# muse_inference_api.py 내부에서 수정
model.predict(windows, batch_size=32)  # 기본값: 128
```

---

## 팀 협업

### MUSE2 팀: EEG 데이터 수집

`eeg_data_source.py`의 `MUSE2Reader` 클래스를 구현합니다.

```python
from eeg_data_source import MUSE2Reader

reader = MUSE2Reader()
reader.connect()
chunk = reader.read_chunk()
```

검증:
```bash
python validate_eeg.py --source muse2
```

### Jetson 팀: 실시간 감지

```bash
python realtime_detector.py
```

참고 자료: [PROTOCOL.md](PROTOCOL.md)

---

## 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| 모델 파일 없음 | `export MUSE_MODEL_PATH=/path` 설정 |
| 포트 8000 사용 중 | `lsof -i :8000 \| xargs kill -9` |
| 메모리 부족 | `uvicorn` 실행 시 `--workers 1` 옵션 사용 |
| 점수 항상 0 | API 상태 확인: `curl http://localhost:8000/health` |
| 경고 미작동 | `config.yaml`에서 `alert.enabled: true` 확인 |

---

## 기술 사양

| 항목 | 값 |
|------|-----|
| EEG 채널 | AF7, AF8 (MUSE2) |
| 샘플 레이트 | 256Hz |
| 윈도우 크기 | 5초 (1280개 샘플) |
| 윈도우 슬라이드 간격 | 1초 (256개 샘플) |
| 전처리 | Bandpass 필터 (0.5~40Hz) → Median 제거 → Z-score 정규화 |
| 신경망 프레임워크 | TensorFlow Keras |
| 웹 프레임워크 | FastAPI |
| 클라이언트 프레임워크 | OpenCV |

---

## 알려진 제한사항

- 정확도 53%: MUSE2 센서의 높은 노이즈로 인함 (개선 진행 중)
- 메모리: 멀티 워커 환경에서는 Redis 기반 백엔드 필요
- 지연시간: 5초 윈도우로 인한 구조적 지연 (빠른 반응 필요 시 별도 협의)

---

## 참고 문서

| 문서 | 설명 |
|------|------|
| PROTOCOL.md | 팀 간 통신 규약 및 EEGChunk 데이터 포맷 |
| CHECKLIST.md | 단계별 구현 및 배포 가이드 |
| INTEGRATION.md | 전체 시스템 통합 설명 |

---

## 라이센스

각 팀별로 상이 — 담당자에 문의

---

## 지원

문제 발생 시 위 문제 해결 섹션을 참고하거나 관련 문서를 확인합니다.
