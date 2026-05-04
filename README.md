# 🧠 Neuro Sync

> **AI 기반 실시간 졸음운전 감지 안전장치**
> 
> Jetson Orin Nano에서 MUSE2 EEG 뇌파를 분석하여 졸음운전을 감지하고 즉시 경고합니다.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.11%2B-orange)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.95%2B-green)](https://fastapi.tiangolo.com/)

---

## 📊 주요 성능

| 항목 | 성능 |
|------|------|
| **정확도** | 53% (프로토타입) |
| **반응속도** | ~5초 (윈도우 크기) |
| **갱신빈도** | 1초마다 |
| **메모리** | ~800MB |
| **전력소비** | ~15W (Jetson) |

---

## 🎯 시스템 흐름

```
┌──────────────┐
│  MUSE2 헤드셋 │─── 256Hz EEG ────┐
│  (AF7, AF8)  │                  │
└──────────────┘                  ▼
                           ┌─────────────────┐
                           │  FastAPI 서버   │
                           │  (신경망)       │
                           │ Bandpass Filter │
                           │ → 모델 추론     │
                           │ → 확률 반환     │
                           └────────┬────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                 ▼
            ┌──────────────────┐           ┌──────────────────┐
            │ 점수화 엔진      │           │ WebSocket        │
            │ (위험도 계산)    │           │ (실시간 스트림)  │
            └────────┬─────────┘           └──────────────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Jetson UI        │
            │ (OpenCV)         │
            │ - 대시보드       │
            │ - 경고 알림      │
            └──────────────────┘
```

---

## 🚀 5분 안에 시작

### 1️⃣ 설치

```bash
pip install fastapi uvicorn tensorflow scipy pandas numpy \
            opencv-python pyyaml websockets requests
```

### 2️⃣ FastAPI 서버 시작 (Terminal 1)

```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
python muse_inference_api.py
```

✓ `http://localhost:8000/health` 접속해서 상태 확인

### 3️⃣ 실시간 감지 실행 (Terminal 2)

```bash
python realtime_detector.py
```

✓ OpenCV 대시보드 자동 표시 (또는 config.yaml 수정)

### 4️⃣ 테스트 (선택)

```bash
python test_api.py              # API 전체 테스트
python drowsiness_scorer.py     # 점수화 로직 테스트
python validate_eeg.py --source file --file test_data.csv  # 데이터 검증
```

---

## ⚙️ 설정

### config.yaml

```yaml
# EEG 데이터 소스
eeg_source:
  type: file                 # muse2 | tcp | file
  kwargs:
    filepath: test_data.csv

# FastAPI
api:
  url: "http://localhost:8000"
  timeout: 5.0

# 점수화 (위험도 판정)
scorer:
  drowsy_threshold: 0.55           # 졸음 기준 확률
  instant_alert_threshold: 0.80    # 즉시 경고 기준
  accumulated_time_limit: 20.0     # 30초 중 20초 = 위험

# 경고
alert:
  enabled: true
  alarm_file: /path/to/alarm.wav
  gpio_pin: 17                     # 진동 모터 GPIO

# UI
ui:
  display_type: opencv            # opencv | web | headless
  fps: 10
```

---

## 📦 프로젝트 구조

```
Neuro Sync/
├── 🔴 Core Engine
│   ├── muse_inference_api.py       # FastAPI 신경망 서버
│   ├── drowsiness_scorer.py        # 위험도 점수화
│   ├── realtime_detector.py        # UI & 경고 제어
│   └── test_api.py                 # API 테스트
│
├── 🟡 Data & Configuration
│   ├── eeg_data_source.py          # 플러그인 아키텍처 (MUSE2/TCP/파일)
│   ├── config.py                   # 설정 관리
│   ├── config.yaml                 # 배포 설정
│   └── validate_eeg.py             # 데이터 검증
│
└── 📚 Documentation
    ├── README.md                   # 이 파일
    ├── PROTOCOL.md                 # 팀 간 통신 규약
    ├── CHECKLIST.md                # 단계별 작업 가이드
    └── INTEGRATION.md              # 전체 통합 설명
```

---

## 🔌 API 주요 엔드포인트

### `GET /health` — 상태 확인

```bash
curl http://localhost:8000/health
```

### `POST /predict/batch` — 배치 추론

**요청**:
```json
{
  "af7": [12.3, 11.9, ...],  // 256개 샘플
  "af8": [10.1, 10.5, ...],
  "apply_minmax": true
}
```

**응답**:
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

### `POST /session/start` → `POST /session/{sid}/append` → `POST /session/{sid}/end` — 스트리밍

1초씩 청크를 보내면 새 윈도우 나올 때마다 결과 반환

### `WebSocket /ws/stream` — 낮은 지연시간

더 빠른 응답이 필요하면 WebSocket 사용

---

## 📈 점수화 로직

API 응답의 확률값을 위험도(0~100)로 변환

```
입력: prob_raw (0~1)
  ↓
[즉각 점수]     = prob_raw × 100
[평활화 점수]   = 최근 30개 평균
[누적 졸음시간] = 30초 중 졸음 지속 시간
  ↓
위험도 판정:
  ≥ 80점  → "위험" ⚠️ (즉시 경고)
  60~80점 → "주의" 🟡
  < 60점  → "안전" ✅
```

### 커스터마이징

```python
from drowsiness_scorer import DrowsinessScorer

scorer = DrowsinessScorer(
    drowsy_threshold=0.55,
    instant_alert_threshold=0.80,
    accumulated_time_limit=20.0,
    window_size=30
)

score = scorer.score(prob_raw=0.7, prob_scaled=0.75)
print(f"{score.risk_level} | {score.smoothed_score:.0f}점")
```

---

## 💻 사용 예제

### 기본 사용

```python
import requests

# 1️⃣ 세션 시작
sid = requests.post("http://localhost:8000/session/start").json()["session_id"]

# 2️⃣ 1초씩 데이터 전송
for i in range(7):
    af7_chunk = [...] # 256 샘플
    af8_chunk = [...] # 256 샘플
    
    resp = requests.post(
        f"http://localhost:8000/session/{sid}/append",
        json={"af7": af7_chunk, "af8": af8_chunk, "apply_minmax": False}
    )
    
    for result in resp.json()["new_results"]:
        print(f"확률: {result['prob_raw']:.2f}")

# 3️⃣ 세션 종료
requests.post(f"http://localhost:8000/session/{sid}/end")
```

### Jetson 통합

```python
from realtime_detector import RealtimeDrowsinessDetector
from eeg_data_source import create_reader

# 1️⃣ 감지기 생성
detector = RealtimeDrowsinessDetector(
    alert_callback=lambda s: print(f"⚠️ {s.risk_level}!")
)
detector.start_session()

# 2️⃣ EEG 리더 선택 (config.yaml로 관리)
reader = create_reader("file", filepath="test_data.csv")
reader.start_stream(detector.process_eeg_chunk)

# 3️⃣ 통계 조회
stats = detector.get_stats()
print(f"처리: {stats['frame_count']} | 경고: {stats['alert_count']}")
```

---

## 🔧 고급 설정

### 커스텀 경고 콜백

```python
def my_alert_handler(score):
    if score.risk_level == "위험":
        # 경고음
        os.system("aplay /path/to/alarm.wav")
        
        # 진동 모터
        os.system("echo 1 > /sys/class/gpio/gpio17/value")
        
        # 원격 서버 알림
        requests.post("https://alert.server.com/incident", json={
            "level": score.risk_level,
            "score": score.smoothed_score,
            "timestamp": score.timestamp
        })

detector = RealtimeDrowsinessDetector(alert_callback=my_alert_handler)
```

### 성능 최적화

```bash
# Jetson에서 GPU 활성화
export CUDA_VISIBLE_DEVICES=0

# 메모리 부족 시 배치 크기 감소
# muse_inference_api.py 수정:
# model.predict(windows, batch_size=32)  # 기본값: 128
```

---

## ✅ 팀 협업 가이드

### MUSE2팀: EEG 데이터 수집

```python
from eeg_data_source import MUSE2Reader

reader = MUSE2Reader()
reader.connect()
chunk = reader.read_chunk()  # EEGChunk 반환
```

**검증**:
```bash
python validate_eeg.py --source muse2
```

### Jetson팀: 실시간 감지

```bash
python realtime_detector.py
```

**필독**: [PROTOCOL.md](PROTOCOL.md) — 팀 간 통신 규약

---

## 🐛 트러블슈팅

| 문제 | 해결 |
|------|------|
| ❌ 모델 파일 없음 | `export MUSE_MODEL_PATH=/path` |
| ❌ 포트 8000 사용 중 | `lsof -i :8000 \| xargs kill -9` |
| ❌ 메모리 부족 | `--workers 1` 사용 |
| ❌ 점수 항상 0 | `curl http://localhost:8000/health` 확인 |
| ❌ 경고 안 울림 | `config.yaml`에서 `alert.enabled: true` |

---

## 📚 상세 문서

| 문서 | 내용 |
|------|------|
| **[PROTOCOL.md](PROTOCOL.md)** | EEGChunk 포맷 & 팀 간 통신 규약 |
| **[CHECKLIST.md](CHECKLIST.md)** | 단계별 구현/배포 가이드 |
| **[INTEGRATION.md](INTEGRATION.md)** | 전체 시스템 통합 설명 |

---

## 📖 기술 사양

| 항목 | 값 |
|------|-----|
| **EEG 채널** | AF7, AF8 (MUSE2) |
| **샘플 레이트** | 256Hz |
| **윈도우 크기** | 5초 (1280 샘플) |
| **슬라이드 간격** | 1초 (256 샘플) |
| **전처리** | Bandpass (0.5~40Hz) → Median 제거 → Z-score |
| **모델** | TensorFlow Keras |
| **프레임워크** | FastAPI + TensorFlow + OpenCV |

---

## 🚨 알려진 제한사항

- **정확도 53%**: MUSE2의 높은 노이즈 → 개선 중
- **메모리**: 멀티 워커 환경에서는 Redis 백엔드 필요
- **지연시간**: 5초 윈도우로 인한 지연 (더 빠른 모델 필요 시 개별 협의)

---

## 📝 라이센스

프로젝트별로 상이 — 각 팀에 문의

---

## 🤝 기여 & 지원

문제 또는 제안: 각 문서의 troubleshooting 섹션 참고

**Jetson팀**: realtime_detector.py & drowsiness_scorer.py  
**MUSE2팀**: eeg_data_source.py & PROTOCOL.md

---

<div align="center">

**[📖 시작하기](CHECKLIST.md)** | **[🔌 API 문서](PROTOCOL.md)** | **[🎯 통합 가이드](INTEGRATION.md)**

Made with ❤️ for Road Safety

</div>
