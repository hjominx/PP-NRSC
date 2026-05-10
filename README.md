# Neuro Sync - AI Sleep/Drowsiness Detection

A real-time EEG-based drowsiness detection system for Muse2. The system analyzes AF7/AF8 channels, applies preprocessing, and infers alertness state using a TensorFlow model.

## Features

- Real-time inference with 1 Hz update rate
- Advanced preprocessing for noise and artifact reduction
- Modular architecture: FastAPI server, scoring engine, and monitoring tools
- Supports multiple input sources: Muse2, CSV files, TCP
- Includes data analysis and validation utilities

## Performance Metrics

| Metric | Value | Notes |
|-------|------|-------|
| Accuracy | 55% | Awake/Sleep classification accuracy |
| Precision | 54% | Sleep prediction precision |
| Recall | 67% | Sleep detection sensitivity |
| Latency | ~5 sec | Based on 5-second window size |
| Update rate | 1 Hz | One window per second |
| Memory | ~800 MB | Includes TensorFlow model |

## Architecture

```
Muse2 headset (256Hz EEG)
         |
         | AF7, AF8 channels
         v
   Advanced preprocessing
   - bandpass filter (0.5-40Hz)
   - 60Hz notch filter
   - EOG artifact rejection
   - signal quality scoring
         |
         v
   TensorFlow inference model
   - CNN-LSTM architecture
   - real-time prediction
   - probability output
         |
         v
   Scoring engine
   - hysteresis smoothing
   - moving average smoothing
   - accumulated drowsy time
   - risk level output
         |
         v
   Alert system
   - visual/audio alerts
   - OpenCV dashboard
   - optional remote integration
```

## Installation and Run

### 1. Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare model

```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
# or use the dummy model for testing
export MUSE_MODEL_PATH=dummy_model.keras
```

### 3. Start server

```bash
python muse_inference_api.py
curl http://localhost:8000/health
```

### 4. Run realtime detection

```bash
python realtime_detector.py
```

## Test and Validation

### Accuracy test

```bash
python test_accuracy_improvement.py
python test_api.py
python validate_eeg.py --source file --file awake_study.csv
```

### Data used

- `awake_study.csv`, `awake_study2.csv`, `awake_study3.csv`, `awake_study4.csv` (awake)
- `bedtime.csv`, `bedtime2.csv` (drowsy)

### Current status

- FastAPI inference pipeline: completed
- Real-time preprocessing and prediction: completed
- Scoring and risk calculation: completed
- Added awake_study4.csv data: completed
- Remaining: model path validation and deployment testing

## Project Structure

```
nrsc/
├── muse_inference_api.py       # FastAPI inference server
├── drowsiness_scorer.py        # scoring and risk logic
├── realtime_detector.py        # realtime monitoring client
├── test_api.py                 # API smoke test
├── advanced_preprocessing.py   # EEG preprocessing
├── advanced_postprocessing.py  # postprocessing and quality filtering
├── analyze_improvements.py     # analysis of preprocessing improvements
├── test_preprocessing_quality.py # preprocessing quality tests
├── eeg_data_source.py          # EEG data input/output
├── config.py                   # configuration management
├── config.yaml                 # YAML configuration
├── validate_eeg.py             # data validation tool
├── train_muse_model_v2.py      # model training script
├── dummy_model.keras           # test model
├── test_accuracy_improvement.py # accuracy evaluation
├── README.md                   # project documentation
├── IMPROVEMENTS_V2.md          # improvement notes
├── PROTOCOL.md                 # API protocol notes
├── CHECKLIST.md                # development checklist
├── INTEGRATION.md              # integration guide
├── alert_handlers.py           # alert handling utilities
├── local_realtime.py           # local testing tool
└── requirements.txt            # Python dependencies
```

## API Endpoints

### GET /health
Check service status and model load state.

### POST /predict/batch
Batch prediction from JSON input.

Example:

```json
{
  "af7": [12.3, 11.9, ...],
  "af8": [10.1, 10.5, ...],
  "apply_minmax": true
}
```

### POST /session/start → /session/{id}/append → /session/{id}/end
Streaming session endpoints for realtime data.

### WebSocket /ws/stream
Low-latency realtime streaming endpoint.

## Configuration

Adjust system settings in `config.yaml`:

```yaml
eeg_source:
  type: muse2
  kwargs:
    device_name: "Muse"

scorer:
  window_size: 30
  drowsy_threshold: 0.55
  instant_alert_threshold: 0.80
  accumulated_time_limit: 20.0

alert:
  enabled: true
  alarm_file: null
  gpio_pin: null
  send_to_server: false
```

## 🎯 개선 사항 (v2)

### 고급 전처리 적용 결과
- **정확도 향상**: 49% → 55% (+6%p)
- **Sleep 감지 개선**: 재현율 56% → 67% (+11%p)
- **노이즈 대응**: 60Hz 노치 필터, EOG 아티팩트 제거
- **품질 기반 판정**: 신호 품질을 고려한 동적 임계값

### 데이터 증강
- 다중 CSV 파일 통합 (5개 파일, 9351개 윈도우)
- 클래스 균형 맞추기 (각 클래스당 2684개 샘플)
- 다양한 시간대 데이터 수집

## 🤝 기여하기

1. 이슈 생성 또는 PR 제출
2. 코드 스타일 준수 (PEP 8)
3. 테스트 코드 작성
4. 문서 업데이트

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

**⚠️ 주의사항**: 이 시스템은 연구/프로토타입 용도로만 사용하세요. 실제 운전 환경에서는 전문가의 검증을 받아야 합니다.
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
