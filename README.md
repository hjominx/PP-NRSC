# Neuro Sync - AI 기반 졸음운전 감지 시스템

**실시간 EEG 기반 졸음운전 감지 안전장치**

Muse2 EEG 헤드셋을 활용하여 운전자의 뇌파를 실시간 분석하고 졸음운전을 감지하여 즉시 경고하는 AI 시스템입니다.

## 🚀 주요 특징

- **실시간 감지**: 1초 단위로 뇌파 분석 및 졸음 판정
- **고급 전처리**: 노이즈 제거, 아티팩트 필터링, 신호 품질 최적화
- **높은 정확도**: 55% 정확도 (기본 전처리 대비 6%p 향상)
- **모듈식 아키텍처**: FastAPI 서버 + 점수화 엔진 + UI 컴포넌트
- **다양한 데이터 소스**: Muse2, 파일, TCP 네트워크 지원

## 📊 성능 지표

| 지표 | 값 | 설명 |
|------|-----|------|
| **정확도** | 55% | Awake/Sleep 분류 정확도 |
| **정밀도** | 54% | Sleep 예측의 정확성 |
| **재현율** | 67% | Sleep 감지 민감도 |
| **반응 속도** | ~5초 | 윈도우 크기 기준 |
| **갱신 주기** | 1Hz | 1초마다 판정 |
| **메모리 사용** | ~800MB | TensorFlow 모델 포함 |

## 🏗️ 시스템 아키텍처

```
Muse2 헤드셋 (256Hz EEG)
         |
         | AF7, AF8 채널
         v
   고급 전처리 엔진
   - 밴드패스 필터 (0.5~40Hz)
   - 60Hz 노치 필터
   - EOG 아티팩트 제거
   - 신호 품질 평가
         |
         v
   TensorFlow 모델
   - CNN-LSTM 아키텍처
   - 실시간 추론
   - 확률값 출력
         |
         v
   점수화 엔진
   - 히스테리시스 필터링
   - 이동평균 평활화
   - 누적 졸음시간 계산
   - 위험도 판정
         |
         v
   경고 시스템
   - 시각/청각/진동 경고
   - OpenCV 대시보드
   - 원격 서버 연동
```

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 준비

```bash
# 환경변수 설정 (실제 모델 경로로 변경)
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras

# 또는 더미 모델 사용 (테스트용)
export MUSE_MODEL_PATH=dummy_model.keras
```

### 3. 서버 실행

```bash
# FastAPI 서버 시작
python muse_inference_api.py

# 상태 확인
curl http://localhost:8000/health
```

### 4. 실시간 감지

```bash
# 새로운 터미널에서
python realtime_detector.py
```

## 🧪 테스트 및 검증

### 정확도 테스트

```bash
# 기본 vs 고급 전처리 비교
python test_accuracy_improvement.py

# API 기능 테스트
python test_api.py

# 데이터 품질 검증
python validate_eeg.py --source file --file awake_study.csv
```

### 데이터 수집

현재 시스템은 다음 데이터를 사용합니다:
- `awake_study.csv`, `awake_study2.csv`, `awake_study3.csv`, `awake_study4.csv` (깨어있는 상태)
- `bedtime.csv`, `bedtime2.csv` (졸음 상태)

### 현재 진행 상태

- FastAPI 서버 구현: 완료
- 실시간 추론 + 전처리 파이프라인: 완료
- 점수화 로직 연결: 완료
- 추가 데이터(`awake_study4.csv`) 반영: 완료
- 남은 작업: 모델 파일 경로/배포 검증 및 운영 테스트

더 많은 데이터를 수집하려면:
```bash
# 30분씩 3번 측정 추천 (총 90분)
# 다양한 시간대에 측정하여 circadian rhythm 반영
```

## 📁 프로젝트 구조

```
nrsc/
├── 🧠 핵심 엔진
│   ├── muse_inference_api.py       # FastAPI 추론 서버
│   ├── drowsiness_scorer.py        # 점수화 및 판정 로직
│   ├── realtime_detector.py        # 실시간 UI 및 경고
│   └── test_api.py                 # API 테스트 도구
│
├── 🔧 전처리 및 분석
│   ├── advanced_preprocessing.py   # 고급 EEG 전처리
│   ├── advanced_postprocessing.py  # 품질 기반 후처리
│   ├── analyze_improvements.py     # 개선 효과 분석
│   └── test_preprocessing_quality.py # 품질 검증
│
├── 📊 데이터 및 설정
│   ├── eeg_data_source.py          # EEG 데이터 입출력
│   ├── config.py                   # 설정 관리
│   ├── config.yaml                 # YAML 설정 파일
│   └── validate_eeg.py             # 데이터 검증 도구
│
├── 🎯 모델 및 학습
│   ├── train_muse_model_v2.py      # 모델 학습 스크립트
│   ├── dummy_model.keras           # 테스트용 모델
│   └── test_accuracy_improvement.py # 정확도 평가
│
├── 📋 문서
│   ├── README.md                   # 이 파일
│   ├── IMPROVEMENTS_V2.md          # 개선 사항 상세
│   ├── PROTOCOL.md                 # 통신 프로토콜
│   ├── CHECKLIST.md                # 개발 체크리스트
│   └── INTEGRATION.md              # 시스템 통합 가이드
│
└── ⚙️ 유틸리티
    ├── alert_handlers.py           # 경고 처리기
    ├── local_realtime.py           # 로컬 테스트 도구
    └── requirements.txt            # Python 의존성
```

## 🔧 API 엔드포인트

### GET /health
시스템 상태 및 모델 로드 확인

### POST /predict/batch
배치 추론 (JSON 입력)

```json
{
  "af7": [12.3, 11.9, ...],
  "af8": [10.1, 10.5, ...],
  "apply_minmax": true
}
```

### POST /session/start → /session/{id}/append → /session/{id}/end
실시간 스트리밍 세션

### WebSocket /ws/stream
저지연 실시간 스트리밍

## ⚙️ 설정

`config.yaml`에서 시스템 설정 조정:

```yaml
# EEG 데이터 소스
eeg_source:
  type: muse2                    # muse2 | file | tcp
  kwargs:
    device_name: "Muse"

# 점수화 임계값
scorer:
  window_size: 30                # 평활화 윈도우
  drowsy_threshold: 0.55         # 졸음 판정 기준
  instant_alert_threshold: 0.80  # 즉시 경고 기준
  accumulated_time_limit: 20.0   # 누적 시간 제한

# 경고 설정
alert:
  enabled: true
  alarm_file: null               # 경고음 파일
  gpio_pin: null                 # 진동 모터 핀
  send_to_server: false         # 원격 서버 연동
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
