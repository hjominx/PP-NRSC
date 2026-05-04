# 🚗 실시간 졸음운전 감지 시스템

**Jetson Orin Nano + OpenCV + MUSE2**를 사용한 차량 안전장치

```
MUSE2 EEG → FastAPI (전처리/모델) → 점수화 로직 → OpenCV UI + 경고
```

## 📁 프로젝트 구조

### 핵심 파일
| 파일 | 목적 | 담당자 |
|------|------|--------|
| **muse_inference_api.py** | FastAPI 서버: EEG 전처리 & 모델 추론 | - |
| **drowsiness_scorer.py** | 점수화 로직: 확률 → 위험도 계산 | Jetson팀 |
| **realtime_detector.py** | 실시간 감지: Jetson + OpenCV UI | Jetson팀 |

### 연동 및 설정
| 파일 | 목적 |
|------|------|
| **eeg_data_source.py** | 📋 EEG 데이터 소스 추상화 (플러그인 아키텍처) |
| **config.py** | ⚙️ 설정 관리 (YAML 기반) |
| **config.yaml** | 🔧 시스템 설정 (실행 환경별 커스터마이징) |
| **validate_eeg.py** | ✅ EEG 데이터 검증 유틸리티 |

### 문서
| 파일 | 내용 |
|------|------|
| **PROTOCOL.md** | 📋 **MUSE2 ↔ Jetson 통신 프로토콜 (MUST READ)** |
| **INTEGRATION.md** | 📖 전체 통합 가이드 |
| **README.md** | 이 파일 |

### 테스트
| 파일 | 목적 |
|------|------|
| **test_api.py** | API 테스트 (모든 엔드포인트 검증) |

## 🚀 빠른 시작

### 1. FastAPI 서버 시작
```bash
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras
python muse_inference_api.py
```

### 2. 실시간 감지 실행
```bash
python realtime_detector.py
```

### 3. 테스트
```bash
python test_api.py
python drowsiness_scorer.py
```

---

## 🔧 FastAPI 서버

EEG raw 데이터(AF7, AF8) → 전처리 → 모델 추론 → 확률/상태 반환까지만 책임지는 FastAPI 서버.
점수화 로직은 호출하는 쪽에서 응답을 받아서 붙이면 됨.

## 학습 코드와의 일치
- 전처리 (`preprocess`): bandpass 0.5~40Hz (Butter 4차) → median 제거 → ±100 clip → z-score
- 후처리: min-max 스케일링 (옵션) → hysteresis(high=0.55, low=0.45)
- FS=256, SEQ_LEN=1280 (5초), STRIDE=256 (1초)

## 실행
```bash
pip install fastapi uvicorn tensorflow scipy pandas numpy python-multipart websockets

# 모델 경로는 환경변수로 지정 (기본값은 /mnt/user-data/uploads/...)
export MUSE_MODEL_PATH=/path/to/MUSE_activity_model.keras

# 개발용
python muse_inference_api.py

# 프로덕션
uvicorn muse_inference_api:app --host 0.0.0.0 --port 8000 --workers 1
```
> ⚠️ TensorFlow 모델은 멀티 워커에서 메모리 중복되니 `--workers 1` 권장. 동시성 필요하면 워커 늘리거나 모델 서빙 분리.

## 엔드포인트

### `GET /health`
모델 로드 상태와 input/output shape 확인.

### `POST /predict/batch` — JSON 배치
긴 신호 한 번에 보내고 모든 윈도우 결과 받기.

```json
{
  "af7": [12.3, 11.9, ...],   // 256Hz raw
  "af8": [10.1, 10.5, ...],   // 256Hz raw, af7과 같은 길이
  "apply_minmax": true,        // 학습 코드 동일하게 0~1로 펼치기
  "hys_high": 0.55,
  "hys_low": 0.45
}
```

응답:
```json
{
  "n_samples": 2560,
  "n_windows": 6,
  "fs": 256, "seq_len": 1280, "stride": 256,
  "prob_raw_min": 0.55, "prob_raw_max": 0.56,
  "results": [
    {"t_start_sec": 0.0, "t_end_sec": 5.0,
     "prob_raw": 0.551, "prob_scaled": 0.0, "state": 0},
    ...
  ]
}
```

### `POST /predict/csv` — CSV 업로드
`AF7`, `AF8` 컬럼이 있는 CSV. multipart/form-data.
```bash
curl -X POST "http://localhost:8000/predict/csv?apply_minmax=true" \
     -F "file=@bedtime.csv"
```

### 스트리밍 (HTTP) — `/session/*`
1초마다 측정하는 환경에 맞춤. 세션 만들고 청크 보내면, 5초 모인 시점부터 매번 새 윈도우 결과 떨어짐.

```python
import requests
sid = requests.post("http://localhost:8000/session/start").json()["session_id"]

# 1초씩 들어올 때마다
r = requests.post(f"http://localhost:8000/session/{sid}/append",
                  json={"af7": chunk_af7, "af8": chunk_af8,
                        "apply_minmax": False})
for win in r.json()["new_results"]:
    score = my_scoring_logic(win["prob_raw"], win["state"])  # ← 본인 로직

requests.post(f"http://localhost:8000/session/{sid}/end")
```
- 스트리밍에서는 `apply_minmax=False` 권장 — 단일 호출 안 윈도우들에만 min-max 적용되어 의미 없어지는 경우 많음. 점수화는 `prob_raw`로 하시고, 분포 정규화가 꼭 필요하면 본인 쪽에서 누적 윈도우로 처리.
- `state`는 hysteresis가 세션 전체에 걸쳐 이어짐 (한 번 1로 들어가면 0.45 밑으로 떨어지기 전까진 1 유지).

### 스트리밍 (WebSocket) — `/ws/stream`
지연 더 줄이고 싶으면 WS. 동일 페이로드.

```python
import asyncio, json, websockets

async def run():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        for chunk in stream:
            await ws.send(json.dumps({"af7": chunk[:,0].tolist(),
                                      "af8": chunk[:,1].tolist()}))
            msg = json.loads(await ws.recv())
            for win in msg["new_results"]:
                ...  # 점수화

asyncio.run(run())
```

## 응답 필드 요약 (점수화 붙일 때)
| 필드 | 의미 |
|---|---|
| `prob_raw` | 모델의 sigmoid 출력 (0~1). **awake 확률** |
| `prob_scaled` | apply_minmax=True일 때만. 응답 내 윈도우들끼리 0~1로 펼친 값 |
| `state` | hysteresis 적용 후 0=sleep, 1=awake |
| `t_start_sec`, `t_end_sec` | 절대 시간 (세션이면 세션 시작 기준) |

## 알려진 제약
- 세션 저장소는 in-memory dict. 멀티 워커/멀티 인스턴스 환경이면 Redis 등으로 빼야 함.
- 추론은 `_predict_lock`으로 직렬화 — 단일 GPU/CPU에서 안전하지만 처리량 한계 있음. 부하 커지면 모델을 별도 서빙(TF Serving / Triton)으로 분리 권장.
- 보고서대로 정확도 53% 수준이니, 점수화 단에서 다수결/이동평균 같은 후처리 추가 고려.
