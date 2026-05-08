"""
로컬 실시간 테스트 스크립트
---------------------------------
LSL(Muse) 또는 시뮬레이션 모드에서 `EEGChunk`를 읽어
간단한 대체 확률(prob_raw)을 계산한 뒤 `DrowsinessScorer`로
점수화하여 경고를 출력한다.

사용법:
  python3 local_realtime.py --duration 30
"""
import time
import argparse
import numpy as np

from eeg_data_source import create_reader, EEGChunk
from drowsiness_scorer import DrowsinessScorer


def estimate_prob_from_chunk(chunk: EEGChunk) -> float:
    """단순한 대체 모델: 알파(8-13Hz) 파워 비율을 사용해 0~1로 매핑
    (실제 모델을 대체하는 용도이며, 테스트/디버깅 전용)
    """
    af7 = np.asarray(chunk.af7)
    af8 = np.asarray(chunk.af8)
    x = (af7 + af8) * 0.5
    # FFT
    N = len(x)
    freqs = np.fft.rfftfreq(N, d=1/256)
    P = np.abs(np.fft.rfft(x))**2

    # 밴드 파워 계산
    def band_power(p, f, lo, hi):
        mask = (f >= lo) & (f <= hi)
        return p[mask].sum()

    alpha = band_power(P, freqs, 8, 13)
    total = P.sum() + 1e-9
    ratio = alpha / total

    # ratio는 보통 작음 -> 부드러운 시그모이드로 스케일링 (더 관대하게)
    # 경험적으로 시뮬레이션/실데이터 모두에서 과민하지 않게 동작하도록 조정
    x = (ratio - 0.03) / 0.06
    prob = 1.0 / (1.0 + np.exp(-6.0 * x))
    return float(np.clip(prob, 0.0, 1.0))


def main(duration: int = 30, source: str = "muse2"):
    print(f"로컬 실시간 테스트 시작 (source={source}, duration={duration}s)")
    reader = create_reader(source)
    if not reader.connect():
        print("리더 연결 실패")
        return

    # 보수적 임계값: 시뮬레이션 및 실환경에서 과민하게 경고가 뜨는 것을 방지
    scorer = DrowsinessScorer(
        window_size=60,
        drowsy_threshold=0.7,
        accumulated_time_limit=25.0,
        instant_alert_threshold=0.95,
    )
    start = time.time()
    last_seq = -1
    seq_counter = 0
    from collections import deque
    prob_history = deque(maxlen=5)

    try:
        while time.time() - start < duration:
            chunk = reader.read_chunk(timeout=2.0)
            if chunk is None:
                print("타임아웃: 청크 수신 없음")
                continue

            prob = estimate_prob_from_chunk(chunk)
            # 간단한 평활화: 최근 N 프레임 평균을 사용
            prob_history.append(prob)
            smoothed_prob = float(sum(prob_history) / len(prob_history))

            # sequence id 보정 (read_chunk가 직접 호출될 때 증가)
            chunk.sequence_id = seq_counter
            seq_counter += 1

            score = scorer.score(prob_raw=smoothed_prob, prob_scaled=None, state=0)

            ts = time.strftime('%H:%M:%S')
            alert = "ALERT" if score.should_alert else ""
            print(f"[{ts}] seq={chunk.sequence_id} prob={prob:.3f} smoothed={score.smoothed_score:.1f} {score.risk_level} {alert}")

            last_seq = chunk.sequence_id
    except KeyboardInterrupt:
        print("중단됨")
    finally:
        reader.disconnect()
        print("종료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--source', choices=['muse2','file','tcp'], default='muse2')
    args = parser.parse_args()
    main(duration=args.duration, source=args.source)
