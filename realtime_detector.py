"""
실시간 졸음운전 감지 (Jetson Orin Nano + OpenCV + MUSE2)
==========================================================
Jetson에서 실행:
1. MUSE2에서 1초씩 EEG 청크 수신
2. FastAPI 서버로 전송 (실시간 변환)
3. 점수화 로직으로 위험도 계산
4. OpenCV로 화면에 표시 + 경고음/진동 발생

구성:
- muse2_reader.py: MUSE2 데이터 수신 (별도 스레드)
- drowsiness_scorer.py: 위험도 점수 계산
- 이 파일: OpenCV UI + 통합 제어
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

import cv2
import numpy as np
import requests

from drowsiness_scorer import DrowsinessScorer, DrowsinessScore


@dataclass
class RealtimeDrowsinessDetector:
    """
    Jetson Orin Nano에서 실시간 졸음운전 감지를 수행합니다.
    """
    
    api_url: str = "http://localhost:8000"  # FastAPI 서버 주소
    alert_callback: Optional[Callable] = None  # 경고 콜백 (진동/음성/LED 등)
    
    def __post_init__(self):
        self.session_id: Optional[str] = None
        self.scorer = DrowsinessScorer()
        self.running = False
        self.latest_score: Optional[DrowsinessScore] = None
        self.lock = threading.Lock()
        
        # 통계
        self.frame_count = 0
        self.alert_count = 0
        self.start_time = time.time()
    
    def start_session(self):
        """FastAPI 세션 시작"""
        try:
            resp = requests.post(f"{self.api_url}/session/start", timeout=5)
            self.session_id = resp.json()["session_id"]
            print(f"✅ 세션 시작: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ 세션 시작 실패: {e}")
            return False
    
    def end_session(self):
        """FastAPI 세션 종료"""
        if self.session_id:
            try:
                requests.post(f"{self.api_url}/session/{self.session_id}/end", timeout=5)
                print(f"✅ 세션 종료: {self.session_id}")
            except Exception as e:
                print(f"⚠️ 세션 종료 실패: {e}")
    
    def process_eeg_chunk(self, af7_chunk: list[float], af8_chunk: list[float]) -> bool:
        """
        EEG 청크를 처리하고 점수를 계산합니다.
        
        Args:
            af7_chunk: AF7 채널 (256 샘플 = 1초)
            af8_chunk: AF8 채널 (256 샘플 = 1초)
        
        Returns:
            True if 새 윈도우 결과가 있음
        """
        if not self.session_id:
            return False
        
        try:
            # API에 전송
            payload = {
                "af7": af7_chunk,
                "af8": af8_chunk,
                "apply_minmax": False,
            }
            resp = requests.post(
                f"{self.api_url}/session/{self.session_id}/append",
                json=payload,
                timeout=5
            )
            api_response = resp.json()
            
            # 새 윈도우 결과 처리
            if api_response.get("new_results"):
                result = api_response["new_results"][-1]  # 최신 윈도우만
                
                # 점수화
                score = self.scorer.score(
                    prob_raw=result["prob_raw"],
                    prob_scaled=result.get("prob_scaled"),
                    state=result["state"],
                )
                
                with self.lock:
                    self.latest_score = score
                    self.frame_count += 1
                    
                    if score.should_alert:
                        self.alert_count += 1
                        if self.alert_callback:
                            self.alert_callback(score)
                
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ API 호출 실패: {e}")
            return False
    
    def get_latest_score(self) -> Optional[DrowsinessScore]:
        """최신 점수 조회 (스레드 안전)"""
        with self.lock:
            return self.latest_score
    
    def get_stats(self) -> dict:
        """통계 조회"""
        elapsed = time.time() - self.start_time
        return {
            "frame_count": self.frame_count,
            "alert_count": self.alert_count,
            "elapsed_sec": elapsed,
            "fps": self.frame_count / elapsed if elapsed > 0 else 0,
        }


class OpenCVDisplay:
    """OpenCV를 사용한 UI 렌더링"""
    
    @staticmethod
    def create_dashboard(score: Optional[DrowsinessScore], stats: dict) -> np.ndarray:
        """
        졸음운전 대시보드 이미지 생성
        
        Returns:
            (1080, 1920, 3) BGR 이미지
        """
        h, w = 1080, 1920
        img = np.ones((h, w, 3), dtype=np.uint8) * 240  # 밝은 배경
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ===== 제목 =====
        cv2.putText(img, "Real-time Drowsiness Detection", (50, 80),
                    font, 2, (0, 0, 0), 3)
        
        if score is None:
            cv2.putText(img, "Waiting for data...", (50, 200),
                        font, 1.5, (100, 100, 100), 2)
            return img
        
        # ===== 중앙: 위험도 게이지 (원형) =====
        cx, cy = 300, 400
        radius = 150
        
        # 배경 원
        cv2.circle(img, (cx, cy), radius, (200, 200, 200), -1)
        
        # 위험도에 따른 색상
        if score.risk_level == "위험":
            color = (0, 0, 255)  # 빨강
        elif score.risk_level == "주의":
            color = (0, 165, 255)  # 주황
        else:
            color = (0, 255, 0)  # 녹색
        
        # 위험도 게이지 (호)
        angle = int(score.smoothed_score * 1.8)  # 0~100 -> 0~180도
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, angle, color, 15)
        
        # 중앙 점수 표시
        cv2.circle(img, (cx, cy), 80, (255, 255, 255), -1)
        cv2.putText(img, f"{score.smoothed_score:.0f}", (cx-40, cy+20),
                    font, 2.5, color, 3)
        
        # ===== 우측: 상태 표시 =====
        text_x = 600
        y_offset = 200
        
        cv2.putText(img, f"Status: {score.risk_level}", (text_x, y_offset),
                    font, 1.8, color, 2)
        
        y_offset += 80
        cv2.putText(img, f"Instant: {score.instant_score:.1f}", (text_x, y_offset),
                    font, 1.2, (0, 0, 0), 2)
        
        y_offset += 60
        cv2.putText(img, f"Smoothed: {score.smoothed_score:.1f}", (text_x, y_offset),
                    font, 1.2, (0, 0, 0), 2)
        
        y_offset += 60
        cv2.putText(img, f"Drowsy Time: {score.accumulated_drowsy_time:.1f}s", 
                    (text_x, y_offset), font, 1.2, (0, 0, 0), 2)
        
        # ===== 하단: 경고 메시지 =====
        if score.should_alert:
            cv2.rectangle(img, (50, h-150), (w-50, h-50), (0, 0, 255), -1)
            cv2.putText(img, "⚠️ DROWSINESS DETECTED - STAY ALERT ⚠️", 
                        (100, h-80), font, 2, (255, 255, 255), 3)
        
        # ===== 좌측 하단: 통계 =====
        stat_y = h - 250
        cv2.putText(img, f"Frames: {stats['frame_count']}", (50, stat_y),
                    font, 1, (0, 0, 0), 1)
        cv2.putText(img, f"Alerts: {stats['alert_count']}", (50, stat_y + 40),
                    font, 1, (0, 0, 0), 1)
        cv2.putText(img, f"FPS: {stats['fps']:.1f}", (50, stat_y + 80),
                    font, 1, (0, 0, 0), 1)
        
        return img


class SimulatedMUSE2Reader:
    """
    테스트용: 시뮬레이션된 MUSE2 데이터 생성
    실제에서는 muse2-lsl 라이브러리 사용
    """
    
    def __init__(self, callback: Callable, interval: float = 1.0):
        self.callback = callback
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._generate_stream, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _generate_stream(self):
        """시뮬레이션: 1초씩 EEG 데이터 생성"""
        FS = 256
        seed = 0
        
        while self.running:
            seed += 1
            np.random.seed(seed)
            t = np.arange(FS) / FS
            
            # 인위적 EEG 신호
            af7 = (15 * np.sin(2*np.pi*10*t) + 5 * np.sin(2*np.pi*20*t) +
                   np.random.normal(0, 8, FS)).astype(float).tolist()
            af8 = (12 * np.sin(2*np.pi*10*t + 0.5) + 4 * np.sin(2*np.pi*20*t) +
                   np.random.normal(0, 8, FS)).astype(float).tolist()
            
            self.callback(af7, af8)
            time.sleep(self.interval)


def main():
    """메인 루프: 실시간 졸음운전 감지"""
    
    print("🚀 졸음운전 감지 시스템 시작")
    
    # ===== 초기화 =====
    detector = RealtimeDrowsinessDetector(
        api_url="http://localhost:8000",
        alert_callback=lambda s: print(f"  ⚠️ ALERT! {s.risk_level} | score={s.smoothed_score:.1f}"),
    )
    
    if not detector.start_session():
        print("❌ 시작 실패")
        return
    
    # ===== MUSE2 스트림 시작 (시뮬레이션) =====
    muse_reader = SimulatedMUSE2Reader(
        callback=detector.process_eeg_chunk,
        interval=1.0
    )
    muse_reader.start()
    
    # ===== OpenCV 디스플레이 =====
    try:
        while True:
            score = detector.get_latest_score()
            stats = detector.get_stats()
            
            # 대시보드 렌더링
            img = OpenCVDisplay.create_dashboard(score, stats)
            
            # 디스플레이 (Jetson의 경우 DSP 사용, 시뮬레이션은 창)
            cv2.imshow("Drowsiness Detection", img)
            
            # 'q' 키로 종료
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️ 종료 중...")
    
    finally:
        muse_reader.stop()
        detector.end_session()
        cv2.destroyAllWindows()
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()
