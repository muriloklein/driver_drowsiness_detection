import time

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.capture import VideoSource
from src.demo_etapa1 import GREEN, YELLOW, draw_contour, draw_ear_points
from src.landmarks import FaceLandmarkDetector
from src.metrics import (EMA, LEFT_EYE, LEFT_EYE_CONTOUR, MOUTH_CONTOUR,
                         RIGHT_EYE, RIGHT_EYE_CONTOUR, compute_metrics)
from src.preprocessing import Preprocessor


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.use_preprocessing = True
        self._running = True

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        try:
            src = VideoSource(self.source)
            detector = FaceLandmarkDetector(video_mode=not src.is_image)
        except (IOError, FileNotFoundError) as e:
            self.failed.emit(str(e))
            return

        pre = Preprocessor()
        ear_ema = EMA(alpha=0.5)
        frame_time = 0.0 if src.is_camera else 1.0 / src.fps
        fps = 0.0

        while self._running:
            start = time.time()
            ok, frame = src.read()
            if not ok:
                break

            proc = pre.process(frame) if self.use_preprocessing else frame
            pts = detector.detect(proc)
            vis = proc.copy()

            metrics = {"face": False, "low_light": pre.last_low_light and self.use_preprocessing}
            if pts is not None:
                m = compute_metrics(pts)
                if m["ear"] is not None:
                    m["ear"] = ear_ema.update(m["ear"])
                    metrics.update(m)
                    metrics["face"] = True
                draw_contour(vis, pts, RIGHT_EYE_CONTOUR, GREEN)
                draw_contour(vis, pts, LEFT_EYE_CONTOUR, GREEN)
                draw_contour(vis, pts, MOUTH_CONTOUR, YELLOW)
                draw_ear_points(vis, pts, RIGHT_EYE)
                draw_ear_points(vis, pts, LEFT_EYE)

            h, w = vis.shape[:2]
            self.frame_ready.emit(QImage(vis.data, w, h, 3 * w, QImage.Format_BGR888).copy())

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            fps = 0.9 * fps + 0.1 / max(time.time() - start, 1e-6)
            metrics["fps"] = fps
            self.metrics_ready.emit(metrics)

        src.release()
        detector.close()
