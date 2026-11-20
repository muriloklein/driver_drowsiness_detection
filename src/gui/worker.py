import time

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.calibration import EarCalibrator
from src.capture import VideoSource
from src.demo_etapa1 import GREEN, YELLOW, draw_contour, draw_ear_points
from src.fatigue import FatigueMonitor
from src.landmarks import FaceLandmarkDetector
from src.metrics import (EMA, LEFT_EYE, LEFT_EYE_CONTOUR, MOUTH_CONTOUR,
                         RIGHT_EYE, RIGHT_EYE_CONTOUR, compute_metrics)
from src.preprocessing import Preprocessor


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(object)
    state_ready = Signal(object)
    failed = Signal(str)
    calibration_finished = Signal(object, object)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.use_preprocessing = True
        self._running = True
        self._calibrate_requested = False

    def request_calibration(self):
        self._calibrate_requested = True

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        try:
            src = VideoSource(self.source)
        except (IOError, FileNotFoundError) as e:
            self.failed.emit(str(e))
            return
        try:
            detector = FaceLandmarkDetector(video_mode=not src.is_image)
        except (IOError, FileNotFoundError) as e:
            src.release()
            self.failed.emit(str(e))
            return

        try:
            self._loop(src, detector)
        except Exception as e:  # nunca deixa a thread morrer em silêncio
            self.failed.emit(f"Erro no processamento: {e}")
        finally:
            src.release()
            detector.close()

    def _loop(self, src, detector):
        pre = Preprocessor()
        ear_ema = EMA(alpha=0.5)
        calibrator = EarCalibrator()
        monitor = FatigueMonitor()  # novo worker = nova fonte = monitor zerado
        frame_time = 0.0 if src.is_camera else 1.0 / src.fps
        fps = 0.0
        frame_index = 0

        while self._running:
            start = time.time()
            ok, frame = src.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            # webcam: relógio real; arquivo: tempo do vídeo (índice / fps)
            t = time.monotonic() if src.is_camera else frame_index / src.fps
            dt = None if src.is_camera else 1.0 / src.fps
            frame_index += 1

            proc = pre.process(frame) if self.use_preprocessing else frame
            pts = detector.detect(proc)
            vis = proc.copy()

            metrics = {"face": False, "low_light": pre.last_low_light and self.use_preprocessing}
            if pts is not None:
                m = compute_metrics(pts, (w, h))
                if m["ear"] is not None:
                    m["ear"] = ear_ema.update(m["ear"])
                    metrics.update(m)
                    metrics["face"] = True
                draw_contour(vis, pts, RIGHT_EYE_CONTOUR, GREEN)
                draw_contour(vis, pts, LEFT_EYE_CONTOUR, GREEN)
                draw_contour(vis, pts, MOUTH_CONTOUR, YELLOW)
                draw_ear_points(vis, pts, RIGHT_EYE)
                draw_ear_points(vis, pts, LEFT_EYE)
            else:
                ear_ema.value = None  # sem rosto: não arrasta EAR antigo

            if self._calibrate_requested:
                self._calibrate_requested = False
                calibrator.start(t)
            metrics["calibration_progress"] = None
            if calibrator.active:
                finished = calibrator.update(metrics if metrics["face"] else None, t)
                metrics["calibration_progress"] = calibrator.progress
                if finished:
                    if calibrator.result is not None and not calibrator.error:
                        monitor.apply_calibration(calibrator.result)
                    self.calibration_finished.emit(calibrator.result, calibrator.error)

            state = monitor.update(metrics if metrics["face"] else None, dt)

            self.frame_ready.emit(QImage(vis.data, w, h, 3 * w, QImage.Format_BGR888).copy())

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            fps = 0.9 * fps + 0.1 / max(time.time() - start, 1e-6)
            metrics["fps"] = fps
            self.metrics_ready.emit(metrics)
            self.state_ready.emit(state)
