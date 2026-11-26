import time
from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from src.alert import AlertSound
from src.calibration import DEFAULT_EAR_THRESHOLD
from src.fatigue import FatigueLevel
from src.gui.alert_banner import AlertBanner
from src.gui.worker import VideoWorker

MEDIA_FILTER = "Vídeos e imagens (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp)"
GRAPH_SECONDS = 10
STATE_COLORS = {FatigueLevel.OK: "#2e7d32", FatigueLevel.ATENCAO: "#ef6c00",
                FatigueLevel.ALERTA: "#d32f2f"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Sono e Fadiga")
        self.worker = None
        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.calibration = None

        self.video = QLabel("Sem vídeo")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background: #111; color: #aaa;")

        self.webcam_btn = QPushButton("Ligar webcam")
        self.webcam_btn.setCheckable(True)
        self.webcam_btn.toggled.connect(self.toggle_webcam)
        file_btn = QPushButton("Abrir arquivo")
        file_btn.clicked.connect(self.open_file)
        self.calibrate_btn = QPushButton("Calibrar")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.pre_check = QCheckBox("Pré-processamento")
        self.pre_check.setChecked(True)
        self.pre_check.toggled.connect(self.toggle_preprocessing)

        buttons = QHBoxLayout()
        buttons.addWidget(self.webcam_btn)
        buttons.addWidget(file_btn)
        buttons.addWidget(self.calibrate_btn)
        buttons.addWidget(self.pre_check)
        buttons.addStretch()

        self.state = QLabel("Estado: -")
        self.state.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.ear = QLabel("EAR: -")
        self.mar = QLabel("MAR: -")
        self.perclos = QLabel("PERCLOS: -")
        self.yawns = QLabel("Bocejos: -")
        self.fps = QLabel("FPS: -")
        self.light = QLabel("")
        self.threshold_label = QLabel(f"Limiar EAR: {self.ear_threshold:.2f}")
        self.calibration_bar = QProgressBar()
        self.calibration_bar.setRange(0, 100)
        self.calibration_bar.hide()

        panel = QVBoxLayout()
        for label in (self.state, self.ear, self.mar, self.perclos, self.yawns, self.fps, self.light, self.threshold_label):
            panel.addWidget(label)
        panel.addWidget(self.calibration_bar)
        panel.addStretch()
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(180)

        body = QHBoxLayout()
        body.addWidget(self.video, stretch=1)
        body.addWidget(panel_widget)

        self.history = deque()
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(140)
        self.plot.setYRange(0, 0.5)
        self.plot.setXRange(-GRAPH_SECONDS, 0)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.setLabel("left", "EAR")
        self.plot.setLabel("bottom", "segundos")
        self.ear_curve = self.plot.plot(pen=pg.mkPen("#4caf50", width=2))
        self.threshold_line = pg.InfiniteLine(pos=self.ear_threshold, angle=0,
                                              pen=pg.mkPen("#d32f2f", style=Qt.DashLine))
        self.plot.addItem(self.threshold_line)

        self.alert_banner = AlertBanner()
        self.alert_sound = AlertSound()

        root = QVBoxLayout()
        root.addWidget(self.alert_banner)
        root.addLayout(buttons)
        root.addLayout(body)
        root.addWidget(self.plot)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def start(self, source):
        self.stop_worker()
        self.reset_alert()
        self.history.clear()
        self.ear_curve.setData([], [])
        # Nova fonte = novo FatigueMonitor (no worker) com limiar padrão.
        self.calibration = None
        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.threshold_line.setValue(self.ear_threshold)
        self.threshold_label.setText(f"Limiar EAR: {self.ear_threshold:.2f}")
        self.worker = VideoWorker(source)
        self.worker.use_preprocessing = self.pre_check.isChecked()
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.metrics_ready.connect(self.show_metrics)
        self.worker.state_ready.connect(self.show_state)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.show_error)
        self.worker.calibration_finished.connect(self.finish_calibration)
        self.worker.start()
        self.calibrate_btn.setEnabled(True)

    def stop_worker(self):
        if self.worker is not None:
            worker, self.worker = self.worker, None  # sinais pendentes viram "antigos"
            worker.stop()
        self.reset_alert()

    def is_stale_signal(self):
        """True para sinal atrasado: fonte desligada ou worker que já foi trocado.

        Com a fonte desligada, sender() também pode ser None (worker destruído),
        por isso comparar só o remetente não basta.
        """
        return self.worker is None or self.sender() is not self.worker

    def reset_alert(self):
        self.alert_banner.set_state(FatigueLevel.OK)
        self.alert_sound.stop()

    def on_worker_finished(self):
        if self.is_stale_signal():
            return
        self.reset_alert()
        self.calibrate_btn.setEnabled(False)
        self.calibration_bar.hide()
        if self.webcam_btn.isChecked():
            self.set_webcam_button(False)

    def toggle_webcam(self, on):
        self.webcam_btn.setText("Desligar webcam" if on else "Ligar webcam")
        if on:
            self.start(0)
        else:
            self.stop_worker()
            self.clear_display()

    def set_webcam_button(self, on):
        self.webcam_btn.blockSignals(True)
        self.webcam_btn.setChecked(on)
        self.webcam_btn.setText("Desligar webcam" if on else "Ligar webcam")
        self.webcam_btn.blockSignals(False)

    def clear_display(self):
        self.video.clear()
        self.video.setText("Sem vídeo")
        self.state.setText("Estado: -")
        self.ear.setText("EAR: -")
        self.mar.setText("MAR: -")
        self.fps.setText("FPS: -")
        self.perclos.setText("PERCLOS: -")
        self.yawns.setText("Bocejos: -")
        self.state.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.light.setText("")
        self.calibrate_btn.setEnabled(False)
        self.calibration_bar.hide()
        self.history.clear()
        self.ear_curve.setData([], [])
        self.reset_alert()

    def calibrate(self):
        if self.worker is not None:
            self.calibrate_btn.setEnabled(False)
            self.calibration_bar.setValue(0)
            self.calibration_bar.show()
            self.worker.request_calibration()

    def finish_calibration(self, result, error):
        if self.is_stale_signal():
            return
        self.calibrate_btn.setEnabled(True)
        self.calibration_bar.hide()
        if error:
            QMessageBox.warning(self, "Calibração", error)
            return
        self.calibration = result
        self.ear_threshold = result.ear_threshold
        self.threshold_line.setValue(self.ear_threshold)
        self.threshold_label.setText(f"Limiar EAR: {self.ear_threshold:.2f} (calibrado)")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo ou imagem", "", MEDIA_FILTER)
        if path:
            self.set_webcam_button(False)
            self.start(path)

    def toggle_preprocessing(self, checked):
        if self.worker is not None:
            self.worker.use_preprocessing = checked

    def show_frame(self, image):
        if self.is_stale_signal():
            return
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_metrics(self, m):
        if self.is_stale_signal():
            return
        self.fps.setText(f"FPS: {m['fps']:.0f}")
        self.light.setText("Baixa luz" if m["low_light"] else "")
        progress = m["calibration_progress"]
        if progress is not None:
            self.calibration_bar.setValue(int(progress * 100))
        if not m["face"]:
            self.ear.setText("EAR: -")
            self.mar.setText("MAR: -")
            return
        self.ear.setText(f"EAR: {m['ear']:.3f}")
        self.mar.setText(f"MAR: {m['mar']:.3f}")
        now = time.monotonic()
        self.history.append((now, m["ear"]))
        while now - self.history[0][0] > GRAPH_SECONDS:
            self.history.popleft()
        self.ear_curve.setData([t - now for t, _ in self.history], [e for _, e in self.history])

    def show_state(self, state):
        """Único ponto que lê o FatigueState vindo do worker."""
        if self.is_stale_signal():
            return
        if not state.face_present and state.level == FatigueLevel.OK:
            self.state.setText("Rosto não detectado")
        else:
            self.state.setText(f"Estado: {state.level.name}")
        color = STATE_COLORS[state.level]
        self.state.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        self.perclos.setText(f"PERCLOS: {state.perclos * 100:.0f} %" if state.perclos_ready
                             else "PERCLOS: calculando...")
        self.yawns.setText(f"Bocejos: {state.yawn_count}")
        self.update_alert(state)

    def update_alert(self, state):
        """Recebe um FatigueState: faixa vermelha + som."""
        self.alert_banner.set_state(state.level, state.reasons)
        self.alert_sound.update(state.level)

    def show_error(self, message):
        if self.is_stale_signal():
            return
        self.set_webcam_button(False)
        self.clear_display()
        QMessageBox.warning(self, "Erro", message)

    def closeEvent(self, event):
        self.stop_worker()
        self.alert_sound.close()
        super().closeEvent(event)
