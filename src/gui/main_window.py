from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from src.demo_etapa1 import EAR_PREVIEW_THRESHOLD
from src.gui.worker import VideoWorker

MEDIA_FILTER = "Vídeos e imagens (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector de Sono e Fadiga")
        self.worker = None

        self.video = QLabel("Sem vídeo")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background: #111; color: #aaa;")

        self.webcam_btn = QPushButton("Ligar webcam")
        self.webcam_btn.setCheckable(True)
        self.webcam_btn.toggled.connect(self.toggle_webcam)
        file_btn = QPushButton("Abrir arquivo")
        file_btn.clicked.connect(self.open_file)
        self.pre_check = QCheckBox("Pré-processamento")
        self.pre_check.setChecked(True)
        self.pre_check.toggled.connect(self.toggle_preprocessing)

        buttons = QHBoxLayout()
        buttons.addWidget(self.webcam_btn)
        buttons.addWidget(file_btn)
        buttons.addWidget(self.pre_check)
        buttons.addStretch()

        self.state = QLabel("Estado: -")
        self.state.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.ear = QLabel("EAR: -")
        self.mar = QLabel("MAR: -")
        self.fps = QLabel("FPS: -")
        self.light = QLabel("")

        panel = QVBoxLayout()
        for label in (self.state, self.ear, self.mar, self.fps, self.light):
            panel.addWidget(label)
        panel.addStretch()
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(180)

        body = QHBoxLayout()
        body.addWidget(self.video, stretch=1)
        body.addWidget(panel_widget)

        root = QVBoxLayout()
        root.addLayout(buttons)
        root.addLayout(body)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def start(self, source):
        self.stop_worker()
        self.worker = VideoWorker(source)
        self.worker.use_preprocessing = self.pre_check.isChecked()
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.metrics_ready.connect(self.show_metrics)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

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
        self.light.setText("")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo ou imagem", "", MEDIA_FILTER)
        if path:
            self.set_webcam_button(False)
            self.start(path)

    def toggle_preprocessing(self, checked):
        if self.worker is not None:
            self.worker.use_preprocessing = checked

    def show_frame(self, image):
        if self.worker is None:
            return
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_metrics(self, m):
        if self.worker is None:
            return
        self.fps.setText(f"FPS: {m['fps']:.0f}")
        self.light.setText("Baixa luz" if m["low_light"] else "")
        if not m["face"]:
            self.state.setText("Rosto não detectado")
            self.ear.setText("EAR: -")
            self.mar.setText("MAR: -")
            return
        self.ear.setText(f"EAR: {m['ear']:.3f}")
        self.mar.setText(f"MAR: {m['mar']:.3f}")
        self.state.setText("Estado: ATENÇÃO" if m["ear"] < EAR_PREVIEW_THRESHOLD else "Estado: OK")

    def show_error(self, message):
        self.set_webcam_button(False)
        self.clear_display()
        QMessageBox.warning(self, "Erro", message)

    def closeEvent(self, event):
        self.stop_worker()
        super().closeEvent(event)
