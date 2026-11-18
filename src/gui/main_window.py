from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

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

        file_btn = QPushButton("Abrir arquivo")
        file_btn.clicked.connect(self.open_file)

        buttons = QHBoxLayout()
        buttons.addWidget(file_btn)
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
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.metrics_ready.connect(self.show_metrics)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo ou imagem", "", MEDIA_FILTER)
        if path:
            self.start(path)

    def show_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_metrics(self, m):
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
        QMessageBox.warning(self, "Erro", message)

    def closeEvent(self, event):
        self.stop_worker()
        super().closeEvent(event)
