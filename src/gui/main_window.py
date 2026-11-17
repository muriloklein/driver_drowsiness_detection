from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

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

        root = QVBoxLayout()
        root.addLayout(buttons)
        root.addWidget(self.video, stretch=1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def start(self, source):
        self.stop_worker()
        self.worker = VideoWorker(source)
        self.worker.frame_ready.connect(self.show_frame)
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

    def show_error(self, message):
        QMessageBox.warning(self, "Erro", message)

    def closeEvent(self, event):
        self.stop_worker()
        super().closeEvent(event)
