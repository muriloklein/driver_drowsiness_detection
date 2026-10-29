"""
capture.py — Fonte de imagens unificada: webcam, arquivo de vídeo ou imagem.

Uso:
    src = VideoSource(0)                 # webcam 0
    src = VideoSource("teste.mp4")       # vídeo
    src = VideoSource("foto.jpg")        # imagem (repete o mesmo quadro)
"""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class VideoSource:
    def __init__(self, source=0, width=None, height=None):
        # "0", "1"... vindos da linha de comando chegam como texto
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.source = source
        self.is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS
        self.is_camera = isinstance(source, int)
        self._image = None
        self._cap = None

        if self.is_image:
            # cv2.imread não entende acentos em caminhos no Windows; esse truque resolve
            data = np.fromfile(source, dtype=np.uint8)
            self._image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if self._image is None:
                raise IOError(f"Não consegui ler a imagem: {source}")
        else:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise IOError(f"Não consegui abrir a fonte de vídeo: {source}")
            if width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def fps(self):
        """FPS nominal da fonte (vídeos têm um valor fixo; webcam é aproximado)."""
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else 30.0

    def read(self):
        """Retorna (ok, frame_BGR). Para imagem, devolve sempre uma cópia da mesma."""
        if self.is_image:
            return True, self._image.copy()
        ok, frame = self._cap.read()
        if ok and self.is_camera:
            frame = cv2.flip(frame, 1)  # espelha a webcam (fica natural, como espelho)
        return ok, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
