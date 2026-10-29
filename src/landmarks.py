"""
landmarks.py — Localiza 478 pontos do rosto com o MediaPipe Face Landmarker.

Como funciona por dentro (resumo):
    É um pipeline de duas redes neurais:
      1) um detector encontra ONDE está o rosto (caixa delimitadora);
      2) uma segunda rede recebe o recorte do rosto e regride as coordenadas
         (x, y, z) de 478 pontos fixos: cada índice é sempre o mesmo lugar
         anatômico (ex.: o ponto 33 é sempre o canto externo do olho direito).
    No modo VIDEO, ele reaproveita a posição do quadro anterior (tracking) e
    só roda o detector de novo quando perde o rosto, por isso é rápido.

ATENÇÃO (MediaPipe >= 1.0): o antigo `mp.solutions.face_mesh` foi REMOVIDO.
A maioria dos tutoriais online usa ele e não vai funcionar. Usamos a API
"Tasks" (mediapipe.tasks.python.vision.FaceLandmarker).
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class FaceLandmarkDetector:
    def __init__(self, model_path=DEFAULT_MODEL, video_mode=True,
                 min_detection_conf=0.5, min_tracking_conf=0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_path}.\n"
                f"Rode primeiro:  python tools/download_models.py"
            )

        self.video_mode = video_mode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,  # só o motorista
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr):
        """
        Retorna um array NumPy (478, 2) com as coordenadas EM PIXELS de cada
        ponto, ou None se nenhum rosto foi encontrado.
        """
        h, w = frame_bgr.shape[:2]

        # A OpenCV trabalha em BGR; o MediaPipe espera RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.video_mode:
            # O modo VIDEO exige timestamps (ms) sempre crescentes.
            ts = int((time.monotonic() - self._t0) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        # O MediaPipe devolve coordenadas NORMALIZADAS (0 a 1). Convertemos para
        # pixels porque o EAR é uma razão de distâncias: com x e y normalizados
        # por valores diferentes (largura != altura), as distâncias ficariam
        # distorcidas e o EAR mudaria só por mudar a resolução da câmera.
        pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
                       dtype=np.float32)
        return pts

    def close(self):
        self._landmarker.close()
