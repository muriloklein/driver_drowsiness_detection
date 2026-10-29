"""
preprocessing.py — Melhoria da imagem antes da detecção.

Problema real: motorista dirigindo à noite ou num túnel = imagem escura e com
pouco contraste. A rede de landmarks erra mais nessas condições.

Técnicas usadas (todas cobradas no critério "Aplicação de filtros e técnicas"):

1. Correção gama (transformação de intensidade não linear)
   saída = 255 * (entrada/255) ** (1/gama)
   gama > 1 clareia os tons escuros sem estourar os claros.
   Implementado com uma LUT (look-up table): pré-calculamos as 256 saídas
   possíveis uma vez e depois é só "consultar a tabela", que é muito rápido.

2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   A equalização de histograma comum espalha os tons da imagem inteira de uma
   vez. A CLAHE divide a imagem em blocos (tiles) e equaliza cada um, com um
   limite (clipLimit) para não amplificar ruído. Resultado: realça detalhes
   locais, como o contorno dos olhos numa região escura.
   Aplicamos só no canal L (luminosidade) do espaço LAB, para mexer no brilho
   sem distorcer as cores.

3. Filtro Gaussiano (suavização / redução de ruído)
   Câmeras em pouca luz geram ruído; o blur leve atenua isso.
"""

import cv2
import numpy as np


def mean_brightness(frame_bgr):
    """Brilho médio da imagem (0 = preto, 255 = branco)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def build_gamma_lut(gamma):
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv) * 255
    return np.clip(table, 0, 255).astype(np.uint8)


class Preprocessor:
    def __init__(self, low_light_threshold=80, gamma=1.6, clip_limit=2.0,
                 tile_grid=(8, 8), blur_ksize=3):
        self.low_light_threshold = low_light_threshold
        self.gamma_lut = build_gamma_lut(gamma)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.blur_ksize = blur_ksize
        self.last_brightness = 0.0
        self.last_low_light = False

    def apply_clahe(self, frame_bgr):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def process(self, frame_bgr):
        """
        Pipeline adaptativo: só aplica as correções pesadas quando a cena está
        escura. Em cena bem iluminada, mexer na imagem pode até piorar.
        """
        self.last_brightness = mean_brightness(frame_bgr)
        self.last_low_light = self.last_brightness < self.low_light_threshold

        out = frame_bgr
        if self.last_low_light:
            out = cv2.LUT(out, self.gamma_lut)         # 1) clareia
            out = self.apply_clahe(out)                # 2) realça contraste local
        if self.blur_ksize and self.blur_ksize > 1:
            out = cv2.GaussianBlur(out, (self.blur_ksize, self.blur_ksize), 0)  # 3) tira ruído
        return out
