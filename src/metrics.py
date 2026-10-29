"""
metrics.py — Medidas geométricas extraídas dos landmarks.

EAR — Eye Aspect Ratio (Soukupová & Čech, 2016)
------------------------------------------------
Seis pontos ao redor do olho:

        p2    p3
    p1            p4
        p6    p5

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

Numerador = duas "alturas" do olho; denominador = largura (x2 para ficar na
mesma escala). Por ser uma RAZÃO, o EAR não depende de quão perto o rosto está
da câmera: se você chega mais perto, altura e largura crescem juntas.
    Olho aberto  -> ~0.25 a 0.35 (varia por pessoa!)
    Olho fechado -> ~0.05 a 0.15

MAR — Mouth Aspect Ratio
------------------------
Mesma ideia na boca (contorno interno dos lábios): média de três alturas
dividida pela largura. Boca fechada ~0; bocejo -> valor alto e sustentado.
"""

import numpy as np

# Índices do MediaPipe Face Landmarker, na ordem [p1, p2, p3, p4, p5, p6].
# "Direito/esquerdo" é do ponto de vista da PESSOA (não da imagem).
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Boca (lábios internos): cantos e três pares verticais (cima, baixo)
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICAL = [(82, 87), (13, 14), (312, 317)]

# Contornos completos, só para desenhar na tela
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                    466, 388, 387, 386, 385, 384, 398]
MOUTH_CONTOUR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width)


def mouth_aspect_ratio(pts):
    left, right = MOUTH_CORNERS
    width = _dist(pts[left], pts[right])
    if width < 1e-6:
        return 0.0
    heights = [_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL]
    return float(np.mean(heights)) / width


def compute_metrics(pts):
    """Calcula EAR de cada olho, EAR médio e MAR a partir dos 478 pontos."""
    ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
    ear_l = eye_aspect_ratio(pts, LEFT_EYE)
    return {
        "ear_right": ear_r,
        "ear_left": ear_l,
        "ear": (ear_r + ear_l) / 2.0,  # a média reduz o erro de um olho só
        "mar": mouth_aspect_ratio(pts),
    }


class EMA:
    """
    Média Móvel Exponencial: suaviza o sinal quadro a quadro.
        valor_suave = alpha * novo + (1 - alpha) * valor_suave_anterior
    alpha alto = reage rápido (mas oscila); alpha baixo = estável (mas atrasa).
    Os landmarks "tremem" alguns pixels de um quadro para o outro; sem isso o
    EAR oscilaria e geraria falsos alarmes.
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value
