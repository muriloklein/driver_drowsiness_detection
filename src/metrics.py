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

PERCLOS — PERcentage of eyelid CLOSure
--------------------------------------
Fração do TEMPO, numa janela recente (ex.: últimos 60 s), em que os olhos
ficaram fechados. Diferente do EAR, não olha um quadro só: é uma medida
temporal, por isso é uma classe com estado (PerclosMeter).
    Desperto  -> poucos %, só das piscadas (~2 a 8 %)
    Sonolento -> piscadas longas e lentas fazem o valor subir (> 15 a 20 %)

Pose da cabeça (pitch, yaw, roll)
---------------------------------
Estimada com cv2.solvePnP ("Perspective-n-Point"): dados N pontos de um
modelo 3D de rosto genérico e onde esses mesmos pontos aparecem na imagem
(2D), a OpenCV encontra a rotação e a translação do rosto em relação à
câmera. Usamos 6 pontos: ponta do nariz, queixo, cantos externos dos olhos
e cantos da boca. O pitch (cabeça para baixo/cima) detecta o "cabeceio".
"""

import math
from collections import deque

import cv2
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


# --------------------------------------------------------------------------
# Pose da cabeça
# --------------------------------------------------------------------------

# Índices do MediaPipe dos 6 pontos usados no solvePnP, na MESMA ordem de
# FACE_MODEL_3D. "Esquerda/direita" aqui é da IMAGEM: o MediaPipe rotula pela
# aparência, então o ponto 33 fica sempre à esquerda num rosto de frente,
# mesmo com a webcam espelhada.
POSE_LANDMARKS = [
    1,    # ponta do nariz
    152,  # queixo
    33,   # canto externo do olho à esquerda da imagem
    263,  # canto externo do olho à direita da imagem
    61,   # canto da boca à esquerda da imagem
    291,  # canto da boca à direita da imagem
]

# Modelo 3D genérico de rosto (milímetros, aproximado), com a origem na ponta
# do nariz e os MESMOS eixos da câmera na OpenCV:
#   x -> direita da imagem, y -> para BAIXO, z -> para longe da câmera.
# Com eixos iguais aos da câmera, um rosto olhando de frente dá rotação ~zero,
# e os ângulos saem fáceis de interpretar (sem os famosos "pitch = 180°").
FACE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # nariz: o ponto mais próximo da câmera
    (0.0, 330.0, 65.0),       # queixo: abaixo e um pouco atrás do nariz
    (-225.0, -170.0, 135.0),  # olho esquerdo (imagem): acima e bem atrás
    (225.0, -170.0, 135.0),   # olho direito (imagem)
    (-150.0, 150.0, 125.0),   # boca esquerda (imagem)
    (150.0, 150.0, 125.0),    # boca direita (imagem)
], dtype=np.float64)

# SQPNP (OpenCV >= 4.5.3) encontra a solução ótima global, sem chute inicial.
# O ITERATIVE com poucos pontos às vezes "inverte" o rosto; fica de reserva.
_PNP_FLAG = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)


def camera_matrix(image_size):
    """
    Matriz intrínseca APROXIMADA da câmera. Não calibramos a webcam, então
    usamos a aproximação usual: distância focal ~ largura da imagem (em pixels)
    e centro óptico no meio da imagem. Para ângulos de cabeça é suficiente.
    """
    w, h = image_size
    f = float(w)
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def rotation_to_angles(R):
    """
    Converte a matriz de rotação do rosto em ângulos intuitivos (graus).

    Em vez de fórmulas de Euler decoradas, olhamos para dois vetores do rosto
    já rotacionados para o sistema da câmera:
      - frente = para onde o nariz aponta  = R @ (0, 0, -1)
      - lado   = da esquerda p/ direita    = R @ (1, 0, 0)
    pitch > 0: nariz aponta para baixo (cabeça caindo)
    yaw   > 0: rosto virado para a direita da imagem
    roll  > 0: cabeça inclinada com o lado direito (da imagem) para baixo
    """
    fwd = -R[:, 2]
    side = R[:, 0]
    pitch = math.degrees(math.atan2(fwd[1], -fwd[2]))
    yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
    roll = math.degrees(math.atan2(side[1], side[0]))
    return pitch, yaw, roll


def head_pose(pts, image_size):
    """
    Estima (pitch, yaw, roll) em graus a partir dos landmarks em pixels.
    Retorna None se o solvePnP falhar.

    Atenção: o modelo 3D é genérico, então mesmo olhando "reto" o pitch de uma
    pessoa pode não ser exatamente 0 (depende do rosto e da altura da câmera).
    Por isso a calibração guarda o pitch de referência de cada pessoa.
    """
    image_points = pts[POSE_LANDMARKS].astype(np.float64)
    ok, rvec, _tvec = cv2.solvePnP(FACE_MODEL_3D, image_points,
                                   camera_matrix(image_size), None, flags=_PNP_FLAG)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)  # vetor de rotação (3) -> matriz de rotação (3x3)
    return rotation_to_angles(R)


# --------------------------------------------------------------------------
# Métricas por quadro
# --------------------------------------------------------------------------

def compute_metrics(pts, image_size=None):
    """
    Calcula as métricas de UM quadro a partir dos 478 pontos (em pixels).

    image_size = (largura, altura) do quadro. É necessário para a pose da
    cabeça (vira a matriz da câmera). Sem ele, head_pitch/yaw/roll = None.
    Ex.: compute_metrics(pts, image_size=(frame.shape[1], frame.shape[0]))

    O PERCLOS NÃO está aqui porque depende do histórico: use PerclosMeter.
    """
    ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
    ear_l = eye_aspect_ratio(pts, LEFT_EYE)
    metrics = {
        "ear_right": ear_r,
        "ear_left": ear_l,
        "ear": (ear_r + ear_l) / 2.0,  # a média reduz o erro de um olho só
        "mar": mouth_aspect_ratio(pts),
        "head_pitch": None,
        "head_yaw": None,
        "head_roll": None,
    }
    if image_size is not None:
        pose = head_pose(pts, image_size)
        if pose is not None:
            metrics["head_pitch"], metrics["head_yaw"], metrics["head_roll"] = pose
    return metrics


# --------------------------------------------------------------------------
# Métricas temporais
# --------------------------------------------------------------------------

class PerclosMeter:
    """
    PERCLOS com janela deslizante POR TEMPO (não por número de quadros).

    Por que por tempo? O FPS varia (CPU ocupada, cena escura, vídeo de outra
    câmera). Contando quadros, uma janela de "1800 quadros" seria 60 s a 30 FPS
    mas 120 s a 15 FPS. Guardando a duração de cada amostra, a janela é sempre
    de `window` segundos e o PERCLOS é a fração do TEMPO com olho fechado.

    Uso (a cada quadro com rosto detectado):
        perclos = meter.update(ear < limiar, t)   # t em segundos
    Quadros sem rosto: simplesmente não chame update (o intervalo não conta).

    O relógio é passado de fora (t), o que deixa a classe fácil de testar: o
    teste inventa os tempos, sem precisar esperar 60 s de verdade.
    """

    def __init__(self, window=60.0, max_gap=0.25, min_coverage=10.0):
        self.window = window              # tamanho da janela, em segundos
        self.max_gap = max_gap            # intervalo máximo creditado a 1 amostra
        self.min_coverage = min_coverage  # segundos mínimos para o valor ser confiável
        self.reset()

    def reset(self):
        self._samples = deque()  # (t, duração, fechado)
        self._closed_time = 0.0
        self._total_time = 0.0
        self._last_t = None

    def update(self, closed, t):
        # Duração desta amostra = tempo desde a anterior. Se houve um buraco
        # grande (rosto sumiu, vídeo pausou), credita só max_gap, para o
        # buraco não contar como "olho aberto" nem como "olho fechado".
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), self.max_gap)
        self._last_t = t

        self._samples.append((t, dt, bool(closed)))
        self._total_time += dt
        if closed:
            self._closed_time += dt

        # Remove o que saiu da janela
        while self._samples and self._samples[0][0] < t - self.window:
            _, old_dt, old_closed = self._samples.popleft()
            self._total_time -= old_dt
            if old_closed:
                self._closed_time -= old_dt

        return self.value

    @property
    def value(self):
        """PERCLOS entre 0 e 1 (multiplique por 100 para %)."""
        if self._total_time <= 1e-9:
            return 0.0
        return min(max(self._closed_time / self._total_time, 0.0), 1.0)

    @property
    def coverage(self):
        """Quantos segundos de dados a janela contém agora."""
        return self._total_time

    @property
    def ready(self):
        """False nos primeiros segundos: poucos dados, o valor ainda oscila muito."""
        return self._total_time >= self.min_coverage


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
