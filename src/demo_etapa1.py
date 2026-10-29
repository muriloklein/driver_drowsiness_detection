"""
demo_etapa1.py — Demonstração da Etapa 1: rosto -> landmarks -> EAR/MAR ao vivo.

Rodar (o "-m" é importante, faz os imports funcionarem):
    python -m src.demo_etapa1             # webcam 0
    python -m src.demo_etapa1 1           # webcam 1
    python -m src.demo_etapa1 video.mp4   # arquivo de vídeo
    python -m src.demo_etapa1 foto.jpg    # imagem

Teclas:
    q  sair
    p  liga/desliga o pré-processamento (ambiente escuro)
    m  mostra/esconde todos os 478 pontos
    e  liga/desliga a suavização (EMA) do EAR
"""

import sys
import time
from collections import deque

import cv2
import numpy as np

from src.capture import VideoSource
from src.landmarks import FaceLandmarkDetector
from src.metrics import (EMA, LEFT_EYE, LEFT_EYE_CONTOUR, MOUTH_CONTOUR,
                         RIGHT_EYE, RIGHT_EYE_CONTOUR, compute_metrics)
from src.preprocessing import Preprocessor

# Limiar PROVISÓRIO só para visualização. Na Etapa 2 ele vira calibrado por pessoa.
EAR_PREVIEW_THRESHOLD = 0.20

GREEN, RED, YELLOW, WHITE, CYAN = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255), (255, 200, 0)


def draw_contour(frame, pts, idx, color):
    poly = pts[idx].astype(np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def draw_ear_points(frame, pts, idx):
    """Desenha os 6 pontos usados no EAR e as linhas de altura/largura."""
    p = pts[idx].astype(np.int32)
    cv2.line(frame, tuple(p[0]), tuple(p[3]), CYAN, 1)    # largura p1-p4
    cv2.line(frame, tuple(p[1]), tuple(p[5]), YELLOW, 1)  # altura p2-p6
    cv2.line(frame, tuple(p[2]), tuple(p[4]), YELLOW, 1)  # altura p3-p5
    for q in p:
        cv2.circle(frame, tuple(q), 2, WHITE, -1)


def draw_graph(frame, values, threshold, x, y, w=260, h=90, vmax=0.45):
    """Mini gráfico do EAR nos últimos N quadros, desenhado só com OpenCV."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # fundo semitransparente

    ty = int(y + h - (threshold / vmax) * h)
    cv2.line(frame, (x, ty), (x + w, ty), RED, 1)
    cv2.putText(frame, f"limiar {threshold:.2f}", (x + 4, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, RED, 1)

    if len(values) > 1:
        pts = [(int(x + i * w / (values.maxlen - 1)),
                int(y + h - min(v, vmax) / vmax * h)) for i, v in enumerate(values)]
        cv2.polylines(frame, [np.array(pts, np.int32)], False, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, "EAR", (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def put(frame, text, row, color=WHITE):
    cv2.putText(frame, text, (10, 25 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)  # contorno preto para ler em qualquer fundo
    cv2.putText(frame, text, (10, 25 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    src = VideoSource(source)
    detector = FaceLandmarkDetector(video_mode=not src.is_image)
    pre = Preprocessor()
    ear_ema = EMA(alpha=0.5)
    history = deque(maxlen=150)  # ~5 s a 30 FPS

    use_pre, show_mesh, use_ema = True, False, True
    fps, prev = 0.0, time.time()

    while True:
        ok, frame = src.read()
        if not ok:
            break  # fim do vídeo

        # 1) Pré-processamento (só atua de verdade em cena escura)
        proc = pre.process(frame) if use_pre else frame

        # 2) Landmarks
        pts = detector.detect(proc)
        vis = proc.copy()

        if pts is None:
            put(vis, "Rosto nao detectado", 0, RED)
        else:
            # 3) Métricas
            m = compute_metrics(pts)
            ear = ear_ema.update(m["ear"]) if use_ema else m["ear"]
            history.append(ear)

            # 4) Desenho
            if show_mesh:
                for q in pts.astype(np.int32):
                    cv2.circle(vis, tuple(q), 1, (180, 180, 180), -1)
            draw_contour(vis, pts, RIGHT_EYE_CONTOUR, GREEN)
            draw_contour(vis, pts, LEFT_EYE_CONTOUR, GREEN)
            draw_contour(vis, pts, MOUTH_CONTOUR, YELLOW)
            draw_ear_points(vis, pts, RIGHT_EYE)
            draw_ear_points(vis, pts, LEFT_EYE)

            closed = ear < EAR_PREVIEW_THRESHOLD
            put(vis, f"EAR: {ear:.3f}  (D {m['ear_right']:.2f} | E {m['ear_left']:.2f})", 0,
                RED if closed else GREEN)
            put(vis, f"MAR: {m['mar']:.3f}", 1, YELLOW)
            if closed:
                put(vis, "OLHOS FECHADOS", 2, RED)

            draw_graph(vis, history, EAR_PREVIEW_THRESHOLD, vis.shape[1] - 270, 10)

        # Status na parte de baixo
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        h = vis.shape[0]
        status = (f"FPS {fps:.1f} | brilho {pre.last_brightness:.0f}"
                  f"{' (BAIXA LUZ)' if pre.last_low_light and use_pre else ''}"
                  f" | [p] pre:{'on' if use_pre else 'off'} [e] ema:{'on' if use_ema else 'off'}"
                  f" [m] malha [q] sair")
        cv2.putText(vis, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.imshow("Etapa 1 - Landmarks e EAR", vis)
        key = cv2.waitKey(0 if src.is_image else 1) & 0xFF
        if key == ord("q") or (src.is_image and key == 27):
            break
        elif key == ord("p"):
            use_pre = not use_pre
        elif key == ord("m"):
            show_mesh = not show_mesh
        elif key == ord("e"):
            use_ema = not use_ema
            ear_ema.value = None

    src.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
