"""
demo_etapa1.py — Demonstração no terminal: landmarks, EAR/MAR, PERCLOS, pose
da cabeça e calibração, ao vivo. Serve para testar as métricas sem a GUI.

Rodar (o "-m" é importante, faz os imports funcionarem):
    python -m src.demo_etapa1             # webcam 0
    python -m src.demo_etapa1 1           # webcam 1
    python -m src.demo_etapa1 video.mp4   # arquivo de vídeo
    python -m src.demo_etapa1 foto.jpg    # imagem

Teclas:
    q  sair
    c  calibrar (olhe para a câmera, olhos abertos, por 3 s)
    p  liga/desliga o pré-processamento (ambiente escuro)
    m  mostra/esconde todos os 478 pontos
    e  liga/desliga a suavização (EMA) do EAR
"""

import math
import sys
import time
from collections import deque

import cv2
import numpy as np

from src.calibration import DEFAULT_EAR_THRESHOLD, EarCalibrator
from src.capture import VideoSource
from src.landmarks import FaceLandmarkDetector
from src.metrics import (EMA, LEFT_EYE, LEFT_EYE_CONTOUR, MOUTH_CONTOUR,
                         RIGHT_EYE, RIGHT_EYE_CONTOUR, PerclosMeter, compute_metrics)
from src.preprocessing import Preprocessor

NOSE_TIP = 1

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


def draw_head_direction(frame, pts, pitch, yaw, length=80):
    """Seta saindo do nariz para onde o rosto aponta (yaw -> x, pitch -> y)."""
    nose = pts[NOSE_TIP]
    end = (int(nose[0] + length * math.sin(math.radians(yaw))),
           int(nose[1] + length * math.sin(math.radians(pitch))))
    cv2.arrowedLine(frame, (int(nose[0]), int(nose[1])), end, CYAN, 2, cv2.LINE_AA, tipLength=0.25)


def draw_progress(frame, progress, text):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = w // 4, h // 2 - 15, 3 * w // 4, h // 2 + 15
    cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * progress), y1), CYAN, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 1)
    cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


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
    perclos = PerclosMeter(window=60.0)
    calibrator = EarCalibrator(duration=3.0)
    history = deque(maxlen=150)  # ~5 s a 30 FPS

    ear_threshold = DEFAULT_EAR_THRESHOLD
    pitch_baseline = 0.0
    message, message_until = "", 0.0

    use_pre, show_mesh, use_ema = True, False, True
    fps, prev = 0.0, time.time()
    frame_idx = 0

    while True:
        ok, frame = src.read()
        if not ok:
            break  # fim do vídeo

        # Relógio: na webcam, tempo real. No vídeo, o tempo DO VÍDEO (quadro / FPS),
        # senão o PERCLOS mudaria conforme a velocidade do computador.
        t = time.monotonic() if src.is_camera or src.is_image else frame_idx / src.fps
        frame_idx += 1

        # 1) Pré-processamento (só atua de verdade em cena escura)
        proc = pre.process(frame) if use_pre else frame

        # 2) Landmarks
        pts = detector.detect(proc)
        vis = proc.copy()

        m = None
        if pts is None:
            put(vis, "Rosto nao detectado", 0, RED)
        else:
            # 3) Métricas (image_size liga a pose da cabeça)
            m = compute_metrics(pts, image_size=(proc.shape[1], proc.shape[0]))
            ear = ear_ema.update(m["ear"]) if use_ema else m["ear"]
            m["ear"] = ear  # calibração e PERCLOS usam o MESMO EAR que aparece no gráfico
            history.append(ear)
            closed = ear < ear_threshold
            if not calibrator.active:
                perclos.update(closed, t)

            # 4) Desenho
            if show_mesh:
                for q in pts.astype(np.int32):
                    cv2.circle(vis, tuple(q), 1, (180, 180, 180), -1)
            draw_contour(vis, pts, RIGHT_EYE_CONTOUR, GREEN)
            draw_contour(vis, pts, LEFT_EYE_CONTOUR, GREEN)
            draw_contour(vis, pts, MOUTH_CONTOUR, YELLOW)
            draw_ear_points(vis, pts, RIGHT_EYE)
            draw_ear_points(vis, pts, LEFT_EYE)

            put(vis, f"EAR: {ear:.3f}  (D {m['ear_right']:.2f} | E {m['ear_left']:.2f})", 0,
                RED if closed else GREEN)
            put(vis, f"MAR: {m['mar']:.3f}", 1, YELLOW)
            put(vis, f"PERCLOS: {perclos.value * 100:.1f}%"
                     f"{'' if perclos.ready else ' (coletando...)'}", 2, WHITE)
            if m["head_pitch"] is not None:
                rel = m["head_pitch"] - pitch_baseline
                draw_head_direction(vis, pts, m["head_pitch"], m["head_yaw"])
                put(vis, f"Cabeca: pitch {rel:+.0f} | yaw {m['head_yaw']:+.0f}"
                         f" | roll {m['head_roll']:+.0f}", 3, CYAN)
            if closed:
                put(vis, "OLHOS FECHADOS", 4, RED)

            draw_graph(vis, history, ear_threshold, vis.shape[1] - 270, 10)

        # 5) Calibração
        if calibrator.active:
            finished = calibrator.update(m, t)
            draw_progress(vis, calibrator.progress, "Calibrando: olhe para a camera, olhos abertos")
            if finished:
                if calibrator.result:
                    r = calibrator.result
                    ear_threshold = r.ear_threshold
                    pitch_baseline = r.pitch_baseline if r.pitch_baseline is not None else 0.0
                    perclos.reset()  # o limiar mudou: o histórico antigo não vale mais
                    message = (f"Calibrado: EAR aberto {r.ear_open:.3f} -> limiar {ear_threshold:.3f}"
                               f" ({r.rejected} piscadas descartadas)")
                else:
                    message = calibrator.error
                message_until = time.monotonic() + 4.0
                print(message)
        if message and time.monotonic() < message_until:
            put(vis, message, 5, YELLOW)

        # Status na parte de baixo
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        h = vis.shape[0]
        status = (f"FPS {fps:.1f} | brilho {pre.last_brightness:.0f}"
                  f"{' (BAIXA LUZ)' if pre.last_low_light and use_pre else ''}"
                  f" | [p] pre:{'on' if use_pre else 'off'} [e] ema:{'on' if use_ema else 'off'}"
                  f" [c] calibrar [m] malha [q] sair")
        cv2.putText(vis, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.imshow("Detector de fadiga - demo das metricas", vis)
        # Imagem parada espera uma tecla; mas durante a calibração precisa seguir rodando
        wait = 0 if src.is_image and not calibrator.active and time.monotonic() > message_until else 1
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q") or (src.is_image and key == 27):
            break
        elif key == ord("c"):
            calibrator.start(t)
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
