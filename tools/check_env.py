"""
check_env.py — Verifica se o ambiente está pronto para o projeto.

Rode com:  python tools/check_env.py
Cada integrante do grupo deve rodar isso na própria máquina.

O que ele faz:
  1. Mostra versões do Python, OpenCV, NumPy e MediaPipe
  2. Verifica se a OpenCV tem suporte a CUDA (GPU NVIDIA)
  3. Abre a webcam e mostra o vídeo com o FPS na tela (tecla 'q' para sair)
"""

import sys
import time


def check_libs():
    print("=" * 50)
    print("1) Bibliotecas")
    print("=" * 50)
    print(f"Python ....... {sys.version.split()[0]}")

    try:
        import cv2
        print(f"OpenCV ....... {cv2.__version__}")
    except ImportError:
        print("OpenCV ....... NÃO INSTALADA -> pip install -r requirements.txt")
        sys.exit(1)

    try:
        import numpy as np
        print(f"NumPy ........ {np.__version__}")
    except ImportError:
        print("NumPy ........ NÃO INSTALADO")

    try:
        import mediapipe as mp
        print(f"MediaPipe .... {mp.__version__}")
    except ImportError:
        print("MediaPipe .... NÃO INSTALADO")

    # O módulo cv2.face só existe no opencv-contrib-python
    print(f"Contrib ...... {'OK' if hasattr(cv2, 'face') else 'ausente (instale opencv-contrib-python)'}")
    return cv2


def check_cuda(cv2):
    print("\n" + "=" * 50)
    print("2) CUDA (GPU NVIDIA)")
    print("=" * 50)
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        n = 0

    if n > 0:
        print(f"OpenCV com CUDA: SIM ({n} GPU(s) disponível(is))")
    else:
        print("OpenCV com CUDA: NÃO")
        print("  -> Normal! A OpenCV do pip vem SEM CUDA. Vamos desenvolver em CPU")
        print("     e compilar a OpenCV com CUDA só na máquina com NVIDIA (Etapa 4).")


def test_webcam(cv2, index=0):
    print("\n" + "=" * 50)
    print("3) Webcam")
    print("=" * 50)

    cap = cv2.VideoCapture(index)          # 0 = primeira câmera do sistema
    if not cap.isOpened():
        print(f"Não consegui abrir a câmera {index}. Tente outro índice: 1, 2...")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Câmera aberta: {w}x{h}. Pressione 'q' na janela para sair.")

    prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()             # frame é uma matriz NumPy (altura x largura x 3 canais BGR)
        if not ok:
            print("Falha ao ler quadro da câmera.")
            break

        frame = cv2.flip(frame, 1)          # espelha, fica mais natural (como um espelho)

        # FPS suavizado: média móvel para o número não ficar "pulando"
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"shape: {frame.shape}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Teste de ambiente - detector de fadiga", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"FPS médio aproximado: {fps:.1f}")


if __name__ == "__main__":
    cv2 = check_libs()
    check_cuda(cv2)
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_webcam(cv2, cam)
