"""
download_models.py — Baixa os modelos pré-treinados usados pelo projeto.

Rodar uma vez: python tools/download_models.py

Os arquivos vão para a pasta models/.
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODELS = {
    # Rede neural do MediaPipe que localiza 478 pontos do rosto (incluindo íris)
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"[ok] {name} já existe ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[..] baixando {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[ok] {name} salvo ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
