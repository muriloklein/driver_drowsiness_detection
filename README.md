# Detector de Sono e Fadiga de Motoristas

Trabalho prático da disciplina **CCC309 – Processamento de Imagens e Visão Computacional** (UPF, 2026/2), Prof. Dr. Rafael Rieder.

Sistema que monitora o rosto do motorista pela câmera e emite alertas quando detecta sinais de sonolência: olhos fechados por tempo prolongado (EAR e PERCLOS), bocejos (MAR) e queda da cabeça. Combina landmarks faciais com uma CNN própria para classificação do estado dos olhos, executada via OpenCV DNN (com suporte a CUDA).

> 🎬 **Vídeo demonstrativo:** _link será adicionado aqui_

## Integrantes

- Murilo 
- Guilherme 
- Giuliano

## Requisitos

- Python **3.13** (testado; todos do grupo devem usar a mesma versão)
- Webcam
- Opcional: GPU NVIDIA + OpenCV compilada com CUDA

## Instalação

```bash
git clone <url-do-repositorio>
cd detector-fadiga

# Criar e ativar o ambiente virtual com Python 3.13
# Windows:
py -3.13 -m venv .venv
# Linux/macOS:
python3.13 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Testar o ambiente e a webcam
python tools/check_env.py

# Baixar os modelos pré-treinados (uma vez)
python tools/download_models.py
```

## Como rodar

Sempre a partir da raiz do projeto, com o `.venv` ativado:

```bash
python -m src.demo_etapa1              # webcam
python -m src.demo_etapa1 video.mp4    # arquivo de vídeo
python -m src.demo_etapa1 foto.jpg     # imagem
```

## Estrutura

```
src/
  capture.py         # leitura de webcam / vídeo / imagem
  preprocessing.py   # CLAHE, correção gama, filtros
  landmarks.py       # MediaPipe Face Mesh
  metrics.py         # EAR, MAR, PERCLOS, pose da cabeça
  eye_classifier.py  # CNN de olhos via cv2.dnn
  fatigue.py         # lógica de decisão
  gui/               # interface PySide6
models/              # modelos .onnx
training/            # notebook de treino da CNN
tools/               # scripts utilitários
```