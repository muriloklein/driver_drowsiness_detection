# Detector de Sono e Fadiga de Motoristas

Trabalho prático da disciplina **CCC309: Processamento de Imagens e Visão Computacional** (UPF, 2026/2), Prof. Dr. Rafael Rieder.

Sistema que monitora o rosto do motorista pela câmera e emite alertas quando detecta sinais de sonolência: olhos fechados por tempo prolongado (EAR e PERCLOS), bocejos (MAR) e queda da cabeça. Combina landmarks faciais com uma CNN própria para classificação do estado dos olhos, executada via OpenCV DNN (com suporte a CUDA).

> 🎬 **Vídeo demonstrativo:** _link será adicionado aqui_

## Integrantes

- Murilo K. Klein
- Guilherme S. Machado
- Giuliano L. Chiochietta

## Requisitos

- Python **3.13** (versão testada. Todos do grupo devem usar a mesma)
- Webcam
- Opcional: GPU NVIDIA + OpenCV compilada com CUDA

## Instalação

```bash
git clone https://github.com/muriloklein/driver_drowsiness_detection.git
cd driver_drowsiness_detection
git checkout develop

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
  landmarks.py       # MediaPipe Face Landmarker (478 pontos)
  metrics.py         # EAR, MAR, PERCLOS, pose da cabeça e suavização
  calibration.py     # limiar de EAR por pessoa
  fatigue.py         # lógica de decisão
  alert.py           # alerta sonoro
  gui/               # interface PySide6 (janela, gráfico, alertas)
docs/                # plano do projeto
models/              # modelos baixados por tools/download_models.py
tests/               # testes unitários de metrics e fatigue
training/            # notebook de treino da CNN de olhos (extra)
tools/               # scripts utilitários
```

## Desenvolvimento

O trabalho acontece na branch `develop` e a `main` recebe só versões estáveis. O plano, a divisão de tarefas e os prazos estão em [docs/plano-projeto-final.md](docs/plano-projeto-final.md).
