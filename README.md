# Detector de Sono e Fadiga de Motoristas

Trabalho prático da disciplina **CCC309: Processamento de Imagens e Visão Computacional** (UPF, 2026/2), Prof. Dr. Rafael Rieder.

Sistema que monitora o rosto do motorista pela câmera e emite alertas quando detecta sinais de sonolência: olhos fechados por tempo prolongado (EAR e PERCLOS), bocejos (MAR) e queda da cabeça. Combina landmarks faciais com uma CNN própria para classificação do estado dos olhos, executada via OpenCV DNN (com suporte a CUDA).

> 🎬 **Vídeo demonstrativo:** _link será adicionado aqui_

## Integrantes

- Murilo K. Klein
- Guilherme S. Machado
- Giuliano L. Chiochietta

## Requisitos

- Python 3.11 ou superior (testado na 3.11 e na 3.13)
- Webcam
- Opcional: GPU NVIDIA + OpenCV compilada com CUDA

## Instalação

Clone o repositório e entre na branch `develop`:

```bash
git clone https://github.com/muriloklein/driver_drowsiness_detection.git
cd driver_drowsiness_detection
git checkout develop
```

Rode o setup. Ele cria o `.venv`, instala as dependências, baixa o modelo e testa o ambiente:

```bash
python setup.py
```

Ative o ambiente virtual. Isso precisa ser feito a cada novo terminal:

| Terminal | Comando |
| --- | --- |
| PowerShell ou CMD (Windows) | `.venv\Scriptsctivate` |
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| Linux ou macOS | `source .venv/bin/activate` |

Quando estiver ativo, o terminal mostra `(.venv)` no começo da linha.

## Como rodar

Sempre a partir da raiz do projeto, com o `.venv` ativado:

```bash
python -m src.demo_etapa1              # webcam
python -m src.demo_etapa1 video.mp4    # arquivo de vídeo
python -m src.demo_etapa1 foto.jpg     # imagem
```

Teclas na demo: `q` sai, `p` liga ou desliga o pré-processamento, `m` mostra os 478 pontos e `e` liga ou desliga a suavização.

Se a webcam não abrir, teste outro índice (`python -m src.demo_etapa1 1`) e confira em Configurações, Privacidade e segurança, Câmera se os aplicativos de desktop têm acesso.

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
