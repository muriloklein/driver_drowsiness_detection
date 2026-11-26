# Detector de Sono e Fadiga de Motoristas

Trabalho prático da disciplina **CCC309: Processamento de Imagens e Visão Computacional** (UPF, 2026/2), Prof. Dr. Rafael Rieder.

Sistema que monitora o rosto do motorista pela câmera e emite alertas quando detecta sinais de sonolência: olhos fechados por tempo prolongado (EAR e PERCLOS), bocejos (MAR) e queda da cabeça.

> 🎬 **Vídeo demonstrativo:** _link será adicionado aqui_

## Integrantes

- Murilo K. Klein
- Guilherme S. Machado
- Giuliano L. Chiochetta

## Como funciona

1. **Pré-processamento:** em cena escura, aplica correção gama e CLAHE para clarear e realçar o contraste. Um filtro Gaussiano leve reduz o ruído.
2. **Landmarks faciais:** o MediaPipe Face Landmarker, uma rede neural pronta (Deep Learning), localiza 478 pontos do rosto.
3. **Métricas geométricas:** a partir dos pontos são calculados o EAR (abertura dos olhos), o MAR (abertura da boca), o PERCLOS (fração do tempo com os olhos fechados) e a pose da cabeça (via `cv2.solvePnP`).
4. **Calibração:** o limiar de EAR e a posição de referência da cabeça são ajustados para cada pessoa.
5. **Máquina de estados temporal:** combina as métricas ao longo do tempo e define o nível `OK`, `ATENÇÃO` ou `ALERTA`. No `ALERTA`, uma faixa vermelha pisca e um som é tocado.

| Sinal | Condição | Nível |
| --- | --- | --- |
| Olhos fechados | EAR abaixo do limiar por mais de 2 s | `ALERTA` |
| Cabeça caída | Inclinação 20° acima da referência por mais de 2 s | `ALERTA` |
| PERCLOS | Mais de 20 % nos últimos 60 s | `ATENÇÃO` |
| Bocejos | 3 bocejos em 5 min (MAR acima de 0,6 por mais de 1,5 s) | `ATENÇÃO` |
| Rosto ausente | Mais de 3 s sem rosto | `ATENÇÃO` |

## Requisitos

- Python 3.11 ou superior (testado na 3.11, 3.13 e 3.14)
- Webcam (dispensável para analisar arquivos de vídeo ou imagem)

## Instalação

Clone o repositório:

```bash
git clone https://github.com/muriloklein/driver_drowsiness_detection.git
cd driver_drowsiness_detection
```

A branch `main` traz a versão estável. Para desenvolver, mude para a `develop` com `git checkout develop`.

Rode o setup. Ele cria o `.venv`, instala as dependências, baixa o modelo e testa o ambiente:

```bash
python setup.py
```

Ative o ambiente virtual. Isso precisa ser feito a cada novo terminal:

| Terminal | Comando |
| --- | --- |
| PowerShell ou CMD (Windows) | `.venv\Scripts\activate` |
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| Linux ou macOS | `source .venv/bin/activate` |

Quando estiver ativo, o terminal mostra `(.venv)` no começo da linha.

## Como rodar

Sempre a partir da raiz do projeto, com o `.venv` ativado:

```bash
python -m src.gui
```

A janela abre já com a webcam ligada.

| Controle | O que faz |
| --- | --- |
| **Ligar webcam / Desligar webcam** | Inicia ou encerra a captura da webcam |
| **Abrir arquivo** | Analisa um vídeo (`.mp4`, `.avi`, `.mov`, `.mkv`) ou uma imagem (`.jpg`, `.jpeg`, `.png`, `.bmp`) |
| **Calibrar** | Mede por 3 s o seu EAR de olhos abertos e a posição da cabeça. Olhe para a câmera até a barra de progresso terminar |
| **Pré-processamento** | Liga ou desliga os filtros, para comparar o antes e o depois em cena escura |

O painel lateral mostra o estado (com o motivo quando há `ATENÇÃO` ou `ALERTA`), EAR, MAR, Pitch (inclinação da cabeça, positiva para baixo), PERCLOS, bocejos, FPS e o limiar de EAR. O gráfico mostra o EAR dos últimos 10 s junto com a linha do limiar. A calibração vale para a fonte atual: ao trocar de fonte, calibre de novo.

### Demo da etapa 1

A demo original, sem a interface gráfica, continua disponível:

```bash
python -m src.demo_etapa1              # webcam
python -m src.demo_etapa1 video.mp4    # arquivo de vídeo
python -m src.demo_etapa1 foto.jpg     # imagem
```

Teclas na demo: `q` sai, `p` liga ou desliga o pré-processamento, `m` mostra os 478 pontos e `e` liga ou desliga a suavização.

Se a webcam não abrir, teste outro índice na demo (`python -m src.demo_etapa1 1`) e confira em Configurações, Privacidade e segurança, Câmera se os aplicativos de desktop têm acesso.

## Testes

Com o `.venv` ativado, na raiz do projeto:

```bash
pytest
```

## Estrutura

```
src/
  capture.py         # leitura de webcam / vídeo / imagem
  preprocessing.py   # CLAHE, correção gama, filtro Gaussiano
  landmarks.py       # MediaPipe Face Landmarker (478 pontos)
  metrics.py         # EAR, MAR, PERCLOS, pose da cabeça e suavização
  calibration.py     # limiar de EAR por pessoa
  fatigue.py         # máquina de estados temporal
  alert.py           # alerta sonoro
  gui/               # interface PySide6 (janela, gráfico, alertas)
docs/                # plano do projeto
models/              # modelos baixados por tools/download_models.py
tests/               # testes de metrics, calibration, fatigue e alert
tools/               # scripts utilitários
```

## Desenvolvimento

O trabalho acontece na branch `develop` e a `main` recebe só versões estáveis. O plano, a divisão de tarefas e os prazos estão em [docs/plano-projeto-final.md](docs/plano-projeto-final.md).
