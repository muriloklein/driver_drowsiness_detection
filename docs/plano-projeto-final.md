# Detector de Sono e Fadiga — Plano do Projeto Final

## Visão geral

Sistema que filma o rosto do motorista pela webcam e emite alerta quando detecta sinais de sonolência: olhos fechados por tempo prolongado, bocejos e queda da cabeça. Roda em desktop (Python + OpenCV + MediaPipe + PySide6).

O repositório hoje tem a **Etapa 1 pronta** (10 commits, todos do Murilo): captura de vídeo, pré-processamento para cena escura, 478 landmarks faciais, cálculo de EAR/MAR com suavização e uma demo no terminal com gráfico ao vivo. Tudo abaixo parte desse ponto.

**Decisão de escopo:** o componente de Deep Learning é o MediaPipe Face Landmarker (rede neural pronta). O enunciado pede "Deep Learning **e/ou** CUDA", então atende. A CNN própria de olhos (ONNX + `cv2.dnn`) fica como **extra**, só se sobrar tempo após a sprint de 23/11.

**O que é nosso (e deve ser destacado na apresentação):**

- Pré-processamento adaptativo (gama, CLAHE, Gaussiano)
- Métricas: EAR, MAR, PERCLOS e pose da cabeça
- Calibração do EAR por pessoa (o limiar fixo de 0,20 da demo não serve para todos)
- Lógica de decisão com janelas de tempo
- GUI com vídeo, gráfico e alertas

**Fora do escopo:** múltiplos rostos, detecção de celular/distração, versão mobile, gravação de histórico em banco.

## Requisitos do enunciado

| Requisito | Onde é atendido | O que mostrar |
| --- | --- | --- |
| Solução com OpenCV | Captura, filtros, desenho e conversões de cor em todo o `src/` | Código |
| Deep Learning e/ou CUDA | MediaPipe Face Landmarker (2 redes: detector + regressor de landmarks) | Slide explicando o pipeline |
| GUI | Janela PySide6 com vídeo, gráfico, indicadores e botões | Demo ao vivo |
| Aplicabilidade prática | Prevenção de acidentes por sono ao volante | Contexto na abertura |
| Processamento de imagens | Gama, CLAHE, filtro Gaussiano, comparação antes/depois | Tecla/botão liga-desliga o pré-processamento no escuro |
| Demo em vídeo (30–60 s) | Gravação com webcam: normal → olhos fechados → bocejo → cabeça caindo → alerta | Link no README |
| Flexibilidade de arquivos | Webcam, vídeo (.mp4/.avi) e imagem (.jpg/.png) pela GUI | Abrir um arquivo que o professor trouxer |
| Repositório GitHub com professor | Convidar `rafaelrieder` como colaborador | Aba Contributors |
| Todos commitam, contribuição balanceada | Divisão de tarefas abaixo; commits na `develop` | Gráfico de contribuições |
| README com links do vídeo | `README.md` | Repositório |

## Arquitetura

Fluxo de dados, uma direção só, cada módulo recebe a saída do anterior:

```
câmera/arquivo → Preprocessor → FaceLandmarkDetector → compute_metrics → FatigueMonitor → GUI (vídeo, gráfico, alerta)
   capture.py    preprocessing.py     landmarks.py          metrics.py        fatigue.py       gui/
```

```
src/
  capture.py         ✅ leitura de webcam / vídeo / imagem
  preprocessing.py   ✅ gama, CLAHE, Gaussiano (adaptativo por brilho)
  landmarks.py       ✅ MediaPipe Face Landmarker
  metrics.py         ✅ EAR, MAR, EMA   ⬜ PERCLOS, pose da cabeça
  calibration.py     ⬜ limiar de EAR por pessoa
  fatigue.py         ⬜ estados e regras de alerta
  alert.py           ⬜ alerta sonoro
  gui/               ⬜ janela PySide6
tests/               ⬜ testes de metrics e fatigue
```

**Contrato entre módulos (definir no dia 1 para os três trabalharem em paralelo):**

- `compute_metrics(pts)` devolve `dict` com `ear`, `mar` e, depois, `perclos` e `head_pitch`.
- `FatigueMonitor.update(metrics, dt)` devolve um `FatigueState`: `level` (`OK`, `ATENCAO`, `ALERTA`), `reasons` (lista de textos) e os valores atuais.
- A GUI só lê o `FatigueState`; não calcula nada.

### Regras de decisão (proposta inicial, ajustar na prática)

- **Olhos fechados:** EAR abaixo do limiar calibrado por mais de 2 s → `ALERTA`.
- **PERCLOS:** % de quadros com olho fechado nos últimos 60 s acima de 20 % → `ATENCAO`.
- **Bocejo:** MAR acima de 0,6 por mais de 1,5 s → conta 1 bocejo; 3 bocejos em 5 min → `ATENCAO`.
- **Cabeça:** inclinação para baixo acima de 20° por mais de 2 s → `ALERTA`.
- **Rosto ausente** por mais de 3 s → `ATENCAO` ("rosto não detectado").

Os números acima são chutes razoáveis da literatura e precisam ser testados com os próprios integrantes.

## Telas

Uma janela só (`MainWindow`):

```
┌──────────────────────────────────────────────┐
│ [Webcam ▾] [Abrir arquivo] [Calibrar] [Pré-proc ☑] │
├───────────────────────────┬──────────────────┤
│                           │ Estado: OK       │
│   vídeo com landmarks     │ EAR   0.31       │
│                           │ MAR   0.05       │
│                           │ PERCLOS 4 %      │
│                           │ Bocejos 0        │
│                           │ FPS 28           │
├───────────────────────────┴──────────────────┤
│ gráfico do EAR (últimos 10 s) com limiar     │
└──────────────────────────────────────────────┘
```

Faixa vermelha piscando e som quando o estado vira `ALERTA`.

## Git, branches e divisão do trio

- **`develop`**: onde o trabalho acontece. Os três commitam nela.
- **`main`**: só versões estáveis; merge da `develop` no fim de cada sprint.
- Antes de começar o dia: `git pull`. Cada um commita da própria conta.
- Pull request opcional; o importante é cada um aparecer como autor de commits de verdade, em arquivos diferentes, para evitar conflito.

| # | Tarefa | Responsável | Depende de | Entrega | Situação |
| --- | --- | --- | --- | --- | --- |
| 1 | Etapa 1: captura, pré-processamento, landmarks, EAR/MAR, demo | Murilo | — | Código atual | ✅ Concluída |
| 2 | Setup do repositório | Guilherme | — | Branch `develop`, professor convidado, README com nomes e instruções corrigidas (o `git clone` cita `detector-fadiga`), contrato entre módulos documentado | ⬜ |
| 3 | PERCLOS e pose da cabeça | Murilo | 1 | `metrics.py` com `perclos` (janela deslizante) e `head_pitch` (via `cv2.solvePnP` com 6 pontos do rosto) | ⬜ |
| 4 | Calibração por pessoa | Murilo | 1 | `calibration.py`: 3 s de olhos abertos → limiar = 70 % do EAR médio | ⬜ |
| 5 | Lógica de decisão | Giuliano | 2 (contrato) | `fatigue.py` com `FatigueMonitor` e as regras acima, usando relógio injetável para testar | ⬜ |
| 6 | Testes unitários | Giuliano | 3, 5 | `tests/` cobrindo EAR/MAR/PERCLOS com pontos sintéticos e as regras de `fatigue.py` | ⬜ |
| 7 | GUI: janela e vídeo | Guilherme | 2 | `gui/main_window.py`: `QThread` de captura, vídeo com landmarks, painel de indicadores, botões de fonte (webcam/arquivo) | ⬜ |
| 8 | GUI: gráfico e calibração | Guilherme | 4, 7 | Gráfico `pyqtgraph` do EAR com limiar; botão Calibrar | ⬜ |
| 9 | Alertas | Giuliano | 5 | `alert.py` (som, via `winsound`/`QSoundEffect`) e faixa vermelha na GUI | ⬜ |
| 10 | Integração e teste com arquivos | Giuliano + Guilherme | 7, 9 | Fluxo completo; testar com vídeos e fotos diferentes, incluindo cena escura | ⬜ |
| 11 | Ajuste de limiares com dados reais | Os três | 10 | Cada um grava 1 min de teste; limiares ajustados | ⬜ |
| 12 | Vídeo demonstrativo | Os três | 11 | Vídeo de 30–60 s, link no README | ⬜ |
| 13 | README final e ensaio | Os três | 12 | README completo, apresentação de 10 min ensaiada | ⬜ |
| 14 | Extra: CNN de olhos + CUDA | Murilo | 11 | Só se sobrar tempo: notebook em `training/`, `.onnx` em `models/`, `eye_classifier.py` via `cv2.dnn` | ⬜ Opcional |

As tarefas 3, 4, 5 e 7 andam em paralelo depois da 2. A 2 deve sair no primeiro dia.

### Conventional Commits

Formato: `tipo: descrição no imperativo`, em português, sem acento (como os commits atuais).

```
feat: adiciona calculo de perclos com janela deslizante
feat: implementa calibracao do limiar de ear por pessoa
feat: cria maquina de estados de fadiga com regras de alerta
feat: adiciona janela principal com video e painel de indicadores
test: cobre regras de alerta do fatigue monitor
fix: corrige timestamp duplicado no modo video
docs: adiciona link do video demonstrativo ao readme
```

Tipos: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Sprints e checklist de entrega

Hoje é 15/11; a apresentação é 30/11. O app precisa estar completo em 23/11.

1. **Sprint 16/11 – 22/11 (núcleo):** tarefas 2, 3, 4, 5, 6, 7. Ao fim: métricas completas, regras testadas, janela abre e mostra vídeo com indicadores. Merge `develop` → `main`.
2. **Sprint 23/11 (integração):** tarefas 8, 9, 10. Ao fim: app completo, alerta funcionando, arquivos abrindo. Merge na `main`. **Congelar funcionalidades.**
3. **24/11 – 29/11 (entrega):** tarefas 11, 12, 13 (e 14 se der). Só correções e polimento.
4. **30/11:** apresentação.

### Checklist

- [ ] Professor (`rafaelrieder`) adicionado como colaborador
- [ ] Branch `develop` criada; `main` recebe só merges da `develop`
- [ ] Os 3 integrantes com commits de código (não só README)
- [ ] Ambiente roda do zero com `pip install -r requirements.txt` + `download_models.py`
- [ ] Alerta dispara com olhos fechados por mais de 2 s
- [ ] Calibração muda o limiar e o gráfico reflete isso
- [ ] Abre webcam, vídeo e imagem pela GUI
- [ ] Pré-processamento visível funcionando em cena escura (antes/depois)
- [ ] Testes passando (`pytest`)
- [ ] Vídeo de 30–60 s gravado e linkado no README
- [ ] README com integrantes, descrição, instruções e link do vídeo
- [ ] Roteiro da apresentação (10 min): problema → pipeline → filtros → métricas → demo ao vivo → arquivo do professor → o que é nosso vs. modelo pronto
