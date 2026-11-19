"""
fatigue.py — Lógica de decisão: transforma as métricas de cada quadro em um
nível de alerta (OK / ATENCAO / ALERTA).

Regras (ver docs/plano-projeto-final.md):
    - Olhos fechados: EAR abaixo do limiar por mais de 2 s            -> ALERTA
    - PERCLOS: mais de 20 % de olho fechado nos últimos 60 s          -> ATENCAO
    - Bocejo: MAR > 0.6 por mais de 1.5 s conta 1 bocejo;
      3 bocejos em 5 min                                              -> ATENCAO
    - Cabeça: (pitch - pitch_baseline) > 20° por mais de 2 s          -> ALERTA
    - Rosto ausente por mais de 3 s                                   -> ATENCAO

O nível final é o MAIOR entre as regras disparadas; `reasons` lista todas.

Relógio injetável
-----------------
O monitor nunca chama time.* diretamente nas regras: ele só enxerga um tempo
`t` em segundos. Há duas formas de alimentá-lo:

    monitor.update(metrics, dt)    # dt = segundos desde o quadro anterior.
                                   # É o contrato do plano; vídeo: dt = 1/fps.
    monitor.update(metrics)        # sem dt: usa o relógio injetado
                                   # (padrão time.monotonic; nos testes,
                                   # FatigueMonitor(clock=meu_relogio_falso)).

Em ambos os casos o teste não precisa esperar tempo de verdade.

Sinal de EAR: o monitor usa `metrics["ear"]` como vem. Para ser coerente com a
calibração, passe o MESMO EAR (suavizado ou não) que foi usado em EarCalibrator.
O pitch da cabeça, ao contrário, é suavizado aqui dentro (EMA), porque o
solvePnP às vezes salta alguns graus de um quadro para o outro.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum

from src.calibration import DEFAULT_EAR_THRESHOLD
from src.metrics import EMA, PerclosMeter


class FatigueLevel(IntEnum):
    """IntEnum para poder comparar/maximizar: OK < ATENCAO < ALERTA."""
    OK = 0
    ATENCAO = 1
    ALERTA = 2


@dataclass
class FatigueState:
    """Resultado de um quadro. A GUI só lê isto; não calcula nada."""
    level: FatigueLevel = FatigueLevel.OK
    reasons: list = field(default_factory=list)  # textos prontos para exibir
    face_present: bool = False
    ear: float | None = None            # EAR atual (None sem rosto)
    mar: float | None = None
    ear_threshold: float = DEFAULT_EAR_THRESHOLD
    eyes_closed: bool = False           # EAR abaixo do limiar neste quadro
    closed_duration: float = 0.0        # s contínuos de olho fechado
    perclos: float = 0.0                # 0 a 1
    perclos_ready: bool = False         # False nos primeiros 10 s
    yawn_count: int = 0                 # bocejos nos últimos 5 min
    head_pitch: float | None = None     # relativo à calibração, suavizado (graus)
    head_down_duration: float = 0.0     # s contínuos com a cabeça caída
    face_absent_duration: float = 0.0   # s contínuos sem rosto


class FatigueMonitor:
    def __init__(self, clock=time.monotonic,
                 closed_seconds=2.0,
                 perclos_window=60.0, perclos_limit=0.20,
                 mar_threshold=0.6, yawn_seconds=1.5,
                 yawn_limit=3, yawn_window=300.0,
                 head_limit=20.0, head_seconds=2.0,
                 absent_seconds=3.0,
                 pitch_alpha=0.3):
        self.clock = clock
        self.closed_seconds = closed_seconds
        self.perclos_limit = perclos_limit
        self.mar_threshold = mar_threshold
        self.yawn_seconds = yawn_seconds
        self.yawn_limit = yawn_limit
        self.yawn_window = yawn_window
        self.head_limit = head_limit
        self.head_seconds = head_seconds
        self.absent_seconds = absent_seconds

        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.pitch_baseline = 0.0

        self._perclos = PerclosMeter(window=perclos_window)
        self._pitch_ema = EMA(alpha=pitch_alpha)
        self._t = 0.0  # tempo acumulado quando se usa dt
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Zera todo o histórico (mantém limiar e baseline da calibração)."""
        self._perclos.reset()
        self._pitch_ema.value = None
        self._closed_since = None
        self._yawn_since = None
        self._yawn_counted = False
        self._yawns = []  # instantes em que cada bocejo foi contado
        self._head_since = None
        self._absent_since = None

    def apply_calibration(self, result):
        """Aplica um CalibrationResult (de EarCalibrator.result)."""
        self.ear_threshold = result.ear_threshold
        self.pitch_baseline = result.pitch_baseline if result.pitch_baseline is not None else 0.0
        # O limiar mudou: PERCLOS e contagens antigas não valem mais.
        self.reset()

    # ------------------------------------------------------------------
    def _now(self, dt):
        if dt is None:
            self._t = self.clock()
        else:
            self._t += max(dt, 0.0)
        return self._t

    def update(self, metrics, dt=None):
        """
        metrics: dict de compute_metrics() ou None se não há rosto no quadro.
        dt: segundos desde o quadro anterior (opcional; sem ele usa o relógio).
        Retorna um FatigueState.
        """
        t = self._now(dt)
        if metrics is None or metrics.get("ear") is None:
            return self._update_absent(t)
        return self._update_present(metrics, t)

    # ------------------------------------------------------------------
    def _update_absent(self, t):
        # Sem rosto não dá para julgar olho/boca/cabeça: esses cronômetros
        # recomeçam quando o rosto voltar. O PERCLOS não recebe amostra (o
        # intervalo sem rosto não conta, como documentado no PerclosMeter).
        self._closed_since = None
        self._yawn_since = None
        self._yawn_counted = False
        self._head_since = None
        self._pitch_ema.value = None
        if self._absent_since is None:
            self._absent_since = t
        absent = t - self._absent_since

        self._prune_yawns(t)
        reasons = []
        level = FatigueLevel.OK
        if absent > self.absent_seconds:
            level = FatigueLevel.ATENCAO
            reasons.append("Rosto não detectado")
        level, reasons = self._add_long_term(level, reasons)

        return FatigueState(
            level=level, reasons=reasons, face_present=False,
            ear_threshold=self.ear_threshold,
            perclos=self._perclos.value, perclos_ready=self._perclos.ready,
            yawn_count=len(self._yawns), face_absent_duration=absent,
        )

    def _update_present(self, metrics, t):
        self._absent_since = None
        ear = float(metrics["ear"])
        mar = metrics.get("mar")

        # --- olhos fechados ------------------------------------------------
        closed = ear < self.ear_threshold
        if closed:
            if self._closed_since is None:
                self._closed_since = t
            closed_dur = t - self._closed_since
        else:
            self._closed_since = None
            closed_dur = 0.0
        self._perclos.update(closed, t)

        # --- bocejo ----------------------------------------------------------
        if mar is not None and mar > self.mar_threshold:
            if self._yawn_since is None:
                self._yawn_since = t
                self._yawn_counted = False
            if not self._yawn_counted and t - self._yawn_since > self.yawn_seconds:
                self._yawn_counted = True  # 1 bocejo por episódio
                self._yawns.append(t)
        else:
            self._yawn_since = None
            self._yawn_counted = False
        self._prune_yawns(t)

        # --- cabeça ----------------------------------------------------------
        pitch_rel = None
        head_dur = 0.0
        raw_pitch = metrics.get("head_pitch")
        if raw_pitch is not None:
            pitch_rel = self._pitch_ema.update(float(raw_pitch)) - self.pitch_baseline
            if pitch_rel > self.head_limit:
                if self._head_since is None:
                    self._head_since = t
                head_dur = t - self._head_since
            else:
                self._head_since = None
        else:
            self._head_since = None

        # --- decisão ---------------------------------------------------------
        level = FatigueLevel.OK
        reasons = []
        if closed_dur > self.closed_seconds:
            level = FatigueLevel.ALERTA
            reasons.append(f"Olhos fechados há {closed_dur:.1f} s")
        if head_dur > self.head_seconds:
            level = FatigueLevel.ALERTA
            reasons.append("Cabeça caída")
        level, reasons = self._add_long_term(level, reasons)

        return FatigueState(
            level=level, reasons=reasons, face_present=True,
            ear=ear, mar=mar, ear_threshold=self.ear_threshold,
            eyes_closed=closed, closed_duration=closed_dur,
            perclos=self._perclos.value, perclos_ready=self._perclos.ready,
            yawn_count=len(self._yawns),
            head_pitch=pitch_rel, head_down_duration=head_dur,
        )

    # ------------------------------------------------------------------
    def _prune_yawns(self, t):
        self._yawns = [y for y in self._yawns if t - y <= self.yawn_window]

    def _add_long_term(self, level, reasons):
        """Regras de janela longa (PERCLOS e bocejos): valem com ou sem rosto."""
        if self._perclos.ready and self._perclos.value > self.perclos_limit:
            level = max(level, FatigueLevel.ATENCAO)
            reasons.append(f"PERCLOS alto ({self._perclos.value * 100:.0f} %)")
        if len(self._yawns) >= self.yawn_limit:
            level = max(level, FatigueLevel.ATENCAO)
            reasons.append(f"{len(self._yawns)} bocejos em {self.yawn_window / 60:.0f} min")
        return level, reasons
