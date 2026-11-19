"""
calibration.py — Limiar de EAR calibrado para cada pessoa.

Por que calibrar?
    O EAR de olho ABERTO varia muito entre pessoas (formato do olho, óculos,
    ângulo da câmera): numa pessoa pode ser 0.32, noutra 0.22. Com o limiar
    fixo de 0.20 da demo, a segunda pessoa estaria quase sempre "fechando o
    olho" e a primeira teria que fechar MUITO para o sistema perceber.

Como funciona:
    1) A pessoa olha para a câmera com os olhos abertos por ~3 s.
    2) Juntamos o EAR de cada quadro com rosto detectado.
    3) Descartamos as PISCADAS (EAR bem abaixo da mediana): em 3 s é normal
       piscar 1 ou 2 vezes, e elas puxariam a média para baixo.
    4) ear_aberto = média do que sobrou
       limiar     = 70 % do ear_aberto
    5) De brinde, guardamos o pitch "normal" da cabeça. O modelo 3D do rosto
       é genérico, então o pitch de quem está olhando reto não é exatamente 0:
       a regra de cabeceio deve comparar com essa referência.

Uso (o relógio t é passado de fora, para poder testar sem esperar):
    cal = EarCalibrator()
    cal.start(t)
    a cada quadro:  cal.update(metrics_ou_None, t)
    quando cal.done:   cal.result.ear_threshold   (ou cal.error se falhou)

Importante: calibre com o MESMO sinal de EAR que a lógica de fadiga usa
(se a fadiga usa o EAR suavizado pela EMA, calibre com o EAR suavizado).
"""

from dataclasses import dataclass
import statistics

DEFAULT_EAR_THRESHOLD = 0.20  # usado enquanto ninguém calibrou


@dataclass
class CalibrationResult:
    ear_open: float               # EAR médio de olho aberto
    ear_threshold: float          # limiar: abaixo disso = olho fechado
    pitch_baseline: float | None  # pitch "olhando reto" (graus) ou None
    samples: int                  # quadros usados na média
    rejected: int                 # quadros descartados como piscada


class EarCalibrator:
    def __init__(self, duration=3.0, ratio=0.70, blink_ratio=0.80,
                 min_samples=10, min_face_ratio=0.70,
                 valid_range=(0.15, 0.50)):
        self.duration = duration              # segundos de coleta
        self.ratio = ratio                    # limiar = ratio * ear_aberto
        self.blink_ratio = blink_ratio        # abaixo de 80 % da mediana = piscada
        self.min_samples = min_samples        # mínimo de quadros válidos
        self.min_face_ratio = min_face_ratio  # % mínima de quadros com rosto
        self.valid_range = valid_range        # EAR aberto plausível
        self._reset_state()

    def _reset_state(self):
        self.active = False
        self.done = False
        self.result = None
        self.error = None
        self._t0 = None
        self._t = None
        self._ears = []
        self._pitches = []
        self._frames = 0

    # ------------------------------------------------------------------
    def start(self, t):
        """Começa uma nova coleta (descarta qualquer calibração em andamento)."""
        self._reset_state()
        self.active = True
        self._t0 = self._t = t

    def cancel(self):
        self._reset_state()

    @property
    def progress(self):
        """De 0 a 1, para desenhar uma barra de progresso."""
        if not self.active or self._t0 is None:
            return 1.0 if self.done else 0.0
        # max(0.0): se o relógio voltar (ex.: troca de fonte), não fica negativo
        return min(max((self._t - self._t0) / self.duration, 0.0), 1.0)

    def update(self, metrics, t):
        """
        metrics: dict de compute_metrics() ou None se não há rosto no quadro.
        Retorna True no quadro em que a calibração termina (com sucesso ou não).
        """
        if not self.active:
            return False

        self._t = t
        self._frames += 1
        if metrics is not None and metrics.get("ear") is not None:
            self._ears.append(float(metrics["ear"]))
            if metrics.get("head_pitch") is not None:
                self._pitches.append(float(metrics["head_pitch"]))

        if t - self._t0 >= self.duration:
            self._finish()
            return True
        return False

    # ------------------------------------------------------------------
    def _finish(self):
        self.active = False
        self.done = True
        try:
            self.result = self._compute()
        except ValueError as e:
            self.error = str(e)

    def _compute(self):
        n = len(self._ears)
        if self._frames == 0 or n / self._frames < self.min_face_ratio or n < self.min_samples:
            raise ValueError("Rosto não detectado em boa parte do tempo. "
                             "Fique de frente para a câmera e tente de novo.")

        # Mediana é robusta: algumas piscadas não a mudam. Tudo muito abaixo
        # dela é piscada e sai da conta.
        med = statistics.median(self._ears)
        kept = [e for e in self._ears if e >= self.blink_ratio * med]
        if len(kept) < self.min_samples:
            raise ValueError("Muitas piscadas durante a calibração. "
                             "Mantenha os olhos abertos e tente de novo.")

        ear_open = statistics.fmean(kept)
        lo, hi = self.valid_range
        if not lo <= ear_open <= hi:
            raise ValueError(f"EAR de olho aberto fora do esperado ({ear_open:.2f}). "
                             "Olhe para a câmera com os olhos abertos e tente de novo.")

        pitch = statistics.median(self._pitches) if self._pitches else None
        return CalibrationResult(
            ear_open=ear_open,
            ear_threshold=self.ratio * ear_open,
            pitch_baseline=pitch,
            samples=len(kept),
            rejected=n - len(kept),
        )
