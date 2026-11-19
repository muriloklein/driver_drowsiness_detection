"""
alert.py — Alerta sonoro, ligado ao nível do FatigueMonitor.

Uso:
    alerta = AlertSound()
    ...
    alerta.update(state.level)   # a cada quadro; barato, nunca bloqueia
    ...
    alerta.close()               # ao fechar a janela

Enquanto o nível for ALERTA, toca um bipe repetido numa thread própria
(daemon), então o loop de vídeo/GUI nunca trava. Ao sair de ALERTA o som para
na hora (no máximo termina o bipe corrente, ~0,4 s). Sem Windows (`winsound`
ausente) usa o sino do terminal como fallback.

O som em si é injetável (`beep=`), para testar sem tocar nada.
"""

import sys
import threading

from src.fatigue import FatigueLevel

try:
    import winsound
except ImportError:  # Linux / macOS
    winsound = None

BEEP_HZ = 1000
BEEP_MS = 400
BEEP_PERIOD = 0.8  # s entre inícios de bipes


def _default_beep():
    if winsound is not None:
        winsound.Beep(BEEP_HZ, BEEP_MS)  # bloqueia só a thread do alerta
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


class AlertSound:
    def __init__(self, beep=_default_beep, period=BEEP_PERIOD, enabled=True):
        self._beep = beep
        self._period = period
        self.enabled = enabled  # permite silenciar (ex.: checkbox "Som")
        self._active = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="alert-sound", daemon=True)
        self._thread.start()

    @property
    def playing(self):
        return self._active.is_set()

    def update(self, level):
        """Chame a cada quadro com state.level. Liga em ALERTA, desliga fora."""
        if self.enabled and level == FatigueLevel.ALERTA:
            self._active.set()
        else:
            self._active.clear()

    def stop(self):
        self._active.clear()

    def close(self):
        self._closed.set()
        self._active.clear()
        self._thread.join(timeout=1.0)

    def _loop(self):
        while not self._closed.is_set():
            if not self._active.wait(timeout=0.1):
                continue
            try:
                self._beep()
            except Exception:  # som nunca pode derrubar o app
                pass
            # espera o intervalo, mas acorda se o alerta cessar ou fechar
            self._closed.wait(self._period)
