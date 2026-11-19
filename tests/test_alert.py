import time

from src.alert import AlertSound
from src.fatigue import FatigueLevel


def _wait(cond, timeout=1.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_toca_so_em_alerta():
    calls = []
    a = AlertSound(beep=lambda: calls.append(1), period=0.05)
    try:
        a.update(FatigueLevel.OK)
        a.update(FatigueLevel.ATENCAO)
        time.sleep(0.3)
        assert calls == [] and not a.playing

        a.update(FatigueLevel.ALERTA)
        assert _wait(lambda: len(calls) >= 2)

        a.update(FatigueLevel.OK)
        time.sleep(0.2)
        n = len(calls)
        time.sleep(0.3)
        assert len(calls) == n and not a.playing
    finally:
        a.close()


def test_silenciado_nao_toca():
    calls = []
    a = AlertSound(beep=lambda: calls.append(1), period=0.05, enabled=False)
    try:
        a.update(FatigueLevel.ALERTA)
        time.sleep(0.3)
        assert calls == []
    finally:
        a.close()


def test_erro_no_som_nao_derruba():
    def boom():
        raise RuntimeError("sem placa de som")
    a = AlertSound(beep=boom, period=0.05)
    try:
        a.update(FatigueLevel.ALERTA)
        time.sleep(0.2)
        assert a._thread.is_alive()
    finally:
        a.close()
