"""Testes das regras de fatigue.py. O tempo e simulado via `dt` ou relogio falso."""

import pytest

from src.calibration import DEFAULT_EAR_THRESHOLD, CalibrationResult
from src.fatigue import FatigueLevel, FatigueMonitor

OPEN, CLOSED = 0.30, 0.10  # EAR de olho aberto / fechado (limiar padrao 0.20)


def face(ear=OPEN, mar=0.05, pitch=None):
    return {"ear": ear, "mar": mar, "head_pitch": pitch}


def run(monitor, metrics, seconds, dt=0.5):
    """Envia o mesmo quadro por `seconds` segundos. Devolve todos os estados."""
    return [monitor.update(metrics, dt) for _ in range(int(round(seconds / dt)))]


def calibration(threshold=0.20, pitch_baseline=None):
    return CalibrationResult(ear_open=0.3, ear_threshold=threshold,
                             pitch_baseline=pitch_baseline, samples=90, rejected=0)


# ------------------------------------------------------------------ estado base

def test_normal_face_is_ok():
    states = run(FatigueMonitor(), face(), 10)
    assert all(s.level == FatigueLevel.OK for s in states)
    assert states[-1].face_present
    assert states[-1].reasons == []
    assert states[-1].ear == OPEN


def test_level_ordering():
    assert FatigueLevel.OK < FatigueLevel.ATENCAO < FatigueLevel.ALERTA


def test_default_threshold_used_before_calibration():
    assert FatigueMonitor().ear_threshold == DEFAULT_EAR_THRESHOLD


# ---------------------------------------------------------------- olhos fechados

def test_eyes_closed_alerts_only_after_two_seconds():
    m = FatigueMonitor()
    states = run(m, face(ear=CLOSED), 3.0)  # duracoes: 0, .5, 1, 1.5, 2, 2.5
    assert [s.level for s in states[:5]] == [FatigueLevel.OK] * 5  # 2.0 s exatos ainda nao
    assert states[5].level == FatigueLevel.ALERTA
    assert any("Olhos fechados" in r for r in states[5].reasons)
    assert states[5].eyes_closed
    assert states[5].closed_duration == pytest.approx(2.5)


def test_blink_does_not_alert_and_resets_timer():
    m = FatigueMonitor()
    run(m, face(ear=CLOSED), 1.5)
    s = m.update(face(ear=OPEN), 0.5)
    assert s.level == FatigueLevel.OK and s.closed_duration == 0.0
    states = run(m, face(ear=CLOSED), 1.5)  # recomeca do zero
    assert all(x.level == FatigueLevel.OK for x in states)


def test_alert_clears_when_eyes_open():
    m = FatigueMonitor()
    run(m, face(ear=CLOSED), 3.0)
    assert m.update(face(ear=OPEN), 0.5).level == FatigueLevel.OK


def test_ear_exactly_at_threshold_is_not_closed():
    s = FatigueMonitor().update(face(ear=DEFAULT_EAR_THRESHOLD), 0.5)
    assert not s.eyes_closed


def test_calibrated_threshold_changes_decision():
    m = FatigueMonitor()
    assert not m.update(face(ear=0.25), 0.5).eyes_closed
    m.apply_calibration(calibration(threshold=0.28))
    s = m.update(face(ear=0.25), 0.5)
    assert s.eyes_closed and s.ear_threshold == 0.28


# ----------------------------------------------------------------------- PERCLOS

def test_perclos_gives_attention_when_high():
    m = FatigueMonitor()
    # ciclo de 1 s: 0.3 s de olho fechado, sem nunca chegar a 2 s seguidos
    states = []
    for _ in range(20):
        for i in range(10):
            states.append(m.update(face(ear=CLOSED if i < 3 else OPEN), 0.1))
    last = states[-1]
    assert last.perclos_ready
    assert last.perclos > 0.20
    assert last.level == FatigueLevel.ATENCAO
    assert any("PERCLOS" in r for r in last.reasons)


def test_perclos_not_used_before_ready():
    m = FatigueMonitor()
    states = []
    for _ in range(5):  # so 5 s de dados
        for i in range(10):
            states.append(m.update(face(ear=CLOSED if i < 3 else OPEN), 0.1))
    assert not states[-1].perclos_ready
    assert states[-1].level == FatigueLevel.OK


def test_low_perclos_stays_ok():
    m = FatigueMonitor()
    for _ in range(20):
        for i in range(10):
            s = m.update(face(ear=CLOSED if i < 1 else OPEN), 0.1)  # ~10 %
    assert s.perclos_ready and s.level == FatigueLevel.OK


# ------------------------------------------------------------------------ bocejo

def yawn(m, seconds=3.0):  # duracoes 0..2.5 s, passa de 1.5 s
    return run(m, face(mar=0.8), seconds)


def test_short_mouth_opening_is_not_a_yawn():
    m = FatigueMonitor()
    s = run(m, face(mar=0.8), 1.5)[-1]  # dur 1.0 <= 1.5
    assert s.yawn_count == 0


def test_long_mouth_opening_counts_one_yawn_only_once():
    m = FatigueMonitor()
    states = run(m, face(mar=0.8), 6.0)  # um unico episodio longo
    assert states[-1].yawn_count == 1
    assert states[-1].level == FatigueLevel.OK


def test_mar_at_threshold_is_not_a_yawn():
    m = FatigueMonitor()
    assert run(m, face(mar=0.6), 5.0)[-1].yawn_count == 0


def test_three_yawns_in_window_give_attention():
    m = FatigueMonitor()
    for _ in range(3):
        yawn(m)
        s = m.update(face(mar=0.05), 1.0)
    assert s.yawn_count == 3
    assert s.level == FatigueLevel.ATENCAO
    assert any("bocejos" in r for r in s.reasons)


def test_two_yawns_are_not_enough():
    m = FatigueMonitor()
    for _ in range(2):
        yawn(m)
        s = m.update(face(mar=0.05), 1.0)
    assert s.yawn_count == 2 and s.level == FatigueLevel.OK


def test_yawns_expire_after_five_minutes():
    m = FatigueMonitor()
    for _ in range(3):
        yawn(m)
        m.update(face(mar=0.05), 1.0)
    s = m.update(face(mar=0.05), 301.0)
    assert s.yawn_count == 0 and s.level == FatigueLevel.OK


# ------------------------------------------------------------------------ cabeca

def test_head_down_alerts_after_two_seconds():
    m = FatigueMonitor()
    states = run(m, face(pitch=30.0), 3.0)
    assert states[3].level == FatigueLevel.OK  # 1.5 s
    assert states[5].level == FatigueLevel.ALERTA
    assert "Cabeça caída" in states[5].reasons
    assert states[5].head_down_duration > 2.0


def test_head_below_limit_is_ok():
    m = FatigueMonitor()
    assert all(s.level == FatigueLevel.OK for s in run(m, face(pitch=15.0), 5.0))


def test_head_uses_pitch_baseline_from_calibration():
    m = FatigueMonitor()
    m.apply_calibration(calibration(pitch_baseline=15.0))
    # pitch absoluto 30 = relativo 15: abaixo do limite de 20
    states = run(m, face(pitch=30.0), 5.0)
    assert all(s.level == FatigueLevel.OK for s in states)
    assert states[-1].head_pitch == pytest.approx(15.0)
    # pitch absoluto 40 = relativo 25: acima do limite
    states = run(m, face(pitch=40.0), 6.0)
    assert states[-1].level == FatigueLevel.ALERTA


def test_head_pitch_none_is_ignored():
    m = FatigueMonitor()
    s = run(m, face(pitch=None), 5.0)[-1]
    assert s.level == FatigueLevel.OK and s.head_pitch is None


def test_single_pitch_spike_is_smoothed():
    m = FatigueMonitor()
    run(m, face(pitch=0.0), 2.0)
    s = m.update(face(pitch=60.0), 0.5)  # salto de 1 quadro, EMA amortece
    assert s.head_pitch < 60.0
    assert m.update(face(pitch=0.0), 0.5).head_down_duration == 0.0


def test_head_recovering_resets_timer():
    m = FatigueMonitor()
    run(m, face(pitch=40.0), 1.5)
    run(m, face(pitch=0.0), 5.0)
    s = run(m, face(pitch=40.0), 1.0)[-1]
    assert s.level == FatigueLevel.OK


# ------------------------------------------------------------------ rosto ausente

def test_absent_face_gives_attention_after_three_seconds():
    m = FatigueMonitor()
    states = [m.update(None, 1.0) for _ in range(5)]  # duracoes 0..4
    assert [s.level for s in states[:4]] == [FatigueLevel.OK] * 4  # 3.0 s exatos ainda nao
    assert states[4].level == FatigueLevel.ATENCAO
    assert "Rosto não detectado" in states[4].reasons
    assert not states[4].face_present
    assert states[4].face_absent_duration == pytest.approx(4.0)


def test_metrics_without_ear_counts_as_absent():
    s = FatigueMonitor().update({"ear": None}, 1.0)
    assert not s.face_present


def test_face_returning_clears_absence():
    m = FatigueMonitor()
    for _ in range(5):
        m.update(None, 1.0)
    s = m.update(face(), 0.5)
    assert s.level == FatigueLevel.OK and s.face_present and s.face_absent_duration == 0.0


def test_absence_resets_closed_eye_timer():
    m = FatigueMonitor()
    run(m, face(ear=CLOSED), 1.5)
    m.update(None, 0.5)
    s = run(m, face(ear=CLOSED), 1.5)[-1]
    assert s.level == FatigueLevel.OK


# ------------------------------------------------------------------ combinacoes

def test_final_level_is_the_maximum_and_lists_all_reasons():
    m = FatigueMonitor()
    states = run(m, face(ear=CLOSED, pitch=40.0), 3.0)
    s = states[-1]
    assert s.level == FatigueLevel.ALERTA
    assert len(s.reasons) == 2


def test_reset_clears_history_but_keeps_calibration():
    m = FatigueMonitor()
    m.apply_calibration(calibration(threshold=0.25, pitch_baseline=5.0))
    run(m, face(ear=0.1, pitch=40.0), 3.0)
    m.reset()
    s = m.update(face(ear=0.1, pitch=40.0), 0.5)
    assert s.closed_duration == 0.0 and s.head_down_duration == 0.0
    assert m.ear_threshold == 0.25 and m.pitch_baseline == 5.0


def test_apply_calibration_none_baseline_becomes_zero():
    m = FatigueMonitor()
    m.apply_calibration(calibration(pitch_baseline=None))
    assert m.pitch_baseline == 0.0


# ----------------------------------------------------------------- relogio injetado

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_injected_clock_drives_rules_without_dt():
    clock = FakeClock()
    m = FatigueMonitor(clock=clock)
    levels = []
    for _ in range(6):
        levels.append(m.update(face(ear=CLOSED)).level)
        clock.t += 0.5
    assert levels[:5] == [FatigueLevel.OK] * 5
    assert levels[5] == FatigueLevel.ALERTA


def test_negative_dt_does_not_rewind_time():
    m = FatigueMonitor()
    run(m, face(ear=CLOSED), 1.5)  # duracoes 0, 0.5, 1.0
    s = m.update(face(ear=CLOSED), -5.0)
    assert s.closed_duration == pytest.approx(1.0)
