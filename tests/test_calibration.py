"""Testes de calibration.py com relogio inventado (sem webcam)."""

import pytest

from src.calibration import EarCalibrator


def test_progress_is_zero_before_start():
    assert EarCalibrator().progress == 0.0


def test_progress_grows_with_time():
    cal = EarCalibrator(duration=3.0)
    cal.start(10.0)
    cal.update({"ear": 0.3}, 11.5)
    assert cal.progress == pytest.approx(0.5)


def test_progress_never_negative_when_clock_goes_back():
    # Ex.: calibracao iniciada com o relogio da webcam (monotonic, valor grande)
    # e continuada com o relogio de um video (indice / fps, perto de zero).
    cal = EarCalibrator(duration=3.0)
    cal.start(1000.0)
    cal.update({"ear": 0.3}, 2.0)
    assert cal.progress == 0.0


def test_progress_capped_at_one():
    cal = EarCalibrator(duration=3.0)
    cal.start(0.0)
    cal._t = 10.0  # alem da duracao, sem terminar pelo update
    assert cal.progress == 1.0
