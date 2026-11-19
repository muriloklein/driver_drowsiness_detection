"""Testes de metrics.py com pontos sinteticos (sem webcam nem MediaPipe)."""

import cv2
import numpy as np
import pytest

from src.metrics import (
    EMA, FACE_MODEL_3D, LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICAL,
    POSE_LANDMARKS, RIGHT_EYE, PerclosMeter, camera_matrix, compute_metrics,
    eye_aspect_ratio, head_pose, mouth_aspect_ratio,
)

IMAGE_SIZE = (640, 480)


def empty_pts():
    return np.zeros((478, 2), dtype=np.float64)


def set_eye(pts, idx, half_height, width=10.0, x0=0.0):
    """Olho com largura `width` e altura total 2*half_height (EAR = 2*hh/width)."""
    p1, p2, p3, p4, p5, p6 = idx
    pts[p1] = (x0, 0)
    pts[p4] = (x0 + width, 0)
    pts[p2] = (x0 + 0.3 * width, half_height)
    pts[p6] = (x0 + 0.3 * width, -half_height)
    pts[p3] = (x0 + 0.7 * width, half_height)
    pts[p5] = (x0 + 0.7 * width, -half_height)


def set_mouth(pts, height, width=10.0):
    left, right = MOUTH_CORNERS
    pts[left] = (0, 0)
    pts[right] = (width, 0)
    for i, (top, bottom) in enumerate(MOUTH_VERTICAL):
        x = width * (0.25 + 0.25 * i)
        pts[top] = (x, -height / 2)
        pts[bottom] = (x, height / 2)


def project_face(pitch=0.0, yaw=0.0, roll=0.0, depth=600.0):
    """Projeta o modelo 3D com uma rotacao conhecida e devolve 478 pontos 2D."""
    rvec = np.radians([pitch, -yaw, roll]).astype(np.float64).reshape(3, 1)
    tvec = np.array([[0.0], [0.0], [depth]])
    img, _ = cv2.projectPoints(FACE_MODEL_3D, rvec, tvec, camera_matrix(IMAGE_SIZE), None)
    pts = empty_pts()
    pts[POSE_LANDMARKS] = img.reshape(-1, 2)
    return pts


# ---------------------------------------------------------------- EAR / MAR

def test_ear_open_eye_value():
    pts = empty_pts()
    set_eye(pts, RIGHT_EYE, half_height=2.0)  # alturas 4 e 4, largura 10
    assert eye_aspect_ratio(pts, RIGHT_EYE) == pytest.approx(0.4)


def test_ear_closed_eye_is_zero():
    pts = empty_pts()
    set_eye(pts, RIGHT_EYE, half_height=0.0)
    assert eye_aspect_ratio(pts, RIGHT_EYE) == pytest.approx(0.0)


def test_ear_is_scale_invariant():
    small, big = empty_pts(), empty_pts()
    set_eye(small, LEFT_EYE, half_height=1.5, width=10.0)
    set_eye(big, LEFT_EYE, half_height=4.5, width=30.0)
    assert eye_aspect_ratio(small, LEFT_EYE) == pytest.approx(eye_aspect_ratio(big, LEFT_EYE))


def test_ear_zero_width_returns_zero():
    assert eye_aspect_ratio(empty_pts(), RIGHT_EYE) == 0.0


def test_mar_value_and_closed_mouth():
    pts = empty_pts()
    set_mouth(pts, height=5.0, width=10.0)
    assert mouth_aspect_ratio(pts) == pytest.approx(0.5)
    set_mouth(pts, height=0.0)
    assert mouth_aspect_ratio(pts) == pytest.approx(0.0)


def test_mar_zero_width_returns_zero():
    assert mouth_aspect_ratio(empty_pts()) == 0.0


def test_compute_metrics_without_image_size_has_no_pose():
    pts = empty_pts()
    set_eye(pts, RIGHT_EYE, 2.0, x0=0.0)
    set_eye(pts, LEFT_EYE, 1.0, x0=50.0)
    set_mouth(pts, 5.0)
    m = compute_metrics(pts)
    assert m["ear_right"] == pytest.approx(0.4)
    assert m["ear_left"] == pytest.approx(0.2)
    assert m["ear"] == pytest.approx(0.3)
    assert m["mar"] == pytest.approx(0.5)
    assert m["head_pitch"] is None and m["head_yaw"] is None and m["head_roll"] is None


def test_compute_metrics_contract_keys_with_image_size():
    m = compute_metrics(project_face(), IMAGE_SIZE)
    assert set(m) == {"ear", "ear_right", "ear_left", "mar",
                      "head_pitch", "head_yaw", "head_roll"}
    assert all(m[k] is not None for k in ("head_pitch", "head_yaw", "head_roll"))


# ------------------------------------------------------------- pose da cabeca

def test_head_pose_frontal_is_near_zero():
    pitch, yaw, roll = head_pose(project_face(), IMAGE_SIZE)
    assert abs(pitch) < 1 and abs(yaw) < 1 and abs(roll) < 1


@pytest.mark.parametrize("angle", [-20.0, 15.0, 30.0])
def test_head_pose_recovers_pitch(angle):
    pitch, yaw, roll = head_pose(project_face(pitch=angle), IMAGE_SIZE)
    assert pitch == pytest.approx(angle, abs=1.5)
    assert abs(yaw) < 2 and abs(roll) < 2


def test_head_pose_head_down_is_positive_pitch():
    pitch, _, _ = head_pose(project_face(pitch=25.0), IMAGE_SIZE)
    assert pitch > 0


@pytest.mark.parametrize("angle", [-20.0, 20.0])
def test_head_pose_recovers_yaw(angle):
    _, yaw, _ = head_pose(project_face(yaw=angle), IMAGE_SIZE)
    assert yaw == pytest.approx(angle, abs=1.5)


@pytest.mark.parametrize("angle", [-15.0, 15.0])
def test_head_pose_recovers_roll(angle):
    _, _, roll = head_pose(project_face(roll=angle), IMAGE_SIZE)
    assert roll == pytest.approx(angle, abs=1.5)


# -------------------------------------------------------------------- PERCLOS

def feed(meter, closed_fn, seconds, dt=0.1, t0=0.0):
    """Alimenta o medidor de t0 ate t0+seconds. Devolve o ultimo t."""
    n = int(round(seconds / dt))
    t = t0
    for i in range(n + 1):
        t = t0 + i * dt
        meter.update(closed_fn(i), t)
    return t


def test_perclos_all_open_is_zero():
    m = PerclosMeter()
    feed(m, lambda i: False, 20)
    assert m.value == 0.0


def test_perclos_all_closed_is_one():
    m = PerclosMeter()
    feed(m, lambda i: True, 20)
    assert m.value == pytest.approx(1.0)


def test_perclos_half_closed():
    m = PerclosMeter()
    feed(m, lambda i: i % 2 == 0, 30)
    assert m.value == pytest.approx(0.5, abs=0.05)


def test_perclos_old_samples_leave_window():
    m = PerclosMeter(window=60)
    t = feed(m, lambda i: True, 30)               # 30 s fechado
    feed(m, lambda i: False, 70, t0=t + 0.1)      # depois 70 s aberto
    assert m.value == pytest.approx(0.0, abs=1e-6)


def test_perclos_depends_on_time_not_on_frame_count():
    a, b = PerclosMeter(), PerclosMeter()
    feed(a, lambda i: (i * 0.1) % 10 < 2, 40, dt=0.1)    # 20 % fechado a 10 FPS
    feed(b, lambda i: (i * 0.05) % 10 < 2, 40, dt=0.05)  # 20 % fechado a 20 FPS
    assert a.value == pytest.approx(b.value, abs=0.03)
    assert a.value == pytest.approx(0.2, abs=0.03)


def test_perclos_large_gap_is_capped():
    m = PerclosMeter(max_gap=0.25)
    m.update(False, 0.0)
    m.update(True, 100.0)  # rosto sumiu por 100 s: so credita max_gap
    assert m.coverage == pytest.approx(0.25)


def test_perclos_ready_after_min_coverage():
    m = PerclosMeter()
    t = feed(m, lambda i: False, 5)
    assert not m.ready
    feed(m, lambda i: False, 6, t0=t + 0.1)
    assert m.ready


def test_perclos_reset():
    m = PerclosMeter()
    feed(m, lambda i: True, 15)
    m.reset()
    assert m.value == 0.0 and m.coverage == 0.0 and not m.ready


def test_perclos_empty_is_zero():
    assert PerclosMeter().value == 0.0


# ------------------------------------------------------------------------ EMA

def test_ema_first_value_passes_through():
    e = EMA(alpha=0.5)
    assert e.update(10.0) == 10.0


def test_ema_smooths_step():
    e = EMA(alpha=0.5)
    e.update(0.0)
    assert e.update(10.0) == pytest.approx(5.0)
    assert e.update(10.0) == pytest.approx(7.5)


def test_ema_converges_to_constant():
    e = EMA(alpha=0.3)
    e.update(0.0)
    for _ in range(50):
        v = e.update(4.0)
    assert v == pytest.approx(4.0, abs=1e-3)
