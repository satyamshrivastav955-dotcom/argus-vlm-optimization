"""
Unit tests for frame gating module (FrameGate, compute_mad).
"""

import numpy as np
from PIL import Image
import pytest

from src.frame_optimization.frame_gate import FrameGate, compute_mad, pil_to_gray_array


def test_compute_mad_identical():
    """Identical frames must have a MAD of 0.0."""
    frame = np.zeros((100, 100), dtype=np.uint8)
    score = compute_mad(frame, frame)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_compute_mad_maximum_difference():
    """All black vs all white frames must have a MAD of 1.0."""
    black = np.zeros((50, 50), dtype=np.uint8)
    white = np.full((50, 50), 255, dtype=np.uint8)
    score = compute_mad(black, white)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_compute_mad_shape_mismatch():
    """Frames of different sizes should be resized and compared without error."""
    f1 = np.zeros((50, 50), dtype=np.uint8)
    f2 = np.zeros((100, 100), dtype=np.uint8)
    score = compute_mad(f1, f2)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_frame_gate_first_frame_always_processed():
    """The very first frame must always be processed."""
    gate = FrameGate(method="mad", threshold=0.05)
    gate.reset()

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    decision = gate.check(img)
    assert decision.should_process is True
    assert "first_frame" in decision.reason.lower()


def test_frame_gate_skip_static():
    """Subsequent identical frames must be skipped."""
    gate = FrameGate(method="mad", threshold=0.05)
    gate.reset()

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    d1 = gate.check(img)
    assert d1.should_process is True
    gate.update_previous(img)

    # Second identical frame
    d2 = gate.check(img)
    assert d2.should_process is False
    assert d2.score == pytest.approx(0.0, abs=1e-5)


def test_frame_gate_max_skip_frames():
    """Frames exceeding max_skip_frames must be forced to process."""
    gate = FrameGate(method="mad", threshold=0.10, max_skip_frames=3)
    gate.reset()

    img = Image.new("RGB", (64, 64), color=(50, 50, 50))
    gate.check(img)
    gate.update_previous(img)

    # 1st skip
    d1 = gate.check(img)
    assert d1.should_process is False

    # 2nd skip
    d2 = gate.check(img)
    assert d2.should_process is False

    # 3rd skip
    d3 = gate.check(img)
    assert d3.should_process is False

    # 4th check: max_skip_frames (3) reached, force process
    d4 = gate.check(img)
    assert d4.should_process is True
    assert "forced" in d4.reason.lower()


def test_pil_to_gray_array():
    """Verify PIL to grayscale numpy array conversion."""
    img = Image.new("RGB", (20, 20), color=(255, 0, 0))
    gray = pil_to_gray_array(img)
    assert isinstance(gray, np.ndarray)
    assert gray.shape == (20, 20)
    assert gray.dtype == np.uint8
