"""
Unit tests for spatial redundancy reduction module.
"""

from PIL import Image, ImageDraw
import pytest

from src.frame_optimization.spatial_redundancy import (
    PatchChangeDetector,
    crop_changed_region,
)


def test_detector_identical_frames():
    """Identical frames should report no changed region."""
    detector = PatchChangeDetector(patch_size=32, change_threshold=0.10)
    frame = Image.new("RGB", (128, 128), color=(128, 128, 128))

    res = detector.detect(frame, frame)
    assert res.has_changed_region is False
    assert res.changed_patch_count == 0
    assert res.bbox is None
    assert res.crop_area_ratio == 1.0


def test_detector_localized_change():
    """A localized motion block should produce a valid bounding box covering the change."""
    detector = PatchChangeDetector(patch_size=16, change_threshold=0.05, min_crop_fraction=0.01)
    f1 = Image.new("RGB", (128, 128), color=(0, 0, 0))
    f2 = Image.new("RGB", (128, 128), color=(0, 0, 0))

    # Draw a 32x32 white box in the center (48, 48) to (80, 80)
    draw = ImageDraw.Draw(f2)
    draw.rectangle([48, 48, 80, 80], fill=(255, 255, 255))

    res = detector.detect(f1, f2)
    assert res.has_changed_region is True
    assert res.changed_patch_count > 0
    assert res.bbox is not None

    x1, y1, x2, y2 = res.bbox
    assert x1 <= 48
    assert y1 <= 48
    assert x2 >= 80
    assert y2 >= 80


def test_crop_changed_region_padding():
    """Cropping should apply padding and return a valid PIL image within dimensions."""
    detector = PatchChangeDetector(patch_size=16, change_threshold=0.05, min_crop_fraction=0.01)
    f1 = Image.new("RGB", (100, 100), color=(0, 0, 0))
    f2 = Image.new("RGB", (100, 100), color=(0, 0, 0))

    draw = ImageDraw.Draw(f2)
    draw.rectangle([30, 30, 60, 60], fill=(255, 255, 255))

    res = detector.detect(f1, f2)
    cropped, ratio = crop_changed_region(f2, res, padding=10, min_crop_fraction=0.01, max_crop_fraction=0.95)

    assert isinstance(cropped, Image.Image)
    assert ratio <= 1.0
    cw, ch = cropped.size
    assert cw <= 100
    assert ch <= 100


def test_crop_fallback_when_too_large():
    """If the entire frame changes, crop should fall back to full frame."""
    detector = PatchChangeDetector(patch_size=16, change_threshold=0.05)
    f1 = Image.new("RGB", (64, 64), color=(0, 0, 0))
    f2 = Image.new("RGB", (64, 64), color=(255, 255, 255))

    res = detector.detect(f1, f2)
    cropped, ratio = crop_changed_region(f2, res, padding=0, max_crop_fraction=0.80)
    # Since 100% changed, ratio > 0.80 -> fallback to full frame
    assert cropped.size == (64, 64)
    assert ratio == 1.0
