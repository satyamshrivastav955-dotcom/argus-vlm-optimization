"""
Spatial Redundancy Reduction — Change-Region Cropping.

IMPORTANT DISCLAIMER
--------------------
This module implements CHANGE-REGION CROPPING, NOT internal ViT-token pruning.

The pipeline is:
  previous frame + current frame
    → grid-based patch change map
    → bounding box of changed patches
    → add padding
    → crop image
    → return cropped PIL Image for VLM input

This reduces redundant background content BEFORE the VLM sees the image.
It does NOT modify internal attention masks or ViT token tensors.

The actual token reduction inside the VLM depends on:
  - The VLM processor's internal image tiling logic
  - The cropped image's final resolution
  - How the VLM counts visual tokens

We estimate token reduction as: (crop_area / full_area), but this is
an approximation. Measure actual input_token_count from VLMResult to verify.

Usage
-----
    detector = PatchChangeDetector(patch_size=32, change_threshold=0.10)
    result = detector.detect(previous_frame, current_frame)
    if result.has_changed_region:
        cropped = crop_changed_region(current_frame, result, padding=32)
    else:
        cropped = current_frame  # No meaningful change — use full frame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Data types
# ============================================================

@dataclass
class ChangedRegionResult:
    """
    Result of change-region detection on a frame pair.

    Attributes
    ----------
    has_changed_region:
        True if at least one patch exceeded the change threshold.
    change_score:
        Mean change score across all patches (0.0–1.0).
    changed_patch_count:
        Number of patches that exceeded the threshold.
    total_patches:
        Total number of grid patches.
    bbox:
        Bounding box of changed patches as (x1, y1, x2, y2) in pixels.
        None if no patches changed.
    frame_width:
        Width of the full frame in pixels.
    frame_height:
        Height of the full frame in pixels.
    crop_area_ratio:
        (crop_w × crop_h) / (frame_w × frame_h) after padding.
        Approximates the fraction of the frame being sent to the VLM.
    """

    has_changed_region: bool
    change_score: float
    changed_patch_count: int
    total_patches: int
    bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    frame_width: int
    frame_height: int
    crop_area_ratio: float = 1.0

    def to_dict(self):
        return {
            "has_changed_region": self.has_changed_region,
            "change_score": self.change_score,
            "changed_patch_count": self.changed_patch_count,
            "total_patches": self.total_patches,
            "bbox": list(self.bbox) if self.bbox else None,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "crop_area_ratio": self.crop_area_ratio,
        }


# ============================================================
# PatchChangeDetector
# ============================================================

class PatchChangeDetector:
    """
    Grid-based patch change detector for spatial redundancy reduction.

    Divides frames into a regular grid of patches and computes the mean
    absolute difference per patch. Changed patches define the crop region.

    Parameters
    ----------
    patch_size:
        Size of each grid cell in pixels (applied to both width and height).
        Smaller = finer detection, more computation.
        Typical values: 16, 32, 64.
    change_threshold:
        Normalized per-patch change threshold (0.0–1.0).
        Patches with mean absolute difference > threshold are "changed".
        Typical values: 0.05–0.20. Must be swept, not assumed.
    min_crop_fraction:
        If the detected changed area is smaller than this fraction of the
        full frame, fall back to full frame. (Tiny crops may lose context.)
    max_crop_fraction:
        If the detected changed area is larger than this fraction, fall back
        to full frame. (Cropping adds no benefit if most of the frame changed.)
    """

    def __init__(
        self,
        patch_size: int = 32,
        change_threshold: float = 0.10,
        min_crop_fraction: float = 0.10,
        max_crop_fraction: float = 0.80,
    ) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if not (0.0 <= change_threshold <= 1.0):
            raise ValueError(
                f"change_threshold must be in [0, 1], got {change_threshold}"
            )
        self.patch_size = patch_size
        self.change_threshold = change_threshold
        self.min_crop_fraction = min_crop_fraction
        self.max_crop_fraction = max_crop_fraction

    def detect(
        self,
        prev_frame: Image.Image,
        curr_frame: Image.Image,
    ) -> ChangedRegionResult:
        """
        Detect the changed region between two frames.

        Parameters
        ----------
        prev_frame:
            Previous frame (PIL Image, RGB or L).
        curr_frame:
            Current frame (PIL Image, RGB or L).

        Returns
        -------
        ChangedRegionResult
        """
        W, H = curr_frame.size
        prev_gray = np.array(prev_frame.convert("L"), dtype=np.float32)
        curr_gray = np.array(curr_frame.convert("L"), dtype=np.float32)

        # Resize prev to match curr if different sizes
        if prev_gray.shape != curr_gray.shape:
            prev_pil = Image.fromarray(prev_gray.astype(np.uint8)).resize(
                (W, H), Image.BILINEAR
            )
            prev_gray = np.array(prev_pil, dtype=np.float32)

        # Compute per-patch change scores
        n_rows = max(1, H // self.patch_size)
        n_cols = max(1, W // self.patch_size)

        changed_patches = []

        for row in range(n_rows):
            for col in range(n_cols):
                r0 = row * self.patch_size
                r1 = min(r0 + self.patch_size, H)
                c0 = col * self.patch_size
                c1 = min(c0 + self.patch_size, W)

                patch_prev = prev_gray[r0:r1, c0:c1]
                patch_curr = curr_gray[r0:r1, c0:c1]

                if patch_prev.size == 0 or patch_curr.size == 0:
                    continue

                patch_diff = np.abs(patch_prev - patch_curr).mean() / 255.0

                if patch_diff > self.change_threshold:
                    changed_patches.append((c0, r0, c1, r1))  # (x1, y1, x2, y2)

        total_patches = n_rows * n_cols
        changed_count = len(changed_patches)

        if changed_count == 0:
            return ChangedRegionResult(
                has_changed_region=False,
                change_score=0.0,
                changed_patch_count=0,
                total_patches=total_patches,
                bbox=None,
                frame_width=W,
                frame_height=H,
                crop_area_ratio=1.0,
            )

        # Compute overall change score (fraction of changed patches)
        change_score = changed_count / total_patches

        # Union bounding box of all changed patches
        x1 = min(p[0] for p in changed_patches)
        y1 = min(p[1] for p in changed_patches)
        x2 = max(p[2] for p in changed_patches)
        y2 = max(p[3] for p in changed_patches)

        bbox = (x1, y1, x2, y2)

        # Estimate crop ratio (before padding)
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_ratio = (crop_w * crop_h) / (W * H)

        return ChangedRegionResult(
            has_changed_region=True,
            change_score=change_score,
            changed_patch_count=changed_count,
            total_patches=total_patches,
            bbox=bbox,
            frame_width=W,
            frame_height=H,
            crop_area_ratio=crop_ratio,
        )


# ============================================================
# Crop function
# ============================================================

def crop_changed_region(
    frame: Image.Image,
    detection_result: ChangedRegionResult,
    padding: int = 32,
    min_crop_fraction: float = 0.10,
    max_crop_fraction: float = 0.80,
) -> Tuple[Image.Image, float]:
    """
    Crop the changed region from the frame, with padding.

    If the cropped area is smaller than min_crop_fraction or larger than
    max_crop_fraction of the full frame, returns the full frame unchanged.

    Parameters
    ----------
    frame:
        Current frame (PIL Image, RGB).
    detection_result:
        Output of PatchChangeDetector.detect().
    padding:
        Pixels to add around the detected bounding box on each side.
    min_crop_fraction:
        Minimum allowed crop area fraction. Below this → full frame.
    max_crop_fraction:
        Maximum allowed crop area fraction. Above this → full frame.

    Returns
    -------
    (cropped_image, actual_crop_area_ratio)
        cropped_image: PIL Image — either a crop or the original full frame.
        actual_crop_area_ratio: float — actual fraction of frame sent to VLM.
    """
    W, H = frame.size

    if not detection_result.has_changed_region or detection_result.bbox is None:
        return frame, 1.0

    x1, y1, x2, y2 = detection_result.bbox

    # Add padding, clamp to frame bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)

    crop_w = x2 - x1
    crop_h = y2 - y1
    actual_ratio = (crop_w * crop_h) / (W * H)

    # Sanity checks — fall back to full frame if crop is trivially small or large
    if actual_ratio < min_crop_fraction:
        logger.debug(
            f"Crop ratio {actual_ratio:.3f} < min {min_crop_fraction} — "
            "using full frame."
        )
        return frame, 1.0

    if actual_ratio > max_crop_fraction:
        logger.debug(
            f"Crop ratio {actual_ratio:.3f} > max {max_crop_fraction} — "
            "using full frame (most of frame changed)."
        )
        return frame, 1.0

    cropped = frame.crop((x1, y1, x2, y2))
    return cropped, actual_ratio
