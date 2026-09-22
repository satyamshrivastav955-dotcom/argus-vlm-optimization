"""
Frame-level gating module.

Provides cheap pixel-level change detection to determine whether the
current frame differs enough from the previous frame to warrant a VLM
inference call.

This is NOT VLM token pruning. It is a pre-inference skip decision.

Supported methods
-----------------
- mad  : Mean Absolute Difference (grayscale, normalized 0–1)
         Extremely fast (~0.1ms per frame). Perceptually naive.
- ssim : Structural Similarity Index (slower but more perceptually motivated)
         Requires skimage. ~1–5ms per frame depending on resolution.

Usage
-----
    gate = FrameGate(method="mad", threshold=0.05, max_skip_frames=30)
    decision = gate.check(previous_frame, current_frame)
    if decision.should_process:
        result = vlm.generate(current_frame, prompt)
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
class FrameDiff:
    """Result of frame difference computation."""

    score: float            # Normalized difference score (0.0–1.0 for MAD)
    method: str             # "mad" | "ssim"
    should_process: bool    # True = invoke VLM; False = skip
    consecutive_skips: int  # Number of consecutive skips so far
    reason: str             # Human-readable decision reason

    def to_dict(self):
        return {
            "score": self.score,
            "method": self.method,
            "should_process": self.should_process,
            "consecutive_skips": self.consecutive_skips,
            "reason": self.reason,
        }


# ============================================================
# Core functions
# ============================================================

def compute_mad(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> float:
    """
    Compute the Mean Absolute Difference between two grayscale frames.

    Both inputs must be 2D numpy arrays (uint8, values 0–255).
    Returns a normalized score in [0.0, 1.0].

    Parameters
    ----------
    prev_frame:
        Previous frame as grayscale uint8 array (H, W).
    curr_frame:
        Current frame as grayscale uint8 array (H, W).

    Returns
    -------
    float: Normalized MAD score. 0.0 = identical, 1.0 = maximally different.
    """
    if prev_frame.shape != curr_frame.shape:
        # Resize curr to match prev if shapes differ
        from PIL import Image as PILImage
        curr_pil = PILImage.fromarray(curr_frame).resize(
            (prev_frame.shape[1], prev_frame.shape[0]), PILImage.BILINEAR
        )
        curr_frame = np.array(curr_pil)

    diff = np.abs(prev_frame.astype(np.float32) - curr_frame.astype(np.float32))
    return float(diff.mean() / 255.0)


def compute_ssim(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> float:
    """
    Compute SSIM-based change score between two grayscale frames.

    Returns a *difference* score: 0.0 = identical, ~1.0 = very different.
    (SSIM itself is a similarity score; we return 1 - SSIM.)

    Requires: scikit-image

    Parameters
    ----------
    prev_frame:
        Previous frame as grayscale uint8 array (H, W).
    curr_frame:
        Current frame as grayscale uint8 array (H, W).

    Returns
    -------
    float: 1 - SSIM (0.0 = identical, up to ~1.0 = very different).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError(
            "scikit-image is required for SSIM. "
            "Install it with: pip install scikit-image"
        )

    if prev_frame.shape != curr_frame.shape:
        from PIL import Image as PILImage
        curr_pil = PILImage.fromarray(curr_frame).resize(
            (prev_frame.shape[1], prev_frame.shape[0]), PILImage.BILINEAR
        )
        curr_frame = np.array(curr_pil)

    similarity, _ = ssim(
        prev_frame.astype(np.float32),
        curr_frame.astype(np.float32),
        data_range=255.0,
        full=True,
    )
    return float(1.0 - similarity)


def pil_to_gray_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale uint8 numpy array."""
    return np.array(image.convert("L"))


# ============================================================
# FrameGate class
# ============================================================

class FrameGate:
    """
    Frame-level skip gate.

    Computes a cheap pixel-level change score between consecutive frames.
    If the score is below `threshold`, the frame is skipped.
    If `max_skip_frames` consecutive frames have been skipped, forces process.

    Parameters
    ----------
    method:
        Change detection method: "mad" (fast) or "ssim" (slower).
    threshold:
        Normalized change score threshold. Frames with score < threshold
        are skipped. Sweep this parameter — do not assume a single value
        is correct for all scene types.
    max_skip_frames:
        Maximum consecutive frames to skip before forcing VLM call.
        Prevents indefinite skipping during slow drift.

    Usage
    -----
        gate = FrameGate(method="mad", threshold=0.05)
        gate.reset()
        for frame in frames:
            decision = gate.check(frame)
            if decision.should_process:
                result = vlm.generate(...)
                gate.update_previous(frame)
    """

    SUPPORTED_METHODS = ("mad", "ssim")

    def __init__(
        self,
        method: str = "mad",
        threshold: float = 0.05,
        max_skip_frames: int = 30,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.threshold = threshold
        self.max_skip_frames = max_skip_frames

        self._prev_gray: Optional[np.ndarray] = None
        self._consecutive_skips: int = 0

    def reset(self) -> None:
        """Reset gate state (call at start of each new video)."""
        self._prev_gray = None
        self._consecutive_skips = 0

    def check(self, current_frame: Image.Image) -> FrameDiff:
        """
        Check whether the current frame should be processed by the VLM.

        Parameters
        ----------
        current_frame:
            Current frame as PIL Image.

        Returns
        -------
        FrameDiff with should_process flag and change score.
        """
        curr_gray = pil_to_gray_array(current_frame)

        # First frame — always process
        if self._prev_gray is None:
            self._consecutive_skips = 0
            return FrameDiff(
                score=1.0,
                method=self.method,
                should_process=True,
                consecutive_skips=0,
                reason="first_frame",
            )

        # Compute change score
        if self.method == "mad":
            score = compute_mad(self._prev_gray, curr_gray)
        elif self.method == "ssim":
            score = compute_ssim(self._prev_gray, curr_gray)
        else:
            raise RuntimeError(f"Unsupported method: {self.method}")

        # Force process after max_skip_frames consecutive skips
        if self._consecutive_skips >= self.max_skip_frames:
            self._consecutive_skips = 0
            return FrameDiff(
                score=score,
                method=self.method,
                should_process=True,
                consecutive_skips=self._consecutive_skips,
                reason=f"forced_after_{self.max_skip_frames}_consecutive_skips",
            )

        # Normal gate decision
        if score >= self.threshold:
            self._consecutive_skips = 0
            reason = f"change_detected_score={score:.4f}_threshold={self.threshold}"
            should_process = True
        else:
            self._consecutive_skips += 1
            reason = f"static_frame_score={score:.4f}_threshold={self.threshold}"
            should_process = False

        return FrameDiff(
            score=score,
            method=self.method,
            should_process=should_process,
            consecutive_skips=self._consecutive_skips,
            reason=reason,
        )

    def update_previous(self, frame: Image.Image) -> None:
        """
        Update the stored previous frame.

        Call this after successfully processing a frame (whether or not
        the VLM was invoked). This keeps the gate reference current.

        Parameters
        ----------
        frame:
            The frame to store as the new "previous" frame.
        """
        self._prev_gray = pil_to_gray_array(frame)
