"""
Benchmarking metrics module.

Provides consistent metric computations used across all experiments.

Important notes
---------------
ROUGE-L is included as an AUXILIARY text similarity metric only.
It is NOT a surveillance event correctness metric.
Event recall requires ground-truth event annotations (see sample_data/README.md).

Do not use ROUGE-L as a proxy for hazard detection or event preservation quality.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================
# Schema
# ============================================================

RESULT_SCHEMA_COLUMNS: List[str] = [
    "experiment",
    "configuration",
    "frame_gate_enabled",
    "spatial_reduction_enabled",
    "delta_caption_enabled",
    "kv_cache_enabled",
    "kv_backend",
    "kv_nbits",
    "peak_vram_gb",
    "latency_seconds",
    "tokens_per_second",
    "vlm_calls",
    "frames_processed",
    "frames_skipped",
    "input_tokens",
    "output_tokens",
    "token_reduction_pct",
    "memory_reduction_pct",
    "quality_metric",          # ROUGE-L (auxiliary only)
    "event_recall",            # Requires ground truth; null if unavailable
    "false_negative_count",    # Requires ground truth; null if unavailable
    "status",
]


# ============================================================
# ROUGE-L
# ============================================================

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between hypothesis and reference texts.

    AUXILIARY METRIC ONLY. Not a surveillance correctness metric.

    Parameters
    ----------
    hypothesis:
        Generated text (the text to evaluate).
    reference:
        Reference text (e.g., the FP16 baseline output).

    Returns
    -------
    float: ROUGE-L F1 score in [0.0, 1.0].
           Returns 0.0 if either string is empty.
           Returns -1.0 if rouge-score library is not available.
    """
    if not hypothesis or not reference:
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return float(scores["rougeL"].fmeasure)
    except ImportError:
        logger.warning(
            "rouge-score not installed. "
            "Run: pip install rouge-score. Returning -1.0 sentinel."
        )
        return -1.0
    except Exception as e:
        logger.warning(f"ROUGE-L computation failed: {e}")
        return -1.0


# ============================================================
# Token reduction
# ============================================================

def compute_token_reduction(
    baseline_tokens: int,
    optimized_tokens: int,
) -> float:
    """
    Compute percentage token reduction relative to baseline.

    Parameters
    ----------
    baseline_tokens:
        Token count in the baseline (unoptimized) configuration.
    optimized_tokens:
        Token count in the optimized configuration.

    Returns
    -------
    float: Reduction percentage. Positive = fewer tokens.
           0.0 if baseline_tokens is 0 (avoid division by zero).
    """
    if baseline_tokens <= 0:
        return 0.0
    reduction = (baseline_tokens - optimized_tokens) / baseline_tokens * 100.0
    return float(reduction)


# ============================================================
# Memory reduction
# ============================================================

def compute_memory_reduction(
    baseline_vram_gb: float,
    optimized_vram_gb: float,
) -> float:
    """
    Compute percentage VRAM reduction relative to baseline.

    Parameters
    ----------
    baseline_vram_gb:
        Peak VRAM in the baseline configuration (GB).
    optimized_vram_gb:
        Peak VRAM in the optimized configuration (GB).

    Returns
    -------
    float: Reduction percentage. Positive = less VRAM.
           0.0 if baseline is 0.
    """
    if baseline_vram_gb <= 0.0:
        return 0.0
    reduction = (baseline_vram_gb - optimized_vram_gb) / baseline_vram_gb * 100.0
    return float(reduction)


# ============================================================
# Effective FPS
# ============================================================

def compute_effective_fps(
    frames_processed: int,
    total_wall_time_seconds: float,
) -> float:
    """
    Compute effective frames-per-second processing rate.

    Parameters
    ----------
    frames_processed:
        Total number of frames processed by VLM (not skipped).
    total_wall_time_seconds:
        Total wall-clock time for the entire pipeline (seconds).

    Returns
    -------
    float: Effective FPS. 0.0 if time is 0.
    """
    if total_wall_time_seconds <= 0.0:
        return 0.0
    return float(frames_processed / total_wall_time_seconds)


# ============================================================
# Event recall (placeholder — requires ground truth)
# ============================================================

def compute_event_recall(
    event_frame_ranges: List[tuple],
    processed_frame_indices: List[int],
) -> Dict[str, Any]:
    """
    Compute event recall given ground truth event frame ranges.

    An event is "detected" if at least one frame in the event's range
    was processed by the VLM (not skipped by the gate).

    IMPORTANT: Only call this if genuine ground truth is available.
    Do NOT fabricate event_frame_ranges.

    Parameters
    ----------
    event_frame_ranges:
        List of (start_frame, end_frame) tuples marking event windows.
    processed_frame_indices:
        List of frame indices that were processed by the VLM.

    Returns
    -------
    dict with keys:
        total_events, detected_events, missed_events,
        recall, false_negative_count
    """
    if not event_frame_ranges:
        return {
            "total_events": 0,
            "detected_events": 0,
            "missed_events": 0,
            "recall": None,  # Cannot compute without ground truth
            "false_negative_count": 0,
            "note": "No ground truth event annotations provided.",
        }

    processed_set = set(processed_frame_indices)
    detected = 0
    missed = 0

    for (start, end) in event_frame_ranges:
        event_frames = set(range(start, end + 1))
        if event_frames & processed_set:
            detected += 1
        else:
            missed += 1

    total = len(event_frame_ranges)
    recall = detected / total if total > 0 else 0.0

    return {
        "total_events": total,
        "detected_events": detected,
        "missed_events": missed,
        "recall": recall,
        "false_negative_count": missed,
    }
