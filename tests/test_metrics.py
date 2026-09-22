"""
Unit tests for benchmarking metrics module.
"""

import pytest
from src.benchmarking.metrics import (
    compute_rouge_l,
    compute_token_reduction,
    compute_memory_reduction,
    compute_effective_fps,
    compute_event_recall,
    RESULT_SCHEMA_COLUMNS,
)


def test_schema_columns_defined():
    assert isinstance(RESULT_SCHEMA_COLUMNS, list)
    assert "experiment" in RESULT_SCHEMA_COLUMNS
    assert "peak_vram_gb" in RESULT_SCHEMA_COLUMNS
    assert "status" in RESULT_SCHEMA_COLUMNS


def test_compute_token_reduction():
    assert compute_token_reduction(100, 40) == pytest.approx(60.0)
    assert compute_token_reduction(100, 100) == pytest.approx(0.0)
    assert compute_token_reduction(0, 50) == pytest.approx(0.0)


def test_compute_memory_reduction():
    assert compute_memory_reduction(10.0, 5.0) == pytest.approx(50.0)
    assert compute_memory_reduction(0.0, 5.0) == pytest.approx(0.0)


def test_compute_effective_fps():
    assert compute_effective_fps(100, 10.0) == pytest.approx(10.0)
    assert compute_effective_fps(50, 0.0) == pytest.approx(0.0)


def test_compute_rouge_l_empty():
    assert compute_rouge_l("", "some reference") == 0.0
    assert compute_rouge_l("some hypothesis", "") == 0.0


def test_compute_event_recall_no_ground_truth():
    res = compute_event_recall([], [1, 2, 3])
    assert res["total_events"] == 0
    assert res["recall"] is None


def test_compute_event_recall_with_ground_truth():
    ranges = [(10, 20), (50, 60)]
    processed = [15, 70]  # First event captured, second missed
    res = compute_event_recall(ranges, processed)
    assert res["total_events"] == 2
    assert res["detected_events"] == 1
    assert res["missed_events"] == 1
    assert res["recall"] == pytest.approx(0.5)
    assert res["false_negative_count"] == 1
