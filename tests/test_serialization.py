"""
Unit tests for data structure serialization and CSV/JSON output runners.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.benchmarking.benchmark_runner import save_json, append_csv, save_csv
from src.frame_optimization.frame_gate import FrameDiff
from src.frame_optimization.spatial_redundancy import ChangedRegionResult
from src.frame_optimization.delta_caption import CaptionRecord, SceneState
from src.kv_cache.benchmark import KVCacheConfig


def test_frame_diff_to_dict():
    fd = FrameDiff(score=0.04, method="mad", should_process=False, consecutive_skips=3, reason="Test")
    d = fd.to_dict()
    assert d["score"] == 0.04
    assert d["should_process"] is False
    # Ensure json serializable
    dumped = json.dumps(d)
    assert "score" in dumped


def test_changed_region_to_dict():
    cr = ChangedRegionResult(
        has_changed_region=True,
        change_score=0.15,
        changed_patch_count=5,
        total_patches=100,
        bbox=(10, 20, 50, 60),
        frame_width=640,
        frame_height=480,
        crop_area_ratio=0.05,
    )
    d = cr.to_dict()
    assert d["has_changed_region"] is True
    assert d["bbox"] == [10, 20, 50, 60]
    dumped = json.dumps(d)
    assert "bbox" in dumped


def test_caption_record_to_dict():
    rec = CaptionRecord(
        frame_idx=1,
        caption_type="delta",
        output_text="person walked by",
        input_token_count=120,
        output_token_count=15,
        latency_seconds=0.25,
        tokens_per_second=60.0,
        peak_vram_gb=4.5,
        diff_score=0.12,
        triggered_by="normal_delta",
    )
    d = rec.to_dict()
    assert d["caption_type"] == "delta"
    assert json.dumps(d)


def test_kv_cache_config_to_dict():
    cfg = KVCacheConfig(name="hqq_4bit", backend="hqq", nbits=4)
    d = cfg.to_dict()
    assert d["name"] == "hqq_4bit"
    assert d["nbits"] == 4
    assert json.dumps(d)


def test_csv_and_json_io(tmp_path: Path):
    json_path = tmp_path / "test.json"
    data = {"status": "success", "count": 42}
    save_json(data, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data

    csv_path = tmp_path / "test.csv"
    row1 = {"id": 1, "val": "a"}
    row2 = {"id": 2, "val": "b"}
    append_csv(row1, csv_path)
    append_csv(row2, csv_path)

    pd.options.mode.string_storage = "python"
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert list(df["val"]) == ["a", "b"]
