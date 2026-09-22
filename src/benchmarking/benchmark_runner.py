"""
Benchmark runner — manages result collection, incremental CSV/JSON saves,
and failure handling across all experiment types.

Key design decisions
---------------------
- Results are saved incrementally after each experiment config.
  If the kernel crashes mid-sweep, partial results are preserved.
- Failed experiments are recorded with FAILED status, not silently dropped.
- All saves are append-safe (append_csv creates file if missing).
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================
# Safe generation helper (used by notebooks directly)
# ============================================================

def safe_generate(wrapper, image, prompt, max_new_tokens=256, kv_cache_config=None):
    """
    Wrapper around QwenVLMWrapper.generate() with explicit OOM handling.

    Returns VLMResult. On OOM, clears GPU cache and returns failed result.
    """
    try:
        return wrapper.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            kv_cache_config=kv_cache_config,
        )
    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        from src.vlm.qwen_vlm import VLMResult
        return VLMResult(
            generated_text="",
            input_token_count=0,
            output_token_count=0,
            latency_seconds=0.0,
            tokens_per_second=0.0,
            peak_vram_gb=0.0,
            status="oom",
            error=str(e),
        )
    except Exception as e:
        from src.vlm.qwen_vlm import VLMResult
        return VLMResult(
            generated_text="",
            input_token_count=0,
            output_token_count=0,
            latency_seconds=0.0,
            tokens_per_second=0.0,
            peak_vram_gb=0.0,
            status="failed",
            error=str(e),
        )


def clear_gpu() -> None:
    """Clear GPU cache and run Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure_peak_vram_gb() -> float:
    """Return current peak GPU VRAM allocated in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_vram_peak() -> None:
    """Reset VRAM peak counter (call before timing a block)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# IO helpers
# ============================================================

def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file. Creates parent directories if needed.

    Parameters
    ----------
    data:
        JSON-serializable data (dict, list, etc.).
    path:
        Output file path.
    indent:
        JSON indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Saved JSON: {path}")


def append_csv(row: Dict[str, Any], path: str | Path) -> None:
    """
    Append one row to a CSV file. Creates file with header if it does not exist.

    Safe for incremental saves — call after each experiment config to
    avoid data loss if the kernel crashes.

    Parameters
    ----------
    row:
        Dict mapping column names to values.
    path:
        CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.debug(f"Appended row to CSV: {path}")


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """
    Save a list of rows to a CSV file (overwrites if exists).

    Parameters
    ----------
    rows:
        List of dicts (all must have the same keys).
    path:
        Output file path.
    """
    if not rows:
        logger.warning("save_csv called with empty rows list.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved CSV ({len(rows)} rows): {path}")


# ============================================================
# BenchmarkRunner
# ============================================================

@dataclass
class ExperimentResult:
    """General experiment result record for the final summary."""

    experiment: str
    configuration: str
    frame_gate_enabled: bool = False
    spatial_reduction_enabled: bool = False
    delta_caption_enabled: bool = False
    kv_cache_enabled: bool = False
    kv_backend: str = "dynamic"
    kv_nbits: int = 16
    peak_vram_gb: float = 0.0
    latency_seconds: float = 0.0
    tokens_per_second: float = 0.0
    vlm_calls: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_reduction_pct: float = 0.0
    memory_reduction_pct: float = 0.0
    quality_metric: float = 0.0
    event_recall: Optional[float] = None
    false_negative_count: Optional[int] = None
    status: str = "ok"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """
    Coordinates experiment execution, result collection, and saving.

    Usage
    -----
        runner = BenchmarkRunner(results_dir="results/combined")
        runner.add_result(ExperimentResult(...))
        runner.save_all()
    """

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult) -> None:
        """Add a result and immediately append it to CSV (incremental save)."""
        self._results.append(result)
        csv_path = self.results_dir / "results.csv"
        append_csv(result.to_dict(), csv_path)
        logger.info(
            f"Result recorded: [{result.experiment}] status={result.status}"
        )

    def save_all(self, prefix: str = "results") -> None:
        """Save all accumulated results to CSV and JSON."""
        if not self._results:
            logger.warning("No results to save.")
            return
        rows = [r.to_dict() for r in self._results]
        save_csv(rows, self.results_dir / f"{prefix}.csv")
        save_json(rows, self.results_dir / f"{prefix}.json")

    def get_results(self) -> List[ExperimentResult]:
        return list(self._results)

    def clear(self) -> None:
        self._results.clear()
