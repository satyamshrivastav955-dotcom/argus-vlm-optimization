"""
KV cache benchmark module.

Provides:
  - KVCacheConfig: Configuration for one KV cache experiment variant.
  - KVCacheResult: Result record for one experiment run.
  - run_kv_cache_experiment(): Run a single configuration safely.
  - run_kv_cache_benchmark(): Sweep all configurations and return results list.

Experiment schema (per result row):
  config, backend, nbits, peak_memory_gb, latency_seconds,
  tokens_per_second, generated_tokens, quality_score, status, error

Failure handling:
  If any backend raises an exception (including OOM or ImportError),
  the result is recorded as FAILED and the sweep continues.
  The entire benchmark will never abort due to a single config failure.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Data types
# ============================================================

@dataclass
class KVCacheConfig:
    """Configuration for one KV cache experiment variant."""

    name: str
    backend: str         # "dynamic" | "HQQ" | "quanto"
    nbits: int           # 16 (FP16) | 8 | 4 | 2
    enabled: bool = True
    description: str = ""

    def to_kv_config_dict(self) -> Optional[Dict[str, Any]]:
        """Convert to the dict format expected by QwenVLMWrapper.generate()."""
        if self.backend.lower() in ("dynamic", "fp16"):
            return None  # Use HF default
        return {"backend": self.backend, "nbits": self.nbits}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return {
            "name": self.name,
            "backend": self.backend,
            "nbits": self.nbits,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def defaults(cls) -> List["KVCacheConfig"]:
        """Return the four standard experiment configurations."""
        return [
            cls(
                name="baseline_fp16_dynamic",
                backend="dynamic",
                nbits=16,
                description="FP16 HF DynamicCache — unmodified baseline.",
            ),
            cls(
                name="quantized_hqq_4bit",
                backend="HQQ",
                nbits=4,
                description="HQQ INT4 KV cache quantization.",
            ),
            cls(
                name="quantized_hqq_2bit",
                backend="HQQ",
                nbits=2,
                description="HQQ INT2 KV cache quantization.",
            ),
            cls(
                name="quantized_quanto_4bit",
                backend="quanto",
                nbits=4,
                description="Quanto INT4 KV cache quantization.",
            ),
        ]


@dataclass
class KVCacheResult:
    """
    Result record for one KV cache experiment run.

    Matches the canonical result schema for CSV/JSON serialization.
    """

    config: str
    backend: str
    nbits: int
    peak_memory_gb: float
    latency_seconds: float
    tokens_per_second: float
    generated_tokens: int
    quality_score: float       # ROUGE-L vs FP16 baseline (0.0–1.0)
    status: str                # "ok" | "failed" | "oom"
    error: Optional[str] = None
    generated_text: str = ""
    input_tokens: int = 0
    context_length: int = 0    # Used in long-context experiments

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "backend": self.backend,
            "nbits": self.nbits,
            "peak_memory_gb": self.peak_memory_gb,
            "latency_seconds": self.latency_seconds,
            "tokens_per_second": self.tokens_per_second,
            "generated_tokens": self.generated_tokens,
            "quality_score": self.quality_score,
            "status": self.status,
            "error": self.error,
            "generated_text": self.generated_text,
            "input_tokens": self.input_tokens,
            "context_length": self.context_length,
        }


# ============================================================
# Single experiment runner
# ============================================================

def run_kv_cache_experiment(
    wrapper,          # QwenVLMWrapper instance (already loaded)
    image: Image.Image,
    prompt: str,
    config: KVCacheConfig,
    max_new_tokens: int = 256,
    reference_text: Optional[str] = None,
    num_runs: int = 1,
) -> KVCacheResult:
    """
    Run one KV cache configuration and return a result record.

    Parameters
    ----------
    wrapper:
        A loaded QwenVLMWrapper instance.
    image:
        PIL Image for inference.
    prompt:
        Text prompt.
    config:
        KVCacheConfig specifying the backend.
    max_new_tokens:
        Max output tokens.
    reference_text:
        If provided, compute ROUGE-L similarity (used when comparing
        quantized configs against FP16 baseline).
    num_runs:
        Number of inference runs. Average is taken over all runs.

    Returns
    -------
    KVCacheResult
    """
    if not config.enabled:
        logger.info(f"[{config.name}] Disabled — skipping.")
        return KVCacheResult(
            config=config.name,
            backend=config.backend,
            nbits=config.nbits,
            peak_memory_gb=0.0,
            latency_seconds=0.0,
            tokens_per_second=0.0,
            generated_tokens=0,
            quality_score=0.0,
            status="disabled",
            error="Config is disabled.",
        )

    logger.info(
        f"[{config.name}] Running — backend={config.backend}, nbits={config.nbits}"
    )

    kv_config = config.to_kv_config_dict()

    latencies = []
    tps_list = []
    peak_vramas = []
    last_result = None

    for run_idx in range(num_runs):
        result = wrapper.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            kv_cache_config=kv_config,
        )
        last_result = result

        if result.status != "ok":
            logger.warning(
                f"[{config.name}] Run {run_idx+1} failed: {result.error}"
            )
            return KVCacheResult(
                config=config.name,
                backend=config.backend,
                nbits=config.nbits,
                peak_memory_gb=result.peak_vram_gb,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                generated_tokens=0,
                quality_score=0.0,
                status=result.status,
                error=result.error,
                input_tokens=result.input_token_count,
            )

        latencies.append(result.latency_seconds)
        tps_list.append(result.tokens_per_second)
        peak_vramas.append(result.peak_vram_gb)

    # Average metrics across runs
    avg_latency = sum(latencies) / len(latencies)
    avg_tps = sum(tps_list) / len(tps_list)
    avg_vram = max(peak_vramas)  # Use max VRAM (worst-case)

    # Compute quality score
    quality_score = 1.0  # For baseline (self-comparison)
    if reference_text and last_result and last_result.generated_text:
        try:
            from src.benchmarking.metrics import compute_rouge_l
            quality_score = compute_rouge_l(
                hypothesis=last_result.generated_text,
                reference=reference_text,
            )
        except Exception as e:
            logger.warning(f"ROUGE-L computation failed: {e}")
            quality_score = -1.0  # Sentinel: could not compute

    return KVCacheResult(
        config=config.name,
        backend=config.backend,
        nbits=config.nbits,
        peak_memory_gb=avg_vram,
        latency_seconds=avg_latency,
        tokens_per_second=avg_tps,
        generated_tokens=last_result.output_token_count if last_result else 0,
        quality_score=quality_score,
        status="ok",
        generated_text=last_result.generated_text if last_result else "",
        input_tokens=last_result.input_token_count if last_result else 0,
    )


# ============================================================
# Full benchmark sweep
# ============================================================

def run_kv_cache_benchmark(
    wrapper,
    image: Image.Image,
    prompt: str,
    configs: Optional[List[KVCacheConfig]] = None,
    max_new_tokens: int = 256,
    num_runs: int = 1,
) -> List[KVCacheResult]:
    """
    Run all KV cache configurations and return a list of results.

    Failure handling: if any config raises an error, it is recorded
    as FAILED and the sweep continues to the next config. The benchmark
    never aborts due to a single config failure.

    Parameters
    ----------
    wrapper:
        A loaded QwenVLMWrapper instance.
    image:
        PIL Image.
    prompt:
        Text prompt.
    configs:
        List of KVCacheConfig objects. Defaults to KVCacheConfig.defaults().
    max_new_tokens:
        Max output tokens.
    num_runs:
        Number of inference runs per config (averaged).

    Returns
    -------
    List[KVCacheResult]
    """
    if configs is None:
        configs = KVCacheConfig.defaults()

    results: List[KVCacheResult] = []
    baseline_text: Optional[str] = None

    for config in configs:
        try:
            result = run_kv_cache_experiment(
                wrapper=wrapper,
                image=image,
                prompt=prompt,
                config=config,
                max_new_tokens=max_new_tokens,
                reference_text=baseline_text,
                num_runs=num_runs,
            )
        except Exception as unexpected:
            # Catch any exception not already handled inside run_kv_cache_experiment
            logger.error(
                f"[{config.name}] Unexpected exception: {unexpected}", exc_info=True
            )
            result = KVCacheResult(
                config=config.name,
                backend=config.backend,
                nbits=config.nbits,
                peak_memory_gb=0.0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                generated_tokens=0,
                quality_score=0.0,
                status="failed",
                error=f"Unexpected: {unexpected}",
            )

        results.append(result)

        # The first successful result becomes the reference for ROUGE-L
        if (
            baseline_text is None
            and result.status == "ok"
            and result.generated_text
        ):
            baseline_text = result.generated_text
            logger.info(f"Reference text set from config: {config.name}")

        # Clear GPU cache between configs to avoid contamination
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ============================================================
# Long-context simulation
# ============================================================

def build_synthetic_context(
    base_prompt: str,
    target_token_count: int,
    filler_text: str = "A person walks across the frame. The corridor remains empty. ",
) -> str:
    """
    Build a synthetic long-context string by repeating filler text.

    This is used to simulate accumulated long-context KV cache growth.

    IMPORTANT: This is a SIMULATION. It is NOT identical to processing
    actual video frames. It approximates the memory behavior of a growing
    KV cache context. Results should be interpreted accordingly.

    Parameters
    ----------
    base_prompt:
        The actual instruction/question.
    target_token_count:
        Approximate target context size in tokens.
        Actual token count depends on tokenizer; this is a rough target.
    filler_text:
        Text repeated to reach the target context length.

    Returns
    -------
    str: Prompt string approximately reaching the target token count.
    """
    # Rough approximation: ~4 chars per token (varies by tokenizer)
    target_chars = target_token_count * 4
    filler_chars = len(filler_text)

    if filler_chars == 0:
        return base_prompt

    repeats = max(1, (target_chars - len(base_prompt)) // filler_chars)
    context_prefix = filler_text * repeats
    return context_prefix + "\n\n" + base_prompt
