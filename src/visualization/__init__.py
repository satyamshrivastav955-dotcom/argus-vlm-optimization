"""
Argus VLM Optimization - Visualization Module.
Provides presentation-ready plotting utilities for experimental benchmarks.
"""

from src.visualization.plots import (
    plot_vram_vs_context,
    plot_latency_vs_context,
    plot_token_reduction,
    plot_vlm_calls_comparison,
    plot_ablation_comparison,
)

__all__ = [
    "plot_vram_vs_context",
    "plot_latency_vs_context",
    "plot_token_reduction",
    "plot_vlm_calls_comparison",
    "plot_ablation_comparison",
]
