"""
Plotting and visualization routines for Argus VLM Optimization benchmarks.
Generates presentation-ready, high-resolution figures for papers, slides, and reports.
"""

import os
from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _setup_style():
    """Configure matplotlib defaults for publication-quality aesthetic."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    })


def _ensure_dir(path: Optional[str]):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_vram_vs_context(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "KV Cache Peak Memory Usage vs. Context Size",
) -> plt.Figure:
    """
    Plot Peak VRAM (GB) as a function of context token length across KV cache configurations.

    Expected columns in df:
        - 'context_tokens' or 'context_size'
        - 'peak_memory_gb'
        - 'config' or 'backend'
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ctx_col = "context_tokens" if "context_tokens" in df.columns else "context_size"
    cfg_col = "config" if "config" in df.columns else "backend"

    # Filter out failed runs if status column exists
    plot_df = df[df["status"] != "FAILED"].copy() if "status" in df.columns else df.copy()

    if plot_df.empty or ctx_col not in plot_df.columns:
        ax.text(0.5, 0.5, "No completed benchmark data available", ha="center", va="center")
        ax.set_title(title)
        return fig

    palette = plt.cm.get_cmap("tab10")
    unique_cfgs = plot_df[cfg_col].unique()

    for i, cfg in enumerate(unique_cfgs):
        sub = plot_df[plot_df[cfg_col] == cfg].sort_values(by=ctx_col)
        ax.plot(
            sub[ctx_col],
            sub["peak_memory_gb"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=str(cfg),
            color=palette(i % 10),
        )

    ax.set_xlabel("Context Size (Tokens)")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(title="KV Cache Config", frameon=True, framealpha=0.9)
    fig.tight_layout()

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_latency_vs_context(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Generation Latency vs. Context Size",
) -> plt.Figure:
    """
    Plot generation latency (seconds) or tokens/sec vs. context size.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ctx_col = "context_tokens" if "context_tokens" in df.columns else "context_size"
    cfg_col = "config" if "config" in df.columns else "backend"
    lat_col = "latency_seconds" if "latency_seconds" in df.columns else "latency"

    plot_df = df[df["status"] != "FAILED"].copy() if "status" in df.columns else df.copy()

    if plot_df.empty or ctx_col not in plot_df.columns:
        ax.text(0.5, 0.5, "No completed benchmark data available", ha="center", va="center")
        ax.set_title(title)
        return fig

    palette = plt.cm.get_cmap("tab10")
    unique_cfgs = plot_df[cfg_col].unique()

    for i, cfg in enumerate(unique_cfgs):
        sub = plot_df[plot_df[cfg_col] == cfg].sort_values(by=ctx_col)
        ax.plot(
            sub[ctx_col],
            sub[lat_col],
            marker="s",
            linewidth=2,
            markersize=6,
            label=str(cfg),
            color=palette(i % 10),
        )

    ax.set_xlabel("Context Size (Tokens)")
    ax.set_ylabel("Total Latency (seconds)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(title="KV Cache Config", frameon=True, framealpha=0.9)
    fig.tight_layout()

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_token_reduction(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Vision Token Reduction via Spatial Cropping",
) -> plt.Figure:
    """
    Plot comparison of vision tokens between full frame and spatial change crop.

    Expected columns:
        - 'frame_id' or 'strategy'
        - 'baseline_tokens' or 'full_tokens'
        - 'optimized_tokens' or 'crop_tokens'
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    if "baseline_tokens" in df.columns and "optimized_tokens" in df.columns:
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width / 2, df["baseline_tokens"], width, label="Full Frame", color="#4a90e2", alpha=0.9)
        ax.bar(x + width / 2, df["optimized_tokens"], width, label="Spatial Crop", color="#50e3c2", alpha=0.9)
        ax.set_xlabel("Sample Frame Index")
        ax.set_ylabel("Estimated Vision Tokens")
    elif "strategy" in df.columns and "avg_tokens" in df.columns:
        ax.bar(df["strategy"], df["avg_tokens"], color="#4a90e2", width=0.5)
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Average Tokens per Frame")
    else:
        ax.text(0.5, 0.5, "Insufficient columns for token reduction plot", ha="center", va="center")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_vlm_calls_comparison(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "VLM Inference Calls: Baseline vs. Frame Gating",
) -> plt.Figure:
    """
    Plot breakdown of processed vs. skipped frames under different gating thresholds.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if "threshold" in df.columns and "processed_frames" in df.columns and "skipped_frames" in df.columns:
        thresholds = [str(t) for t in df["threshold"]]
        processed = df["processed_frames"].values
        skipped = df["skipped_frames"].values

        ax.bar(thresholds, processed, label="Processed (VLM Invoked)", color="#e94e77", alpha=0.85)
        ax.bar(thresholds, skipped, bottom=processed, label="Skipped (Gated)", color="#2ecc71", alpha=0.85)
        ax.set_xlabel("Motion Difference Threshold")
        ax.set_ylabel("Frame Count")
        ax.legend(frameon=True)
    elif "method" in df.columns and "vlm_calls" in df.columns:
        ax.bar(df["method"], df["vlm_calls"], color="#3498db", width=0.5)
        ax.set_xlabel("Method")
        ax.set_ylabel("Total VLM Calls")
    else:
        ax.text(0.5, 0.5, "Data not formatted for VLM calls comparison", ha="center", va="center")

    ax.set_title(title, fontweight="bold", pad=12)
    fig.tight_layout()

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_ablation_comparison(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Argus VLM Optimization Pipeline Ablation (E0 - E7)",
) -> plt.Figure:
    """
    Grouped bar or summary chart comparing ablation configurations across multiple dimensions:
    VLM calls, Total Latency, and Caption Quality.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    exp_col = "experiment_id" if "experiment_id" in df.columns else "name"

    if exp_col in df.columns:
        labels = df[exp_col].astype(str).tolist()

        # Subplot 1: Latency & VLM calls
        ax1 = axes[0]
        if "total_latency_seconds" in df.columns:
            ax1.barh(labels, df["total_latency_seconds"], color="#4a90e2", alpha=0.85)
            ax1.set_xlabel("Total Pipeline Latency (s)")
        elif "vlm_calls" in df.columns:
            ax1.barh(labels, df["vlm_calls"], color="#4a90e2", alpha=0.85)
            ax1.set_xlabel("VLM Calls")
        ax1.set_title("Runtime / Computation Efficiency", fontweight="bold")
        ax1.invert_yaxis()

        # Subplot 2: Quality or Token Savings
        ax2 = axes[1]
        if "avg_rouge_l" in df.columns:
            ax2.barh(labels, df["avg_rouge_l"], color="#2ecc71", alpha=0.85)
            ax2.set_xlim(0, 1.05)
            ax2.set_xlabel("Average ROUGE-L vs. Baseline")
            ax2.set_title("Semantic Quality Retention", fontweight="bold")
        elif "token_reduction_pct" in df.columns:
            ax2.barh(labels, df["token_reduction_pct"], color="#f39c12", alpha=0.85)
            ax2.set_xlabel("Token Reduction (%)")
            ax2.set_title("Input Token Reduction", fontweight="bold")
        ax2.invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "No experiment ID column", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No experiment ID column", ha="center", va="center")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
