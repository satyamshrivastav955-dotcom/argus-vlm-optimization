# Argus — VLM Optimization Module

> **Real-Time AI Surveillance, Semantic Understanding and Query System Using Hybrid Graph Memory with Efficient KV Cache Management**  
> *Sub-Module: Vision-Language Model (VLM) Optimization & Redundancy Reduction*

[![Tests](https://img.shields.io/badge/tests-27%2F27%20passing-brightgreen.svg)](#testing--verification)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Context & Motivation

**Project Argus** is an intelligent, real-time AI surveillance and semantic reasoning system engineered for long-duration continuous video streams. In continuous surveillance, streaming multi-camera HD video directly into a heavy Vision-Language Model (VLM) creates severe computational bottlenecks:

1. **Massive Redundancy:** Static scenes (empty hallways, stationary parking lots, nighttime views) generate near-identical frames, squandering GPU compute on redundant visual tokens.
2. **KV Cache Explosion:** Long-duration conversational context and multi-frame histories consume high gigabytes of VRAM in attention Key-Value (KV) caches, rapidly triggering Out-Of-Memory (OOM) failures.
3. **High Latency:** Naive full-frame generation induces seconds of latency per frame, breaking real-time surveillance requirements.

This repository implements the **VLM Optimization Sub-Module** for Argus. It delivers a modular, empirically benchmarked optimization pipeline designed to minimize inference costs, visual token counts, and memory footprint while preserving surveillance scene understanding.

```
                  RAW SURVEILLANCE VIDEO STREAM
                               │
                               ▼
               ┌───────────────────────────────┐
               │   1. Frame Skip Gating (MAD)  │ ──(Score < Thresh)──> [SKIP FRAME]
               └───────────────────────────────┘
                               │ (Motion Detected)
                               ▼
               ┌───────────────────────────────┐
               │ 2. Spatial Redundancy Cropping │ ──> Crop Bounding Box + Padding
               └───────────────────────────────┘     (Change-Region Only)
                               │
                               ▼
               ┌───────────────────────────────┐
               │    3. Delta Scene Captioner   │ ──> Build Dynamic Prompt
               └───────────────────────────────┘     (Full vs. Delta State)
                               │
                               ▼
               ┌───────────────────────────────┐
               │  4. VLM Inference with        │
               │     Quantized KV-Cache        │
               │  (Qwen2.5-VL / Florence-2)    │
               │  • Dynamic FP16               │
               │  • HQQ INT4 / INT2            │
               │  • Quanto INT4                │
               └───────────────────────────────┘
                               │
                               ▼
                     STRUCTURED SCENE STATE
                   (Feed to Argus Graph Memory)
```

---

## 2. Assigned Scope & Responsibilities

This sub-module focuses on verifiable, reproducible engineering:

- **KV-Cache Quantization:** Evaluating FP16 vs. INT4 (HQQ, Quanto) vs. INT2 (HQQ) quantization schemes for memory reduction and generation fidelity.
- **Static & Redundant Frame Mitigation:** Implementing pre-inference pixel-difference gating to drop static frames without GPU invocation.
- **Spatial Redundancy Reduction:** Detecting localized motion patches and feeding tightly bounded crops to the VLM instead of redundant full frames.
- **Delta Captioning & Scene State:** Maintaining conversational state and prompting the VLM to describe only incremental scene deltas.
- **VLM Architecture Support:**
  - Prototype model: `Qwen/Qwen2.5-VL-3B-Instruct`
  - Architectural design: Abstract `VLMBackend` base class allowing seamless drop-in of `Florence-2` or alternative VLMs.
- **Evidence-Based Engineering:** Standardized metrics, reproducible sweep scripts, and zero fabricated claims.

---

## 3. Repository Structure

```
argus-vlm-optimization/
├── .gitignore                      # Git ignore for ML, checkpoints, logs, and venvs
├── LICENSE                         # MIT License
├── requirements.txt                # Pinned production and research dependencies
├── README.md                       # Comprehensive project documentation
├── configs/                        # YAML configuration files
│   ├── default.yaml                # Global execution defaults (model, device, paths)
│   ├── kv_cache.yaml               # KV cache quantization backends & context sweeps
│   └── frame_optimization.yaml     # Motion thresholds, patch sizes, and intervals
├── docs/                           # Detailed technical documentation
│   ├── experiment_plan.md          # 7-day experimental schedule & protocols
│   ├── methodology.md              # Theoretical foundations, formulas & limitations
│   └── results_template.md         # Markdown template for recording empirical data
├── notebooks/                      # Self-contained Colab-ready experimental notebooks
│   ├── 01_kv_cache_baseline.ipynb  # Single-frame KV-cache quantization benchmark
│   ├── 02_kv_cache_long_context.ipynb # Multi-frame long context scaling (1K–50K tokens)
│   ├── 03_static_frame_baseline.ipynb # Unoptimized naive every-N-frame baseline
│   ├── 04_frame_gating.ipynb       # Frame gating motion threshold parameter sweeps
│   ├── 05_spatial_redundancy.ipynb # Grid patch detection & change-region cropping
│   ├── 06_delta_captioning.ipynb   # Full vs. delta captioning with state tracking
│   ├── 07_combined_vlm_optimization.ipynb # Full pipeline ablation study (E0–E7)
│   └── 08_final_comparison.ipynb   # Statistical aggregation and final report synthesis
├── sample_data/                    # Surveillance dataset guide & synthetic fixtures
│   └── README.md                   # Instructions for sample video feeds & annotations
├── scripts/                        # Automation & builder utilities
│   └── generate_notebooks.py       # Notebook generation and verification script
├── src/                            # Production Python source modules
│   ├── __init__.py
│   ├── vlm/                        # VLM wrappers and abstractions
│   │   ├── __init__.py
│   │   └── qwen_vlm.py             # Qwen2.5-VL wrapper & VLMBackend interface
│   ├── kv_cache/                   # KV cache quantization engine
│   │   ├── __init__.py
│   │   └── benchmark.py            # HQQ/Quanto/Dynamic runner with OOM guard
│   ├── frame_optimization/         # Frame & token redundancy reducers
│   │   ├── __init__.py
│   │   ├── frame_gate.py           # Pixel-level MAD/SSIM frame gating
│   │   ├── spatial_redundancy.py   # PatchChangeDetector & crop_changed_region
│   │   └── delta_caption.py        # DeltaCaptioner & SceneState tracker
│   ├── benchmarking/               # Metrics and experiment management
│   │   ├── __init__.py
│   │   ├── metrics.py              # ROUGE-L, FPS, token & memory reduction formulas
│   │   └── benchmark_runner.py     # Safe execution and incremental CSV/JSON saves
│   └── visualization/              # Publication-grade plotting routines
│       ├── __init__.py
│       └── plots.py                # VRAM, latency, token, and ablation charts
├── results/                        # Experimental outputs (git-tracked directory structure)
│   ├── kv_cache/                   # Raw CSV/JSON benchmarks for KV cache
│   ├── static_frames/              # Results from gating, spatial, and delta tests
│   ├── combined/                   # Ablation study outputs (E0–E7)
│   └── figures/                    # High-resolution generated charts
└── tests/                          # Automated test suite (Pytest)
    ├── __init__.py
    ├── test_frame_gate.py          # Unit tests for FrameGate and MAD difference
    ├── test_spatial_redundancy.py  # Tests for patch detection & bbox cropping
    ├── test_delta_caption.py       # Tests for SceneState & prompt builders
    ├── test_metrics.py             # Validation of metric calculations
    └── test_serialization.py      # Verification of CSV/JSON serialization
```

---

## 4. Core Optimization Techniques

### 4.1 KV-Cache Quantization
Autoregressive token generation requires storing Key and Value projection tensors for all preceding tokens. For extended surveillance sessions, this KV cache consumes gigabytes of VRAM.
- **Dynamic FP16:** Standard PyTorch floating-point cache (baseline).
- **HQQ (Half-Quadratic Quantization):** On-the-fly INT4 and INT2 quantization of Key and Value tensors without requiring calibration datasets.
- **Quanto:** Post-training INT4 integer quantization.
- **OOM Safety:** The execution runner encapsulates generation in guarded try-except blocks, recording Out-Of-Memory events gracefully and continuing the sweep.

### 4.2 Frame Skip Gating (Temporal Reduction)
Surveillance cameras monitor scenes with long idle intervals. Before invoking the VLM:
- Computes Mean Absolute Difference (MAD) between consecutive grayscale frames:
  $$\text{MAD} = \frac{1}{H \times W \times 255} \sum_{i=1}^H \sum_{j=1}^W |I_t(i,j) - I_{t-1}(i,j)|$$
- If $\text{MAD} < \text{threshold}$, the frame is discarded immediately, costing less than $0.1\,\text{ms}$ on CPU and consuming $0$ VLM compute.
- A configurable `max_skip_frames` heartbeat forces periodic inference to prevent blind drift during gradual illumination changes.

### 4.3 Spatial Redundancy Reduction (Change-Region Cropping)
When motion occurs, it frequently occupies only a small fraction of the surveillance field (e.g., a person walking through a doorway):
- The frame is divided into regular grid patches (e.g., $32 \times 32$ pixels).
- Patches exceeding a per-patch change threshold are aggregated into a union bounding box.
- Safety padding is applied, and the cropped region is passed to the VLM processor.
- Fallback logic reverts to full-frame mode if the changed area is trivially small ($<10\%$) or dominates the frame ($>80\%$).

### 4.4 Delta Captioning & Scene State Tracking
Standard pipelines re-prompt the VLM with a generic prompt every frame, causing repeated generation of redundant background text:
- **Frame 0:** A full prompt generates a comprehensive baseline description initializing `SceneState`.
- **Changed Frames:** The VLM is supplied with the previous scene description and instructed to output **only** the incremental delta.
- **Periodic Full Re-captioning:** Refreshes state every $K$ processed frames to prevent semantic drift.
- **Scene Cut Detection:** High frame differences automatically force a full re-caption.

### 4.5 Full Pipeline Ablation Matrix (E0 – E7)

| Experiment | Frame Gating | Spatial Cropping | Delta Captioning | KV Quantization | Purpose |
|:---:|:---:|:---:|:---:|:---:|:---|
| **E0** | ❌ | ❌ | ❌ | ❌ | Naive Unoptimized Baseline |
| **E1** | ✅ | ❌ | ❌ | ❌ | Temporal Skip Benefit Only |
| **E2** | ❌ | ✅ | ❌ | ❌ | Spatial Token Reduction Only |
| **E3** | ❌ | ❌ | ✅ | ❌ | Output Token Reduction Only |
| **E4** | ❌ | ❌ | ❌ | ✅ | Memory Footprint Reduction Only |
| **E5** | ✅ | ✅ | ❌ | ❌ | Gating + Spatial Cropping |
| **E6** | ✅ | ✅ | ✅ | ❌ | Gating + Spatial + Delta Captioning |
| **E7** | ✅ | ✅ | ✅ | ✅ | **Full Argus Combined Architecture** |

---

## 5. Engineering Disclaimers & Honest Limitations

In accordance with strict research integrity:

1. **Change-Region Cropping $\neq$ ViT Token Pruning:** This module performs bounding box cropping on the PIL image prior to input processing. It does **not** alter internal Vision Transformer (ViT) patch attention matrices or remove embedded patch tokens inside the transformer layers. Actual visual token savings depend on the VLM processor's internal tiling algorithm.
2. **ROUGE-L Limitations:** ROUGE-L is an auxiliary textual similarity metric compared against uncompressed FP16 output. It is **not** a direct metric for surveillance hazard detection or event recall.
3. **Event Recall Requires Ground Truth:** Real event recall and false negative rates require temporal bounding box annotations on genuine surveillance footage.
4. **Delta Caption Drift:** Without periodic full re-captioning, iterative delta captions are subject to perceptual drift over long horizons.
5. **No Fabricated Data:** The repository contains no placeholder benchmark numbers masquerading as real results. All results files are generated from actual executions.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.0+ recommended for VLM inference (CPU execution is supported for testing, gating, spatial cropping, and mock execution).

### Setup
```bash
# Clone repository
git clone https://github.com/your-username/argus-vlm-optimization.git
cd argus-vlm-optimization

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Test Suite
```bash
python -m pytest tests/ -v
```

All 27 test cases cover:
- Frame difference calculation (MAD), identical frames, inverted frames, shape resizing
- First-frame processing guarantee, skip logic, forced periodic heartbeat
- Patch detection, bounding box unions, crop padding, and fallback thresholds
- Scene state updates, delta prompt formatting, recaption triggering
- Token, memory, FPS, and event recall metric computations
- JSON/CSV incremental persistence and data structure serialization

---

## 7. Running the Experiments

The experiment workflow can be executed on Google Colab (with a T4/A100 GPU) or locally:

1. **KV Cache Benchmarks:** Open `notebooks/01_kv_cache_baseline.ipynb` and `notebooks/02_kv_cache_long_context.ipynb`.
2. **Frame Redundancy Experiments:** Run `notebooks/03_static_frame_baseline.ipynb`, `notebooks/04_frame_gating.ipynb`, and `notebooks/05_spatial_redundancy.ipynb`.
3. **Delta Captioning & Ablation:** Execute `notebooks/06_delta_captioning.ipynb` and `notebooks/07_combined_vlm_optimization.ipynb`.
4. **Synthesis & Final Report:** Run `notebooks/08_final_comparison.ipynb` to aggregate all CSVs into `results/final_summary.csv` and generate `results/final_report.md`.

---

## 8. License & Attribution

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. Developed as part of **Project Argus** for college research on real-time surveillance optimization.
