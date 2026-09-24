# ARGUS — Real-Time AI Surveillance & Efficient VLM Reasoning

> **Real-Time AI Surveillance, Semantic Understanding and Query System Using Hybrid Graph Memory with Efficient KV Cache Management**  
> *Subsystem Focus: Vision-Language Model (VLM) Token Optimization & Quantized KV-Cache Architecture*

[![Tests](https://img.shields.io/badge/tests-31%2F31%20passing-brightgreen.svg)](#testing--verification)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/VLM-Qwen2.5--VL--3B-blueviolet.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Prototype-FF4B4B.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Presentation Scope

Argus is an end-to-end intelligent surveillance platform designed to ingest continuous camera feeds, maintain structured semantic memory of events, and answer natural-language security queries.

### Current Validated Milestone (Today's Scope)
This repository contains the verified implementation and empirical evaluation of the **VLM Token & KV Cache Optimization Subsystem**, evaluated on `Qwen/Qwen2.5-VL-3B-Instruct` across continuous long-context surveillance video on an **NVIDIA Tesla T4 GPU**.

### The Computational Bottleneck in Surveillance VLMs
In continuous surveillance, streaming multi-frame video into an autoregressive Vision-Language Model creates extreme resource demands:
1. **Visual Token Redundancy**: Static cameras observe unchanged backgrounds (walls, hallways, parking lots) 90%+ of the time. Unoptimized pipelines generate hundreds of visual tokens per frame, flooding the Transformer with redundant inputs.
2. **KV-Cache Memory Wall**: Accumulated video tokens must be stored in the Key-Value (KV) cache. Across a 100-frame sequence, context exceeds **44,000 tokens**, causing rapid Out-Of-Memory (OOM) failures on 16GB GPUs.
3. **Quadratic Prefill Latency**: Attention prefill complexity scales quadratically $O(N^2)$, causing 60+ second delays per inference call and breaking real-time surveillance requirements.

```
SURVEILLANCE VIDEO ──► FRAME SAMPLING ──► TEMPORAL TOKEN PRUNING ──► QUANTIZED KV CACHE ──► VLM REASONING
  (120.3s / 100 frames)    (39,100 tokens)        (▼ 39.9% pruned)         (HQQ INT8 / FP16)     (Surveillance Scene Text)
```

---

## 2. Verified Empirical Benchmark Results

> **Ground Truth Notice:** All numbers reported below are directly extracted from the experimental run recorded in `edi (1).ipynb` on Google Colab (NVIDIA Tesla T4 16GB, Transformers 4.57.2, HQQ backend). No values have been simulated or fabricated.

### Long-Context Video Benchmark (100 Frames, 120.3 Seconds Video)

| Configuration | Context Tokens | Peak GPU VRAM | VRAM Saved vs Base | Prefill Latency | Relative Speedup | Key Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Unpruned Baseline (FP16)** | 44,200 | **10.91 GB** | Baseline (0.0 GB) | 60.43 s | 1.00x | Full visual tokens, standard DynamicCache |
| **2. Temporal Pruning + FP16** | 28,605 | **9.91 GB** | **+0.997 GB (9.1%)** | **33.14 s** | **1.82x (FASTER)** | **Optimal for Real-Time Speed & Latency** |
| **3. Temporal Pruning + INT8** | 28,605 | **9.45 GB** | **+1.459 GB (13.4%)**| 78.06 s | 0.77x (Quant delay) | **Optimal for VRAM-Constrained Edge Hardware** |

### Key Quantitative Findings:
- **Visual Token Reduction**: Visual tokens dropped from **39,100 to 23,505**, an empirical **39.88% reduction** across 100 frames (15,595 redundant background tokens eliminated).
- **Total Context Reduction**: Total sequence length dropped from **44,200 to 28,605 tokens** (a **35.28% overall sequence reduction**).
- **Prefill Speedup (FP16)**: Temporal pruning alone cuts prefill time nearly in half (**1.82x speedup**, 33.14s vs 60.43s) by eliminating quadratic self-attention over static background patches.
- **Peak VRAM Reduction (INT8)**: Combining temporal pruning with HQQ INT8 achieves a **1.459 GB (13.4%) peak VRAM reduction**, keeping memory comfortably within the 10GB boundary.
- **Hardware-Specific Trade-off**: INT8 prefill is **slower (78.06s)** on the Tesla T4 due to software dequantization kernel overhead during prefill accumulation. Hence, **Pruned FP16 is optimal for throughput**, whereas **Pruned INT8 is optimal for VRAM-capped edge setups**.
- **Qualitative Semantic Inspection**: Generated semantic output is displayed for qualitative inspection to verify whether useful scene information is retained.

---

## 3. Why Long-Context Evaluation is Essential (Single-Image vs Long-Context)

In single-image tests, KV-cache quantization appears ineffective. In continuous video surveillance, it is indispensable:

| Metric | Single Image (Cell 6) | Long-Context Video (Cell 10) | Why the Difference Matters |
| :--- | :---: | :---: | :--- |
| **Sequence Length** | 442 tokens | 44,200 tokens | Single image is ~100x smaller |
| **Model Weights** | ~7.50 GB | ~7.50 GB | Static weights dominate short context |
| **KV Cache Size** | ~0.014 GB (14 MB) | ~3.40 GB (3,400 MB) | KV cache expands linearly with frame count |
| **VRAM Saved by Quantization** | **+0.013 GB (+0.2%)** | **+1.459 GB (+13.4%)** | **Quantization only matters at scale** |

---

## 4. Technical Architecture & Scope Boundary

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Surveillance Video Stream                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 Perception Pipeline (YOLOv8 / ByteTrack)               │ [Next Integration]
│      - Motion detection, object tracking, event proposal triggers      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│          ARGUS VLM OPTIMIZATION ENGINE (CURRENT SUBSYSTEM)             │ [COMPLETED & VALIDATED]
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. ViT Patch Hooking & Pre-merge Feature Extraction              │  │
│  │ 2. Temporal Token Pruning (Cosine similarity threshold = 0.98)   │  │
│  │ 3. 3D RoPE (Rotary Position Embedding) Re-alignment             │  │
│  │ 4. Quantized KV Cache Management (HQQ INT8 / INT4)               │  │
│  │ 5. Qwen2.5-VL-3B Multimodal Conditioning                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                      Argus Semantic Memory Engine                      │ [Next Integration]
│  ┌─────────────────────────────────┬────────────────────────────────┐  │
│  │ Episodic Vector Store           │ Dynamic Knowledge Graph        │  │
│  │ (Dense scene embeddings)        │ (Entities, Actions, Relations) │  │
│  └─────────────────────────────────┴────────────────────────────────┘  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│               Natural Language Operator Query Interface                │ [Planned]
│      "Did any unauthorized vehicle enter between 14:00 and 15:00?"     │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Status Matrix

| Component | Status | Details |
| :--- | :---: | :--- |
| **ViT Patch Hooking** | ✅ VALIDATED | Intercepts pre-merge patch embeddings from `Qwen2.5-VL` vision merger |
| **Temporal Token Pruning** | ✅ VALIDATED | 39.9% visual token reduction using cosine similarity ($\ge 0.98$) |
| **3D RoPE Alignment** | ✅ VALIDATED | Slices positional indices to maintain correct spatio-temporal coordinates |
| **KV Cache Quantization** | ✅ VALIDATED | Evaluated DynamicCache (FP16) vs HQQ INT8 and INT4 |
| **Long-Context Accumulation** | ✅ VALIDATED | 100 frames across 120.3s evaluated on Tesla T4 with text generation |
| **Interactive Dashboard** | ✅ COMPLETED | Streamlit engineering prototype with Demo & Live modes |
| **Event-Triggered Perception** | ⏳ IN PROGRESS | Motion proposals via YOLOv8 / ByteTrack to gate VLM invocation |
| **Hybrid Graph Memory** | ⏳ ROADMAP | Spatio-temporal scene graphs (NetworkX/Neo4j) for long-term history |
| **Natural Language Query** | ⏳ ROADMAP | RAG-driven operator retrieval over structured graph memory |

---

## 5. Quick Start: Running the Prototype

### 1. Launch the Streamlit Application (One Command)

```powershell
streamlit run app.py
```

The application opens automatically in your browser at `http://localhost:8501`.

### 2. Available Modes
- **Demo Mode (Default & Recommended for Presentations)**:
  - 100% reliable, zero GPU required, zero network download required.
  - Loads verified empirical data from `results/benchmark_results.json`.
  - Includes interactive video preview (`sample_data/sample_surveillance.mp4`), pipeline animation, metric cards, trajectory plots, and raw model output.
- **Live / Experimental Mode**:
  - Automatically checks for local CUDA hardware.
  - Safely falls back if CUDA or model weights are missing, ensuring no crash during demonstrations.

---

## 6. Presentation Assets

All presentation assets are prepared in the `presentation/` directory:

- 📊 **PowerPoint Slide Deck**: `presentation/argus_vlm_presentation.pptx` (Professional 16:9 8-slide widescreen deck with modern dark styling).
- 📑 **Slide Outline Markdown**: `presentation/SLIDES.md` (Slide-by-slide headers, bullet points, and speaking notes).
- 🎙️ **2-Minute Demo Script**: `presentation/DEMO_SCRIPT.md` (Exact, timed walkthrough of what to say and click).
- 🛡️ **Viva & Technical Defense Q&A**: `presentation/VIVA_QA.md` (10 technical answers to tough examiner questions).

---

## 7. Running Unit Tests

To verify code integrity, run the automated test suite:

```powershell
pytest
```

Output:
```text
======================== 31 passed in 3.80s ========================
```

Test coverage includes:
- `tests/test_temporal_pruning.py`: Pre-merge cosine similarity masks, window re-indexing, minimum keep enforcement.
- `tests/test_frame_gate.py`: Frame-level pixel change gating (MAD and SSIM).
- `tests/test_spatial_redundancy.py`: Change-region detection and bounding box cropping.
- `tests/test_delta_caption.py`: State tracking and prompt building.
- `tests/test_metrics.py`: Token reduction, memory reduction, and FPS calculations.

---

## 8. Limitations & Engineering Realities

- **Prefill Latency of Quantized Cache**: In current PyTorch/Transformers implementations on Pascal/Turing GPUs (such as T4), INT8 dequantization is performed in software, introducing prefill overhead. Pruned FP16 should be selected when latency is paramount.
- **Scene-Level Collapse Prevention**: For extended static periods, `enforce_minimum_keep_ratio(0.05)` ensures at least 5% of tokens are retained, preventing attention failure.
- **Future Integration Scope**: Full-system components (YOLO tracking, graph memory, RAG querying) are architectural extensions and are not claimed as completed in this milestone.

---

## 9. Citation & Project Metadata

```bibtex
@software{argus_vlm_optimization_2026,
  author = {Shrivastav, Satyam},
  title = {Argus: Real-Time AI Surveillance & Efficient VLM Reasoning},
  year = {2026},
  url = {https://github.com/satyamshrivastav955-dotcom/argus-vlm-optimization}
}
```
