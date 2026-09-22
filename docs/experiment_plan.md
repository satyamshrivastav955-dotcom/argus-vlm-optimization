# Experiment Plan — Argus VLM Optimization

## Overview

This document describes the day-by-day experimental plan for the VLM optimization
module of Project Argus. The goal is to produce reproducible, measured evidence for
which optimization techniques reduce compute cost while preserving surveillance quality.

---

## Day 1 — Baseline & Repository

**Notebook:** `01_kv_cache_baseline.ipynb`

**Goal:** Establish a clean, reproducible baseline for VLM inference.

**Tasks:**
1. Initialize repository structure
2. Create `QwenVLMWrapper` — configurable, CUDA/CPU-aware
3. Record environment metadata (model, torch, transformers, CUDA, GPU)
4. Set fixed random seed for reproducibility
5. Run deterministic inference on one fixed image
6. Record: input tokens, output tokens, latency, tokens/sec, peak VRAM, generated text

**Deliverables:**
- `results/kv_cache/baseline_results.csv`
- `results/kv_cache/baseline_results.json`
- Baseline text output (saved verbatim)

**Success criteria:**
- Notebook runs top-to-bottom in Colab without manual intervention
- Results saved before notebook end

---

## Day 2 — KV Cache Quantization

**Notebooks:** `01_kv_cache_baseline.ipynb` (extended) + `02_kv_cache_long_context.ipynb`

**Goal:** Measure the impact of KV cache quantization on memory and quality.

**Configurations:**
| Config | Backend | Bits |
|--------|---------|------|
| baseline_fp16_dynamic | HF DynamicCache | 16 |
| quantized_hqq_4bit | HQQ | 4 |
| quantized_hqq_2bit | HQQ | 2 |
| quantized_quanto_4bit | Quanto | 4 |

**For each configuration, measure:**
- Peak GPU allocated VRAM (GB)
- Inference latency (seconds)
- Tokens generated
- Tokens/sec
- Generated text
- ROUGE-L similarity to FP16 baseline

**Failure protocol:**
- If a backend fails → log FAILED + exception → continue to next config
- Never terminate the entire benchmark on a single failure

**Long-context experiment (Notebook 02):**
- Synthetic contexts: 1K, 5K, 10K, 20K, 50K tokens
- Compare FP16 vs INT4 vs INT2 at each context length
- Record: peak VRAM, latency, tokens/sec, success/failure
- Plots: VRAM vs context, latency vs context, throughput vs context

**Deliverables:**
- `results/kv_cache/kv_configs_results.csv`
- `results/kv_cache/long_context_results.csv`
- `results/figures/kv_vram_vs_context.png`
- `results/figures/kv_latency_vs_context.png`
- `results/figures/kv_throughput_vs_context.png`
- `results/figures/kv_memory_savings.png`

---

## Day 3 — Static Frame Baseline

**Notebook:** `03_static_frame_baseline.ipynb`

**Goal:** Measure the cost of naively processing every frame — no optimization.

**Pipeline:**
```
Video → sample every N frames → full frame → VLM → full caption
```

**For each sampled frame, record:**
- Frame index
- VLM call count
- Input token count
- Output token count
- Per-frame latency
- Generated caption

**Aggregate records:**
- Total VLM calls
- Total input/output tokens
- Total processing time
- Effective processing FPS
- Peak VRAM

**Input:** Synthetic frames (auto-generated) or external surveillance video (not committed)

**Deliverables:**
- `results/static_frames/baseline_results.csv`
- `results/static_frames/baseline_captions.json`

---

## Day 4 — Frame-Level Gating

**Notebook:** `04_frame_gating.ipynb`

**Goal:** Determine whether cheap change detection can safely skip frames without missing events.

**Algorithm:**
```
previous frame → current frame → grayscale diff → MAD score
   if score > threshold: process frame with VLM
   else: skip frame (inherit previous caption)
   fallback: force VLM every max_skip_frames regardless
```

**Parameter sweep:**
- `motion_thresh`: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

**For each threshold, record:**
- Frames processed
- Frames skipped
- VLM calls saved (vs baseline)
- Total latency
- Average latency/frame
- Event frames detected (if annotation available)
- Event frames missed (false negatives)

**Key question:** At what threshold do we start missing meaningful events?

**Deliverables:**
- `results/static_frames/frame_gate_results.csv`
- `results/figures/frame_gate_vmlcalls_vs_threshold.png`
- `results/figures/frame_gate_recall_vs_threshold.png`

---

## Day 5 — Spatial Redundancy Reduction

**Notebook:** `05_spatial_redundancy.ipynb`

**Goal:** Reduce redundant visual content (sky, walls, static background) before VLM input.

**IMPORTANT:** This implementation is **change-region cropping**, NOT internal ViT-token pruning.
Internal VLM token modification is out of scope for this phase.

**Algorithm:**
```
previous frame + current frame
   → grid-based change map (patch_size × patch_size cells)
   → bounding box of changed patches
   → add padding
   → crop image
   → VLM on cropped region only
```

**Parameter sweep:**
- `patch_size`: [16, 32, 64]
- `change_threshold`: [0.05, 0.10, 0.15, 0.20]
- `padding`: [0, 16, 32, 64]

**For each config, record:**
- Full frame dimensions
- Crop dimensions
- Crop area ratio (crop / full frame)
- Approximate input token reduction
- Per-frame latency
- Caption quality (ROUGE-L vs full-frame caption)
- Events/hazards preserved (manual check)

**Deliverables:**
- `results/static_frames/spatial_results.csv`
- `results/figures/spatial_crop_ratio_vs_patch.png`
- `results/figures/spatial_quality_vs_crop.png`

---

## Day 6 — Delta Captioning + Scene State

**Notebook:** `06_delta_captioning.ipynb`

**Goal:** Reduce output token count and repetition by prompting for incremental changes only.

**Algorithm:**
```
Frame 0: Full caption (establish scene state)
Frame N (changed, N > 0):
   delta prompt = previous scene + "describe only what changed"
   → VLM → delta caption → update scene state
Every K processed frames: force full re-caption
On scene cut (high diff score): force full re-caption
```

**Scene state structure:**
```python
{
    "objects": [],
    "locations": [],
    "activities": [],
    "hazards": [],
    "last_full_caption_frame": 0,
    "last_update_frame": 0
}
```

**For each frame, record:**
- Caption type (full / delta)
- Output token count
- Latency
- Caption text
- Scene state after update

**Aggregate records:**
- Total output tokens (vs naive full-caption baseline)
- Average latency/frame
- Missed changes (manual review)
- Caption drift over time (ROUGE-L degradation)

**Deliverables:**
- `results/static_frames/delta_caption_results.csv`
- `results/static_frames/delta_captions.json`
- `results/figures/delta_token_reduction.png`

---

## Day 7 — Combined Optimization + Ablation + Final Comparison

**Notebooks:** `07_combined_vlm_optimization.ipynb` + `08_final_comparison.ipynb`

**Goal:** Run the full pipeline and ablation study. Produce the final factual comparison.

**Full pipeline:**
```
VIDEO
 → Frame sampling (stride N)
 → Frame-level gate (MAD threshold)
 → Spatial redundancy reduction (change-region crop)
 → Delta captioning (full or delta prompt)
 → VLM (Qwen2.5-VL-3B-Instruct)
 → KV cache (FP16 / INT4 / INT2)
 → Scene state update
```

**Ablation experiments:**
| Exp | Frame Gate | Spatial | Delta | KV Cache |
|-----|-----------|---------|-------|----------|
| E0  | ✗ | ✗ | ✗ | FP16 |
| E1  | ✓ | ✗ | ✗ | FP16 |
| E2  | ✗ | ✓ | ✗ | FP16 |
| E3  | ✗ | ✗ | ✓ | FP16 |
| E4  | ✓ | ✓ | ✗ | FP16 |
| E5  | ✓ | ✓ | ✓ | FP16 |
| E6  | ✓ | ✓ | ✓ | INT4 |
| E7  | ✓ | ✓ | ✓ | INT2 |

**Final comparison (Notebook 08):**
- Read all result CSVs
- Generate aggregated summary
- Generate all comparison plots
- Produce factual engineering selection table (NO rankings)
- Save `results/final_summary.csv`
- Save `results/final_report.md`

**Deliverables:**
- `results/combined/ablation_results.csv`
- `results/final_summary.csv`
- `results/final_report.md`
- All comparison figures in `results/figures/`

---

## Notes

- Do not claim superiority of any configuration without measured evidence
- Record all failures explicitly with exception messages
- Save intermediate results incrementally (do not wait until end)
- All thresholds are configurable via YAML; nothing is hardcoded
