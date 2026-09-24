# ARGUS: Real-Time AI Surveillance & Efficient VLM Reasoning
## Project Progress Presentation Deck (8 Slides)

---

### SLIDE 1: TITLE SLIDE
**Title**: ARGUS: Real-Time AI Surveillance & Efficient VLM Reasoning  
**Subtitle**: Semantic Understanding and Query System with Optimized Visual Tokens & KV Cache Architecture  
**Presenter**: Satyam Shrivastav  
**Project Scope**: VLM Optimization Subsystem Milestone  
**Status**: Experimental Validation Complete (NVIDIA Tesla T4)  

---

### SLIDE 2: THE PROBLEM — Why Continuous VLM Processing is Prohibitive
**Header**: Continuous Video Surveillance Overwhelms Modern VLMs
- **The Data Explosion Pipeline**:
  $$\text{Continuous Video Stream} \longrightarrow \text{High-FPS Frames} \longrightarrow \text{Dense Visual Tokens} \longrightarrow \text{Quadratic KV Cache} \longrightarrow \text{OOM / Crash}$$
- **Key Bottlenecks**:
  1. **Massive Visual Redundancy**: Static cameras observe unchanged backgrounds (walls, floors, empty roads) 90%+ of the time, yet traditional pipelines encode full image patches into hundreds of tokens per frame.
  2. **KV Cache Memory Scaling**: As surveillance context accumulates over minutes, the Transformer Key-Value (KV) cache grows linearly in length and quadratically in attention compute, rapidly exhausting GPU VRAM.
  3. **High Prefill Latency**: Processing hundreds of accumulated frames causes extreme prefill delays (60+ seconds), breaking real-time surveillance requirements.

---

### SLIDE 3: CURRENT IMPLEMENTATION — The Argus Optimization Subsystem
**Header**: What is Actually Built and Tested Today
- **Model Backbone**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Component 1 — Internal ViT Hooking**: Intercepts patch embeddings prior to the spatial merger module to analyze raw visual feature dynamics.
- **Component 2 — Temporal Token Pruning**: Evaluates cosine similarity between consecutive frame patches ($\text{threshold} = 0.98$). Automatically preserves dynamic regions while discarding redundant background tokens.
- **Component 3 — Positional Alignment**: Recalculates 3D RoPE (Rotary Position Embeddings) to maintain temporal and spatial coherence after token pruning.
- **Component 4 — KV Cache Quantization**: Evaluated FP16 DynamicCache vs. HQQ INT8 and INT4 quantized cache configurations.
- **Component 5 — Long-Context Benchmarking**: End-to-end evaluation across a real 100-frame video sequence.

---

### SLIDE 4: THE LONG-CONTEXT EXPERIMENT
**Header**: Real-World Continuous Video Evaluation Setup
- **Dataset / Input**:
  - Surveillance video duration: **120.3 seconds** (2.0 minutes continuous capture)
  - Raw video count: **3,610 frames** at 30.0 FPS (640x480 resolution)
  - Evaluated sample: **100 evenly distributed frames** across the entire duration
- **Unpruned Baseline Sequence**:
  - **442 tokens** per frame (391 visual tokens + 51 text/system prompt tokens)
  - **44,200 total context tokens** accumulated in memory
  - **39,100 total visual tokens** before compression
- **Evaluation Environment**:
  - Hardware: NVIDIA Tesla T4 GPU (15.36 GB VRAM)
  - Framework: PyTorch, HuggingFace Transformers 4.57.2, HQQ quantization backend

---

### SLIDE 5: TEMPORAL TOKEN COMPRESSION RESULTS
**Header**: 39.9% Reduction in Visual Token Load
- **Visual Token Reduction**:
  - **Before Pruning**: 39,100 visual tokens
  - **After Pruning**: 23,505 visual tokens
  - **Pruning Rate**: **39.88% of visual tokens eliminated** (15,595 redundant tokens pruned)
- **Total Context Reduction**:
  - **44,200 total tokens $\longrightarrow$ 28,605 total tokens** (**35.3% overall reduction**)
- **Safety Guarantee**:
  - Implemented `enforce_minimum_keep_ratio` (5% minimum retention) to prevent frame representation collapse during static scenes.
- **Surveillance Significance**:
  - Camera preserves moving entities, pedestrians, and changed objects while omitting redundant static tiles.
  - Generated semantic output is displayed for qualitative inspection.

---

### SLIDE 6: KV CACHE BENCHMARK COMPARISON
**Header**: Empirical Memory and Latency Results (100 Frames / 120.3s)

| Configuration | Context Tokens | Peak VRAM (GB) | VRAM Saved | Prefill Latency (s) | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unpruned + FP16 (Baseline)** | 44,200 | **10.91 GB** | Baseline (0.0 GB) | 60.43 s | 1.00x |
| **2. Temporal Pruned + FP16** | 28,605 | **9.91 GB** | **+0.997 GB (9.1%)** | **33.14 s** | **1.82x Speedup** |
| **3. Temporal Pruned + INT8** | 28,605 | **9.45 GB** | **+1.459 GB (13.4%)** | 78.06 s | 0.77x (Quant overhead) |

*Recorded benchmark from current experiment on NVIDIA Tesla T4.*

---

### SLIDE 7: KEY TECHNICAL FINDINGS & TRADE-OFFS
**Header**: What the Empirical Data Teaches Us
1. **Temporal Pruning Delivers Dual Wins**:
   - Pruning alone cuts prefill latency by **1.82x** (from 60.4s down to 33.1s) and saves **~1.0 GB VRAM** due to reduced self-attention complexity ($O(N^2)$ prefill savings). Qualitative semantic output is provided to inspect whether useful scene information is retained.
2. **INT8 Maximizes Memory Savings (13.4% Total Reduction)**:
   - Combining pruning with HQQ INT8 achieves the lowest peak VRAM (**9.45 GB** vs 10.91 GB baseline).
3. **The Quantization Latency Trade-Off**:
   - INT8 is slower in prefill (**78.06s**) on the tested T4 setup because software dequantization kernels add per-token overhead during prefill. **Conclusion: Pruned FP16 is optimal for latency-critical streams; Pruned INT8 is optimal for VRAM-constrained edge devices.**
4. **Single-Image vs. Long-Context Insight**:
   - Single-image tests saved `< 0.2%` VRAM (static weights dominate). Long-context testing is essential to reveal real KV-cache dynamics.

---

### SLIDE 8: CURRENT PROGRESS & NEXT STEPS
**Header**: Roadmap to the Full Argus Surveillance System

```
[VALIDATED COMPUTE FOUNDATION]                [DOWNSTREAM INTEGRATION ROADMAP]
✓ Qwen2.5-VL-3B Integration                   → Upstream Perception (YOLOv8 + ByteTrack)
✓ Pre-merge Visual Token Hooking              → Event-Triggered Reasoning (Salience gating)
✓ Temporal Pruning (39.9% visual reduction)   → Spatio-Temporal Hybrid Graph Memory (NetworkX/Neo4j)
✓ KV Cache Quantization (FP16 / INT8 / INT4)  → Vector Store for dense scene retrieval
✓ 100-Frame Long-Context Benchmarks           → Natural Language Operator Query Engine
```

- **Bottom Line**: The VLM and memory optimization engine is verified and functional. Next milestone connects motion perception triggers and hybrid graph memory.
