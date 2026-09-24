# ARGUS DEMO SCRIPT: 2-MINUTE PRESENTATION WALKTHROUGH

This document provides the exact step-by-step presentation script for demonstrating the **Argus VLM Surveillance Prototype**.

---

## ⏱️ Pre-Demo Setup (Before the Presentation Begins)

1. Open your terminal in the project directory:
   ```powershell
   streamlit run app.py
   ```
2. The browser will open to `http://localhost:8501`.
3. Verify that the sidebar has:
   - **Execution Mode**: `Demo Mode (Recorded Benchmark)` [Selected by default]
   - **Video Input**: `Sample Surveillance Footage (5s demo)`
   - **Temporal Token Pruning**: `[Checked]`
   - **KV Cache Precision**: `Dynamic FP16 (Baseline)` or `HQQ INT8 (Quantized)`

---

## 🎙️ The 2-Minute Presentation Script

### [0:00 - 0:25] Opening: The Surveillance VLM Bottleneck
> *"Good morning everyone. Today I'm presenting the core foundational milestone of **Argus** — our Real-Time AI Surveillance and Semantic Understanding System.*
>
> *Our goal is to allow Vision-Language Models to continuously observe surveillance video feeds, remember events, and answer natural-language security queries.*
>
> *However, there is a fundamental computational roadblock: VLMs produce hundreds of visual tokens per frame. In continuous surveillance, accumulating these tokens causes the Transformer KV cache to explode, crashing GPU memory and creating 60-second prefill delays. Today, we demonstrate our working solution: **Temporal Token Pruning coupled with KV-Cache Quantization**."*

### [0:25 - 0:50] Walking Through the UI Pipeline & Running the Analysis
*(Action: Point to the top pipeline diagram on the screen)*
> *"Here on the screen is our working Argus optimization pipeline. Instead of feeding every raw frame to the LLM:*
> 1. *We sample frames across the video.*
> 2. *We hook into the Qwen2.5-VL Vision Transformer before spatial merging.*
> 3. *We calculate patch-level cosine similarity to prune static background tokens.*
> 4. *We re-align 3D Rotary Position Embeddings.*
> 5. *We ingest the remaining tokens into an optimized KV cache.*
>
> *(Action: Click the green **'▶ Run Surveillance VLM Analysis'** button)*
>
> *Let's run the analysis on this surveillance clip. The system samples 100 frames across 120.3 seconds of footage, prunes redundant background tokens, and runs multimodal inference."*

### [0:50 - 1:20] Highlighting the Key Metric Cards & Experimental Proof
*(Action: Point to the 5 Metric Cards)*
> *"Look at the empirical results from our benchmark on an NVIDIA Tesla T4:*
> - *Across 100 frames, visual tokens dropped from **39,100 down to 23,505 — an immediate 39.9% reduction in visual tokens**.*
> - *Total context tokens shrank from **44,200 to 28,605**.*
> - *Under Temporal Pruning with FP16, prefill latency dropped from **60.43 seconds down to 33.14 seconds — a 1.82x speedup** — while saving nearly **1 GB of VRAM**.*
> - *When we activate HQQ INT8 KV-Cache Quantization, peak VRAM drops further to **9.45 GB — saving 1.46 GB, or 13.4% of total baseline GPU memory**.*
> - *Generated semantic output is displayed for qualitative inspection to verify whether useful scene information is retained.*
>
> *Notice our honest engineering finding: INT8 prefill is slower (78 seconds) due to software dequantization on T4. Thus, **Pruned FP16 is optimal for real-time speed**, while **Pruned INT8 is optimal for memory-constrained edge hardware**."*

### [1:20 - 1:45] Showing the Long-Context Trajectory Graph
*(Action: From the sidebar navigation, select **'Long-Context Benchmark & Visualizations'**)*
> *"To prove why this matters, look at this trajectory graph: **Context Tokens vs. Peak GPU VRAM**.*
>
> *In short single-image tests, quantization does almost nothing because static model weights dominate memory. But as surveillance context accumulates past 10,000, 20,000, and 40,000 tokens, the memory curves diverge dramatically. Pruning and quantization prevent the memory wall from crashing the server."*

### [1:45 - 2:00] Conclusion & Roadmap to Full Argus
*(Action: From the sidebar navigation, select **'Argus System Architecture & Roadmap'**)*
> *"To conclude: The VLM and KV-cache optimization engine is validated and fully operational today.*
>
> *Our next step is integrating upstream motion perception using YOLO and ByteTrack to trigger VLM processing only on salient events, and feeding these semantic descriptions into a spatio-temporal hybrid graph memory for natural language query retrieval.*
>
> *Thank you, and I welcome any questions."*

---

## 🎯 Quick Demo Troubleshooting Tips
- **If the evaluator asks to change parameters**:
  - Uncheck "Temporal Token Pruning" in the sidebar and click "Run Analysis" — show how it reflects the Unpruned Baseline (44,200 tokens, 10.91 GB VRAM, 60.43s prefill).
  - Check it back on and select "HQQ INT8" — show how peak VRAM drops to 9.45 GB.
- **If asked if this is running live**:
  - Be honest: *"The interface is in Demo Mode displaying our verified experimental benchmarks from our Tesla T4 run on Qwen2.5-VL-3B with 100 frames. We also have a Live Experimental Mode if local CUDA GPU hardware is attached."*
