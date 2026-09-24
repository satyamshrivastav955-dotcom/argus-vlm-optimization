# ARGUS VLM OPTIMIZATION: VIVA & TECHNICAL DEFENSE Q&A

This guide prepares you for questions from examiners, professors, or reviewers regarding the technical implementation and empirical results of the Argus VLM optimization subsystem.

---

### Q1: Why did you choose Temporal Pruning over standard frame dropping?
**Answer:**
> *"Frame dropping (frame gating) operates at a coarse, all-or-nothing granularity: it either discards an entire frame or keeps the entire frame. If a person enters a large hallway, frame dropping forces the VLM to process all 391 visual tokens of the room, 95% of which is unchanged static background.*
>
> *Temporal pruning operates at the internal token patch level. By computing cosine similarity between consecutive frame patches in the Vision Transformer before the spatial merger, we discard only the static background tokens (walls, floors) while preserving the dynamic foreground tokens (the moving person). In our 100-frame test, this eliminated 39.9% of visual tokens. Qualitative semantic output is provided to inspect whether useful scene information is retained."*

---

### Q2: Why is KV-cache optimization necessary for video surveillance?
**Answer:**
> *"In autoregressive Vision-Language Models, every token processed must be stored as Key-Value activation vectors in GPU memory so future tokens can attend to them. In standard NLP tasks, context is typically a few hundred words. In continuous video surveillance, each frame introduces hundreds of visual tokens. Across just 100 frames, we accumulated 44,200 tokens.*
>
> *The KV cache expands linearly with token count: for a 3B model, 44,000 tokens of FP16 KV cache consumes over 3.4 GB of VRAM alone. Without KV-cache optimization and token pruning, continuous surveillance quickly causes Out-Of-Memory (OOM) crashes on standard 16GB GPUs like the Tesla T4."*

---

### Q3: Why did you choose Qwen2.5-VL-3B-Instruct?
**Answer:**
> *"We selected Qwen2.5-VL-3B for three specific technical reasons:*
> 1. *It natively supports dynamic resolution and variable token lengths via its spatial merger architecture, allowing us to inspect pre-merge and post-merge patches.*
> 2. *It employs 3D Rotary Position Embeddings (RoPE), which explicitly separate spatial (height/width) coordinates from temporal (time) coordinates. This allows us to slice out pruned tokens and recompute positional indices without destroying spatial coherence.*
> 3. *At 3 billion parameters, it represents the ideal sweet spot: high semantic reasoning capabilities for surveillance anomaly detection while still being deployable on edge/datacenter GPUs with 16GB VRAM."*

---

### Q4: Why focus on INT8 instead of INT4 or INT2 for KV-cache quantization?
**Answer:**
> *"In our initial single-image benchmarks (Cell 6), we tested INT8, INT4, and INT2 using the HQQ backend. While INT4 and INT2 theoretically compress memory further, we observed severe semantic degradation and repetitive generation loops at INT4 and INT2.*
>
> *INT8 with residual length 128 maintained generation fidelity closely aligned with the unpruned FP16 output, while yielding an extra 0.46 GB of VRAM savings over pruning alone in our 100-frame experiment. Qualitative semantic output is provided to inspect whether useful scene information is retained."*

---

### Q5: Why doesn't quantization reduce VRAM much for a single image?
**Answer:**
> *"This was one of our key empirical findings:*
> *For a single image (442 tokens), model parameters dominate total GPU memory. A 3B parameter model in FP16 takes approximately 7.5 GB of static weight memory. The KV cache for 442 tokens is only ~14 MB. Compressing 14 MB by 50% saves only 7 MB (<0.1%), which is imperceptible.*
>
> *However, in long-context surveillance (44,200 tokens across 100 frames), the KV cache grows to gigabytes. That is why single-image benchmarks are misleading, and why our 100-frame long-context test was necessary to prove that KV-cache quantization saves 1.46 GB (13.4% of total memory)."*

---

### Q6: Why did INT8 increase prefill time from 33.14s to 78.06s? Is INT8 not supposed to be faster?
**Answer:**
> *"No, in this implementation INT8 is NOT faster, and we are explicit about this finding. In HuggingFace Transformers using the HQQ QuantizedCache on an NVIDIA Tesla T4, the quantized keys and values must be dequantized on-the-fly during attention computation.*
>
> *Because standard PyTorch layers on T4 execute dequantization kernels in software rather than dedicated INT8 tensor cores during prefill accumulation, this adds per-token computational overhead. INT8's value is purely VRAM reduction (saving 1.46 GB to avoid OOM), not latency. If low latency is the priority, Temporal Pruning with FP16 is the optimal configuration, delivering a 1.82x speedup (33.14s vs 60.43s)."*

---

### Q7: What exactly is your novel contribution in this milestone?
**Answer:**
> *"Our contribution is a verified, hardware-validated VLM optimization pipeline tailored for video surveillance:*
> 1. *We hooked into the internal Vision Transformer merger of Qwen2.5-VL to perform pre-merge patch cosine similarity comparison across time.*
> 2. *We designed a window re-indexing algorithm that preserves 3D RoPE spatial-temporal alignment after pruning static background tokens, achieving a 39.9% visual token reduction.*
> 3. *We evaluated the combined impact of temporal token pruning and HQQ KV-cache quantization across a 100-frame long-context sequence on an NVIDIA T4 GPU, providing concrete empirical tradeoffs between memory savings (up to 13.4%) and prefill speedup (up to 1.82x)."*

---

### Q8: What remains to be implemented for the complete Argus system?
**Answer:**
> *"Today's milestone validates the VLM computation and memory engine. The remaining roadmap consists of:*
> 1. *Upstream Perception Pipeline: Integrating YOLOv8 and ByteTrack to propose motion-salient time windows, preventing the VLM from running during long periods of zero activity.*
> 2. *Spatio-Temporal Hybrid Graph Memory: Storing the VLM's generated semantic descriptions in a knowledge graph (entities, actions, timestamps) and dense vector store.*
> 3. *Natural Language Query Interface: Enabling operators to search surveillance history using conversational queries (e.g., 'Did a white van enter the rear gate after 3 PM?')."*

---

### Q9: How does this optimization subsystem fit into the broader Argus architecture?
**Answer:**
> *"This subsystem serves as the core reasoning engine of Argus. A surveillance system cannot store raw video indefinitely or query it efficiently. Argus transforms continuous raw video into structured semantic memory.*
>
> *Without this optimization engine, the VLM would crash every 2 minutes due to KV-cache exhaustion. By compressing visual tokens by 39.9% and managing KV cache memory, our subsystem enables the VLM to run continuously over long surveillance intervals, producing the semantic scene descriptions that populate the Argus knowledge graph."*

---

### Q10: What happens with longer videos (e.g., 10 minutes or 1 hour)?
**Answer:**
> *"Even with token pruning and INT8 quantization, an infinitely accumulating KV cache will eventually exhaust GPU memory on hour-long streams. That is why Argus employs a hierarchical architecture:*
> - *The VLM operates on a rolling 'hot window' of 1 to 2 minutes using our temporal pruning and KV cache management.*
> - *Once the VLM extracts scene semantics (objects, actions, anomalies), the raw token cache is flushed, and the structured semantic text is transferred into our long-term Hybrid Graph Memory.*
> - *Thus, memory scaling over hours or days is shifted from expensive GPU VRAM to cheap CPU-based graph and vector databases."*
