# Results Template

Use this document to record your experimental findings after running each notebook.
Fill in actual numbers from your run — do NOT copy placeholder values.

---

## Environment

| Field | Value |
|-------|-------|
| Date | |
| Hardware (GPU) | |
| GPU Memory | |
| CUDA version | |
| Python version | |
| PyTorch version | |
| Transformers version | |
| HQQ version | |
| Quanto version | |

---

## Notebook 01 — KV Cache Baseline

### Short-context results (single image)

| Config | Backend | Bits | Peak VRAM (GB) | Latency (s) | Tokens/sec | Output tokens | ROUGE-L | Status |
|--------|---------|------|---------------|-------------|------------|---------------|---------|--------|
| baseline_fp16_dynamic | dynamic | 16 | | | | | 1.00 | |
| quantized_hqq_4bit | HQQ | 4 | | | | | | |
| quantized_hqq_2bit | HQQ | 2 | | | | | | |
| quantized_quanto_4bit | quanto | 4 | | | | | | |

**Observations:**

*Fill in after running notebook 01.*

---

## Notebook 02 — Long-Context KV Cache

### VRAM vs context length

| Context (tokens) | FP16 VRAM (GB) | INT4 VRAM (GB) | INT2 VRAM (GB) |
|-----------------|---------------|---------------|---------------|
| 1K | | | |
| 5K | | | |
| 10K | | | |
| 20K | | | |
| 50K | | | |

**Observations:**

*Fill in after running notebook 02.*

---

## Notebook 03 — Static Frame Baseline

| Metric | Value |
|--------|-------|
| Total frames sampled | |
| VLM calls | |
| Total input tokens | |
| Total output tokens | |
| Total latency (s) | |
| Avg latency/frame (s) | |
| Effective FPS | |
| Peak VRAM (GB) | |

---

## Notebook 04 — Frame Gating

| Threshold | Frames processed | Frames skipped | VLM calls | Latency (s) | Events missed |
|-----------|-----------------|----------------|-----------|-------------|---------------|
| 0.01 | | | | | |
| 0.02 | | | | | |
| 0.05 | | | | | |
| 0.10 | | | | | |
| 0.15 | | | | | |
| 0.20 | | | | | |
| 0.30 | | | | | |

**Threshold selected for further experiments:** `____`
**Reason:** *(based on measured recall, not assumption)*

---

## Notebook 05 — Spatial Redundancy

| Patch size | Change thresh | Padding | Crop ratio | Token reduction % | ROUGE-L | Latency (s) |
|-----------|--------------|---------|-----------|------------------|---------|-------------|
| 16 | 0.05 | 0 | | | | |
| 32 | 0.10 | 32 | | | | |
| 64 | 0.15 | 64 | | | | |
| ... | ... | ... | | | | |

**Configuration selected for further experiments:**
- patch_size: `____`
- change_threshold: `____`
- padding: `____`

**Reason:** *(based on measured crop ratio and quality)*

---

## Notebook 06 — Delta Captioning

| Metric | Full caption only | Delta captioning |
|--------|------------------|-----------------|
| Total output tokens | | |
| Avg tokens/frame | | |
| Total latency (s) | | |
| Avg latency/frame (s) | | |
| ROUGE-L (vs full) | N/A | |
| Missed changes (manual) | N/A | |

---

## Notebook 07 — Ablation Study

| Exp | Frame Gate | Spatial | Delta | KV Cache | VRAM (GB) | Latency (s) | VLM calls | Input tokens | ROUGE-L |
|-----|-----------|---------|-------|----------|-----------|-------------|-----------|-------------|---------|
| E0 | ✗ | ✗ | ✗ | FP16 | | | | | |
| E1 | ✓ | ✗ | ✗ | FP16 | | | | | |
| E2 | ✗ | ✓ | ✗ | FP16 | | | | | |
| E3 | ✗ | ✗ | ✓ | FP16 | | | | | |
| E4 | ✓ | ✓ | ✗ | FP16 | | | | | |
| E5 | ✓ | ✓ | ✓ | FP16 | | | | | |
| E6 | ✓ | ✓ | ✓ | INT4 | | | | | |
| E7 | ✓ | ✓ | ✓ | INT2 | | | | | |

---

## Engineering Selection Table

> NOTE: This table presents facts only — no rankings. The team selects the
> appropriate configuration based on their deployment requirements.

| Config | GPU memory reduction | VLM call reduction | Token reduction | Event preservation | Latency impact | Stability |
|--------|--------------------|--------------------|-----------------|-------------------|----------------|-----------|
| E0 (baseline) | — | — | — | — | — | High |
| E1 | | | | | | |
| E2 | | | | | | |
| E3 | | | | | | |
| E4 | | | | | | |
| E5 | | | | | | |
| E6 | | | | | | |
| E7 | | | | | | |

---

## Limitations Observed During Experiments

*Document any runtime failures, package incompatibilities, or unexpected behaviors here.*

---

## Integration Recommendation

*Based on the measured results above, describe which configuration meets the
project requirements (not which is "best" in the abstract).*

Requirements:
1. Reduce GPU memory — Configuration that achieved this: `____`
2. Reduce VLM calls — Configuration: `____`
3. Preserve event recall — Configuration: `____`
4. Acceptable latency — Configuration: `____`
5. Works in limited GPU memory — Configuration: `____`
