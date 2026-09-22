# Methodology — Argus VLM Optimization

## Overview

This document describes the technical methodology for each optimization technique
investigated in this repository. It explains what each technique does, why it
matters for long-duration surveillance, what it does NOT do, and its limitations.

---

## A. KV Cache Quantization

### What is a KV Cache?

During transformer inference, every input token generates a **Key (K)** and
**Value (V)** vector. These are cached in memory so that the model does not
recompute attention for previously-seen tokens at each generation step.

The KV cache size grows linearly with:
- Input sequence length (context)
- Number of transformer layers
- Embedding dimension
- Precision (bytes per element)

For a model like Qwen2.5-VL-3B-Instruct with FP16 (2 bytes/element):
```
KV cache memory ≈ 2 × layers × heads × head_dim × context_length × 2 bytes
```

### Why this matters for long-duration surveillance

A 1-hour surveillance video at 25 FPS = 90,000 frames.
Even after aggressive frame sampling (e.g., 1 FPS = 3,600 frames), accumulating
KV-state across frames can quickly exhaust GPU memory.

More critically, if each VLM call processes a growing context (e.g., full video
transcript as prefix), the KV cache becomes the dominant memory consumer.

### Quantization approach

Instead of storing KV tensors in FP16, we quantize them to lower precision:

| Backend | Precision | Memory reduction (approx.) | Risk |
|---------|-----------|--------------------------|------|
| HF DynamicCache | FP16 | 1× (baseline) | None |
| HQQ INT4 | INT4 | ~4× | Small quality loss |
| HQQ INT2 | INT2 | ~8× | Higher quality risk |
| Quanto INT4 | INT4 | ~4× | Depends on quantizer |

**HQQ (Half-Quadratic Quantization):**
HQQ minimizes a half-quadratic objective to find quantization parameters,
making it robust to outliers without requiring calibration data.

**Quanto:**
Hugging Face Quanto is a general quantization toolkit integrated with `transformers`.
It applies symmetric or asymmetric INT quantization to KV tensors.

### What quantization does NOT change

KV cache quantization in this repository applies **only to the KV cache tensors**.
The model weights remain in FP16.
Visual token projection, attention computation, and FFN layers are unaffected.

---

## B. Frame-Level Gating

### Problem

The naive approach invokes the VLM on every sampled frame.
For static or slowly-changing surveillance footage, most frames are nearly
identical to the previous frame. Invoking the VLM on these frames is wasteful.

### Approach

Compute a cheap pixel-level **change score** between consecutive frames.
If the score is below a threshold, skip the current frame and reuse the
previous caption.

```
change_score = mean(|current_gray - previous_gray|) / 255.0
```

This is the **Mean Absolute Difference (MAD)** — extremely fast (~0.1ms per frame).

### Configurable parameters

- `threshold`: Frame change threshold. Frames with score < threshold are skipped.
- `max_skip_frames`: Force VLM invocation after N consecutive skips (prevents
  indefinite skipping during slow drift).

### Limitations

- **MAD is perceptually naive.** A uniform brightness shift (lighting change) may
  trigger a large MAD without a meaningful scene change, or vice versa.
- **Threshold is scene-dependent.** A threshold of 0.05 may be appropriate for a
  static indoor camera but too sensitive for an outdoor camera with wind/foliage.
- **SSIM** is a more perceptually motivated alternative but ~10× slower.
- **Threshold must be validated against event recall.** Incorrectly skipping an
  event frame (person entering) is a false negative.

---

## C. Spatial Redundancy Reduction (Change-Region Cropping)

> **IMPORTANT:** The current implementation is **NOT** internal ViT-token pruning.
> It is a preprocessing step: crop the changed region before passing to the VLM.
> This is labeled "Spatial Redundancy Reduction / Change-Region Cropping" throughout
> the codebase and documentation.

### Problem

When a person crosses a corridor, only a small portion of the frame changes.
The VLM still processes the entire frame including static walls, ceiling, floor.
These static pixels consume input tokens (via visual patch embeddings) without
contributing new information.

### Approach

1. Divide the frame into an N×N grid of patches (e.g., 32×32 pixels each)
2. Compute the mean absolute difference per patch
3. Mark patches where difference > `change_threshold` as "changed"
4. Compute the bounding box of changed patches
5. Add configurable padding around the bounding box
6. Crop and resize to VLM input size

```
VLM input = crop(current_frame, changed_bounding_box + padding)
```

### What this reduces

- Redundant background pixels before VLM encoding
- Number of visual patches submitted to the vision encoder
- (Indirectly) the number of visual tokens in the attention context

### What this does NOT do

- Does NOT modify internal ViT attention masks
- Does NOT prune tokens inside the VLM
- Does NOT guarantee linear reduction in FLOPs (VLM may resize crop to same dims)
- Does NOT remove text prompt tokens

### Limitations

- If the crop is very small, resizing to VLM input resolution may introduce
  visible artifacts or blur (upsampling from a tiny region)
- If the changed region is near the frame boundary, padding may clip
- Quality depends on patch_size, threshold, and padding — these must be swept
- Qwen2.5-VL processes images by dividing them into internal visual patches;
  the exact token reduction depends on the processor's internal tiling logic

### Future extension (out of scope for v0.1)

True internal ViT-token pruning would require modifying the model's attention
mechanism to skip or mask visual tokens before they enter the LLM layers. This
is research-level work requiring:
- Access to internal patch token indices
- Modification of the Qwen2.5-VL ViT's forward pass
- Careful validation to avoid attention mask corruption

This is NOT implemented here. Do not claim it is.

---

## D. Delta Captioning

### Problem

Every VLM call on a surveillance frame with a naive prompt generates a full
scene description including static background elements:

> "The image shows a grey corridor with fluorescent lighting, white walls,
> a tiled floor, a fire extinguisher on the left, and an emergency exit sign..."

This description is repeated at every frame, consuming output tokens and
making it harder to extract meaningful changes.

### Approach

Maintain a **scene state** capturing the last known scene description.
On subsequent frames, prompt the VLM with:

```
Previous scene: [last full description]
Changed region image attached.
Describe ONLY what is new, changed, moved, removed, or different.
```

This reduces output token count and focuses the model on changes.

### Full re-captioning triggers

Full re-captioning (resetting scene state) is triggered when:
1. Every K processed frames (`full_recaption_every` config)
2. When the MAD score exceeds `scene_cut_threshold` (scene cut)
3. On the first frame of a new video segment

### Scene state structure

```python
{
    "objects": [],          # List of detected objects
    "locations": [],        # Spatial descriptions
    "activities": [],       # Ongoing activities
    "hazards": [],          # Detected hazards/alerts
    "last_full_caption": "", # Raw text of last full caption
    "last_delta_caption": "",# Raw text of last delta
    "last_full_caption_frame": 0,
    "last_update_frame": 0
}
```

### Limitations

- **Caption drift**: Delta captions accumulated over many frames may drift from
  the true scene state. Periodic full re-captioning mitigates this.
- **LLM context dependency**: The delta prompt is longer than a naive prompt
  (includes previous scene text), so per-token latency may not decrease even if
  output tokens decrease.
- **Quality depends on scene state accuracy**: If the previous caption was wrong,
  delta captions may compound errors.
- **No semantic parsing**: The scene state is stored as raw text, not structured
  data. The full Argus system uses a knowledge graph (Cognee/KuzuDB) for
  structured state — that is out of scope here.

---

## E. Combined Pipeline

The combined pipeline chains all optimizations:

```
Video
  → Frame sampling (stride N)
  → Frame-level gate (MAD threshold)
  → Spatial redundancy reduction (change-region crop)
  → Delta captioning (full or delta prompt)
  → VLM inference (Qwen2.5-VL)
  → Quantized KV cache (INT4 or INT2)
  → Scene state update
```

**Key property:** Each technique is independently configurable via YAML.
The ablation study (notebooks/07) isolates each technique's contribution.

---

## F. Experimental Limitations

1. **Single GPU, single model**: All experiments run on one GPU. Multi-GPU
   or distributed setups are not tested.

2. **Prototype model ≠ Production model**: Qwen2.5-VL-3B-Instruct is used for
   all experiments. The target Argus VLM (Florence-2) has a different
   architecture, tokenizer, and visual encoder. Quantitative results may differ.

3. **Synthetic frames**: Most frame experiments use synthesized frames
   (static background + moving object). Results on real surveillance footage
   may differ.

4. **ROUGE-L is not a surveillance metric**: ROUGE-L measures n-gram overlap
   between generated texts. It does not measure whether a hazard was detected,
   whether a person was correctly localized, or whether an event was described
   accurately. It is used only as an auxiliary text similarity check.

5. **KV cache quantization requires CUDA**: HQQ and Quanto KV backends are
   not supported on CPU. CPU-only environments will log FAILED for these configs.

6. **HQQ version sensitivity**: HQQ API may differ between versions. The
   implementation includes a compatibility shim and logs the exact version used.

7. **No ground truth event labels**: Unless you provide annotated video,
   event recall cannot be computed. The framework provides annotation format
   and placeholder computation, but fabricated numbers are never inserted.
