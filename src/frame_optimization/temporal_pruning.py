"""
Temporal Visual Token Pruning Module for Vision-Language Models (Qwen2.5-VL).

This module implements pre-merge cosine similarity evaluation between
consecutive video frames to identify and prune spatially redundant visual tokens
before sequence packaging and KV-cache ingestion.

Key characteristics:
- Computes cosine similarity at the pre-merge patch level.
- Groups similarities according to the model's spatial merge unit (e.g. 4 for 2x2 merger).
- Reverses spatial window indexing to preserve 3D RoPE positional integrity.
- Enforces a minimum keep ratio to prevent complete frame collapse.
- Adjusts input IDs, position IDs (with RoPE deltas), and attention masks.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_temporal_pruning_mask(
    previous_premerge: torch.Tensor,
    current_premerge: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_unit: int = 4,
    threshold: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a boolean mask indicating which merged visual tokens to KEEP.

    Parameters
    ----------
    previous_premerge:
        Feature tensor of the previous frame before spatial merging, shape [num_patches, hidden_dim].
    current_premerge:
        Feature tensor of the current frame before spatial merging, shape [num_patches, hidden_dim].
    window_index:
        Window index mapping tensor from vision_model.get_window_index(image_grid_thw).
    spatial_merge_unit:
        Number of patch tokens per merged visual token (typically 4 for 2x2 merger).
    threshold:
        Cosine similarity threshold above which a token is marked redundant (pruned).
        Default is 0.98.

    Returns
    -------
    keep_original_order:
        Boolean mask of shape [num_merged_tokens] in original sequence order.
    sim:
        Raw cosine similarity per patch, shape [num_patches].
    grouped_similarity:
        Similarity grouped by spatial merge unit, shape [num_merged_tokens, spatial_merge_unit].
    reverse_indices:
        Arg-sorted reverse indices to map window order back to original order.
    """
    if previous_premerge.shape != current_premerge.shape:
        raise ValueError(
            f"Feature shapes do not match: {previous_premerge.shape} vs {current_premerge.shape}."
        )

    # 1. Compute cosine similarity per unmerged patch token
    sim = F.cosine_similarity(previous_premerge.float(), current_premerge.float(), dim=-1)

    # 2. Group by spatial merge unit (e.g. 4 patches per merged token)
    grouped_similarity = sim.view(-1, spatial_merge_unit)

    # 3. If any patch within a merged unit changed (sim < threshold), keep the merged token
    keep_window_order = (grouped_similarity < threshold).any(dim=1)

    # 4. Map window order back to original sequence order
    reverse_indices = torch.argsort(window_index)
    keep_original_order = keep_window_order[reverse_indices]

    return keep_original_order, sim, grouped_similarity, reverse_indices


def enforce_minimum_keep_ratio(
    keep_mask: torch.Tensor,
    similarity_per_merged_token: torch.Tensor,
    min_keep_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Ensure at least a minimum fraction of visual tokens are preserved.

    Parameters
    ----------
    keep_mask:
        Boolean mask of shape [num_merged_tokens].
    similarity_per_merged_token:
        Mean cosine similarity per merged token, shape [num_merged_tokens].
    min_keep_ratio:
        Minimum ratio of tokens to keep (e.g. 0.05 = 5%).

    Returns
    -------
    safe_mask:
        Updated boolean mask with at least ceil(num_tokens * min_keep_ratio) tokens set to True.
    """
    total = keep_mask.numel()
    min_keep = max(1, math.ceil(total * min_keep_ratio))

    if int(keep_mask.sum()) >= min_keep:
        return keep_mask

    # If too few tokens kept, select the ones with the lowest similarity (highest change)
    _, indices = torch.topk(similarity_per_merged_token, k=min_keep, largest=False)
    safe_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
    safe_mask[indices] = True
    return safe_mask


def build_pruned_multimodal_package(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    merged_image_features: torch.Tensor,
    keep_mask: torch.Tensor,
) -> Dict[str, Any]:
    """
    Build pruned multimodal input tensors for the language model.

    Slices input_ids, position_ids, attention_mask, and injects remaining visual features.

    Parameters
    ----------
    model:
        Qwen2.5-VL model instance.
    inputs:
        Output dict from processor containing input_ids, image_grid_thw, attention_mask.
    merged_image_features:
        Merged visual feature tensor [num_merged_tokens, hidden_dim].
    keep_mask:
        Boolean tensor [num_merged_tokens] indicating tokens to retain.

    Returns
    -------
    dict with pruned input_ids, inputs_embeds, position_ids, attention_mask, rope_delta, lengths.
    """
    original_input_ids = inputs["input_ids"]
    image_token_id = model.config.image_token_id
    image_positions = torch.where(original_input_ids[0] == image_token_id)[0]

    if image_positions.numel() != merged_image_features.shape[0]:
        raise RuntimeError(
            f"Image token count ({image_positions.numel()}) != visual feature count ({merged_image_features.shape[0]})."
        )
    if keep_mask.numel() != merged_image_features.shape[0]:
        raise RuntimeError(
            f"Keep-mask length ({keep_mask.numel()}) != visual feature count ({merged_image_features.shape[0]})."
        )

    # 1. Build sequence-level keep mask
    sequence_keep_mask = torch.ones(
        original_input_ids.shape[1], dtype=torch.bool, device=original_input_ids.device
    )
    sequence_keep_mask[image_positions[~keep_mask]] = False

    # 2. Prune input_ids
    pruned_input_ids = original_input_ids[:, sequence_keep_mask]

    # 3. Compute 3D RoPE index on original sequence, then prune to keep coordinates aligned
    full_position_ids, rope_delta = model.model.get_rope_index(
        original_input_ids,
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs.get("attention_mask"),
    )
    pruned_position_ids = full_position_ids[:, :, sequence_keep_mask]

    # 4. Attention mask
    original_attention_mask = inputs.get(
        "attention_mask", torch.ones_like(original_input_ids)
    )
    pruned_attention_mask = original_attention_mask[:, sequence_keep_mask]

    # 5. Embed pruned sequence and inject visual features into remaining image placeholders
    pruned_embeds = model.get_input_embeddings()(pruned_input_ids).to(dtype=model.dtype)
    remaining_image_positions = torch.where(pruned_input_ids[0] == image_token_id)[0]
    selected_visual_features = merged_image_features[keep_mask].to(
        device=pruned_embeds.device, dtype=pruned_embeds.dtype
    )

    if remaining_image_positions.numel() != selected_visual_features.shape[0]:
        raise RuntimeError(
            f"Remaining image placeholders ({remaining_image_positions.numel()}) != "
            f"selected visual features ({selected_visual_features.shape[0]})."
        )

    pruned_embeds[0, remaining_image_positions, :] = selected_visual_features

    return {
        "input_ids": pruned_input_ids,
        "inputs_embeds": pruned_embeds,
        "position_ids": pruned_position_ids,
        "attention_mask": pruned_attention_mask,
        "rope_delta": rope_delta,
        "original_length": original_input_ids.shape[1],
        "pruned_length": pruned_input_ids.shape[1],
        "visual_tokens_before": merged_image_features.shape[0],
        "visual_tokens_after": int(keep_mask.sum()),
        "token_reduction_pct": 100.0 * (1.0 - pruned_input_ids.shape[1] / original_input_ids.shape[1]),
        "visual_reduction_pct": 100.0 * (1.0 - int(keep_mask.sum()) / merged_image_features.shape[0]),
    }
