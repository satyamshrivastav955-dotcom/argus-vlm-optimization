"""
Unit tests for temporal token pruning module.
"""

import pytest
import torch
from src.frame_optimization.temporal_pruning import (
    build_temporal_pruning_mask,
    enforce_minimum_keep_ratio,
)


def test_build_temporal_pruning_mask_identical_tensors():
    # If consecutive frames are identical, cosine similarity is 1.0 everywhere.
    # Therefore, with threshold=0.98, keep_mask should be all False (all redundant).
    num_patches = 16
    hidden_dim = 32
    spatial_merge_unit = 4
    num_merged = num_patches // spatial_merge_unit  # 4

    prev = torch.randn(num_patches, hidden_dim)
    curr = prev.clone()  # identical
    window_idx = torch.arange(num_merged)

    keep_mask, sim, grouped_sim, rev_idx = build_temporal_pruning_mask(
        prev, curr, window_idx, spatial_merge_unit=spatial_merge_unit, threshold=0.98
    )

    assert keep_mask.shape == (num_merged,)
    assert not keep_mask.any(), "Identical frames should mark all tokens as redundant"
    assert torch.allclose(sim, torch.ones_like(sim))


def test_build_temporal_pruning_mask_distinct_tensors():
    # If consecutive frames are orthogonal/different, similarity is near 0.
    num_patches = 16
    hidden_dim = 32
    spatial_merge_unit = 4
    num_merged = num_patches // spatial_merge_unit

    prev = torch.ones(num_patches, hidden_dim)
    curr = -torch.ones(num_patches, hidden_dim)  # opposite vectors, sim = -1.0
    window_idx = torch.arange(num_merged)

    keep_mask, sim, _, _ = build_temporal_pruning_mask(
        prev, curr, window_idx, spatial_merge_unit=spatial_merge_unit, threshold=0.98
    )

    assert keep_mask.all(), "Completely different frames should keep all tokens"


def test_enforce_minimum_keep_ratio():
    num_merged = 100
    # All false mask (0% kept)
    keep_mask = torch.zeros(num_merged, dtype=torch.bool)
    sims = torch.linspace(0.5, 0.99, num_merged)  # lowest to highest

    safe_mask = enforce_minimum_keep_ratio(keep_mask, sims, min_keep_ratio=0.10)
    assert safe_mask.sum().item() >= 10, "Should keep at least 10% (10 tokens)"
    # Should pick tokens with lowest similarity (most changed)
    assert safe_mask[0].item() is True
    assert safe_mask[-1].item() is False


def test_build_temporal_pruning_mask_shape_mismatch():
    prev = torch.randn(16, 32)
    curr = torch.randn(20, 32)
    window_idx = torch.arange(4)

    with pytest.raises(ValueError):
        build_temporal_pruning_mask(prev, curr, window_idx)
