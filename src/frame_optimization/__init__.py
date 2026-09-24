"""Frame optimization subpackage."""
from src.frame_optimization.frame_gate import FrameGate, FrameDiff  # noqa: F401
from src.frame_optimization.spatial_redundancy import (  # noqa: F401
    PatchChangeDetector,
    crop_changed_region,
    ChangedRegionResult,
)
from src.frame_optimization.delta_caption import (  # noqa: F401
    DeltaCaptioner,
    SceneState,
    CaptionRecord,
)
from src.frame_optimization.temporal_pruning import (  # noqa: F401
    build_temporal_pruning_mask,
    enforce_minimum_keep_ratio,
    build_pruned_multimodal_package,
)
