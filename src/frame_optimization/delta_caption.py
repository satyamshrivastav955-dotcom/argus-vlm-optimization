"""
Delta captioning and scene state management.

OVERVIEW
--------
Instead of re-describing the full scene on every frame (which wastes output
tokens on static background), maintain a scene state and prompt the VLM
to describe only what has CHANGED since the last caption.

Pipeline
--------
Frame 0:
  full_prompt → VLM → full scene description → initialize SceneState

Frame N (changed, N > 0):
  delta_prompt(previous_scene) → VLM → delta description → update SceneState

Every K processed frames (full_recaption_every):
  force full_prompt → VLM → refresh SceneState

Scene cut (high diff score > scene_cut_threshold):
  force full_prompt → VLM → refresh SceneState

LIMITATIONS (documented honestly)
----------------------------------
1. Delta captions may drift from the true scene over many frames.
   Periodic full re-captioning mitigates this but does not eliminate it.
2. The delta prompt is longer than a naive prompt (includes previous scene text).
   This means INPUT tokens may not decrease even if output tokens do.
3. This module stores scene state as raw text. The full Argus architecture
   uses a knowledge graph (Cognee/KuzuDB) for structured state.
   That integration is OUT OF SCOPE for this repository.
4. Caption quality depends on the VLM's instruction-following ability.
   Qwen2.5-VL-3B-Instruct may not perfectly separate "new" from "background".

Usage
-----
    captioner = DeltaCaptioner(full_recaption_every=10, scene_cut_threshold=0.40)
    captioner.reset()
    for frame_idx, (frame, diff_score) in enumerate(stream):
        record = captioner.caption(
            frame=frame,
            frame_idx=frame_idx,
            diff_score=diff_score,
            vlm_wrapper=wrapper,
        )
        print(record.caption_type, record.output_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Scene State
# ============================================================

@dataclass
class SceneState:
    """
    Lightweight scene state for delta captioning.

    NOTE: This is a simplified text-based state. The full Argus architecture
    uses a knowledge graph (KuzuDB/Cognee) for structured representation.
    This module is intentionally scoped to experimental VLM optimization only.
    """

    objects: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)

    last_full_caption: str = ""
    last_delta_caption: str = ""
    last_full_caption_frame: int = -1
    last_update_frame: int = -1
    is_initialized: bool = False

    def update_from_full_caption(self, caption: str, frame_idx: int) -> None:
        """Update state after a full re-caption."""
        self.last_full_caption = caption
        self.last_delta_caption = ""
        self.last_full_caption_frame = frame_idx
        self.last_update_frame = frame_idx
        self.is_initialized = True

    def update_from_delta_caption(self, caption: str, frame_idx: int) -> None:
        """Update state after a delta caption."""
        self.last_delta_caption = caption
        self.last_update_frame = frame_idx

    @property
    def current_scene_description(self) -> str:
        """Return the best available scene description for delta prompting."""
        if self.last_full_caption:
            # Append last delta as context if available
            if self.last_delta_caption:
                return (
                    f"{self.last_full_caption}\n"
                    f"[Recent update: {self.last_delta_caption}]"
                )
            return self.last_full_caption
        return "Scene not yet described."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": self.objects,
            "locations": self.locations,
            "activities": self.activities,
            "hazards": self.hazards,
            "last_full_caption": self.last_full_caption,
            "last_delta_caption": self.last_delta_caption,
            "last_full_caption_frame": self.last_full_caption_frame,
            "last_update_frame": self.last_update_frame,
            "is_initialized": self.is_initialized,
        }

    def reset(self) -> None:
        """Reset scene state (call at start of new video)."""
        self.objects = []
        self.locations = []
        self.activities = []
        self.hazards = []
        self.last_full_caption = ""
        self.last_delta_caption = ""
        self.last_full_caption_frame = -1
        self.last_update_frame = -1
        self.is_initialized = False


# ============================================================
# Caption record
# ============================================================

@dataclass
class CaptionRecord:
    """Record of one captioning event."""

    frame_idx: int
    caption_type: str          # "full" | "delta" | "skipped"
    output_text: str
    input_token_count: int
    output_token_count: int
    latency_seconds: float
    tokens_per_second: float
    peak_vram_gb: float
    diff_score: float          # Frame diff score from gate
    triggered_by: str          # "first_frame" | "recaption_interval" |
                               # "scene_cut" | "normal_delta" | "skipped"
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "caption_type": self.caption_type,
            "output_text": self.output_text,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "latency_seconds": self.latency_seconds,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_gb": self.peak_vram_gb,
            "diff_score": self.diff_score,
            "triggered_by": self.triggered_by,
            "status": self.status,
            "error": self.error,
        }


# ============================================================
# Prompt templates
# ============================================================

FULL_PROMPT_DEFAULT = (
    "You are a surveillance AI assistant. "
    "Describe the complete scene in detail. "
    "Include: all people, vehicles, objects, their locations and activities, "
    "any hazards or unusual events. Be specific and thorough."
)

DELTA_PROMPT_TEMPLATE = (
    "Previous scene description:\n"
    "{previous_scene}\n\n"
    "The image shows a region of the same scene that has changed. "
    "Describe ONLY what is new, changed, moved, removed, or different "
    "compared to the previous description. "
    "Do not repeat descriptions of the static background. "
    "Focus on: new people, vehicles, actions, hazards, or events. "
    "If nothing has meaningfully changed, say so briefly."
)


def build_full_prompt(custom_prompt: Optional[str] = None) -> str:
    """Return the full-scene captioning prompt."""
    return custom_prompt or FULL_PROMPT_DEFAULT


def build_delta_prompt(
    previous_scene: str,
    custom_template: Optional[str] = None,
) -> str:
    """
    Build a delta captioning prompt.

    Parameters
    ----------
    previous_scene:
        The last known scene description (from SceneState).
    custom_template:
        Optional custom template string with {previous_scene} placeholder.

    Returns
    -------
    str: Formatted delta prompt.
    """
    template = custom_template or DELTA_PROMPT_TEMPLATE
    return template.format(previous_scene=previous_scene)


# ============================================================
# DeltaCaptioner
# ============================================================

class DeltaCaptioner:
    """
    Manages the full vs delta captioning decision for a video stream.

    Parameters
    ----------
    full_recaption_every:
        Force a full re-caption every N PROCESSED frames (not total frames).
        Set to 0 to disable periodic re-captioning.
    scene_cut_threshold:
        If the frame diff score exceeds this value, force a full re-caption.
        This handles abrupt scene changes.
    full_prompt:
        Custom full-scene prompt. Defaults to FULL_PROMPT_DEFAULT.
    delta_prompt_template:
        Custom delta prompt template with {previous_scene} placeholder.
    max_new_tokens_full:
        Max output tokens for full captions.
    max_new_tokens_delta:
        Max output tokens for delta captions (can be smaller).
    """

    def __init__(
        self,
        full_recaption_every: int = 10,
        scene_cut_threshold: float = 0.40,
        full_prompt: Optional[str] = None,
        delta_prompt_template: Optional[str] = None,
        max_new_tokens_full: int = 256,
        max_new_tokens_delta: int = 128,
    ) -> None:
        self.full_recaption_every = full_recaption_every
        self.scene_cut_threshold = scene_cut_threshold
        self.full_prompt = full_prompt or FULL_PROMPT_DEFAULT
        self.delta_prompt_template = delta_prompt_template or DELTA_PROMPT_TEMPLATE
        self.max_new_tokens_full = max_new_tokens_full
        self.max_new_tokens_delta = max_new_tokens_delta

        self.scene_state = SceneState()
        self._processed_frame_count = 0

    def reset(self) -> None:
        """Reset state (call at start of new video)."""
        self.scene_state.reset()
        self._processed_frame_count = 0

    def should_full_recaption(self, diff_score: float) -> Tuple[bool, str]:
        """
        Determine whether this frame should use a full caption.

        Returns
        -------
        (should_full, reason)
        """
        if not self.scene_state.is_initialized:
            return True, "first_frame"

        if diff_score >= self.scene_cut_threshold:
            return True, "scene_cut"

        if (
            self.full_recaption_every > 0
            and self._processed_frame_count % self.full_recaption_every == 0
        ):
            return True, "recaption_interval"

        return False, "normal_delta"

    def caption(
        self,
        frame: Image.Image,
        frame_idx: int,
        diff_score: float,
        vlm_wrapper,
        kv_cache_config: Optional[Dict[str, Any]] = None,
    ) -> CaptionRecord:
        """
        Caption a single frame using full or delta strategy.

        Parameters
        ----------
        frame:
            Current frame (PIL Image, RGB). May be a cropped region.
        frame_idx:
            Frame index in the video.
        diff_score:
            Change score from FrameGate (or 0.0 if gate is disabled).
        vlm_wrapper:
            Loaded QwenVLMWrapper instance.
        kv_cache_config:
            Optional KV cache config dict.

        Returns
        -------
        CaptionRecord
        """
        use_full, triggered_by = self.should_full_recaption(diff_score)

        if use_full:
            prompt = build_full_prompt(self.full_prompt)
            max_tokens = self.max_new_tokens_full
        else:
            prompt = build_delta_prompt(
                previous_scene=self.scene_state.current_scene_description,
                custom_template=self.delta_prompt_template,
            )
            max_tokens = self.max_new_tokens_delta

        result = vlm_wrapper.generate(
            image=frame,
            prompt=prompt,
            max_new_tokens=max_tokens,
            kv_cache_config=kv_cache_config,
        )

        if result.status == "ok":
            if use_full:
                self.scene_state.update_from_full_caption(
                    result.generated_text, frame_idx
                )
            else:
                self.scene_state.update_from_delta_caption(
                    result.generated_text, frame_idx
                )
            self._processed_frame_count += 1

        return CaptionRecord(
            frame_idx=frame_idx,
            caption_type="full" if use_full else "delta",
            output_text=result.generated_text,
            input_token_count=result.input_token_count,
            output_token_count=result.output_token_count,
            latency_seconds=result.latency_seconds,
            tokens_per_second=result.tokens_per_second,
            peak_vram_gb=result.peak_vram_gb,
            diff_score=diff_score,
            triggered_by=triggered_by,
            status=result.status,
            error=result.error,
        )


# Fix missing Tuple import
from typing import Tuple  # noqa: E402
