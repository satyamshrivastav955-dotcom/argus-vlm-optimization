"""
Unit tests for delta captioning module (SceneState, DeltaCaptioner, prompt builders).
"""

import pytest
from src.frame_optimization.delta_caption import (
    SceneState,
    DeltaCaptioner,
    build_full_prompt,
    build_delta_prompt,
    FULL_PROMPT_DEFAULT,
)


def test_build_full_prompt():
    assert build_full_prompt() == FULL_PROMPT_DEFAULT
    custom = "Describe everything."
    assert build_full_prompt(custom) == custom


def test_build_delta_prompt():
    prev = "A parking lot with two cars."
    prompt = build_delta_prompt(prev)
    assert prev in prompt
    assert "Describe ONLY what is new" in prompt


def test_scene_state_lifecycle():
    state = SceneState()
    assert not state.is_initialized
    assert state.current_scene_description == "Scene not yet described."

    # Update full
    state.update_from_full_caption("Empty hallway with blue door.", frame_idx=0)
    assert state.is_initialized
    assert state.last_full_caption == "Empty hallway with blue door."
    assert state.last_update_frame == 0
    assert state.current_scene_description == "Empty hallway with blue door."

    # Update delta
    state.update_from_delta_caption("A person entered from left.", frame_idx=5)
    assert state.last_delta_caption == "A person entered from left."
    assert state.last_update_frame == 5
    assert "Empty hallway" in state.current_scene_description
    assert "A person entered" in state.current_scene_description

    # Reset
    state.reset()
    assert not state.is_initialized
    assert state.last_full_caption == ""


def test_delta_captioner_decisions():
    captioner = DeltaCaptioner(full_recaption_every=5, scene_cut_threshold=0.40)
    captioner.reset()

    # 1. First frame must be full
    should_full, reason = captioner.should_full_recaption(diff_score=0.05)
    assert should_full is True
    assert reason == "first_frame"

    # Initialize state
    captioner.scene_state.update_from_full_caption("Base scene", frame_idx=0)
    captioner._processed_frame_count = 1

    # 2. Normal small diff -> delta
    should_full, reason = captioner.should_full_recaption(diff_score=0.10)
    assert should_full is False
    assert reason == "normal_delta"

    # 3. Scene cut -> full
    should_full, reason = captioner.should_full_recaption(diff_score=0.55)
    assert should_full is True
    assert reason == "scene_cut"

    # 4. Periodic recaption (when processed count is multiple of 5)
    captioner._processed_frame_count = 5
    should_full, reason = captioner.should_full_recaption(diff_score=0.05)
    assert should_full is True
    assert reason == "recaption_interval"
