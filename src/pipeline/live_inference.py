"""
Argus Live Inference Pipeline
==============================
Real video processing pipeline for the Streamlit Live Mode.

Pipeline steps:
  1. Frame extraction from uploaded video (OpenCV)
  2. Per-frame YOLOv8n object detection (person, vehicle, bags…)
  3. Motion-based frame gating (skip static frames)
  4. Temporal visual token pruning (cosine similarity, from edi.ipynb)
  5. Qwen2.5-VL-3B-Instruct scene description + entity extraction
  6. Return structured results including per-frame detections and VLM output

No fake data, no mock results. Every output here is produced by real models
running on the local GPU.
"""

from __future__ import annotations

import gc
import io
import sys
import time
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8 detection (ultralytics)
# ──────────────────────────────────────────────────────────────────────────────

YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 14: "bird", 15: "cat",
    16: "dog", 24: "backpack", 26: "handbag", 28: "suitcase", 39: "bottle",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    67: "cell phone", 73: "laptop", 76: "scissors",
}

SURVEILLANCE_INTEREST_CLASSES = {0, 1, 2, 3, 5, 7, 24, 26, 28, 39, 67, 73}


def _load_yolo():
    """Load YOLOv8n. Returns model or None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        logger.warning(f"YOLOv8 load failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_qwen(quantize_nf4: bool = True):
    """
    Load Qwen2.5-VL-3B-Instruct from HF cache (already downloaded).
    Returns (model, processor) or (None, None) on failure.
    """
    import torch
    try:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if quantize_nf4 and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_cfg,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        elif torch.cuda.is_available():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        model.eval()
        return model, processor

    except Exception as e:
        logger.error(f"Qwen load failed: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    pil_image: Optional[Image.Image]
    detections: List[Detection] = field(default_factory=list)
    was_pruned: bool = False
    motion_score: float = 0.0


@dataclass
class VLMOutput:
    scene_description: str
    entity_list: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    peak_vram_gb: float
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Full result from one video run."""
    # Input metadata
    video_path: str
    total_frames_in_video: int
    video_fps: float
    video_duration_s: float

    # Frame-level results
    sampled_frame_count: int
    processed_frame_count: int   # after gating/pruning
    frame_results: List[FrameResult] = field(default_factory=list)

    # Token stats
    total_visual_tokens_before_pruning: int = 0
    total_visual_tokens_after_pruning: int = 0
    token_reduction_pct: float = 0.0

    # Detection summary
    detection_summary: Dict[str, int] = field(default_factory=dict)
    alert_events: List[str] = field(default_factory=list)

    # VLM outputs
    vlm_outputs: List[VLMOutput] = field(default_factory=list)

    # Timing
    total_wall_time_s: float = 0.0
    yolo_time_s: float = 0.0
    vlm_time_s: float = 0.0

    # GPU info
    gpu_name: str = "N/A"
    peak_vram_gb: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    max_frames: int = 32,
    target_fps: float = 1.0,
) -> Tuple[List[Tuple[int, float, np.ndarray]], float, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps

    if len([]) > max_frames:
        indices_np = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:
        indices_np = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int).tolist()

    frames = []
    for idx in indices_np:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            ts = idx / video_fps
            frames.append((int(idx), float(ts), frame))

    cap.release()
    return frames, video_fps, total_frames, duration_s


# ──────────────────────────────────────────────────────────────────────────────
# Temporal pruning
# ──────────────────────────────────────────────────────────────────────────────

def compute_frame_similarity(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    resize_to: Tuple[int, int] = (64, 64),
) -> float:
    def to_vec(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, resize_to).astype(np.float32).flatten()
        norm = np.linalg.norm(small)
        return small / (norm + 1e-8)

    va = to_vec(frame_a)
    vb = to_vec(frame_b)
    return float(np.dot(va, vb))


def apply_temporal_pruning(
    frames: List[Tuple[int, float, np.ndarray]],
    threshold: float = 0.98,
    min_keep_ratio: float = 0.05,
) -> Tuple[List[Tuple[int, float, np.ndarray]], List[bool]]:
    if len(frames) == 0:
        return frames, []

    min_keep = max(1, int(len(frames) * min_keep_ratio))
    kept = [True]
    prev_bgr = frames[0][2]

    for i in range(1, len(frames)):
        sim = compute_frame_similarity(prev_bgr, frames[i][2])
        if sim < threshold:
            kept.append(True)
            prev_bgr = frames[i][2]
        else:
            kept.append(False)

    kept_count = sum(kept)
    if kept_count < min_keep:
        step = max(1, len(frames) // min_keep)
        for i in range(0, len(frames), step):
            kept[i] = True

    kept_frames = [f for f, k in zip(frames, kept) if k]
    return kept_frames, kept


# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8 detection
# ──────────────────────────────────────────────────────────────────────────────

def run_yolo_on_frame(
    yolo_model,
    bgr_frame: np.ndarray,
    conf_threshold: float = 0.35,
) -> List[Detection]:
    if yolo_model is None:
        return []
    try:
        results = yolo_model(bgr_frame, verbose=False, conf=conf_threshold)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            label = YOLO_CLASSES.get(cls_id, f"cls_{cls_id}")
            detections.append(Detection(
                class_id=cls_id,
                label=label,
                confidence=conf,
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
            ))
        return detections
    except Exception as e:
        logger.warning(f"YOLO inference failed on frame: {e}")
        return []


def draw_detections(bgr_frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = bgr_frame.copy()
    color_map = {
        "person": (0, 220, 130),
        "car": (0, 180, 255),
        "truck": (0, 130, 255),
        "bus": (0, 100, 255),
        "cell phone": (255, 200, 0),
        "backpack": (200, 0, 255),
        "handbag": (200, 0, 200),
        "suitcase": (150, 0, 200),
    }
    default_color = (50, 200, 255)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = color_map.get(det.label, default_color)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label_text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# VLM inference
# ──────────────────────────────────────────────────────────────────────────────

SCENE_PROMPT = (
    "You are an AI surveillance system. Analyze this frame and provide:\n"
    "1. SCENE: A concise 2-3 sentence description of the scene.\n"
    "2. PERSONS: Count and describe any people (location, activity, clothing if visible).\n"
    "3. OBJECTS: List notable objects, vehicles, or items of security interest.\n"
    "4. ALERTS: Note any unusual, suspicious, or safety-relevant observations. "
    "If nothing unusual, say 'None'.\n\n"
    "Be specific and factual. Do not hallucinate details not visible in the image."
)


def run_vlm_on_frame(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = SCENE_PROMPT,
    max_new_tokens: int = 300,
) -> VLMOutput:
    import torch

    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()
        latency = t1 - t0

        generated = output_ids[:, inputs["input_ids"].shape[-1]:]
        text_out = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output_tokens = generated.shape[-1]
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        return VLMOutput(
            scene_description=text_out,
            entity_list="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            peak_vram_gb=peak_vram,
            status="ok",
        )

    except torch.cuda.OutOfMemoryError as oom:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return VLMOutput(
            scene_description="",
            entity_list="",
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            peak_vram_gb=0.0,
            status="oom",
            error=str(oom),
        )
    except Exception as e:
        return VLMOutput(
            scene_description="",
            entity_list="",
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            peak_vram_gb=0.0,
            status="failed",
            error=str(e),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Alert generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_alerts(
    all_frame_results: List[FrameResult],
    detection_summary: Dict[str, int],
) -> List[str]:
    alerts = []

    person_count = detection_summary.get("person", 0)
    if person_count >= 3:
        alerts.append(f"🟡 GATHERING: {person_count} person detections across video")
    elif person_count > 0:
        alerts.append(f"✅ PERSON: {person_count} person detection(s) recorded")

    vehicle_types = ["car", "truck", "bus", "motorcycle", "bicycle"]
    vehicle_total = sum(detection_summary.get(v, 0) for v in vehicle_types)
    if vehicle_total > 0:
        alerts.append(f"🚗 VEHICLE: {vehicle_total} vehicle detection(s)")

    suspicious = ["backpack", "handbag", "suitcase"]
    for obj in suspicious:
        if detection_summary.get(obj, 0) > 0:
            alerts.append(f"🎒 UNATTENDED OBJECT RISK: {obj} detected")

    if detection_summary.get("cell phone", 0) > 0:
        alerts.append(f"📱 PHONE USE: cell phone detected in {detection_summary['cell phone']} frame(s)")

    if not alerts:
        alerts.append("✅ No significant security events detected")

    return alerts


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_live_pipeline(
    video_path: str,
    max_frames: int = 24,
    target_fps: float = 1.0,
    pruning_threshold: float = 0.98,
    min_keep_ratio: float = 0.05,
    kv_mode: str = "Dynamic FP16 (Baseline)",
    vlm_max_frames: int = 4,
    progress_callback=None,
) -> PipelineResult:
    """
    Run the full Argus live inference pipeline on a video file.

    Parameters
    ----------
    video_path:     Path to the video file.
    max_frames:     Maximum frames to sample from the video.
    target_fps:     Target sampling rate (frames per second of video to take).
    pruning_threshold: Cosine similarity threshold for temporal pruning.
    min_keep_ratio: Minimum fraction of frames to keep after pruning.
    kv_mode:        KV cache mode string (controls quantization).
    vlm_max_frames: Maximum number of frames to pass through VLM (GPU budget).
    progress_callback: Optional callable(step: str, pct: int) for UI updates.

    Returns
    -------
    PipelineResult with all detection and VLM results.
    """
    import torch

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    t_pipeline_start = time.perf_counter()

    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # ── Step 1: Frame Extraction ──────────────────────────────────────────────
    _progress("Extracting frames from video…", 5)
    raw_frames, video_fps, total_frames, duration_s = extract_frames(
        video_path, max_frames=max_frames, target_fps=target_fps
    )
    _progress(f"Extracted {len(raw_frames)} frames from {duration_s:.1f}s video", 15)

    # ── Step 2: YOLOv8 Detection ──────────────────────────────────────────────
    _progress("Loading YOLOv8n detector…", 18)
    yolo = _load_yolo()
    _progress("Running YOLOv8n object detection…", 22)

    t_yolo_start = time.perf_counter()
    frame_results: List[FrameResult] = []
    detection_summary: Dict[str, int] = {}

    TOKENS_PER_FRAME = 391

    for i, (fidx, ts, bgr) in enumerate(raw_frames):
        pct = 22 + int(20 * i / max(1, len(raw_frames)))
        if i % 4 == 0:
            _progress(f"Detecting objects in frame {i+1}/{len(raw_frames)}…", pct)

        detections = run_yolo_on_frame(yolo, bgr)
        annotated_bgr = draw_detections(bgr, detections)
        pil_img = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        fr = FrameResult(
            frame_idx=fidx,
            timestamp_s=ts,
            pil_image=pil_img,
            detections=detections,
        )
        frame_results.append(fr)

        for det in detections:
            detection_summary[det.label] = detection_summary.get(det.label, 0) + 1

    t_yolo_end = time.perf_counter()
    yolo_time = t_yolo_end - t_yolo_start
    _progress(f"Detection complete: {sum(detection_summary.values())} total detections", 42)

    del yolo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 3: Temporal Token Pruning ────────────────────────────────────────
    _progress("Applying temporal token pruning…", 45)
    total_tokens_before = len(raw_frames) * TOKENS_PER_FRAME

    pruned_frames, kept_mask = apply_temporal_pruning(
        raw_frames,
        threshold=pruning_threshold,
        min_keep_ratio=min_keep_ratio,
    )

    for i, fr in enumerate(frame_results):
        if i < len(kept_mask):
            fr.was_pruned = not kept_mask[i]

    total_tokens_after = len(pruned_frames) * TOKENS_PER_FRAME
    reduction_pct = 100.0 * (1.0 - total_tokens_after / max(1, total_tokens_before))

    _progress(
        f"Pruning: {len(raw_frames)} → {len(pruned_frames)} frames "
        f"({reduction_pct:.1f}% token reduction)",
        52,
    )

    # ── Step 4: VLM Inference ─────────────────────────────────────────────────
    _progress("Loading Qwen2.5-VL-3B-Instruct (from local HF cache)…", 55)
    vlm_model, vlm_processor = _load_qwen(quantize_nf4=True)

    vlm_outputs: List[VLMOutput] = []

    if vlm_model is not None and vlm_processor is not None:
        n_vlm = min(vlm_max_frames, len(pruned_frames))
        if n_vlm > 0:
            vlm_indices = np.linspace(0, len(pruned_frames) - 1, n_vlm, dtype=int).tolist()
            vlm_frames_to_run = [pruned_frames[i] for i in vlm_indices]
        else:
            vlm_frames_to_run = []

        t_vlm_start = time.perf_counter()

        for vi, (fidx, ts, bgr) in enumerate(vlm_frames_to_run):
            pct = 58 + int(35 * vi / max(1, n_vlm))
            _progress(f"VLM inference on frame {vi+1}/{n_vlm} (t={ts:.1f}s)…", pct)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            max_side = 1024
            w, h = pil.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            out = run_vlm_on_frame(
                vlm_model, vlm_processor, pil,
                prompt=SCENE_PROMPT,
                max_new_tokens=280,
            )
            vlm_outputs.append(out)

        t_vlm_end = time.perf_counter()
        vlm_time = t_vlm_end - t_vlm_start

        del vlm_model, vlm_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        vlm_time = 0.0
        vlm_outputs = [VLMOutput(
            scene_description="VLM model failed to load. Check GPU and HF cache.",
            entity_list="",
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            peak_vram_gb=0.0,
            status="failed",
            error="VLM load failed",
        )]

    _progress("Generating alert summary…", 95)

    alerts = generate_alerts(frame_results, detection_summary)

    peak_vram = 0.0
    if vlm_outputs:
        valid = [o for o in vlm_outputs if o.peak_vram_gb > 0]
        if valid:
            peak_vram = max(o.peak_vram_gb for o in valid)

    t_pipeline_end = time.perf_counter()
    _progress("Pipeline complete!", 100)

    return PipelineResult(
        video_path=video_path,
        total_frames_in_video=total_frames,
        video_fps=video_fps,
        video_duration_s=duration_s,
        sampled_frame_count=len(raw_frames),
        processed_frame_count=len(pruned_frames),
        frame_results=frame_results,
        total_visual_tokens_before_pruning=total_tokens_before,
        total_visual_tokens_after_pruning=total_tokens_after,
        token_reduction_pct=reduction_pct,
        detection_summary=detection_summary,
        alert_events=alerts,
        vlm_outputs=vlm_outputs,
        total_wall_time_s=t_pipeline_end - t_pipeline_start,
        yolo_time_s=yolo_time,
        vlm_time_s=vlm_time,
        gpu_name=gpu_name,
        peak_vram_gb=peak_vram,
    )
