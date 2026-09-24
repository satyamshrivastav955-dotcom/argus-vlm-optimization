"""
ARGUS: Efficient VLM Surveillance Prototype
Real-Time AI Surveillance & VLM Optimization

Main Streamlit Application
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ARGUS — Efficient VLM Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling (Modern Dark Glassmorphic Engineering Dashboard)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Custom Header Banner */
    .argus-header {
        background: linear-gradient(135deg, rgba(16, 26, 43, 0.95) 0%, rgba(9, 14, 24, 0.95) 100%);
        border: 1px solid rgba(0, 220, 130, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .argus-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00DC82 0%, #36E4DA 50%, #00B4D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .argus-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 14px;
    }

    /* Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .badge-active {
        background: rgba(0, 220, 130, 0.12);
        color: #00DC82;
        border: 1px solid rgba(0, 220, 130, 0.3);
    }

    .badge-progress {
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-info {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: rgba(0, 220, 130, 0.4);
        transform: translateY(-2px);
    }

    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-delta-positive {
        color: #00DC82;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-delta-neutral {
        color: #94A3B8;
        font-size: 0.85rem;
    }

    /* Pipeline Step Box */
    .pipeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 20px 0;
        overflow-x: auto;
    }

    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        min-width: 140px;
    }

    .pipeline-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: rgba(0, 220, 130, 0.15);
        color: #00DC82;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 6px;
        border: 1px solid rgba(0, 220, 130, 0.3);
    }

    .pipeline-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #F1F5F9;
    }

    .pipeline-desc {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 2px;
    }

    .pipeline-arrow {
        color: #475569;
        font-size: 1.3rem;
        font-weight: bold;
    }

    /* Callout & Disclaimer */
    .recorded-disclaimer {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38BDF8;
        padding: 12px 18px;
        border-radius: 4px 8px 8px 4px;
        font-size: 0.85rem;
        color: #CBD5E1;
        margin: 16px 0;
    }

    .finding-alert {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #00DC82;
        padding: 14px 20px;
        border-radius: 4px 8px 8px 4px;
        color: #E2E8F0;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Verified Benchmark Results
# -----------------------------------------------------------------------------
@st.cache_data
def load_benchmark_data() -> Dict[str, Any]:
    json_path = Path("results/benchmark_results.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to hardcoded verified numbers from edi.ipynb if json missing
    return {
        "system_metadata": {
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "tested_hardware": "NVIDIA Tesla T4 (Google Colab)",
            "video_duration_seconds": 120.3,
            "sampled_frames": 100,
        },
        "long_context_benchmark": {
            "configurations": [
                {
                    "config_id": "unpruned_dynamic_fp16",
                    "display_name": "Unpruned Baseline (FP16 Dynamic)",
                    "final_tokens": 44200,
                    "visual_tokens": 39100,
                    "peak_vram_gb": 10.91,
                    "prefill_latency_seconds": 60.43,
                    "speedup_vs_baseline": 1.0,
                    "vram_saved_gb": 0.0,
                },
                {
                    "config_id": "pruned_dynamic_fp16",
                    "display_name": "Temporal Pruning + FP16 Dynamic",
                    "final_tokens": 28605,
                    "visual_tokens": 23505,
                    "peak_vram_gb": 9.91,
                    "prefill_latency_seconds": 33.14,
                    "speedup_vs_baseline": 1.82,
                    "vram_saved_gb": 0.997,
                },
                {
                    "config_id": "pruned_quantized_int8",
                    "display_name": "Temporal Pruning + INT8 Quantized (HQQ)",
                    "final_tokens": 28605,
                    "visual_tokens": 23505,
                    "peak_vram_gb": 9.45,
                    "prefill_latency_seconds": 78.06,
                    "speedup_vs_baseline": 0.77,
                    "vram_saved_gb": 1.459,
                }
            ],
            "trajectory": []
        }
    }

benchmarks = load_benchmark_data()
long_cfg = benchmarks["long_context_benchmark"]["configurations"]
trajectory_data = benchmarks["long_context_benchmark"].get("trajectory", [])

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ ARGUS Control Panel")
    
    execution_mode = st.radio(
        "Execution Mode",
        ["Demo Mode (Recorded Benchmark)", "Live / Experimental Mode"],
        index=0,
        help="Demo Mode is 100% stable, zero GPU required, using verified experimental results."
    )
    
    if execution_mode == "Demo Mode (Recorded Benchmark)":
        st.caption("🟢 **Presentation Safe**: Using verified experimental records from Qwen2.5-VL-3B run on 100 frames (120.3s video).")
    else:
        st.caption("⚠️ **Experimental**: Executes live PyTorch/Transformers on local machine if CUDA GPU and Qwen2.5-VL are loaded.")
    
    st.markdown("---")
    st.markdown("#### 📹 Video Source")
    video_source = st.radio(
        "Select Video Input",
        ["Sample Surveillance Footage (5s demo)", "Upload Video File (.mp4)"],
        index=0
    )
    
    uploaded_file = None
    if video_source == "Upload Video File (.mp4)":
        uploaded_file = st.file_uploader("Upload Surveillance Clip", type=["mp4", "avi", "mov"])
        
    st.markdown("---")
    st.markdown("#### ⚡ Optimization Pipeline")
    enable_temporal_pruning = st.checkbox("Temporal Token Pruning", value=True, help="Prunes static visual tokens across consecutive frames using cosine similarity.")
    
    if enable_temporal_pruning:
        pruning_threshold = st.slider("Cosine Similarity Threshold", min_value=0.90, max_value=0.99, value=0.98, step=0.01,
                                      help="Tokens with similarity above this threshold are deemed static background and pruned.")
        min_keep_ratio = st.slider("Minimum Keep Ratio", min_value=0.02, max_value=0.20, value=0.05, step=0.01,
                                   help="Guarantees at least this ratio of tokens is kept to prevent frame collapse.")
    else:
        pruning_threshold = 1.0
        min_keep_ratio = 1.0
        
    kv_quant = st.radio(
        "KV Cache Precision",
        ["Dynamic FP16 (Baseline)", "HQQ INT8 (Quantized)", "HQQ INT4 (Quantized)"],
        index=0 if not enable_temporal_pruning else 1,
        help="Quantization precision for Past Key-Values during long-context accumulation."
    )

    st.markdown("---")
    st.markdown("#### 🧭 Dashboard Navigation")
    selected_page = st.selectbox(
        "View Section",
        [
            "Interactive Surveillance Demo",
            "Long-Context Benchmark & Visualizations",
            "Single-Image vs Long-Context Analysis",
            "Argus System Architecture & Roadmap",
            "Surveillance System Overview",
            "Verified Technical Logs & Output"
        ]
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748B; line-height: 1.4;'>
    <b>Argus Project Subsystem</b><br>
    Milestone: VLM Token & KV Cache Optimization<br>
    Hardware: NVIDIA Tesla T4 (Colab)<br>
    Status: Presentation Ready
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Hero Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="argus-header">
    <div class="argus-title">ARGUS // VLM SURVEILLANCE OPTIMIZATION</div>
    <div class="argus-subtitle">Real-Time Surveillance Semantic Understanding & Efficient KV Cache Architecture</div>
    <div class="badge-container">
        <span class="status-badge badge-active">● VLM Integration: Qwen2.5-VL-3B</span>
        <span class="status-badge badge-active">● Visual Token Extraction: Validated</span>
        <span class="status-badge badge-active">● Temporal Token Pruning: Validated (39.9% Reduction)</span>
        <span class="status-badge badge-active">● KV Cache Quantization: FP16 / INT8 / INT4 Validated</span>
        <span class="status-badge badge-active">● Long-Context Evaluation: 100 Frames (120.3s)</span>
        <span class="status-badge badge-progress">◐ Hybrid Memory & Perception: In Progress</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Processing Pipeline Diagram
# -----------------------------------------------------------------------------
st.markdown("""
<div class="pipeline-wrapper">
    <div class="pipeline-step">
        <div class="pipeline-icon">🎥</div>
        <div class="pipeline-name">Surveillance Stream</div>
        <div class="pipeline-desc">120.3s Video / 30 FPS</div>
    </div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">⏱️</div>
        <div class="pipeline-name">Temporal Sampling</div>
        <div class="pipeline-desc">100 Sampled Frames</div>
    </div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">👁️</div>
        <div class="pipeline-name">ViT Token Extraction</div>
        <div class="pipeline-desc">39,100 Visual Tokens</div>
    </div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">✂️</div>
        <div class="pipeline-name">Temporal Token Pruning</div>
        <div class="pipeline-desc">Cosine Sim (≥0.98 pruned)</div>
    </div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">💾</div>
        <div class="pipeline-name">KV Cache Optimization</div>
        <div class="pipeline-desc">FP16 / HQQ INT8 Quantized</div>
    </div>
    <div class="pipeline-arrow">➔</div>
    <div class="pipeline-step">
        <div class="pipeline-icon">🧠</div>
        <div class="pipeline-name">VLM Semantic Output</div>
        <div class="pipeline-desc">Surveillance Scene Description</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: Interactive Surveillance Demo
# =============================================================================
if selected_page == "Interactive Surveillance Demo":
    st.subheader("📹 Video Analysis & Optimization Pipeline")

    col_video, col_controls = st.columns([1.1, 1], gap="medium")

    with col_video:
        st.markdown("##### Input Video Feed")
        sample_video_path = Path("sample_data/sample_surveillance.mp4")

        if video_source == "Upload Video File (.mp4)" and uploaded_file is not None:
            st.video(uploaded_file)
            st.caption(f"Custom file loaded: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")
        else:
            if not sample_video_path.exists():
                try:
                    import subprocess
                    import sys as _sys
                    subprocess.run([_sys.executable, "scripts/generate_sample_video.py"], check=True)
                except Exception as _e:
                    st.warning(f"Could not auto-generate sample video: {_e}")
            if sample_video_path.exists():
                st.video(str(sample_video_path))
                st.caption("Surveillance Test Clip: `sample_data/sample_surveillance.mp4`")
            else:
                st.info("No video loaded. Upload a video file or generate a sample video.")

    with col_controls:
        st.markdown("##### Active Configuration")

        cfg_selected = "pruned_quantized_int8"
        if not enable_temporal_pruning:
            cfg_selected = "unpruned_dynamic_fp16"
        elif "FP16" in kv_quant:
            cfg_selected = "pruned_dynamic_fp16"
        else:
            cfg_selected = "pruned_quantized_int8"

        st.markdown(f"""
        - **Pipeline Mode**: `{("Temporal Pruning + " + kv_quant) if enable_temporal_pruning else "Unpruned Baseline"}`
        - **Target Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
        - **Pruning Threshold**: `cos_sim >= {pruning_threshold if enable_temporal_pruning else "N/A"}`
        - **Min Token Retention**: `{min_keep_ratio * 100:.0f}%`
        """)

        if execution_mode == "Live / Experimental Mode":
            import torch as _torch_check
            if _torch_check.cuda.is_available():
                _gn = _torch_check.cuda.get_device_name(0)
                _vm = _torch_check.cuda.get_device_properties(0).total_memory / (1024**3)
                st.success(f"✅ GPU: **{_gn}** ({_vm:.1f} GB VRAM) — Live inference ready")
            else:
                st.warning("⚠️ No CUDA GPU detected — pipeline will run on CPU (very slow)")

        run_btn = st.button("▶ Run Surveillance VLM Analysis", type="primary", use_container_width=True)

        if run_btn:
            if execution_mode == "Demo Mode (Recorded Benchmark)":
                _pb = st.progress(0, text="Initializing Qwen2.5-VL vision pipeline...")
                time.sleep(0.3)
                _pb.progress(25, text="Sampling 100 frames across 120.3s temporal window...")
                time.sleep(0.4)
                if enable_temporal_pruning:
                    _pb.progress(55, text="Calculating patch cosine similarity & pruning static tokens (39.9% pruned)...")
                    time.sleep(0.4)
                else:
                    _pb.progress(55, text="Extracting full visual tokens (no pruning applied)...")
                    time.sleep(0.4)
                _pb.progress(80, text=f"Ingesting into {kv_quant} KV Cache & RoPE 3D position alignment...")
                time.sleep(0.3)
                _pb.progress(100, text="Inference complete! Displaying recorded benchmark metrics.")
                st.session_state["analysis_done"] = True
                st.session_state["live_result"] = None

            else:
                # ── REAL LIVE INFERENCE ─────────────────────────────────
                import sys as _sys_live
                import tempfile as _tf
                import os as _os

                _src_path = str(Path("src").absolute())
                if _src_path not in _sys_live.path:
                    _sys_live.path.insert(0, _src_path)

                # Resolve video path
                _video_tmp = None
                if uploaded_file is not None:
                    _suf = Path(uploaded_file.name).suffix or ".mp4"
                    _tmp = _tf.NamedTemporaryFile(delete=False, suffix=_suf)
                    _tmp.write(uploaded_file.read())
                    _tmp.close()
                    _video_tmp = _tmp.name
                elif sample_video_path.exists():
                    _video_tmp = str(sample_video_path)

                if _video_tmp is None:
                    st.error("No video source available. Upload a video or generate the sample video first.")
                else:
                    from src.pipeline.live_inference import run_live_pipeline as _run_pipeline

                    _pb = st.progress(0, text="Starting pipeline…")
                    _status_txt = st.empty()

                    def _cb(msg, pct):
                        _pb.progress(pct, text=msg)
                        _status_txt.caption(f"⚙️ {msg}")

                    try:
                        _result = _run_pipeline(
                            video_path=_video_tmp,
                            max_frames=20,
                            pruning_threshold=pruning_threshold,
                            min_keep_ratio=min_keep_ratio,
                            kv_mode=kv_quant,
                            vlm_max_frames=3,
                            progress_callback=_cb,
                        )
                        st.session_state["live_result"] = _result
                        st.session_state["analysis_done"] = True
                        _status_txt.empty()

                    except Exception as _err:
                        import traceback as _tb
                        st.error(f"Pipeline error: {_err}")
                        st.code(_tb.format_exc(), language="text")
                    finally:
                        if uploaded_file is not None and _video_tmp and _os.path.exists(_video_tmp):
                            try:
                                _os.unlink(_video_tmp)
                            except Exception:
                                pass

    # ── Results Display ──────────────────────────────────────────────────────
    if st.session_state.get("analysis_done", True):
        st.markdown("---")

        _live = st.session_state.get("live_result", None)

        if _live is not None and execution_mode == "Live / Experimental Mode":
            # ─── LIVE RESULTS ──────────────────────────────────────────────
            st.markdown("### 🔴 Live Inference Results")
            st.markdown(f"""
            <div class="recorded-disclaimer">
            <b>Real inference complete</b> on <b>{_live.gpu_name}</b> &mdash;
            {_live.sampled_frame_count} frames sampled from {_live.video_duration_s:.1f}s video,
            {_live.processed_frame_count} kept after temporal pruning.
            </div>
            """, unsafe_allow_html=True)

            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            with _m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Frames Sampled</div>
                    <div class="metric-value">{_live.sampled_frame_count}</div>
                    <div class="metric-delta-neutral">{_live.video_duration_s:.1f}s video</div>
                </div>""", unsafe_allow_html=True)
            with _m2:
                _tok_b = _live.total_visual_tokens_before_pruning
                _tok_a = _live.total_visual_tokens_after_pruning
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Visual Tokens</div>
                    <div class="metric-value">{_tok_a:,}</div>
                    <div class="metric-delta-positive">&#9660; {_live.token_reduction_pct:.1f}% reduction</div>
                </div>""", unsafe_allow_html=True)
            with _m3:
                _n_det = sum(_live.detection_summary.values())
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Objects Detected</div>
                    <div class="metric-value">{_n_det}</div>
                    <div class="metric-delta-neutral">{len(_live.detection_summary)} class(es)</div>
                </div>""", unsafe_allow_html=True)
            with _m4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Peak VRAM</div>
                    <div class="metric-value">{_live.peak_vram_gb:.2f} <span style="font-size:1rem;color:#94A3B8;">GB</span></div>
                    <div class="metric-delta-positive">Live measurement</div>
                </div>""", unsafe_allow_html=True)
            with _m5:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">VLM Inference</div>
                    <div class="metric-value">{_live.vlm_time_s:.1f} <span style="font-size:1rem;color:#94A3B8;">s</span></div>
                    <div class="metric-delta-neutral">{len(_live.vlm_outputs)} frame(s)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Alerts
            st.markdown("### 🚨 Security Alert Events")
            for _alert in _live.alert_events:
                if any(k in _alert for k in ("GATHERING", "PHONE", "OBJECT")):
                    st.warning(_alert)
                elif any(k in _alert for k in ("PERSON", "VEHICLE")):
                    st.info(_alert)
                else:
                    st.success(_alert)

            # Detection summary
            if _live.detection_summary:
                st.markdown("### 📊 Detection Summary")
                _det_rows = [
                    {"Object Class": k, "Count": v,
                     "Security Relevance":
                         "🔴 HIGH" if k == "person" else
                         "🟡 MEDIUM" if k in ("car", "truck", "bus", "cell phone") else
                         "🟠 WATCH" if k in ("backpack", "handbag", "suitcase") else "🟢 LOW"}
                    for k, v in sorted(_live.detection_summary.items(), key=lambda x: -x[1])
                ]
                st.dataframe(pd.DataFrame(_det_rows), use_container_width=True, hide_index=True)

            # Frame gallery
            st.markdown("### 🖼️ Annotated Frame Gallery (YOLOv8n Detections)")
            _gallery = [fr for fr in _live.frame_results if fr.pil_image is not None and not fr.was_pruned][:8]
            if _gallery:
                _gcols = st.columns(min(4, len(_gallery)))
                for _ci, _fr in enumerate(_gallery):
                    with _gcols[_ci % 4]:
                        _det_labels = list({d.label for d in _fr.detections})
                        _cap = f"t={_fr.timestamp_s:.1f}s | " + (", ".join(_det_labels) if _det_labels else "no detections")
                        st.image(_fr.pil_image, caption=_cap, use_container_width=True)

            # VLM outputs
            st.markdown("### 🧠 VLM Semantic Scene Descriptions (Qwen2.5-VL-3B)")
            st.caption("Qualitative semantic output is provided to inspect whether useful scene information is retained.")
            for _vi, _vo in enumerate(_live.vlm_outputs):
                with st.expander(
                    f"📝 Frame {_vi+1} — {_vo.input_tokens} input tokens | "
                    f"{_vo.latency_s:.1f}s latency | {_vo.peak_vram_gb:.2f} GB VRAM",
                    expanded=(_vi == 0),
                ):
                    if _vo.status == "ok" and _vo.scene_description:
                        st.markdown(_vo.scene_description)
                    elif _vo.status == "oom":
                        st.error(f"⚠️ Out of GPU Memory: {_vo.error}")
                    else:
                        st.error(f"VLM failed ({_vo.status}): {_vo.error}")

            # Timing
            st.markdown("### ⏱️ Timing Breakdown")
            _other_time = max(0.0, _live.total_wall_time_s - _live.yolo_time_s - _live.vlm_time_s)
            st.dataframe(pd.DataFrame({
                "Stage": ["Frame Extraction & YOLO Detection", "Temporal Token Pruning + Overhead", "Qwen2.5-VL-3B Inference", "Total Pipeline"],
                "Time (s)": [round(_live.yolo_time_s, 2), round(_other_time, 2), round(_live.vlm_time_s, 2), round(_live.total_wall_time_s, 2)],
            }), use_container_width=True, hide_index=True)

        else:
            # ─── DEMO MODE (verified benchmark) ────────────────────────────
            st.markdown("### 📊 Benchmark Metrics for Active Configuration")
            st.markdown("""
            <div class="recorded-disclaimer">
            &#9432; <b>Recorded benchmark from current experiment</b> (NVIDIA Tesla T4, Qwen2.5-VL-3B-Instruct, 100 frames spanning 120.3s video).
            </div>
            """, unsafe_allow_html=True)

            active_res = next((c for c in long_cfg if c["config_id"] == cfg_selected), long_cfg[1])

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown("""<div class="metric-card">
                    <div class="metric-title">Frames Processed</div>
                    <div class="metric-value">100</div>
                    <div class="metric-delta-neutral">120.3s Video Coverage</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                _cls2 = "positive" if active_res["visual_tokens"] < 39100 else "neutral"
                _lbl2 = "&#9660; -39.9% (from 39,100)" if active_res["visual_tokens"] < 39100 else "Baseline (100%)"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Visual Tokens</div>
                    <div class="metric-value">{active_res["visual_tokens"]:,}</div>
                    <div class="metric-delta-{_cls2}">{_lbl2}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                _cls3 = "positive" if active_res["final_tokens"] < 44200 else "neutral"
                _lbl3 = "&#9660; -35.3% Total Tokens" if active_res["final_tokens"] < 44200 else "Baseline (44,200)"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Total Context Tokens</div>
                    <div class="metric-value">{active_res["final_tokens"]:,}</div>
                    <div class="metric-delta-{_cls3}">{_lbl3}</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                _cls4 = "positive" if active_res["vram_saved_gb"] > 0 else "neutral"
                _lbl4 = f"&#9660; -{active_res['vram_saved_gb']:.3f} GB saved" if active_res["vram_saved_gb"] > 0 else "Baseline Peak"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Peak GPU VRAM</div>
                    <div class="metric-value">{active_res["peak_vram_gb"]:.2f} <span style="font-size:1rem;color:#94A3B8;">GB</span></div>
                    <div class="metric-delta-{_cls4}">{_lbl4}</div>
                </div>""", unsafe_allow_html=True)
            with m5:
                _spd = active_res["speedup_vs_baseline"]
                _cls5 = "positive" if _spd > 1.0 else "neutral"
                _lbl5 = f"&#9650; {_spd:.2f}x Speedup" if _spd > 1.0 else (f"&#9660; {_spd:.2f}x (Quant overhead)" if _spd < 1.0 else "Baseline (60.43s)")
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">Prefill Latency</div>
                    <div class="metric-value">{active_res["prefill_latency_seconds"]:.2f} <span style="font-size:1rem;color:#94A3B8;">s</span></div>
                    <div class="metric-delta-{_cls5}">{_lbl5}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### &#128269; Side-by-Side Experimental Comparison")
            _rows = []
            for c in long_cfg:
                _rows.append({
                    "Configuration": c["display_name"],
                    "Visual Tokens": f"{c['visual_tokens']:,}",
                    "Final Context Tokens": f"{c['final_tokens']:,}",
                    "Token Reduction (%)": f"{100 * (1 - c['final_tokens']/44200):.1f}%",
                    "Peak VRAM": f"{c['peak_vram_gb']:.2f} GB",
                    "VRAM Saved": f"+{c['vram_saved_gb']:.3f} GB ({100*c['vram_saved_gb']/10.91:.1f}%)" if c["vram_saved_gb"] > 0 else "Baseline",
                    "Prefill Latency": f"{c['prefill_latency_seconds']:.2f} s",
                    "Speedup": f"{c['speedup_vs_baseline']:.2f}x",
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

            st.markdown("""<div class="finding-alert">
                <b>&#128273; Key Empirical Findings from Benchmark:</b><br>
                &bull; <b>Temporal Pruning alone (FP16)</b> delivers <b>1.82x faster prefill</b> (33.14s vs 60.43s) and saves ~1.0 GB VRAM by cutting quadratic attention over static background tokens.<br>
                &bull; <b>Temporal Pruning + INT8 Quantization</b> achieves the <b>highest memory reduction: 1.459 GB (13.4% of total baseline)</b>, bringing peak VRAM down to 9.45 GB.<br>
                &bull; <i>Engineering note:</i> INT8 prefill latency is <b>78.06s</b> (slower than baseline FP16) due to dequantization overhead on the T4 GPU during prefill accumulation.
            </div>""", unsafe_allow_html=True)

            st.markdown("##### &#128221; Generated Semantic Output (Conditioned on 100-Frame Context)")
            st.caption("Qualitative semantic output is provided to inspect whether useful scene information is retained.")
            st.markdown(f"""
```markdown
[VLM Output from {active_res["display_name"]}]:
{active_res.get("generated_text", "Scene processed successfully.")}
```
""")


# =============================================================================
# PAGE 2: Long-Context Benchmark & Visualizations
# =============================================================================
elif selected_page == "Long-Context Benchmark & Visualizations":
    st.subheader("📈 Long-Context Experimental Benchmarks")
    
    st.markdown("""
    <div class="recorded-disclaimer">
    All graphs render <b>verified empirical data</b> from notebook <code>edi (1).ipynb</code> run on Google Colab (NVIDIA Tesla T4 16GB, Qwen2.5-VL-3B-Instruct).
    </div>
    """, unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 1. Context Tokens vs. Peak GPU VRAM (100 Frames)")
        if trajectory_data:
            df_traj = pd.DataFrame(trajectory_data)
            fig_traj = go.Figure()
            
            fig_traj.add_trace(go.Scatter(
                x=df_traj["tokens"], y=df_traj["dynamic_fp16_vram_gb"],
                mode='lines', name='Dynamic FP16 (Baseline)',
                line=dict(color='#F87171', width=2.5)
            ))
            fig_traj.add_trace(go.Scatter(
                x=df_traj["tokens"], y=df_traj["hqq_int8_vram_gb"],
                mode='lines', name='HQQ INT8 (Quantized)',
                line=dict(color='#38BDF8', width=2.5)
            ))
            fig_traj.add_trace(go.Scatter(
                x=df_traj["tokens"], y=df_traj["hqq_int4_vram_gb"],
                mode='lines', name='HQQ INT4 (Quantized)',
                line=dict(color='#00DC82', width=2.5)
            ))
            
            fig_traj.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                xaxis_title="Accumulated Context Tokens",
                yaxis_title="Allocated GPU Memory (GB)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=40, b=40),
                height=380,
            )
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("Trajectory data loading...")
            
    with col_chart2:
        st.markdown("##### 2. Peak VRAM & Prefill Latency Tradeoff")
        
        cfg_names = ["Unpruned FP16", "Pruned FP16", "Pruned INT8"]
        vram_vals = [10.91, 9.91, 9.45]
        latency_vals = [60.43, 33.14, 78.06]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Peak VRAM (GB)', x=cfg_names, y=vram_vals,
            marker_color=['#94A3B8', '#00DC82', '#38BDF8'],
            text=[f"{v:.2f} GB" for v in vram_vals],
            textposition='auto',
            yaxis='y1'
        ))
        fig_bar.add_trace(go.Scatter(
            name='Prefill Time (s)', x=cfg_names, y=latency_vals,
            mode='lines+markers', line=dict(color='#F59E0B', width=3),
            marker=dict(size=10, color='#F59E0B'),
            text=[f"{l:.1f}s" for l in latency_vals],
            yaxis='y2'
        ))
        
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis=dict(title="Peak VRAM (GB)", range=[0, 14]),
            yaxis2=dict(title="Prefill Time (s)", overlaying='y', side='right', range=[0, 95]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("##### 3. Visual Token Reduction Breakdown (100 Frames)")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Retained Dynamic Tokens', 'Pruned Redundant Background'],
            values=[23505, 15595],
            hole=.55,
            marker_colors=['#00DC82', '#334155'],
            textinfo='label+percent',
            pull=[0.05, 0]
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text='39.9%<br>Pruned', x=0.5, y=0.5, font_size=18, showarrow=False, font_color='#00DC82')],
            margin=dict(l=20, r=20, t=30, b=30),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart4:
        st.markdown("##### 4. Memory Reduction Breakdown (GB saved vs Baseline)")
        vram_savings = [0.0, 0.997, 1.459]
        fig_savings = go.Figure(go.Bar(
            x=["Unpruned Baseline", "Temporal Pruning Alone", "Pruning + INT8 Quantized"],
            y=vram_savings,
            marker_color=['#475569', '#00DC82', '#38BDF8'],
            text=[f"{s:+.3f} GB" if s > 0 else "0.0 GB" for s in vram_savings],
            textposition='auto',
        ))
        fig_savings.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis_title="VRAM Saved (GB)",
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
        )
        st.plotly_chart(fig_savings, use_container_width=True)


# =============================================================================
# PAGE 3: Single-Image vs Long-Context Analysis
# =============================================================================
elif selected_page == "Single-Image vs Long-Context Analysis":
    st.subheader("🔬 Empirical Analysis: Single-Image vs. Long-Context Surveillance")
    
    st.markdown("""
    <div class="finding-alert">
        <b>Why single-image VLM tests give a misleading picture:</b><br>
        In single-image tests (442 tokens), model weights dominate GPU memory (~7.5 GB for Qwen2.5-VL-3B). Quantizing the KV cache saves only <b>7 to 13 MB (&lt;0.2%)</b>.<br>
        However, in continuous video surveillance (44,200 tokens across 100 frames), the KV cache expands by <b>100x</b>, becoming a critical memory bottleneck where pruning and quantization deliver <b>1.46 GB (13.4%) peak savings</b>.
    </div>
    """, unsafe_allow_html=True)
    
    col_single, col_long = st.columns(2)
    
    with col_single:
        st.markdown("##### Single Image Experiment (Cell 6)")
        st.caption("Sequence length: 442 tokens (391 visual tokens)")
        
        single_data = benchmarks.get("single_image_benchmark", {}).get("results", [])
        if single_data:
            df_single = pd.DataFrame(single_data)
            df_single.columns = ["Config", "Tokens", "Latency (s)", "tok/s", "Peak VRAM (GB)", "VRAM Saved (GB)"]
            st.dataframe(df_single, use_container_width=True, hide_index=True)
            
            st.metric(
                label="Single-Image Max VRAM Saved",
                value="+0.013 GB",
                delta="+0.17% (negligible)",
                delta_color="off"
            )
            
    with col_long:
        st.markdown("##### Long Context Video Experiment (Cell 10)")
        st.caption("Sequence length: 44,200 tokens (39,100 visual tokens across 100 frames)")
        
        long_summary = [
            {"Config": "Unpruned FP16", "Tokens": "44,200", "Peak VRAM": "10.91 GB", "Prefill": "60.43s", "VRAM Saved": "Baseline"},
            {"Config": "Pruned FP16", "Tokens": "28,605", "Peak VRAM": "9.91 GB", "Prefill": "33.14s", "VRAM Saved": "+0.997 GB (9.1%)"},
            {"Config": "Pruned INT8", "Tokens": "28,605", "Peak VRAM": "9.45 GB", "Prefill": "78.06s", "VRAM Saved": "+1.459 GB (13.4%)"}
        ]
        st.dataframe(pd.DataFrame(long_summary), use_container_width=True, hide_index=True)
        
        st.metric(
            label="Long-Context Max VRAM Saved",
            value="+1.459 GB",
            delta="13.4% of total GPU memory",
            delta_color="normal"
        )
        
    st.markdown("---")
    st.markdown("##### 💡 Technical Explanation for Defense / Viva")
    st.markdown("""
    - **Single-Image Regime**:
      $$\\text{VRAM} = \\text{Model Weights (7.5 GB)} + \\text{KV Cache (0.014 GB)} + \\text{Activations}$$
      Since $0.014\\text{ GB} \\ll 7.5\\text{ GB}$, shrinking the cache by $2\\times$ saves almost nothing on total peak memory.
    - **Long-Context Surveillance Regime**:
      $$\\text{VRAM} = \\text{Model Weights (7.5 GB)} + \\text{KV Cache (3.4 GB)} + \\text{Activations}$$
      At $44,200$ tokens, the KV cache accounts for nearly a third of all allocated VRAM. Shrinking visual tokens by $39.9\\%$ and quantizing keys/values from FP16 to INT8 prevents out-of-memory crashes on consumer and edge hardware (e.g. 16GB T4).
    """)


# =============================================================================
# PAGE 4: Argus System Architecture & Roadmap
# =============================================================================
elif selected_page == "Argus System Architecture & Roadmap":
    st.subheader("🏛️ Argus Architecture: Subsystems & Implementation Roadmap")
    
    st.markdown("""
    <div class="recorded-disclaimer">
    <b>System Scope Boundary:</b> Today's verified prototype implements the <b>VLM Token & KV Cache Optimization Subsystem</b> (the foundational compute engine). Downstream modules (Perception, Hybrid Memory, Natural Language Query) are actively being developed for future integration.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### End-to-End System Hierarchy")
    
    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      Surveillance Video Stream                         │
    └──────────────────────────────────┬─────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────▼─────────────────────────────────────┐
    │                 Perception Pipeline (YOLOv8 / ByteTrack)               │ [Next Integration]
    │      - Motion detection, object tracking, event proposal triggers      │
    └──────────────────────────────────┬─────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────▼─────────────────────────────────────┐
    │          ARGUS VLM OPTIMIZATION ENGINE (CURRENT SUBSYSTEM)             │ [COMPLETED & VALIDATED]
    │  ┌──────────────────────────────────────────────────────────────────┐  │
    │  │ 1. ViT Patch Hooking & Pre-merge Feature Extraction              │  │
    │  │ 2. Temporal Token Pruning (Cosine similarity threshold = 0.98)   │  │
    │  │ 3. 3D RoPE (Rotary Position Embedding) Re-alignment             │  │
    │  │ 4. Quantized KV Cache Management (HQQ INT8 / INT4)               │  │
    │  │ 5. Qwen2.5-VL-3B Multimodal Conditioning                         │  │
    │  └──────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────┬─────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────▼─────────────────────────────────────┐
    │                      Argus Semantic Memory Engine                      │ [Next Integration]
    │  ┌─────────────────────────────────┬────────────────────────────────┐  │
    │  │ Episodic Vector Store           │ Dynamic Knowledge Graph        │  │
    │  │ (Dense scene embeddings)        │ (Entities, Actions, Relations) │  │
    │  └─────────────────────────────────┴────────────────────────────────┘  │
    └──────────────────────────────────┬─────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────▼─────────────────────────────────────┐
    │               Natural Language Operator Query Interface                │ [Planned]
    │      "Did any unauthorized vehicle enter between 14:00 and 15:00?"     │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.markdown("##### Subsystem Status Matrix")
    
    matrix = [
        {"Subsystem": "ViT Patch Hooking & Extraction", "Component": "Qwen2.5-VL vision merger hooks", "Status": "✅ VALIDATED", "Notes": "Pre-merge patch hooks capture 4 unmerged patches per token"},
        {"Subsystem": "Temporal Token Pruning", "Component": "Cosine similarity + window reindexing", "Status": "✅ VALIDATED", "Notes": "Achieved 39.9% visual token reduction across 100 frames"},
        {"Subsystem": "3D RoPE Re-indexing", "Component": "get_rope_index slice adjustment", "Status": "✅ VALIDATED", "Notes": "Preserves temporal/spatial coordinates after token slicing"},
        {"Subsystem": "KV Cache Quantization", "Component": "HQQ INT8 / INT4 QuantizedCache", "Status": "✅ VALIDATED", "Notes": "Achieved 1.46 GB (13.4%) peak VRAM reduction on 44.2K tokens"},
        {"Subsystem": "Long-Context Benchmark", "Component": "100-frame (120.3s) accumulation", "Status": "✅ VALIDATED", "Notes": "Verified on Tesla T4 with text generation output"},
        {"Subsystem": "Event-Triggered Perception", "Component": "YOLOv8 + ByteTrack trigger", "Status": "⏳ IN PROGRESS", "Notes": "Will trigger VLM only during high-salience intervals"},
        {"Subsystem": "Hybrid Graph Memory", "Component": "NetworkX / Neo4j + Vector DB", "Status": "⏳ ROADMAP", "Notes": "Captures spatio-temporal scene relations over hours/days"},
        {"Subsystem": "NL Query Interface", "Component": "RAG-driven surveillance search", "Status": "⏳ ROADMAP", "Notes": "Operator queries over structured graph memory"}
    ]
    st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 5: Verified Technical Logs & Output
# =============================================================================
elif selected_page == "Verified Technical Logs & Output":
    st.subheader("📜 Verified Experimental Logs & Raw Data")
    
    st.markdown("""
    <div class="recorded-disclaimer">
    Direct raw text output from the model and execution metrics logged during the experiment in <code>edi (1).ipynb</code>.
    </div>
    """, unsafe_allow_html=True)
    
    tab_raw, tab_text, tab_json = st.tabs(["📊 Raw Benchmark Table", "📝 Generated Text Per Config", "⚙️ System Config JSON"])
    
    with tab_raw:
        st.markdown("##### Long-Context 100-Frame Benchmark Results")
        st.code("""
================================================================================
Config                     Final tokens  Peak VRAM  Prefill(s)  Speedup vs Base
unpruned + dynamic                44200     10.91G      60.43s            1.00x
pruned + dynamic                  28605      9.91G      33.14s            1.82x
pruned + INT8                     28605      9.45G      78.06s            0.77x
================================================================================
Combined savings vs unpruned+dynamic baseline:
  Token reduction (pruning alone): 35.3% total tokens (39.9% visual tokens)
  VRAM saved (pruned+dynamic):     +0.997 GB (9.1% reduction)
  VRAM saved (pruned+INT8):        +1.459 GB (13.4% reduction)
  Prefill speedup (pruned+dynamic): 1.82x faster than unpruned baseline
  Prefill speedup (pruned+INT8):    0.77x (slower due to HQQ dequant overhead on T4)
================================================================================
        """, language="text")
        
    with tab_text:
        st.markdown("##### Real Text Output Conditioned on 100-Frame Video Context")
        st.caption("Generated semantic output is displayed for qualitative inspection.")
        
        st.markdown("**1. Unpruned Dynamic FP16 (Baseline)**")
        st.info("""
The image shows a section of an outdoor wall with some greenery in the background. The wall appears to be made of concrete and has patches where moss or algae have grown, indicating it might not receive much sunlight for extended periods.

In terms:
- **Objects**: Wall (concrete), Greenery
  - No people visible.
  
Action: 
No specific actions are taking place on this particular segment shown here as per surveillance system's perspective

Unusual Elements:
1) Moss/Algae Growth — This indicates that there is limited exposure time under direct sun which could suggest either shade from nearby
        """)
        
        st.markdown("**2. Pruned Dynamic FP16**")
        st.success("""
The image shows a section of an outdoor wall with some greenery in the background. The wall appears to be made of concrete and has patches where moss or algae have grown, indicating it might not receive much sunlight.

In terms:
- **Objects**: Wall (concrete), vegetation.
  - No people visible on this segment
        """)

        st.markdown("**3. Pruned Quantized INT8**")
        st.success("""
The image shows a section of an outdoor wall with some greenery in the background. The wall appears to be made of concrete and has patches where moss or algae have grown, indicating it might not receive much sunlight.

In terms:
- **Objects**: Wall (concrete), vegetation.
  - No people visible on this segment
        """)
        
    with tab_json:
        st.markdown("##### System Configuration & Checkpoint Data")
        st.json(benchmarks["system_metadata"])
        if st.checkbox("Show full benchmark JSON payload"):
            st.json(benchmarks)

# =============================================================================
# PAGE 6: Surveillance System Overview (from ai-surveillance repo)
# =============================================================================
elif selected_page == "Surveillance System Overview":
    st.subheader("🛡️ Argus Full Surveillance Pipeline — System Overview")

    st.markdown("""
    <div class="recorded-disclaimer">
    <b>Source:</b> Working production pipeline from the companion <code>ai-surveillance</code> repository.
    All detector status, performance numbers, and configuration values below are drawn directly
    from that codebase's <code>configs/pipeline.yaml</code>, <code>configs/vlm.yaml</code>,
    <code>PROGRESS.md</code>, and <code>README.md</code>.
    </div>
    """, unsafe_allow_html=True)

    # ── Phase progress summary ─────────────────────────────────────────────
    st.markdown("### 📋 Project Phase Summary")

    phase_rows = [
        {"Phase": "1 — Core Detection Loop",   "Status": "✅ Complete", "Key Deliverable": "Shared YOLOv8n + ByteTrack multi-object tracking + OpenCV live display"},
        {"Phase": "2 — Fall Detection",          "Status": "✅ Complete", "Key Deliverable": "YOLOv8n-pose keypoints + state-machine (tuned anti-FP); ONNX export → 46% overhead reduction"},
        {"Phase": "3 — ReID + Face",             "Status": "✅ Complete", "Key Deliverable": "OSNet-x0.25 ReID + SCRFD/MobileFaceNet + identity fusion layer"},
        {"Phase": "4 — Event Detectors",         "Status": "✅ Complete", "Key Deliverable": "Fire/smoke, phone-watching, gathering, violence (heuristic), object-left + motion prefilter"},
        {"Phase": "5A — Pose Smoother",          "Status": "✅ Complete", "Key Deliverable": "One-Euro filter + EMA bbox + per-keypoint confidence gating"},
        {"Phase": "5B — PAR Head",               "Status": "✅ Complete", "Key Deliverable": "Hybrid ResNet18 + HSV dominant colour + 15-frame temporal aggregator (disabled: no trained weights)"},
        {"Phase": "5C — ROI Occupancy",          "Status": "✅ Complete", "Key Deliverable": "Polygon-gated gathering detection with wall-clock timing"},
        {"Phase": "5D — Fight Detector + Clips", "Status": "✅ Complete", "Key Deliverable": "Skeleton velocity + proximity + oscillation signals; 5s ring-buffer MP4 clip on trigger"},
        {"Phase": "5E — Smoking Upgrade",        "Status": "✅ Complete", "Key Deliverable": "Gesture oscillation gate + YOLO crop confirmation (cadilak/smoking-detection-yolov8)"},
        {"Phase": "5F — Event Buffer",           "Status": "✅ Complete", "Key Deliverable": "Windowed JSON flush every 10 s; SQLite event log + keyframe storage"},
        {"Phase": "6 — VLM Layer",              "Status": "⏳ In Progress", "Key Deliverable": "Qwen2.5-VL-3B NF4 ambient + escalated inference; query engine; Bitchat mesh alerting"},
    ]
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Detector matrix ────────────────────────────────────────────────────
    st.markdown("### 🔍 Detector Registry")

    detectors = [
        {"Detector": "Fall (YOLOv8n-pose)",       "Signal / Method": "Pose keypoints + bbox aspect ratio + rule-based state machine",                "Status": "✅ Live-tested",  "Notes": "Tuned anti-FP; ONNX-direct runtime"},
        {"Detector": "Fight (5D skeleton)",        "Signal / Method": "Pose-based kinematic interaction: proximity + rapid arm/body keypoint velocity",  "Status": "✅ Live-tested",  "Notes": "Oscillation gate prevents hug false positives"},
        {"Detector": "Fire / Smoke",              "Signal / Method": "YOLOv8n fine-tuned on D-Fire (mAP50=0.754), conf=0.45, multi-frame ≥2/5",         "Status": "✅ Real model",   "Notes": "Source: rabahdev/fire-smoke-yolov8n; HSV fallback if weights missing"},
        {"Detector": "Smoking",                   "Signal / Method": "YOLOv8n fine-tuned for cigarette/vape detection near tracked person",             "Status": "✅ Real model",   "Notes": "Source: cadilak/smoking-detection-yolov8"},
        {"Detector": "Phone Use",                 "Signal / Method": "YOLOv8n COCO class 67, imgsz=480; confirm=3 frames / hold=15 frames hysteresis",  "Status": "✅ Stable",       "Notes": "Head-pose gate: nose below shoulder midpoint"},
        {"Detector": "Gathering",                  "Signal / Method": "Fixed-radius centroid clustering; fires on 3+ people within 150 px",               "Status": "✅ Works",        "Notes": "10 s cooldown; no ML model needed"},
        {"Detector": "Violence (heuristic)",       "Signal / Method": "Bbox IoU ≥ 0.3 + relative motion ≥ 40 px/frame sustained for 1.5 s",              "Status": "⚠️ Heuristic",  "Notes": "Cannot distinguish fight from hug; VLM verification resolves"},
        {"Detector": "Object Left Behind",         "Signal / Method": "Track stationary non-person objects (bags, suitcases) for > 30 s",                  "Status": "✅ Complete",    "Notes": "COCO classes: backpack, handbag, suitcase, bottle"},
        {"Detector": "PAR (attributes)",           "Signal / Method": "ResNet18 hybrid + HSV dominant colour + 15-frame temporal aggregator",              "Status": "⚠️ No weights", "Notes": "Disabled until PA-100K/RAP checkpoint; HSV colour works"},
        {"Detector": "ReID (OSNet-x0.25)",        "Signal / Method": "512-dim FP16 embedding + FAISS index; match_threshold=0.65",                       "Status": "✅ Complete",    "Notes": "Runs every 15 frames to save VRAM"},
        {"Detector": "Face (SCRFD + FaceNet)",    "Signal / Method": "SCRFD detection + MobileFaceNet embedding + FAISS face index",                      "Status": "✅ Complete",    "Notes": "Identity fusion: face > ReID priority"},
        {"Detector": "VLM (Qwen2.5-VL-3B NF4)",  "Signal / Method": "Scene description + entity JSON; escalated on detector flags; ambient every 150 f", "Status": "✅ Live",        "Notes": "NF4 4-bit quant; ~2.3–2.5 GB VRAM; 10–15 s per pass"},
    ]
    st.dataframe(pd.DataFrame(detectors), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Performance numbers ───────────────────────────────────────────────
    st.markdown("### ⚡ Live Performance (RTX 4050 Laptop, 6 GB VRAM)")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Webcam FPS (No VLM)</div>
            <div class="metric-value">~22</div>
            <div class="metric-delta-neutral">18–25 FPS range</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Webcam FPS (+ VLM BG)</div>
            <div class="metric-value">~22</div>
            <div class="metric-delta-neutral">Non-blocking thread</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">VLM Inference Time</div>
            <div class="metric-value">~12 s</div>
            <div class="metric-delta-neutral">10–15 s per pass (BG)</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total VRAM (All + VLM)</div>
            <div class="metric-value">~2.4 GB</div>
            <div class="metric-delta-positive">▼ 40% of 6 GB budget</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # VRAM budget chart
    st.markdown("##### VRAM Budget Breakdown (RTX 4050, 6 GB)")
    vram_components = ["Detectors only\n(YOLO, pose, ReID, face)", "Qwen2.5-VL-3B NF4\n(VLM)", "Remaining\nheadroom"]
    vram_values = [0.085, 2.35, 3.565]
    vram_colors = ["#38BDF8", "#00DC82", "#1E293B"]

    fig_vram = go.Figure(go.Bar(
        x=vram_components,
        y=vram_values,
        marker_color=vram_colors,
        text=[f"{v:.2f} GB" for v in vram_values],
        textposition="auto",
    ))
    fig_vram.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        yaxis_title="VRAM (GB)",
        yaxis=dict(range=[0, 6.2]),
        margin=dict(l=40, r=20, t=30, b=50),
        height=320,
        annotations=[
            dict(x=2, y=5.9, text="6 GB Budget", showarrow=False,
                 font=dict(color="#F87171", size=12))
        ],
        shapes=[
            dict(type="line", x0=-0.5, x1=2.5, y0=6.0, y1=6.0,
                 line=dict(color="#F87171", width=2, dash="dash"))
        ]
    )
    st.plotly_chart(fig_vram, use_container_width=True)

    st.markdown("---")

    # ── FrameRouter scheduling ────────────────────────────────────────────
    st.markdown("### ⏱️ FrameRouter Stage Scheduling (configs/pipeline.yaml)")
    st.caption("The FrameRouter determines how often each pipeline stage runs, ensuring heavy stages don't block the main 30 FPS capture loop.")

    router_stages = [
        {"Stage": "detect",        "Runs every N frames": 1,  "Effective rate @ 30fps": "30 Hz",  "Notes": "Every frame — feeds all downstream stages"},
        {"Stage": "track",         "Runs every N frames": 1,  "Effective rate @ 30fps": "30 Hz",  "Notes": "ByteTrack is CPU-cheap"},
        {"Stage": "pose",          "Runs every N frames": 3,  "Effective rate @ 30fps": "10 Hz",  "Notes": "ONNX-direct; 17% overhead reduction vs every-2"},
        {"Stage": "violence",      "Runs every N frames": 3,  "Effective rate @ 30fps": "10 Hz",  "Notes": "Needs temporal continuity; cheaper than pose"},
        {"Stage": "fight",         "Runs every N frames": 3,  "Effective rate @ 30fps": "10 Hz",  "Notes": "Same cadence as violence — skeleton-based"},
        {"Stage": "face",          "Runs every N frames": 8,  "Effective rate @ 30fps": "3.75 Hz","Notes": "buffalo_s = 5 ONNX models; biggest FPS driver"},
        {"Stage": "smoking",       "Runs every N frames": 10, "Effective rate @ 30fps": "3 Hz",   "Notes": "Low urgency event"},
        {"Stage": "phone",         "Runs every N frames": 10, "Effective rate @ 30fps": "3 Hz",   "Notes": "Dedicated YOLO instance (COCO cls 67)"},
        {"Stage": "par",           "Runs every N frames": 10, "Effective rate @ 30fps": "3 Hz",   "Notes": "Pedestrian attribute recognition on crops"},
        {"Stage": "reid",          "Runs every N frames": 15, "Effective rate @ 30fps": "2 Hz",   "Notes": "Only needed on new/re-appearing tracks"},
        {"Stage": "fire_smoke",    "Runs every N frames": 15, "Effective rate @ 30fps": "2 Hz",   "Notes": "Slow-evolving event"},
        {"Stage": "gathering",     "Runs every N frames": 30, "Effective rate @ 30fps": "1 Hz",   "Notes": "Aggregate statistic"},
        {"Stage": "object_left",   "Runs every N frames": 30, "Effective rate @ 30fps": "1 Hz",   "Notes": "Track stationary non-person objects"},
        {"Stage": "vlm_escalated", "Runs every N frames": 1,  "Effective rate @ 30fps": "On-demand","Notes": "Triggered when detector flags high priority"},
    ]
    st.dataframe(pd.DataFrame(router_stages), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── VLM Layer architecture ────────────────────────────────────────────
    st.markdown("### 🧠 Phase 6: VLM Integration Layer")

    col_vlm1, col_vlm2 = st.columns(2)

    with col_vlm1:
        st.markdown("##### VLM Model Configuration")
        st.code("""
# configs/vlm.yaml (key fields)
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"  # 3B, 7.1 GB on disk
  device: "cuda:0"
  quantization: "nf4"         # bitsandbytes 4-bit NF4
  max_vram_gb: 3.5            # hard ceiling; load aborts if exceeded
  max_pixels: 602112          # reduced for VRAM; was 1003520

escalated:
  max_tokens: 256
  context_window_frames: 15   # frames of context per escalated pass

escalation:
  forced_full_fidelity_interval: 150  # frames (every ~5s at 30 FPS)

integration:
  log_vlm_events: true
  vlm_event_db_table: "vlm_events"    # stored in events.db
        """, language="yaml")

    with col_vlm2:
        st.markdown("##### VLM Module Responsibilities")
        vlm_modules = [
            {"Module": "vlm/core.py (VLMCore)",          "Role": "Qwen2.5-VL-3B NF4 loader; dual-pass inference (SCENE + ENTITY prompts); multi-frame buffer"},
            {"Module": "vlm/__init__.py (VLMIntegration)","Role": "Lifecycle management; escalation routing; SQLite persistence; single model instance shared"},
            {"Module": "vlm/escalation.py",              "Role": "Escalation scoring from detector event priority; threshold gating"},
            {"Module": "vlm/query_engine.py",            "Role": "Natural language query against vlm_events table; shares loaded model (no second load)"},
            {"Module": "vlm/kv_cache.py",               "Role": "Tiered KV cache spec: Hot (VRAM) / Warm (RAM, quantized) / Cold (SQLite) — architectural design"},
            {"Module": "vlm/token_pruning.py",          "Role": "ZSPAPrune-style prompt-aware token pruning; person-token boost; diversity-relevance balance"},
            {"Module": "vlm/temporal_merge.py",         "Role": "Multi-frame entity merging across consecutive VLM passes"},
            {"Module": "core/bitchat.py",               "Role": "Bitchat mesh alert client: HTTP REST, background queue, image + text push to Android"},
        ]
        st.dataframe(pd.DataFrame(vlm_modules), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Connection to VLM optimization research ───────────────────────────
    st.markdown("### 🔗 Connection to This Repository's VLM Optimization Research")

    st.markdown("""
    <div class="finding-alert">
    <b>How the two repositories connect:</b><br><br>
    The <b>ai-surveillance</b> system runs Qwen2.5-VL-3B in production (live webcam, NF4 4-bit quantization).
    This confirms the real-world deployment pressure that motivates the KV cache optimization research in this repository:<br><br>
    &bull; The live system shows VLM inference takes <b>10–15 s per escalated pass</b> on an RTX 4050 (6 GB VRAM).<br>
    &bull; Without token pruning, accumulating video context across hundreds of frames would exhaust VRAM and make real-time operation impossible.<br>
    &bull; The <b>Temporal Token Pruning</b> technique validated in this notebook (39.9% visual token reduction, 1.82× faster prefill)
       is the proposed solution to this exact production bottleneck.<br>
    &bull; The <b>Tiered KV Cache</b> architecture (Hot/Warm/Cold) in <code>vlm/kv_cache.py</code> is the design target for enabling
       infinite-streaming surveillance without VRAM overflow.
    </div>
    """, unsafe_allow_html=True)

    # Integration roadmap chart
    st.markdown("##### Integration Roadmap: Research → Production")

    roadmap = [
        {"Step": "✅ Done",      "Item": "Validated temporal token pruning",       "Impact": "39.9% visual token reduction, 1.82× faster prefill"},
        {"Step": "✅ Done",      "Item": "Validated KV cache quantization (INT8)",  "Impact": "1.459 GB peak VRAM savings on 100-frame context"},
        {"Step": "⏳ Next",     "Item": "Port token pruning into vlm/token_pruning.py", "Impact": "Reduces per-escalation VRAM in live pipeline"},
        {"Step": "⏳ Next",     "Item": "Activate Hot/Warm/Cold KV tier (vlm/kv_cache.py)", "Impact": "Enables infinite-streaming without VRAM overflow"},
        {"Step": "🗺️ Roadmap", "Item": "Natural language query engine over event history", "Impact": "Operator can ask: 'Who entered after 14:00?'"},
        {"Step": "🗺️ Roadmap", "Item": "Hybrid Graph Memory (NetworkX/Neo4j + vector DB)", "Impact": "Spatio-temporal scene relations over hours/days"},
    ]
    st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Bitchat alerting ──────────────────────────────────────────────────
    st.markdown("### 📡 Bitchat Mesh Alert System")
    st.caption("Every surveillance alert and VLM scene description is pushed to an Android phone via Bitchat's mesh network in real time.")

    alert_rows = [
        {"Event": "🔥 Fire / Smoke",   "Bitchat Message": "[FIRE] 15:42:10 — Fire detected in camera view",    "Priority": "🔴 Immediate (bypasses rate limiter)"},
        {"Event": "🧎 Person Falls",   "Bitchat Message": "[FALL] 15:43:22 — Person id:1 has fallen",           "Priority": "🔴 Immediate"},
        {"Event": "👊 Fight",          "Bitchat Message": "[FIGHT] 15:44:01 — Fight between persons detected",   "Priority": "🔴 Immediate"},
        {"Event": "📱 Phone Use",      "Bitchat Message": "[PHONE] 15:45:00 — Person id:1 using phone",          "Priority": "🟡 Rate-limited (5.5 s)"},
        {"Event": "👥 Crowd",          "Bitchat Message": "[GATHERING] 15:46:10 — 4 people gathered",            "Priority": "🟡 Rate-limited"},
        {"Event": "🎥 VLM Scene",      "Bitchat Message": "[CAM] A man is sitting at a desk… (+ keyframe image)","Priority": "🟢 Ambient (every 150 frames)"},
    ]
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.8rem; padding: 12px;'>
    <b>ARGUS Real-Time Surveillance Project</b> — Milestone: VLM Optimization Subsystem<br>
    Validated on Qwen2.5-VL-3B-Instruct with PyTorch, Transformers 4.57.2 & HQQ Backend.
</div>
""", unsafe_allow_html=True)
