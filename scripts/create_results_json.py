import json
import re

# Load extracted trajectory
with open('edi (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell8_out = nb['cells'][8].get('outputs', [])
all_text = ""
for out in cell8_out:
    t = out.get('text', []) or out.get('data', {}).get('text/plain', [])
    all_text += "".join(t)

pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d\.]+)G\s+([\d\.]+)G\s+([\d\.]+)G", re.MULTILINE)
trajectory = []
for match in pattern.finditer(all_text):
    frame, tokens, dyn_v, int8_v, int4_v = match.groups()
    trajectory.append({
        "frame": int(frame),
        "tokens": int(tokens),
        "dynamic_fp16_vram_gb": float(dyn_v),
        "hqq_int8_vram_gb": float(int8_v),
        "hqq_int4_vram_gb": float(int4_v)
    })

benchmark_data = {
    "system_metadata": {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "tested_hardware": "NVIDIA Tesla T4 (15.36 GB VRAM, Google Colab)",
        "precision": "torch.float16",
        "transformers_version": "4.57.2",
        "quantization_backend": "HQQ (Half-Quadratic Quantization)",
        "video_duration_seconds": 120.3,
        "video_total_frames": 3610,
        "video_fps": 30.0,
        "sampled_frames": 100,
        "pruning_threshold_cosine": 0.98,
        "min_keep_ratio": 0.05
    },
    "long_context_benchmark": {
        "configurations": [
            {
                "config_id": "unpruned_dynamic_fp16",
                "display_name": "Unpruned Baseline (FP16 Dynamic)",
                "description": "Full visual tokens, standard DynamicCache FP16",
                "final_tokens": 44200,
                "visual_tokens": 39100,
                "peak_vram_gb": 10.91,
                "prefill_latency_seconds": 60.43,
                "speedup_vs_baseline": 1.0,
                "vram_saved_gb": 0.0,
                "vram_saved_pct": 0.0,
                "token_reduction_pct": 0.0,
                "generated_text": (
                    "The image shows a section of an outdoor wall with some greenery in the background. "
                    "The wall appears to be made of concrete and has patches where moss or algae have grown, "
                    "indicating it might not receive much sunlight for extended periods.\n\n"
                    "In terms:\n- **Objects**: Wall (concrete), Greenery\n  - No people visible.\n\n"
                    "Action:\nNo specific actions are taking place on this particular segment shown here "
                    "as per surveillance system's perspective\n\n"
                    "Unusual Elements:\n1) Moss/Algae Growth — This indicates that there is limited exposure time "
                    "under direct sun which could suggest either shade from nearby"
                )
            },
            {
                "config_id": "pruned_dynamic_fp16",
                "display_name": "Temporal Pruning + FP16 Dynamic",
                "description": "Cosine similarity pruning (threshold=0.98), DynamicCache FP16",
                "final_tokens": 28605,
                "visual_tokens": 23505,
                "peak_vram_gb": 9.91,
                "prefill_latency_seconds": 33.14,
                "speedup_vs_baseline": 1.82,
                "vram_saved_gb": 0.997,
                "vram_saved_pct": 9.14,
                "token_reduction_pct": 35.28,
                "visual_token_reduction_pct": 39.88,
                "generated_text": (
                    "The image shows a section of an outdoor wall with some greenery in the background. "
                    "The wall appears to be made of concrete and has patches where moss or algae have grown, "
                    "indicating it might not receive much sunlight.\n\n"
                    "In terms:\n- **Objects**: Wall (concrete), vegetation.\n  - No people visible on this segment"
                )
            },
            {
                "config_id": "pruned_quantized_int8",
                "display_name": "Temporal Pruning + INT8 Quantized (HQQ)",
                "description": "Cosine similarity pruning + HQQ INT8 KV Cache (q_group=64, residual=128)",
                "final_tokens": 28605,
                "visual_tokens": 23505,
                "peak_vram_gb": 9.45,
                "prefill_latency_seconds": 78.06,
                "speedup_vs_baseline": 0.77,
                "vram_saved_gb": 1.459,
                "vram_saved_pct": 13.37,
                "token_reduction_pct": 35.28,
                "visual_token_reduction_pct": 39.88,
                "generated_text": (
                    "The image shows a section of an outdoor wall with some greenery in the background. "
                    "The wall appears to be made of concrete and has patches where moss or algae have grown, "
                    "indicating it might not receive much sunlight.\n\n"
                    "In terms:\n- **Objects**: Wall (concrete), vegetation.\n  - No people visible on this segment"
                )
            }
        ],
        "trajectory": trajectory
    },
    "single_image_benchmark": {
        "sequence_length": 442,
        "visual_tokens_unpruned": 391,
        "results": [
            {"config": "dynamic_fp16", "tokens": 150, "latency_s": 9.77, "tok_s": 15.3, "vram_gb": 7.60, "vram_saved_gb": 0.0},
            {"config": "hqq_int8_res32", "tokens": 150, "latency_s": 11.37, "tok_s": 13.2, "vram_gb": 7.60, "vram_saved_gb": 0.007},
            {"config": "hqq_int4_res32", "tokens": 21, "latency_s": 1.94, "tok_s": 10.8, "vram_gb": 7.59, "vram_saved_gb": 0.011},
            {"config": "hqq_int4_res64", "tokens": 21, "latency_s": 1.91, "tok_s": 11.0, "vram_gb": 7.59, "vram_saved_gb": 0.011},
            {"config": "hqq_int2_res32", "tokens": 51, "latency_s": 5.04, "tok_s": 10.1, "vram_gb": 7.59, "vram_saved_gb": 0.013}
        ],
        "key_takeaway": (
            "Single-image / short-context testing saves almost no VRAM (<0.2%) because static model weights (3B params = ~7.5 GB) "
            "dominate memory, and KV cache for 442 tokens is only ~14 MB. In contrast, long-context surveillance (44,200 tokens across 100 frames) "
            "makes KV cache a major memory bottleneck, where temporal pruning + INT8 achieves 1.46 GB (13.4%) peak memory reduction."
        )
    },
    "system_status": {
        "vlm_integration": {"name": "Qwen2.5-VL-3B Integration", "status": "COMPLETED", "icon": "✓"},
        "visual_token_extraction": {"name": "Visual Token Extraction & Hooking", "status": "COMPLETED", "icon": "✓"},
        "temporal_pruning": {"name": "Temporal Token Pruning (Cosine Sim)", "status": "COMPLETED", "icon": "✓"},
        "kv_cache_benchmarking": {"name": "KV Cache Quantization (FP16/INT8/INT4/INT2)", "status": "COMPLETED", "icon": "✓"},
        "long_context_testing": {"name": "100-Frame Long Context Benchmarking", "status": "COMPLETED", "icon": "✓"},
        "full_argus_integration": {"name": "Full Argus Integration (Memory & Perception)", "status": "IN_PROGRESS", "icon": "⏳"}
    }
}

with open('results/benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(benchmark_data, f, indent=2)

print("Created results/benchmark_results.json successfully.")
