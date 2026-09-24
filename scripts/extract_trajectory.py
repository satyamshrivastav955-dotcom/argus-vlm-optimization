import json
import re

with open('edi (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 8 output has frame checkpoints: frame, tokens, dyn_vram, int8_vram, int4_vram
cell8_out = nb['cells'][8].get('outputs', [])
all_text = ""
for out in cell8_out:
    t = out.get('text', []) or out.get('data', {}).get('text/plain', [])
    all_text += "".join(t)

trajectory = []
# Lines look like:
# Frame  Tokens   dyn_vram  int8_vram  int4_vram  saved_int8  saved_int4
#     1     442      7.61G      7.61G      7.61G     +0.007G     +0.011G
# ...
pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d\.]+)G\s+([\d\.]+)G\s+([\d\.]+)G", re.MULTILINE)
for match in pattern.finditer(all_text):
    frame, tokens, dyn_v, int8_v, int4_v = match.groups()
    trajectory.append({
        "frame": int(frame),
        "tokens": int(tokens),
        "dynamic_fp16_vram_gb": float(dyn_v),
        "hqq_int8_vram_gb": float(int8_v),
        "hqq_int4_vram_gb": float(int4_v)
    })

print(f"Extracted {len(trajectory)} trajectory points from Cell 8.")
if trajectory:
    print("First point:", trajectory[0])
    print("Last point:", trajectory[-1])
