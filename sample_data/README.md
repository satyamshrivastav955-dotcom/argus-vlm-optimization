# Sample Data

This directory holds test inputs for experiments.

## What is committed

- `README.md` (this file) — instructions only
- Lightweight synthetic test images generated at runtime (PNG < 50KB each)

## What is NOT committed

- Video files (`.mp4`, `.avi`, `.mov`, etc.)
- Large images
- Proprietary or copyrighted datasets

---

## Recommended Sample Videos

For experiments in notebooks `03`–`07`, you need a short surveillance-style video.

### Option 1 — Synth frames (auto-generated, no download needed)

Each notebook includes a `generate_synthetic_frames()` utility that creates:
- A static background (grey gradient or solid color)
- A simulated moving object (rectangle) crossing the frame over N frames
- Optional noise injection to simulate camera grain

This is sufficient to verify the frame gating, cropping, and delta captioning pipelines.

### Option 2 — Creative Commons Surveillance Videos

The following public domain / CC0 / CC-BY sources provide suitable footage:

| Source | URL | Notes |
|--------|-----|-------|
| Pexels | https://www.pexels.com/search/videos/surveillance/ | Free, no attribution needed |
| Pixabay | https://pixabay.com/videos/search/security%20camera/ | CC0 |
| VIRAT Dataset (public) | https://viratdata.org | Academic use — read license carefully |
| UCF Crime Dataset | https://webpages.charlotte.edu/cchen62/dataset.html | Academic only |

### Recommended download command (Colab)

```bash
# Example — Pexels free video (replace URL with chosen video)
!wget -O sample_data/sample_corridor.mp4 "YOUR_VIDEO_URL"
```

> **Important**: Do not commit video files to this repository.
> Add videos to `.gitignore` entries or keep them outside the repo root.

---

## Annotation Format

For event-recall evaluation, label videos using this format in a JSON file:

```json
{
  "video_id": "sample_001",
  "video_path": "sample_data/sample_corridor.mp4",
  "scene_type": "indoor_corridor",
  "fps": 25,
  "total_frames": 500,
  "events": [
    {
      "event_id": "evt_001",
      "type": "person_enters",
      "frame_start": 120,
      "frame_end": 160,
      "description": "Person enters from left side of frame",
      "is_hazard": false
    },
    {
      "event_id": "evt_002",
      "type": "person_leaves",
      "frame_start": 300,
      "frame_end": 330,
      "description": "Same person exits to right side",
      "is_hazard": false
    }
  ],
  "notes": "Manually annotated. Static camera. Indoor lighting."
}
```

Save annotations as `sample_data/annotations_<video_id>.json`.

> **Note**: Only use labels that are genuinely annotated.
> Do NOT fabricate accuracy numbers or invent ground truth.
