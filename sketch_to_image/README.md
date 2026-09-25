# ✋ Sketch-to-Image — Draw with Your Finger, Get Generative Art

A Python video-processing project that uses your **webcam + MediaPipe hand
tracking** so you can **draw in mid-air with your index finger**, then
**pinch (thumb + middle finger)** to turn the sketch into a **generative
image**.

```
Webcam frames ──► HandTracker (MediaPipe) ──► fingertip path
                                                   │
                                                   ▼
                       SketchCanvas (smoothing, strokes, undo)
                                                   │  pinch gesture
                                                   ▼
                    Generator (procedural art engine / optional SD-ControlNet)
                                                   │
                                                   ▼
                              On-screen result + saved PNGs
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Real-time app: webcam loop, gestures, HUD, saving |
| `hand_tracker.py` | MediaPipe hand tracking, fingertip cursor, pinch detection |
| `sketch_canvas.py` | Virtual whiteboard: stroke capture, Catmull-Rom smoothing, undo/clear |
| `generator.py` | Turns a sketch into generative artwork (2 back-ends) |
| `demo.py` | Headless test — no webcam needed; writes sample images to `output/` |
| `requirements.txt` | Dependencies |

## Install & run

```bash
pip install -r requirements.txt
python main.py                 # webcam 0
python main.py --source 1      # other camera
python main.py --source clip.mp4   # or a video file
```

No webcam handy? Verify the whole pipeline headlessly:

```bash
python demo.py     # writes output/demo_sketch.png + output/demo_generated.png
```

## Gestures & keys

| Input | Action |
|---|---|
| Raise **index finger**, move hand | Draw a stroke in the air |
| Open hand / lower finger | Stop current stroke |
| **Pinch** thumb + middle finger (~0.4 s hold) | ✨ Generate image from sketch |
| `g` | Generate without pinching |
| `u` | Undo last stroke |
| `c` | Clear canvas |
| `q` / `Esc` | Quit |

Generated results flash on screen for 4 seconds and are saved to
`output/gen_NNN_sketch.png` and `output/gen_NNN_art.png`.

## How generation works

**Back-end 1 — built-in procedural engine (default, zero heavy deps):**
the sketch is analysed (line extraction → connected-component region
finding), then rendered as unique generative art: random palette
harmonisation, gradient background, per-region textured fills (halftone /
stripes / radial gradients), glowing ink lines, particles, vignette and
saturation post-processing. Every generation is seeded and different.

**Back-end 2 — Stable-Diffusion ControlNet (optional, real AI):**

```bash
pip install torch diffusers transformers accelerate
export HF_TOKEN=hf_xxx...llyasviel/sd-controlnet-scribble>
python main.py --controlnet
```

Uses `lllyasviel/sd-controlnet-scribble` + SD-1.5 to convert the scribble
into an image while preserving its structure. Falls back automatically to
the procedural engine if unavailable.

## Notes / tuning

- Pinch sensitivity: `HandTracker.PINCH_RATIO` in `hand_tracker.py`.
- Stroke thickness: `SketchCanvas.render(line_thickness=...)`.
- Drawing area inside the frame: `ROI` constant in `main.py`.
- The frame is mirrored (selfie view) so drawing feels natural.
