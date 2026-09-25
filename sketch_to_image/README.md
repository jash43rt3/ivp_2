<div align="center">

# ✨ Sketch → Image

### Draw in mid-air with your finger — watch it become generative art.

A real-time computer-vision studio: point your index finger at the webcam,
**draw in the air**, then **pinch** (thumb + middle finger) and the app turns
your floating sketch into a unique piece of generative artwork.

*MediaPipe Hands · OpenCV · NumPy · optional Stable-Diffusion ControlNet*

</div>

---

## 🎬 How it works

```
webcam video ──► hand tracking ──► fingertip cursor ──► air strokes
   (OpenCV)      (MediaPipe)        (neon HUD)         (canvas)
                                                        │
                              pinch gesture (~0.4 s) ◄──┘
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
              procedural art engine          Stable-Diffusion
              (always available)             ControlNet scribble
                                              (--controlnet)
                                        │
                                        ▼
                     artwork flashed on screen + saved to output/
```

## ✋ Gestures & controls

| Action | How |
|---|---|
| ✏️ **Draw** | Raise your **index finger** and move it in front of the camera |
| 🖼️ **Generate** | **Pinch thumb + middle finger** and hold ~0.4 s (a countdown ring fills around the cursor) |
| `g` | Generate instantly (no pinch needed) |
| `u` / `c` | Undo last stroke / clear canvas |
| `s` | Save the gallery contact-sheet |
| `h` / `q` | Toggle help panel / quit |

The interface shows a live status pill (`DRAWING`, `PINCH HOLD`, …), an FPS
meter, corner brackets for the drawing area, a glowing mini-canvas preview of
your sketch, and a cinematic flash + slide-up reveal when art is generated.

## 🚀 Quick start

```bash
cd sketch_to_image
pip install -r requirements.txt

python main.py            # open webcam 0 and start drawing in the air
python main.py --source 1 # different camera, or --source clip.mp4 for a file
python demo.py --ui       # no webcam? render sample artworks + a UI mockup
```

Every artwork (and its source sketch) is written to `output/` as
`gen_001_sketch.png` / `gen_001_art.png`, plus a `gallery.png` contact sheet.

## 🧠 Generation back-ends

**1. Built-in procedural engine (default, zero heavy deps)**
Analyses the sketch (enclosed regions, strokes), then composes a unique
artwork each time: gradient skies, ambient glow orbs, halftone / stripe /
bokeh region fills, glowing ink outlines, star sparkles, film grain and
vignette — from curated colour palettes.

**2. Stable-Diffusion ControlNet (optional, state-of-the-art)**

```bash
pip install torch diffusers transformers accelerate
export HF_TOKEN=hf_xxx          # accept SD-v1.5 + ControlNet on huggingface.co
python main.py --controlnet
```

Your scribble is fed to `lllyasviel/sd-controlnet-scribble` + SD v1.5 and
rendered as a full AI image (falls back to the procedural engine if the
model can't load).

## 📁 Project layout

```
sketch_to_image/
├── main.py           # real-time app: video loop, gestures, HUD, saving
├── hand_tracker.py   # MediaPipe hands, pinch detection, neon cursor/trail
├── sketch_canvas.py  # stroke capture, Catmull-Rom smoothing, tapered ink
├── generator.py      # procedural art engine + optional ControlNet backend
├── theme.py          # shared glass-panel / pill / ring HUD design system
├── demo.py           # headless pipeline test + sample renders (--ui mockup)
├── requirements.txt
└── output/           # sketches, artworks, gallery
```

## 💡 Tips

* Good, even lighting on your hand = stable tracking.
* Keep your hand inside the corner brackets while drawing.
* Hold the pinch *still* for the ring to fill; release before it fires again.
* Simple closed shapes (houses, faces, rockets) produce the most striking
  region-filled artwork.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.21` |
| Camera not found | try `--source 1`, or check OS camera permissions |
| No GUI window on Linux | run inside a desktop session (needs display server) |
| ControlNet falls back silently | ensure `HF_TOKEN` is set and model licenses accepted |
