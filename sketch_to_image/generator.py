"""Generative rendering of hand-drawn sketches.

Two back-ends (no heavy AI dependencies required):

1. ControlNet backend (optional, best quality)
   If a HuggingFace token is available and the `diffusers` stack is
   installed, the sketch is fed to a real Sketch->Image ControlNet model.

2. Built-in procedural "neural-style" generator (always works)
   The sketch is analysed (strokes, enclosed regions, curvature), then each
   region is rendered as a stylised generative artwork: gradient skies,
   textured fills, glow edges, particles, palette harmonisation... every
   generation is unique (seeded randomness).
"""

import hashlib
import math
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
def generate_image(sketch_bgr: np.ndarray, seed: Optional[int] = None,
                   use_controlnet: bool = False) -> np.ndarray:
    """Turn a sketch (white bg, black lines) into a generated image."""
    if seed is None:
        digest = hashlib.md5(sketch_bgr.tobytes()).hexdigest()
        seed = int(digest[:8], 16) ^ random.randrange(1 << 31)

    if use_controlnet:
        img = _try_controlnet(sketch_bgr, seed)
        if img is not None:
            return img

    return _procedural_generate(sketch_bgr, seed)


# --------------------------------------------------------------------- #
# Optional ControlNet backend
# --------------------------------------------------------------------- #
def _try_controlnet(sketch_bgr: np.ndarray, seed: int) -> Optional[np.ndarray]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        pipe_cache = getattr(_try_controlnet, "_pipe", None)
        if pipe_cache is None:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet, torch_dtype=torch.float16)
            pipe.enable_model_cpu_offload()
            _try_controlnet._pipe = pipe
            pipe_cache = pipe

        from PIL import Image
        gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
        pil = Image.fromarray(gray).resize((512, 512))
        out = pipe(prompt="masterpiece, vibrant digital art",
                   image=pil, num_inference_steps=18,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# --------------------------------------------------------------------- #
# Procedural generative engine
# --------------------------------------------------------------------- #
_PALETTES = [
    [(255, 94, 77), (255, 195, 113), (72, 52, 212)],     # sunset pop
    [(0, 210, 190), (12, 94, 110), (236, 242, 141)],     # lagoon
    [(240, 90, 160), (90, 40, 120), (255, 220, 120)],    # neon night
    [(120, 200, 87), (40, 120, 60), (250, 230, 160)],    # meadow
    [(250, 160, 60), (200, 60, 40), (255, 235, 205)],    # ember
    [(150, 120, 250), (60, 60, 160), (255, 180, 220)],   # dreamwave
]


def _procedural_generate(sketch: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    h, w = sketch.shape[:2]
    gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    lines = (gray < 128).astype(np.uint8)

    palette = rng.choice(_PALETTES) + rng.choice(_PALETTES)
    rng.shuffle(palette)

    canvas = _gradient_background(w, h, palette, rng)

    regions = _find_regions(lines)
    for i, mask in enumerate(regions):
        color = np.array(palette[i % len(palette)], dtype=np.float32)
        canvas = _paint_region(canvas, mask, color, rng)

    canvas = _draw_line_art(canvas, lines, palette, rng)
    canvas = _add_particles(canvas, lines, palette, rng)
    canvas = _post_process(canvas, rng)
    return canvas


def _gradient_background(w, h, palette, rng) -> np.ndarray:
    c1 = np.array(palette[rng.randrange(len(palette))], np.float32)
    c2 = np.array(palette[rng.randrange(len(palette))], np.float32)
    angle = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx * math.cos(angle) + yy * math.sin(angle))
    t = (t - t.min()) / max(np.ptp(t), 1e-6)
    t = t[..., None]
    img = c1 * (1 - t) + c2 * t
    noise = np.random.default_rng(rng.randrange(1 << 31)).normal(0, 6, (h, w, 1))
    return np.clip(img + noise.astype(np.float32), 0, 255).astype(np.uint8)


def _find_regions(lines: np.uint8, min_area: int = 400) -> List[np.ndarray]:
    """Fill enclosed contours of the sketch and return per-region masks."""
    dilated = cv2.dilate(lines, np.ones((5, 5), np.uint8))
    inv = cv2.bitwise_not(dilated)
    n, labels = cv2.connectedComponents(inv)
    masks = []
    for lab in range(1, n):
        mask = (labels == lab).astype(np.uint8)
        area = int(mask.sum())
        if area >= min_area:
            ys, xs = np.where(mask)
            touches_border = (ys.min() == 0 or xs.min() == 0 or
                              ys.max() == mask.shape[0] - 1 or
                              xs.max() == mask.shape[1] - 1)
            if not touches_border or area < 0.6 * mask.shape[0] * mask.shape[1]:
                masks.append(cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    masks.sort(key=lambda m: -int(m.sum()))
    return masks[:12]


def _paint_region(canvas: np.ndarray, mask: np.ndarray,
                  color: np.float32, rng: random.Random) -> np.ndarray:
    ys, xs = np.where(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = max(math.sqrt(int(mask.sum()) / math.pi), 1.0)

    # radial gradient fill with slight hue jitter
    jitter = np.array([rng.uniform(-25, 25)] * 3, np.float32)
    inner = np.clip(color + 60 + jitter, 0, 255)
    outer = np.clip(color - 30 + jitter, 0, 255)

    yy, xx = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    t = np.clip(d, 0, 1)[..., None]
    fill = inner * (1 - t) + outer * t

    style = rng.choice(["halftone", "stripes", "solid"])
    if style == "halftone":
        step = rng.randint(8, 14)
        grid_y, grid_x = np.mgrid[0:canvas.shape[0]:step, 0:canvas.shape[1]:step]
        dots = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2) < radius ** 2
        pattern = np.zeros_like(canvas, np.float32)
        for gy, gx, keep in zip(grid_y.ravel(), grid_x.ravel(), dots.ravel()):
            if keep:
                r = rng.uniform(1.5, step / 2.6)
                cv2.circle(pattern, (int(gx), int(gy)), int(r),
                           tuple(float(v) for v in inner), -1)
        fill = np.where(pattern > 0, pattern, fill * 0.85)
    elif style == "stripes":
        ang = rng.uniform(0, math.pi)
        yy2, xx2 = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]].astype(np.float32)
        band = np.sin(xx2 * math.cos(ang) + yy2 * math.sin(ang) / rng.uniform(.4, 1.)
                      * rng.uniform(0.05, 0.12) * 100) > 0
        fill = np.where(band[..., None], fill, fill * 0.75)

    m = mask[..., None].astype(np.float32) * rng.uniform(0.55, 0.85)
    out = canvas.astype(np.float32)
    blended = out * (1 - m) + fill * m
    return np.clip(blended, 0, 255).astype(np.uint8)


def _draw_line_art(canvas: np.ndarray, lines: np.uint8,
                   palette, rng: random.Random) -> np.ndarray:
    """Render sketch strokes as glowing dark ink over the artwork."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glow_mask = cv2.dilate(lines, kernel, iterations=2).astype(np.float32)
    glow_color = np.array(palette[rng.randrange(len(palette))], np.float32) * 1.2

    img = canvas.astype(np.float32)
    g = glow_mask[..., None] * rng.uniform(0.25, 0.45)
    img = img * (1 - g) + glow_color * g

    ink = np.clip(np.mean(palette, axis=0) * 0.25, 10, 60)
    lm = lines[..., None].astype(np.float32)
    img = img * (1 - lm) + ink * lm
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_particles(canvas: np.ndarray, lines: np.uint8,
                   palette, rng: random.Random) -> np.ndarray:
    ys, xs = np.where(lines > 0)
    if len(xs) == 0:
        return canvas
    out = canvas.copy()
    for _ in range(rng.randint(40, 90)):
        i = rng.randrange(len(xs))
        px, py = int(xs[i]) + rng.randint(-30, 30), int(ys[i]) + rng.randint(-30, 30)
        color = palette[rng.randrange(len(palette))]
        cv2.circle(out, (px, py), rng.randint(1, 4),
                   tuple(int(c) for c in color), -1, cv2.LINE_AA)
    return out


def _post_process(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # subtle vignette + saturation boost => "generated art" look
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    vignette = 1 - np.clip((d - 0.35) * 0.9, 0, 0.45)
    img = img.astype(np.float32) * vignette[..., None]

    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int32) + 25, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
