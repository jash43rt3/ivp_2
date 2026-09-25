"""Shared visual theme: colours, fonts and HUD drawing helpers.

All on-screen UI in main.py is rendered through these helpers so the app
has one consistent, modern look (dark glass panels, neon accents).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]  # BGR

# ------------------------------------------------------------------ #
# Palette (BGR)
# ------------------------------------------------------------------ #
INK        = (250, 245, 235)   # warm white
MUTED      = (170, 160, 150)
ACCENT     = (255, 95, 60)     # electric coral
ACCENT_SOFT= (150, 80, 60)
CYAN       = (235, 180, 80)    # teal accent
GOLD       = (80, 200, 255)
DANGER     = (70, 70, 235)
PANEL      = (32, 26, 22)      # dark glass
PANEL_EDGE = (70, 60, 55)


def overlay(frame: np.ndarray, rect: Tuple[int, int, int, int],
            color: Color, alpha: float):
    """Blend a solid colour over a rectangular region (x, y, w, h)."""
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def rounded_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  color: Color = PANEL, alpha: float = 0.72,
                  radius: int = 14, border: Optional[Color] = PANEL_EDGE):
    """Draw a translucent rounded rectangle 'glass' panel."""
    x = max(0, x); y = max(0, y)
    w = min(w, frame.shape[1] - x); h = min(h, frame.shape[0] - y)
    if w <= 4 or h <= 4:
        return
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.rectangle(mask, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(mask, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
        cv2.circle(mask, (cx, cy), radius, color, -1)
    region = frame[y:y + h, x:x + w]
    solid = np.zeros_like(region)
    solid[:] = color
    m = mask[y:y + h, x:x + w]
    blended = cv2.addWeighted(solid, alpha, region, 1 - alpha, 0)
    region[m > 0] = blended[m > 0]
    if border is not None:
        cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), border, 1)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), border, 1)
        for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                       (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
            cv2.circle(frame, (cx, cy), radius, border, 1)


def text(frame: np.ndarray, s: str, org: Tuple[int, int], scale: float = 0.6,
         color: Color = INK, weight: int = 1, anchor_left: bool = True,
         shadow: bool = True):
    """Draw crisp text with an optional soft shadow; returns end-x position."""
    x, y = org
    if shadow:
        cv2.putText(frame, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, weight, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    return x + tw if anchor_left else x - tw


def centered_text(frame: np.ndarray, s: str, cx: int, y: int, scale: float = 0.6,
                  color: Color = INK, weight: int = 1):
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    text(frame, s, (cx - tw // 2, y), scale, color, weight)


def pill(frame: np.ndarray, label: str, org: Tuple[int, int],
         fg: Color = INK, bg: Color = PANEL, pad: int = 10, scale: float = 0.55,
         dot: Optional[Color] = None):
    """Small rounded badge like `● DRAW`. Returns its width."""
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    w = pad * 2 + tw + (14 if dot else 0)
    h = th + pad * 2
    x, y = org
    rounded_panel(frame, x, y, w, h, bg, alpha=0.85, radius=h // 2, border=None)
    cx = x + pad
    if dot is not None:
        cv2.circle(frame, (cx + 4, y + h // 2), 4, dot, -1, cv2.LINE_AA)
        cx += 14
    text(frame, label, (cx, y + h // 2 + th // 2), scale, fg, 1)
    return w


def progress_ring(frame: np.ndarray, center: Tuple[int, int], radius: int,
                  frac: float, color: Color = ACCENT, bg: Color = (60, 50, 50),
                  thickness: int = 4):
    """Circular countdown ring (used for the pinch-to-generate gesture)."""
    cv2.circle(frame, center, radius, bg, thickness, cv2.LINE_AA)
    if frac > 0:
        cv2.ellipse(frame, center, (radius, radius), 90, 0,
                    360 * min(1.0, frac), color, thickness, cv2.LINE_AA)


def letterbox(frame: np.ndarray, size: Tuple[int, int],
              color: Color = (16, 12, 10)) -> np.ndarray:
    """Resize keeping aspect ratio, pad with background colour."""
    w, h = size
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), color, np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas
