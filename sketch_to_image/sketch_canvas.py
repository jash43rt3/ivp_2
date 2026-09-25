"""Virtual whiteboard where finger strokes are collected and rendered.

Strokes are stored as raw point lists; rendering smooths them with a
Catmull-Rom spline and draws them as tapered, pressure-like ink lines so
even shaky air-drawing looks intentional and artistic.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Stroke = List[Point]


class SketchCanvas:
    """Stores strokes; each stroke is a polyline captured while the user
    keeps the index finger raised (drawing mode)."""

    def __init__(self, width: int, height: int, bg_color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.strokes: List[Stroke] = []
        self.bg_color = bg_color
        self._current: Optional[Stroke] = None

    # ------------------------------------------------------------------ #
    @property
    def has_content(self) -> bool:
        return bool(self.strokes) or self._current is not None

    @property
    def stroke_count(self) -> int:
        return len(self.strokes) + (1 if self._current else 0)

    def begin_stroke(self, pt: Point):
        self._current = [pt]

    def add_point(self, pt: Point):
        if self._current is None:
            self.begin_stroke(pt)
        elif self._current[-1] != pt:
            self._current.append(pt)

    def end_stroke(self) -> Optional[Stroke]:
        if self._current:
            self.strokes.append(self._current)
            stroke = self._current
            self._current = None
            return stroke
        return None

    def undo(self):
        if self.strokes:
            self.strokes.pop()

    def clear(self):
        self.strokes.clear()
        self._current = None

    # ------------------------------------------------------------------ #
    def render(self, line_thickness: int = 6, smooth: bool = True) -> np.ndarray:
        """Return the sketch as a BGR image (white background, black ink)."""
        canvas = np.full((self.height, self.width, 3), self.bg_color,
                         dtype=np.uint8)
        all_strokes = list(self.strokes)
        if self._current:
            all_strokes.append(self._current)

        for stroke in all_strokes:
            self._draw_stroke(canvas, stroke, line_thickness, smooth)
        return canvas

    def _draw_stroke(self, img: np.ndarray, stroke: Stroke,
                     thickness: int, smooth: bool):
        pts = np.array(stroke, dtype=np.int32)
        if len(pts) < 2:
            cv2.circle(img, tuple(pts[0]), max(thickness // 2, 2),
                       (0, 0, 0), -1, cv2.LINE_AA)
            return
        if smooth and len(pts) >= 3:
            pts = self._smooth_polyline(pts)

        # Tapered ink: thin at the ends of the stroke, full weight inside.
        n = len(pts)
        for i in range(n - 1):
            t = i / (n - 1)
            taper = 0.45 + 0.55 * min(t, 1 - t) / 0.25
            w = max(2, int(round(thickness * min(taper, 1.0))))
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 0),
                     w, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), max(thickness // 2, 1),
                   (0, 0, 0), -1, cv2.LINE_AA)

    @staticmethod
    def _smooth_polyline(pts: np.ndarray, samples_per_segment: int = 4) -> np.ndarray:
        """Catmull-Rom style smoothing so finger strokes look natural."""
        pts = pts.astype(np.float32)
        extended = np.vstack([pts[:1], pts, pts[-1:]])
        out = []
        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i - 1:i + 3]
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                t2, t3 = t * t, t * t * t
                x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                           + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                           + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
                y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                           + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                           + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
                out.append((x, y))
        out.append(tuple(pts[-1]))
        return np.array(out, dtype=np.int32)
