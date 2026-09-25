"""Headless demo: simulate finger strokes -> sketch -> generated image.

Run:  python demo.py
Outputs files into ./output/ so you can verify the pipeline without a webcam.
"""

import math
import os

import cv2
import numpy as np

from generator import generate_image
from sketch_canvas import SketchCanvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def fake_house(canvas: SketchCanvas):
    w, h = canvas.width, canvas.height

    def line(p1, p2, jitter=2.0):
        pts = []
        n = 40
        for i in range(n + 1):
            t = i / n
            x = p1[0] + (p2[0] - p1[0]) * t + np.random.uniform(-jitter, jitter)
            y = p1[1] + (p2[1] - p1[1]) * t + np.random.uniform(-jitter, jitter)
            pts.append((int(x), int(y)))
        canvas.begin_stroke(pts[0])
        for pt in pts[1:]:
            canvas.add_point(pt)
        canvas.end_stroke()

    def circle(cx, cy, r):
        pts = [(int(cx + r * math.cos(a / 180 * math.pi)),
                int(cy + r * math.sin(a / 180 * math.pi)))
               for a in range(0, 365, 6)]
        canvas.begin_stroke(pts[0])
        for pt in pts[1:]:
            canvas.add_point(pt)
        canvas.end_stroke()

    # house body, roof, door, window, sun, ground
    line((w * .30, h * .55), (w * .30, h * .85))
    line((w * .30, h * .85), (w * .70, h * .85))
    line((w * .70, h * .85), (w * .70, h * .55))
    line((w * .70, h * .55), (w * .30, h * .55))
    line((w * .25, h * .55), (w * .50, h * .30))
    line((w * .50, h * .30), (w * .75, h * .55))
    line((w * .45, h * .85), (w * .45, h * .65))
    line((w * .45, h * .65), (w * .55, h * .65))
    line((w * .55, h * .65), (w * .55, h * .85))
    circle(w * .82, h * .18, h * .08)
    line((w * .05, h * .85), (w * .95, h * .85))


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(42)

    canvas = SketchCanvas(960, 720)
    fake_house(canvas)

    sketch = canvas.render()
    art = generate_image(sketch, seed=7)

    cv2.imwrite(os.path.join(OUT, "demo_sketch.png"), sketch)
    cv2.imwrite(os.path.join(OUT, "demo_generated.png"), art)

    side = np.hstack([sketch, art])
    cv2.imwrite(os.path.join(OUT, "demo_side_by_side.png"), side)
    print("Saved: output/demo_sketch.png, output/demo_generated.png, "
          "output/demo_side_by_side.png")


if __name__ == "__main__":
    main()
