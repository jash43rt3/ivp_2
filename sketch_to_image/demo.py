"""Headless demo: simulate finger strokes -> sketch -> generated images.

Run:  python demo.py            (writes files into ./output/)
      python demo.py --ui       (also renders the full HUD/composited view
                                 of main.py onto fake frames and saves a
                                 screenshot - no webcam needed)
"""

import argparse
import math
import os

import cv2
import numpy as np

from generator import generate_image, make_gallery
from sketch_canvas import SketchCanvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


# --------------------------------------------------------------------- #
def draw_house(canvas: SketchCanvas):
    """Simulated finger strokes: a house with roof, door, sun, ground."""
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


def draw_face(canvas: SketchCanvas):
    """A smiley face with rays - second sample sketch."""
    w, h = canvas.width, canvas.height

    def stroke(pts):
        canvas.begin_stroke(pts[0])
        for p in pts[1:]:
            canvas.add_point(p)
        canvas.end_stroke()

    def circle(cx, cy, r, a0=0, a1=360):
        stroke([(int(cx + r * math.cos(math.radians(a))),
                 int(cy + r * math.sin(math.radians(a))))
                for a in range(a0, a1 + 1, 6)])

    cx, cy, r = w * .5, h * .5, h * .3
    circle(cx, cy, r)
    circle(cx - r * .35, cy - r * .25, r * .10)     # left eye
    circle(cx + r * .35, cy - r * .25, r * .10)     # right eye
    stroke([(int(cx - r * .4 + i * 2),
             int(cy + r * .25 + 18 * math.sin(i / 2 * math.pi / 40)))
            for i in range(40)])                     # smile
    for a in range(0, 360, 45):                     # sun rays
        x0 = cx + (r + 14) * math.cos(math.radians(a))
        y0 = cy + (r + 14) * math.sin(math.radians(a))
        x1 = cx + (r + 46) * math.cos(math.radians(a))
        y1 = cy + (r + 46) * math.sin(math.radians(a))
        stroke([(int(x0), int(y0)), (int(x1), int(y1))])


# --------------------------------------------------------------------- #
def render_ui_screenshot(app, frame):
    """Push one fake frame through the real App pipeline and grab the view."""
    from dataclasses import dataclass, field
    from hand_tracker import HandState

    state = HandState(landmarks=(640, 300), thumb_tip=(660, 330),
                      middle_tip=(680, 320), pinching=True, hand_found=True,
                      trail=[(500 + 10 * i, 300 + 6 * math.sin(i))
                             for i in range(14)])
    app.drawing = True
    view = app._compose_view(frame, state)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", action="store_true",
                    help="also render a mock screenshot of the live UI")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    np.random.seed(42)

    # --- sketches ------------------------------------------------------
    house = SketchCanvas(960, 720)
    draw_house(house)
    face = SketchCanvas(960, 720)
    draw_face(face)

    house_sketch = house.render()
    face_sketch = face.render()
    cv2.imwrite(os.path.join(OUT, "demo_sketch.png"), house_sketch)
    cv2.imwrite(os.path.join(OUT, "demo_sketch_2.png"), face_sketch)

    # --- generations (multiple seeds => unique artworks) ----------------
    arts = [generate_image(house_sketch, seed=s) for s in (7, 21, 99)]
    arts += [generate_image(face_sketch, seed=s) for s in (5, 42)]
    for i, art in enumerate(arts, 1):
        cv2.imwrite(os.path.join(OUT, f"demo_art_{i}.png"), art)

    cv2.imwrite(os.path.join(OUT, "demo_side_by_side.png"),
                np.hstack([house_sketch, arts[0]]))
    cv2.imwrite(os.path.join(OUT, "demo_gallery.png"),
                make_gallery(arts, cols=3, cell=(480, 360)))

    # --- optional mock UI screenshot ------------------------------------
    if args.ui:
        import main as app_main
        from hand_tracker import HandTracker
        app = app_main.App.__new__(app_main.App)   # skip webcam initialisation
        app.tracker = HandTracker()
        app.width, app.height = 1280, 720
        app.canvas = _rescaled(house)
        app.drawing = False
        app.pinch_frames = 6
        app.last_generated = cv2.resize(arts[0], (1280, 720))
        app.show_result_until = 0.0
        app.flash_until = 0.0
        app.use_controlnet = False
        app.generation_count = 3
        app.show_help = True
        app.help_until = math.inf
        app.fps = 29.7
        bg = _fake_camera_bg(1280, 720)
        shot = render_ui_screenshot(app, bg)
        cv2.imwrite(os.path.join(OUT, "demo_ui.png"), shot)
        print("Saved: output/demo_ui.png")

    print("Saved: demo_sketch.png, demo_sketch_2.png, demo_art_1..5.png, "
          "demo_side_by_side.png, demo_gallery.png")


def _rescaled(canvas: SketchCanvas) -> SketchCanvas:
    c = SketchCanvas(1280, 720)
    c.strokes = [[(int(x * 1280 / canvas.width), int(y * 720 / canvas.height))
                  for (x, y) in s] for s in canvas.strokes]
    return c


def _fake_camera_bg(w, h):
    """Blurred gradient with noise to imitate a webcam image."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 90 + 40 * np.sin(xx / 300) + yy / 12
    img[..., 1] = 70 + yy / 9
    img[..., 2] = 60 + 30 * np.cos(yy / 200)
    rng = np.random.default_rng(1)
    img += rng.normal(0, 8, (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 21)


if __name__ == "__main__":
    main()
