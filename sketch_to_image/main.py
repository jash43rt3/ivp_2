"""Sketch-to-Image: draw in the air with your finger, get generative art.

Controls (press key while the video window is focused):
    PINCH (thumb + middle finger)  -> generate an image from your sketch
    c / C                         -> clear canvas
    u / U                         -> undo last stroke
    s / S                         -> save current generated gallery
    q / Q or ESC                  -> quit

Usage:
    python main.py                    # webcam (default)
    python main.py --source video.mp4 # process a video file instead
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import generate_image          # noqa: E402
from hand_tracker import HandTracker         # noqa: E402
from sketch_canvas import SketchCanvas       # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Drawing region inside the camera frame (avoids drawing when the hand
# is right at the edge). Coordinates are fractions of width/height.
ROI = (0.15, 0.08, 0.85, 0.92)


class App:
    def __init__(self, source, width=1280, height=720, use_controlnet=False):
        self.cap = self._open_source(source)
        self.width = width
        self.height = height
        self.tracker = HandTracker()
        self.canvas = SketchCanvas(width, height)
        self.drawing = False
        self.last_generated = None
        self.show_result_until = 0.0
        self.use_controlnet = use_controlnet
        self.generation_count = 0

    @staticmethod
    def _open_source(source):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}. "
                               "Check your webcam index or file path.")
        return cap

    # ------------------------------------------------------------------ #
    def _map_to_canvas(self, pt):
        """Map a fingertip position from the frame ROI to canvas coords."""
        x0, y0, x1, y1 = ROI
        fx = (pt[0] / self.width - x0) / (x1 - x0)
        fy = (pt[1] / self.height - y0) / (y1 - y0)
        cx = int(np.clip(fx, 0, 1) * (self.canvas.width - 1))
        cy = int(np.clip(fy, 0, 1) * (self.canvas.height - 1))
        return (cx, cy)

    def _generate(self):
        if not self.canvas.has_content:
            return
        self.canvas.end_stroke()
        sketch = self.canvas.render()
        t0 = time.time()
        art = generate_image(sketch, use_controlnet=self.use_controlnet)
        art = cv2.resize(art, (self.width, self.height))
        self.last_generated = art
        self.show_result_until = time.time() + 4.0
        self.generation_count += 1

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.join(OUTPUT_DIR, f"gen_{self.generation_count:03d}")
        cv2.imwrite(stem + "_sketch.png", sketch)
        cv2.imwrite(stem + "_art.png", art)
        print(f"[+] Generated #{self.generation_count} in "
              f"{time.time() - t0:.2f}s -> {stem}_art.png")

    # ------------------------------------------------------------------ #
    def _handle_frame(self, frame):
        state = self.tracker.process(frame)

        if state.landmarks is not None and not state.pinching:
            cpt = self._map_to_canvas(state.landmarks)
            if not self.drawing:
                self.drawing = True
                self.canvas.begin_stroke(cpt)
            else:
                self.canvas.add_point(cpt)
        else:
            if self.drawing:
                self.drawing = False
                self.canvas.end_stroke()

        if state.pinching and self.drawing:
            self.drawing = False
            self.canvas.end_stroke()

        # pinch must be held for a few frames, then fires once until released
        if state.pinching:
            self._pinch_frames = getattr(self, "_pinch_frames", 0) + 1
            if self._pinch_frames == 12:      # ~0.4 s hold -> single trigger
                self._generate()
        else:
            self._pinch_frames = 0

        return state

    def _compose_view(self, frame, state):
        view = self.tracker.draw_overlay(frame.copy(), state)
        sketch = self.canvas.render()
        small_sketch = cv2.resize(sketch, (self.width // 3, self.height // 3))
        h, w = small_sketch.shape[:2]
        view[h - 10:view.shape[0] - 10, 10:w + 10] = small_sketch
        cv2.rectangle(view, (9, h - 11), (w + 11, view.shape[0] - 9),
                      (255, 255, 255), 2)

        now = time.time()
        if self.last_generated is not None and now < self.show_result_until:
            alpha = 0.85
            view = cv2.addWeighted(view, 1 - alpha, self.last_generated, alpha, 0)
            cv2.putText(view, "AI GENERATED!", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        self._draw_hud(view, state)
        return view

    @staticmethod
    def _draw_hud(view, state):
        cv2.putText(view, "Draw: raise index finger | Pinch(thumb+middle): GENERATE",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(view, "Draw: raise index finger | Pinch(thumb+middle): GENERATE",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        status = "PINCH -> generating..." if state.pinching else \
            ("DRAWING" if state.landmarks else "show your hand to camera")
        color = (0, 0, 255) if state.pinching else (0, 255, 0)
        cv2.putText(view, status, (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(view, "[c]lear  [u]ndo  [q]uit",
                    (view.shape[1] - 250, view.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ------------------------------------------------------------------ #
    def run(self):
        print(__doc__)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[i] End of video source.")
                break
            frame = cv2.resize(cv2.flip(frame, 1), (self.width, self.height))

            state = self._handle_frame(frame)
            view = self._compose_view(frame, state)
            cv2.imshow("Sketch to Generative Image - Air Drawing", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self.canvas.clear()
                self.drawing = False
            elif key == ord('u'):
                self.canvas.undo()
            elif key == ord('g'):      # manual generate (no pinch needed)
                self._generate()

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Air-drawing -> generative images")
    p.add_argument("--source", default="0",
                   help="webcam index (e.g. 0) or video file path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--controlnet", action="store_true",
                   help="use Stable-Diffusion ControlNet backend "
                        "(requires HF_TOKEN + diffusers installed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    App(args.source, args.width, args.height, args.controlnet).run()
