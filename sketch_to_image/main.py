"""Sketch-to-Image: draw in the air with your finger, get generative art.

Gestures:
    Raise index finger + move   -> draw a stroke in mid-air
    Pinch (thumb + middle) ~0.4s -> generate an image from your sketch

Controls (press key while the video window is focused):
    g / G   -> generate now (no pinch needed)
    u / U   -> undo last stroke
    c / C   -> clear canvas
    h / H   -> toggle help panel
    q / Q or ESC -> quit

Usage:
    python main.py                     # webcam 0
    python main.py --source 1          # another webcam
    python main.py --source clip.mp4   # process a video file instead
    python main.py --controlnet        # real SD ControlNet backend
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme                                   # noqa: E402
from generator import generate_image, make_gallery  # noqa: E402
from hand_tracker import HandTracker           # noqa: E402
from sketch_canvas import SketchCanvas         # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Drawing region inside the camera frame (avoids drawing when the hand
# is right at the edge). Coordinates are fractions of width/height.
ROI = (0.15, 0.10, 0.85, 0.90)
PINCH_HOLD_FRAMES = 12          # ~0.4 s at 30 fps


class App:
    def __init__(self, source, width=1280, height=720, use_controlnet=False):
        self.cap = self._open_source(source)
        fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        if fw != width or fh != height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        self.tracker = HandTracker()
        self.canvas = SketchCanvas(width, height)
        self.drawing = False
        self.pinch_frames = 0
        self.last_generated = None
        self.gallery = []
        self.show_result_until = 0.0
        self.flash_until = 0.0
        self.use_controlnet = use_controlnet
        self.generation_count = 0
        self.show_help = True
        self.help_until = time.time() + 6.0
        self.fps = 0.0
        self._last_t = time.time()

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

    # ------------------------------------------------------------------ #
    def _generate(self):
        if not self.canvas.has_content:
            return
        self.canvas.end_stroke()
        sketch = self.canvas.render()
        t0 = time.time()
        art = generate_image(sketch, use_controlnet=self.use_controlnet)
        art = cv2.resize(art, (self.width, self.height))
        self.last_generated = art
        self.gallery.append(art)
        now = time.time()
        self.show_result_until = now + 4.0
        self.flash_until = now + 0.35
        self.generation_count += 1

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.join(OUTPUT_DIR, f"gen_{self.generation_count:03d}")
        cv2.imwrite(stem + "_sketch.png", sketch)
        cv2.imwrite(stem + "_art.png", art)
        print(f"[+] Generated #{self.generation_count} in "
              f"{time.time() - t0:.2f}s -> {stem}_art.png")

    def _save_gallery(self):
        if not self.gallery:
            print("[i] Nothing generated yet.")
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "gallery.png")
        cv2.imwrite(path, make_gallery(self.gallery))
        print(f"[+] Gallery ({len(self.gallery)} artworks) saved -> {path}")

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
            self.pinch_frames += 1
            if self.pinch_frames == PINCH_HOLD_FRAMES:
                self._generate()
        else:
            self.pinch_frames = 0

        return state

    # ------------------------------------------------------------------ #
    def _compose_view(self, frame, state):
        view = self.tracker.draw_overlay(frame.copy(), state,
                                         drawing=self.drawing)
        self._draw_roi(view)
        self._draw_sketch_panel(view)
        self._draw_hud(view, state)

        now = time.time()
        # white flash on generation
        dt_left = self.flash_until - now
        if dt_left > 0:
            alpha = min(1.0, dt_left / 0.35) * 0.9
            theme.overlay(view, (0, 0, self.width, self.height),
                          (255, 255, 255), alpha)
        # result reveal: artwork slides up over the feed
        reveal_left = self.show_result_until - now
        if self.last_generated is not None and reveal_left > 0:
            total = 4.0
            p = 1.0 - reveal_left / total
            if p < 0.15:                       # ease-in slide
                k = (p / 0.15) ** 2
                y_off = int((1 - k) * self.height)
                blended = cv2.addWeighted(view, 0.15,
                                          self.last_generated, 0.85, 0)
                view[y_off:, :] = blended[y_off:, :]
                self._draw_result_card(view)
        return view

    def _draw_roi(self, view):
        """Subtle corner brackets showing the active drawing area."""
        x0, y0, x1, y1 = ROI
        w, h = self.width, self.height
        px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        L = 28
        col = theme.ACCENT_SOFT if not self.drawing else theme.CYAN
        for (cx, cy, dx, dy) in ((px0, py0, 1, 1), (px1, py0, -1, 1),
                                 (px0, py1, 1, -1), (px1, py1, -1, -1)):
            cv2.line(view, (cx, cy), (cx + dx * L, cy), col, 2, cv2.LINE_AA)
            cv2.line(view, (cx, cy), (cx, cy + dy * L), col, 2, cv2.LINE_AA)

    def _draw_sketch_panel(self, view):
        """Live mini-canvas bottom-left: glass panel + current strokes."""
        pw, ph = self.width // 4, int(self.height / 3.4)
        m = 16
        x, y = m, self.height - ph - m
        theme.rounded_panel(view, x - 6, y - 30, pw + 12, ph + 36,
                            theme.PANEL, alpha=0.78)
        theme.text(view, "SKETCH", (x + 2, y - 10), 0.5, theme.MUTED, 1)
        count = f"{self.canvas.stroke_count} strokes"
        tw, _ = cv2.getTextSize(count, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        theme.text(view, count, (x + pw - tw, y - 10), 0.45, theme.MUTED, 1)

        sketch = self.canvas.render()
        small = cv2.resize(sketch, (pw, ph), interpolation=cv2.INTER_AREA)
        # invert: show sketch as glowing lines on dark panel
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ink = (255 - np.clip(gray.astype(np.int32) * 2 - 120, 0, 255)
               ).astype(np.uint8)
        glow = cv2.merge([np.clip(ink.astype(np.int32) + 40, 0, 255)
                          .astype(np.uint8), ink // 2, ink // 3])
        view[y:y + ph, x:x + pw] = glow
        cv2.rectangle(view, (x, y), (x + pw, y + ph), theme.PANEL_EDGE, 1)

    def _draw_result_card(self, view):
        """Caption card shown while the generated artwork is displayed."""
        w, h = self.width, self.height
        card_w, card_h = 420, 96
        x, y = (w - card_w) // 2, h - card_h - 34
        theme.rounded_panel(view, x, y, card_w, card_h, theme.PANEL,
                            alpha=0.88)
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
        dot = tuple(int(c * (0.6 + 0.4 * pulse)) for c in theme.ACCENT)
        cv2.circle(view, (x + 30, y + card_h // 2), 9, dot, -1, cv2.LINE_AA)
        theme.text(view, "GENERATED!", (x + 52, y + 40), 1.0, theme.INK, 2)
        engine = "ControlNet SD" if self.use_controlnet else "procedural engine"
        theme.text(view, f"#{self.generation_count}  -  {engine}  -  "
                         f"saved to output/", (x + 52, y + 70), 0.5,
                   theme.MUTED, 1)

    def _draw_hud(self, view, state):
        w = self.width
        # ---- top bar ------------------------------------------------------
        theme.rounded_panel(view, 0, 0, w, 54, theme.PANEL, alpha=0.66,
                            radius=0, border=None)
        theme.text(view, "SKETCH > IMAGE", (16, 35), 0.75, theme.ACCENT, 2)
        theme.text(view, "air-drawing studio", (185, 35), 0.45, theme.MUTED, 1)

        # status pill (right of title)
        if state.pinching:
            label, dot = "PINCH HOLD", theme.ACCENT
        elif self.drawing:
            label, dot = "DRAWING", theme.CYAN
        elif state.hand_found:
            label, dot = "READY - RAISE INDEX FINGER", theme.GOLD
        else:
            label, dot = "SHOW YOUR HAND", theme.MUTED
        pw = theme.pill(view, label, (w - 320, 13), dot=dot)
        # FPS badge
        theme.text(view, f"{self.fps:4.1f} fps", (w - 320 + pw + 12, 35),
                   0.5, theme.MUTED, 1)

        # ---- pinch progress ring around cursor ---------------------------
        if state.pinching and state.landmarks:
            frac = self.pinch_frames / PINCH_HOLD_FRAMES
            theme.progress_ring(view, state.landmarks, 34, frac,
                                color=theme.ACCENT, thickness=5)

        # ---- bottom-right control hints ----------------------------------
        hints = [("g", "generate"), ("u", "undo"), ("c", "clear"),
                 ("h", "help"), ("q", "quit")]
        hint = "  ".join(f"[{k}] {v}" for k, v in hints)
        tw, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        theme.rounded_panel(view, w - tw - 44, self.height - 44, tw + 32, 32,
                            theme.PANEL, alpha=0.7, radius=16, border=None)
        theme.text(view, hint, (w - tw - 28, self.height - 22), 0.5,
                   theme.INK, 1)

        # ---- help overlay --------------------------------------------------
        if self.show_help or time.time() < self.help_until:
            self._draw_help(view)

    def _draw_help(self, view):
        w, h = self.width, self.height
        pw, ph = 460, 250
        x, y = w - pw - 18, 66
        theme.rounded_panel(view, x, y, pw, ph, theme.PANEL, alpha=0.85)
        theme.text(view, "HOW IT WORKS", (x + 20, y + 34), 0.7,
                   theme.ACCENT, 2)
        rows = [
            ("1.", "Raise your INDEX FINGER and move to draw."),
            ("2.", "Pinch THUMB + MIDDLE finger (~0.4 s)"),
            ("",  "to turn your sketch into generative art."),
            ("",  "Every generation is unique."),
            ("",  "Artworks are saved under output/."),
        ]
        yy = y + 66
        for num, line in rows:
            if num:
                theme.text(view, num, (x + 20, yy), 0.55, theme.CYAN, 1)
            theme.text(view, line, (x + 46, yy), 0.55, theme.INK, 1)
            yy += 30
        theme.text(view, "press [h] to hide", (x + 20, y + ph - 16), 0.45,
                   theme.MUTED, 1)

    # ------------------------------------------------------------------ #
    def run(self):
        print(__doc__)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[i] End of video source.")
                break
            frame = cv2.resize(cv2.flip(frame, 1), (self.width, self.height))

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self._last_t,
                                                         1e-3))
            self._last_t = now

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
            elif key == ord('g'):
                self._generate()
            elif key == ord('s'):
                self._save_gallery()
            elif key == ord('h'):
                self.show_help = not self.show_help

        self._save_gallery()
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
