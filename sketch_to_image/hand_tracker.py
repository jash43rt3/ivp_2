"""Hand tracking module using MediaPipe Hands.

Detects the index-finger tip for freehand air-drawing and the pinch
gesture (thumb + middle finger) as the trigger to generate an image.
Also renders a neon "energy" cursor and motion trail on the video frame.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

Point = Tuple[int, int]


@dataclass
class HandState:
    """Result of one frame analysis."""
    landmarks: Optional[Point] = None          # index fingertip (px coords)
    thumb_tip: Optional[Point] = None
    middle_tip: Optional[Point] = None
    pinching: bool = False                     # thumb touching middle finger
    hand_found: bool = False
    trail: List[Point] = field(default_factory=list)


class HandTracker:
    INDEX_TIP = 8
    THUMB_TIP = 4
    MIDDLE_TIP = 12
    # Pinch distance threshold as a fraction of hand size (wrist->middle MCP).
    PINCH_RATIO = 0.35
    TRAIL_LENGTH = 16

    def __init__(self, max_hands: int = 1, model_complexity: int = 1):
        self.mp_hands = mp.solutions.hands
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.state = HandState()

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr: np.ndarray) -> HandState:
        """Analyse a BGR frame and return the current HandState."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        state = HandState()
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            ix, iy = int(lm[self.INDEX_TIP].x * w), int(lm[self.INDEX_TIP].y * h)
            tx, ty = int(lm[self.THUMB_TIP].x * w), int(lm[self.THUMB_TIP].y * h)
            mx, my = int(lm[self.MIDDLE_TIP].x * w), int(lm[self.MIDDLE_TIP].y * h)
            wrist_x, wrist_y = int(lm[0].x * w), int(lm[0].y * h)
            mcp_x, mcp_y = int(lm[9].x * w), int(lm[9].y * h)
            hand_size = max(np.hypot(mcp_x - wrist_x, mcp_y - wrist_y), 1.0)

            state.hand_found = True
            state.landmarks = (ix, iy)
            state.thumb_tip = (tx, ty)
            state.middle_tip = (mx, my)
            pinch_dist = np.hypot(tx - mx, ty - my)
            state.pinching = bool(pinch_dist < self.PINCH_RATIO * hand_size)

            if not state.pinching:
                state.trail = self.state.trail[-(self.TRAIL_LENGTH - 1):] + \
                    [(ix, iy)]
        self.state = state
        return state

    def release(self):
        self._hands.close()

    # ------------------------------------------------------------------ #
    def draw_overlay(self, frame: np.ndarray, state: HandState,
                     drawing: bool = False) -> np.ndarray:
        """Neon fingertip cursor + fading comet trail + pinch indicator."""
        from theme import ACCENT, CYAN, INK, GOLD
        if not state.hand_found:
            return frame

        # --- comet trail -------------------------------------------------
        pts = [(int(p[0]), int(p[1])) for p in state.trail]
        for i in range(len(pts) - 1):
            a = (i + 1) / len(pts)
            colour = tuple(int(c * a) for c in (CYAN if drawing else GOLD))
            thickness = max(1, int(5 * a))
            cv2.line(frame, pts[i], pts[i + 1], colour, thickness, cv2.LINE_AA)

        # --- pinch link between thumb & middle tips -----------------------
        if state.thumb_tip and state.middle_tip:
            link = ACCENT if state.pinching else (90, 90, 90)
            cv2.line(frame, state.thumb_tip, state.middle_tip, link,
                     2 if state.pinching else 1, cv2.LINE_AA)

        # --- cursor --------------------------------------------------------
        if state.landmarks is not None:
            x, y = state.landmarks
            color = ACCENT if state.pinching else (CYAN if drawing else INK)
            radius = 26 if state.pinching else 13
            # soft glow
            glow = np.zeros_like(frame)
            cv2.circle(glow, (x, y), radius + 6, color, -1, cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (0, 0), 12)
            cv2.addWeighted(glow, 0.45, frame, 1.0, 0, dst=frame)
            # ring + dot
            cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
            # crosshair ticks (N, E, S, W)
            r6 = radius + 6
            for (ax, ay, bx, by) in ((x + r6 - 4, y, x + r6 + 4, y),
                                     (x - r6 - 4, y, x - r6 + 4, y),
                                     (x, y + r6 - 4, x, y + r6 + 4),
                                     (x, y - r6 - 4, x, y - r6 + 4)):
                cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
        return frame
