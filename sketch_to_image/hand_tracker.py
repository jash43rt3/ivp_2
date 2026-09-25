"""Hand tracking module using MediaPipe Hands.

Detects index-finger tip for freehand drawing and the pinch gesture
(thumb + middle finger) as a trigger to generate an image from the sketch.
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
    trail: List[Point] = field(default_factory=list)


class HandTracker:
    INDEX_TIP = 8
    THUMB_TIP = 4
    MIDDLE_TIP = 12
    # Pinch distance threshold as a fraction of hand size (wrist->middle MCP).
    PINCH_RATIO = 0.35
    TRAIL_LENGTH = 12

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

            state.landmarks = (ix, iy)
            state.thumb_tip = (tx, ty)
            state.middle_tip = (mx, my)
            pinch_dist = np.hypot(tx - mx, ty - my)
            state.pinching = bool(pinch_dist < self.PINCH_RATIO * hand_size)

            state.trail = self.state.trail[-(self.TRAIL_LENGTH - 1):] + [(ix, iy)] \
                if not state.pinching else []

        self.state = state
        return state

    def release(self):
        self._hands.close()

    def draw_overlay(self, frame: np.ndarray, state: HandState) -> np.ndarray:
        """Draw cursor circle, motion trail and pinch indicator on the frame."""
        overlay = frame
        if state.landmarks is not None:
            color = (0, 255, 0) if not state.pinching else (0, 0, 255)
            radius = 22 if state.pinching else 10
            cv2.circle(overlay, state.landmarks, radius, color, 2)
            cv2.circle(overlay, state.landmarks, 3, color, -1)
            for i, pt in enumerate(state.trail[:-1]):
                alpha = (i + 1) / len(state.trail)
                cv2.line(overlay, pt, state.trail[i + 1],
                         (int(255 * alpha), 200, int(255 * alpha)), 2)
        return overlay
