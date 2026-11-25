"""
Eye Tracking Module for Drowsiness Detection
Calculates Eye Aspect Ratio (EAR) for blink and drowsiness detection.
"""

import time
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

from ..utils.config import EYE_CONFIG


class EyeTracker:
    """
    Eye tracking using Eye Aspect Ratio (EAR) calculation.

    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

    Where P1-P6 are the 6 eye landmark points:
        P1: outer corner, P2/P3: upper lid, P4: inner corner, P5/P6: lower lid

    Drowsiness is time based (eyes closed for `closed_seconds`), so behaviour
    does not depend on the camera frame rate. Every `update` accepts an explicit
    timestamp so video files can be processed faster than real time.
    """

    # A closure must span at least this many frames to count as a blink
    MIN_BLINK_FRAMES = 2

    def __init__(
        self,
        ear_threshold: Optional[float] = None,
        closed_seconds: Optional[float] = None,
        history_size: int = 5,
    ):
        self.ear_threshold = EYE_CONFIG.EAR_THRESHOLD if ear_threshold is None else ear_threshold
        self.closed_seconds = EYE_CONFIG.CLOSED_SECONDS if closed_seconds is None else closed_seconds

        self.left_eye_indices = EYE_CONFIG.LEFT_EYE_INDICES
        self.right_eye_indices = EYE_CONFIG.RIGHT_EYE_INDICES

        # State
        self.closed_frame_counter = 0
        self.closed_since: Optional[float] = None
        self.closed_duration = 0.0
        self.blink_counter = 0
        self.is_drowsy = False
        self.baseline_ear: Optional[float] = None

        self.ear_history = deque(maxlen=history_size)

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """
        Eye Aspect Ratio for one eye.

        Args:
            eye_points: 6 (x, y) points ordered P1..P6 as described above
        """
        pts = np.asarray(eye_points, dtype=np.float64)
        vertical_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_2 = np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])
        if horizontal == 0:
            return 0.0
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def get_ear(self, face_data) -> Tuple[float, float, float]:
        """Return (left_ear, right_ear, average_ear) for a FaceData."""
        left_ear = self.calculate_ear(face_data.get_landmarks_float(self.left_eye_indices))
        right_ear = self.calculate_ear(face_data.get_landmarks_float(self.right_eye_indices))
        return left_ear, right_ear, (left_ear + right_ear) / 2.0

    def get_smoothed_ear(self, current_ear: float) -> float:
        """Moving average over the last few frames."""
        self.ear_history.append(current_ear)
        return float(np.mean(self.ear_history))

    def calibrate(self, open_eye_ears: Sequence[float]) -> float:
        """
        Personalise the closed-eye threshold from EAR samples taken while the
        driver looks at the camera with eyes open (blinks are rejected by using
        the upper-middle percentile). Returns the new threshold.
        """
        values = np.asarray(list(open_eye_ears), dtype=np.float64)
        if values.size < 10:
            raise ValueError("need at least 10 EAR samples to calibrate")
        self.baseline_ear = float(np.percentile(values, 60))
        # Never stray far from the sane absolute range
        self.ear_threshold = float(np.clip(EYE_CONFIG.CALIBRATION_RATIO * self.baseline_ear, 0.12, 0.30))
        return self.ear_threshold

    def update(self, face_data, timestamp: Optional[float] = None) -> dict:
        """Update state with one frame's landmarks (timestamp in seconds)."""
        if timestamp is None:
            timestamp = time.time()
        left_ear, right_ear, avg_ear = self.get_ear(face_data)
        return self.update_ear(avg_ear, timestamp, left_ear, right_ear)

    def update_ear(
        self,
        avg_ear: float,
        timestamp: float,
        left_ear: Optional[float] = None,
        right_ear: Optional[float] = None,
    ) -> dict:
        """State machine core, driven directly by an EAR value (used by tests)."""
        smoothed_ear = self.get_smoothed_ear(avg_ear)
        eyes_closed = smoothed_ear < self.ear_threshold

        if eyes_closed:
            if self.closed_since is None:
                self.closed_since = timestamp
            self.closed_frame_counter += 1
            self.closed_duration = timestamp - self.closed_since
        else:
            # A short closure that ended is a blink
            if (self.MIN_BLINK_FRAMES <= self.closed_frame_counter
                    and self.closed_duration < self.closed_seconds):
                self.blink_counter += 1
            self.closed_since = None
            self.closed_frame_counter = 0
            self.closed_duration = 0.0

        self.is_drowsy = eyes_closed and self.closed_duration >= self.closed_seconds

        return {
            'left_ear': avg_ear if left_ear is None else left_ear,
            'right_ear': avg_ear if right_ear is None else right_ear,
            'avg_ear': avg_ear,
            'smoothed_ear': smoothed_ear,
            'eyes_closed': eyes_closed,
            'is_drowsy': self.is_drowsy,
            'closed_frames': self.closed_frame_counter,
            'closed_duration': self.closed_duration,
            'blink_count': self.blink_counter,
            'drowsiness_progress': min(self.closed_duration / self.closed_seconds, 1.0),
        }

    def reset(self):
        """Reset tracking state (e.g. when the face is lost). Keeps blink count."""
        self.closed_frame_counter = 0
        self.closed_since = None
        self.closed_duration = 0.0
        self.is_drowsy = False
        self.ear_history.clear()

    def reset_blink_counter(self):
        """Reset blink counter (e.g., for per-minute statistics)"""
        self.blink_counter = 0


def draw_eye_landmarks(
    frame: np.ndarray,
    face_data,
    left_indices: Tuple[int, ...],
    right_indices: Tuple[int, ...],
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """
    Draw eye landmarks and contours on frame.
    
    Args:
        frame: BGR image
        face_data: FaceData object
        left_indices: Left eye landmark indices
        right_indices: Right eye landmark indices
        color: BGR color for drawing
        
    Returns:
        Frame with eye landmarks drawn
    """
    import cv2
    
    # Get eye coordinates
    left_eye = face_data.get_landmarks(left_indices)
    right_eye = face_data.get_landmarks(right_indices)
    
    # Draw eye contours
    cv2.polylines(frame, [left_eye], True, color, 1)
    cv2.polylines(frame, [right_eye], True, color, 1)
    
    # Draw eye center points
    left_center = left_eye.mean(axis=0).astype(int)
    right_center = right_eye.mean(axis=0).astype(int)
    
    cv2.circle(frame, tuple(left_center), 2, (0, 0, 255), -1)
    cv2.circle(frame, tuple(right_center), 2, (0, 0, 255), -1)
    
    return frame
