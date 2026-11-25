"""
Mouth Tracking Module for Yawn Detection
Calculates Mouth Aspect Ratio (MAR) for yawn detection.
"""

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

from ..utils.config import MOUTH_CONFIG


class MouthTracker:
    """
    Mouth tracking using Mouth Aspect Ratio (MAR).

    MAR = mean(inner-lip openings) / mouth width. A yawn is a MAR above the
    threshold that is sustained for `yawn_duration` seconds (talking and
    smiling do not last that long at high MAR).
    """

    # MediaPipe Face Mesh landmark indices
    UPPER_LIP_TOP = 13      # Inner upper lip (centre)
    LOWER_LIP_BOTTOM = 14   # Inner lower lip (centre)
    LEFT_CORNER = 61        # Mouth corner
    RIGHT_CORNER = 291      # Mouth corner
    UPPER_INNER = 82        # Inner upper lip (off-centre)
    LOWER_INNER = 87        # Inner lower lip (off-centre)

    def __init__(
        self,
        mar_threshold: Optional[float] = None,
        yawn_duration: Optional[float] = None,
        history_size: int = 5,
    ):
        self.mar_threshold = MOUTH_CONFIG.MAR_THRESHOLD if mar_threshold is None else mar_threshold
        self.yawn_duration = MOUTH_CONFIG.YAWN_DURATION if yawn_duration is None else yawn_duration

        self.yawn_start_time: Optional[float] = None
        self.is_yawning = False
        self.yawn_counter = 0
        self.current_yawn_duration = 0.0

        self.mar_history = deque(maxlen=history_size)

    def calculate_mar(self, face_data) -> float:
        """Mouth Aspect Ratio for a FaceData (0 = closed, ~0.6+ = wide open)."""
        pts = face_data.get_landmarks_float((
            self.UPPER_LIP_TOP, self.LOWER_LIP_BOTTOM,
            self.LEFT_CORNER, self.RIGHT_CORNER,
            self.UPPER_INNER, self.LOWER_INNER,
        ))
        vertical_1 = np.linalg.norm(pts[0] - pts[1])
        vertical_2 = np.linalg.norm(pts[4] - pts[5])
        horizontal = np.linalg.norm(pts[2] - pts[3])
        if horizontal == 0:
            return 0.0
        return float((vertical_1 + vertical_2) / 2.0 / horizontal)

    def get_smoothed_mar(self, current_mar: float) -> float:
        """Moving average over the last few frames."""
        self.mar_history.append(current_mar)
        return float(np.mean(self.mar_history))

    def update(self, face_data, timestamp: Optional[float] = None) -> dict:
        """Update state with one frame's landmarks (timestamp in seconds)."""
        if timestamp is None:
            timestamp = time.time()
        return self.update_mar(self.calculate_mar(face_data), timestamp)

    def update_mar(self, mar: float, timestamp: float) -> dict:
        """State machine core, driven directly by a MAR value (used by tests)."""
        smoothed_mar = self.get_smoothed_mar(mar)
        mouth_open = smoothed_mar > self.mar_threshold

        if mouth_open:
            if self.yawn_start_time is None:
                self.yawn_start_time = timestamp
            self.current_yawn_duration = timestamp - self.yawn_start_time
            if self.current_yawn_duration >= self.yawn_duration and not self.is_yawning:
                self.yawn_counter += 1
                self.is_yawning = True
        else:
            self.yawn_start_time = None
            self.current_yawn_duration = 0.0
            self.is_yawning = False

        return {
            'mar': mar,
            'smoothed_mar': smoothed_mar,
            'mouth_open': mouth_open,
            'is_yawning': self.is_yawning,
            'yawn_duration': self.current_yawn_duration,
            'yawn_count': self.yawn_counter,
            'yawn_progress': min(self.current_yawn_duration / self.yawn_duration, 1.0) if mouth_open else 0.0,
        }

    def reset(self):
        """Reset tracking state (e.g. when the face is lost). Keeps yawn count."""
        self.yawn_start_time = None
        self.is_yawning = False
        self.current_yawn_duration = 0.0
        self.mar_history.clear()

    def reset_yawn_counter(self):
        """Reset yawn counter (e.g., for per-session statistics)"""
        self.yawn_counter = 0


def draw_mouth_landmarks(
    frame: np.ndarray,
    face_data,
    color: Tuple[int, int, int] = (255, 0, 255)
) -> np.ndarray:
    """
    Draw mouth landmarks on frame.
    
    Args:
        frame: BGR image
        face_data: FaceData object
        color: BGR color for drawing
        
    Returns:
        Frame with mouth landmarks drawn
    """
    import cv2
    
    # Key mouth points
    indices = [
        MouthTracker.UPPER_LIP_TOP,
        MouthTracker.LOWER_LIP_BOTTOM,
        MouthTracker.LEFT_CORNER,
        MouthTracker.RIGHT_CORNER,
    ]
    
    for idx in indices:
        point = face_data.get_landmark(idx)
        cv2.circle(frame, point, 3, color, -1)
    
    # Draw mouth width line
    left = face_data.get_landmark(MouthTracker.LEFT_CORNER)
    right = face_data.get_landmark(MouthTracker.RIGHT_CORNER)
    cv2.line(frame, left, right, color, 1)
    
    # Draw mouth height line
    upper = face_data.get_landmark(MouthTracker.UPPER_LIP_TOP)
    lower = face_data.get_landmark(MouthTracker.LOWER_LIP_BOTTOM)
    cv2.line(frame, upper, lower, color, 1)
    
    return frame
