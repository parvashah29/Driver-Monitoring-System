"""
Eye Tracking Module for Drowsiness Detection
Calculates Eye Aspect Ratio (EAR) for blink and drowsiness detection.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import distance
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import EYE_CONFIG


class EyeTracker:
    """
    Eye tracking using Eye Aspect Ratio (EAR) calculation.
    
    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    
    Where P1-P6 are the 6 eye landmark points:
        P1: Left corner
        P2: Upper left
        P3: Upper right
        P4: Right corner
        P5: Lower right
        P6: Lower left
    """
    
    def __init__(
        self,
        ear_threshold: float = None,
        consecutive_frames: int = None
    ):
        """
        Initialize eye tracker.
        
        Args:
            ear_threshold: EAR value below which eye is considered closed
            consecutive_frames: Frames with closed eyes to trigger drowsiness
        """
        self.ear_threshold = ear_threshold or EYE_CONFIG.EAR_THRESHOLD
        self.consecutive_frames = consecutive_frames or EYE_CONFIG.CONSECUTIVE_FRAMES
        
        # MediaPipe Face Mesh eye landmark indices
        self.left_eye_indices = EYE_CONFIG.LEFT_EYE_INDICES
        self.right_eye_indices = EYE_CONFIG.RIGHT_EYE_INDICES
        
        # State tracking
        self.closed_frame_counter = 0
        self.blink_counter = 0
        self.is_drowsy = False
        
        # History for smoothing
        self.ear_history = []
        self.history_size = 5
    
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio for a single eye.
        
        Args:
            eye_points: Array of 6 (x, y) coordinates for eye landmarks
            
        Returns:
            EAR value (typically 0.15-0.45 for open eyes)
        """
        # Compute euclidean distances between vertical eye landmarks
        # P2-P6 (upper-left to lower-left)
        vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
        # P3-P5 (upper-right to lower-right)
        vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        # P1-P4 (left corner to right corner)
        horizontal = distance.euclidean(eye_points[0], eye_points[3])
        
        # Avoid division by zero
        if horizontal == 0:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def get_ear(self, face_data) -> Tuple[float, float, float]:
        """
        Calculate EAR for both eyes from face data.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            Tuple of (left_ear, right_ear, average_ear)
        """
        # Get eye landmark coordinates
        left_eye = face_data.get_landmarks(self.left_eye_indices)
        right_eye = face_data.get_landmarks(self.right_eye_indices)
        
        # Calculate EAR for each eye
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, avg_ear
    
    def get_smoothed_ear(self, current_ear: float) -> float:
        """
        Get temporally smoothed EAR value.
        
        Args:
            current_ear: Current frame's EAR value
            
        Returns:
            Smoothed EAR value
        """
        self.ear_history.append(current_ear)
        if len(self.ear_history) > self.history_size:
            self.ear_history.pop(0)
        
        return np.mean(self.ear_history)
    
    def update(self, face_data) -> dict:
        """
        Update eye tracking state with new frame data.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            Dictionary with tracking results
        """
        left_ear, right_ear, avg_ear = self.get_ear(face_data)
        smoothed_ear = self.get_smoothed_ear(avg_ear)
        
        # Check if eyes are closed
        eyes_closed = smoothed_ear < self.ear_threshold
        
        if eyes_closed:
            self.closed_frame_counter += 1
        else:
            # Check for blink (short closure)
            if 2 <= self.closed_frame_counter < self.consecutive_frames:
                self.blink_counter += 1
            self.closed_frame_counter = 0
        
        # Check for drowsiness
        self.is_drowsy = self.closed_frame_counter >= self.consecutive_frames
        
        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'smoothed_ear': smoothed_ear,
            'eyes_closed': eyes_closed,
            'is_drowsy': self.is_drowsy,
            'closed_frames': self.closed_frame_counter,
            'blink_count': self.blink_counter,
            'drowsiness_progress': min(self.closed_frame_counter / self.consecutive_frames, 1.0)
        }
    
    def reset(self):
        """Reset tracking state"""
        self.closed_frame_counter = 0
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
