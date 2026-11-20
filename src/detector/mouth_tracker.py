"""
Mouth Tracking Module for Yawn Detection
Calculates Mouth Aspect Ratio (MAR) for yawn detection.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import distance
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import MOUTH_CONFIG


class MouthTracker:
    """
    Mouth tracking using Mouth Aspect Ratio (MAR) calculation.
    
    MAR = vertical_distance / horizontal_distance
    
    Higher MAR indicates more open mouth (yawning).
    """
    
    # MediaPipe Face Mesh mouth landmark indices
    # Outer lips: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    # Key points for MAR calculation
    UPPER_LIP_TOP = 13      # Top of upper lip
    LOWER_LIP_BOTTOM = 14   # Bottom of lower lip
    LEFT_CORNER = 61        # Left corner of mouth
    RIGHT_CORNER = 291      # Right corner of mouth
    
    # Additional points for better MAR calculation
    UPPER_INNER = 82        # Inner upper lip
    LOWER_INNER = 87        # Inner lower lip
    
    def __init__(
        self,
        mar_threshold: float = None,
        yawn_duration: float = None
    ):
        """
        Initialize mouth tracker.
        
        Args:
            mar_threshold: MAR value above which mouth is considered yawning
            yawn_duration: Duration in seconds mouth must be open to count as yawn
        """
        self.mar_threshold = mar_threshold or MOUTH_CONFIG.MAR_THRESHOLD
        self.yawn_duration = yawn_duration or MOUTH_CONFIG.YAWN_DURATION
        
        # State tracking
        self.yawn_start_time = None
        self.is_yawning = False
        self.yawn_counter = 0
        self.current_yawn_duration = 0.0
        
        # History for smoothing
        self.mar_history = []
        self.history_size = 5
    
    def calculate_mar(self, face_data) -> float:
        """
        Calculate Mouth Aspect Ratio.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            MAR value (0.0-1.0+, higher = more open)
        """
        # Get key mouth landmark coordinates
        upper = face_data.get_landmark(self.UPPER_LIP_TOP)
        lower = face_data.get_landmark(self.LOWER_LIP_BOTTOM)
        left = face_data.get_landmark(self.LEFT_CORNER)
        right = face_data.get_landmark(self.RIGHT_CORNER)
        
        # Additional points for more accurate measurement
        upper_inner = face_data.get_landmark(self.UPPER_INNER)
        lower_inner = face_data.get_landmark(self.LOWER_INNER)
        
        # Calculate vertical distances
        vertical_1 = distance.euclidean(upper, lower)
        vertical_2 = distance.euclidean(upper_inner, lower_inner)
        avg_vertical = (vertical_1 + vertical_2) / 2
        
        # Calculate horizontal distance
        horizontal = distance.euclidean(left, right)
        
        # Avoid division by zero
        if horizontal == 0:
            return 0.0
        
        # Calculate MAR
        mar = avg_vertical / horizontal
        return mar
    
    def get_smoothed_mar(self, current_mar: float) -> float:
        """
        Get temporally smoothed MAR value.
        
        Args:
            current_mar: Current frame's MAR value
            
        Returns:
            Smoothed MAR value
        """
        self.mar_history.append(current_mar)
        if len(self.mar_history) > self.history_size:
            self.mar_history.pop(0)
        
        return np.mean(self.mar_history)
    
    def update(self, face_data) -> dict:
        """
        Update mouth tracking state with new frame data.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            Dictionary with tracking results
        """
        mar = self.calculate_mar(face_data)
        smoothed_mar = self.get_smoothed_mar(mar)
        current_time = time.time()
        
        # Check if mouth is open wide enough to be yawning
        mouth_open = smoothed_mar > self.mar_threshold
        
        if mouth_open:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            
            self.current_yawn_duration = current_time - self.yawn_start_time
            
            # Check if yawn duration threshold exceeded
            if self.current_yawn_duration >= self.yawn_duration:
                if not self.is_yawning:
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
            'yawn_progress': min(self.current_yawn_duration / self.yawn_duration, 1.0) if mouth_open else 0.0
        }
    
    def reset(self):
        """Reset tracking state"""
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
