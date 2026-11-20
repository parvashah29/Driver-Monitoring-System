"""
Configuration settings for Driver Monitoring System
All thresholds and parameters are centralized here for easy tuning.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EyeConfig:
    """Eye Aspect Ratio (EAR) configuration"""
    # EAR threshold - below this value, eye is considered closed
    EAR_THRESHOLD: float = 0.25
    
    # Number of consecutive frames eye must be closed to trigger drowsiness
    CONSECUTIVE_FRAMES: int = 30
    
    # MediaPipe face mesh eye landmark indices
    # Left eye landmarks
    LEFT_EYE_INDICES: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    # Right eye landmarks  
    RIGHT_EYE_INDICES: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)


@dataclass
class MouthConfig:
    """Mouth Aspect Ratio (MAR) configuration"""
    # MAR threshold - above this value, mouth is considered open (yawning)
    MAR_THRESHOLD: float = 0.6
    
    # Duration in seconds mouth must be open to count as yawn
    YAWN_DURATION: float = 2.0
    
    # MediaPipe face mesh mouth landmark indices
    # Outer lip landmarks for MAR calculation
    MOUTH_INDICES: Tuple[int, ...] = (61, 291, 0, 17, 269, 405, 314, 17, 84, 181)
    
    # Simplified indices: top, bottom, left, right
    UPPER_LIP: int = 13
    LOWER_LIP: int = 14
    LEFT_CORNER: int = 61
    RIGHT_CORNER: int = 291


@dataclass
class HeadPoseConfig:
    """Head pose estimation configuration"""
    # Yaw threshold (looking left/right) in degrees
    YAW_THRESHOLD: float = 30.0
    
    # Pitch threshold (looking up/down) in degrees
    PITCH_THRESHOLD: float = 20.0
    
    # Roll threshold (head tilt) in degrees
    ROLL_THRESHOLD: float = 25.0
    
    # Consecutive frames for distraction alert
    DISTRACTION_FRAMES: int = 45
    
    # 3D model points for head pose estimation (nose tip, chin, etc.)
    # These are standard facial landmark 3D coordinates
    MODEL_POINTS_3D: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye corner
        (225.0, 170.0, -135.0),     # Right eye corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0),    # Right mouth corner
    )
    
    # MediaPipe indices for the 3D model points
    LANDMARK_INDICES: Tuple[int, ...] = (1, 152, 263, 33, 287, 57)


@dataclass
class AlertConfig:
    """Alert system configuration"""
    # Alert levels
    LEVEL_NORMAL: int = 0
    LEVEL_WARNING: int = 1
    LEVEL_CRITICAL: int = 2
    
    # Audio alert settings
    WARNING_BEEP_FREQ: int = 800  # Hz
    CRITICAL_BEEP_FREQ: int = 1200  # Hz
    BEEP_DURATION: float = 0.3  # seconds
    
    # Visual alert colors (BGR for OpenCV)
    COLOR_NORMAL: Tuple[int, int, int] = (0, 255, 0)      # Green
    COLOR_WARNING: Tuple[int, int, int] = (0, 165, 255)   # Orange
    COLOR_CRITICAL: Tuple[int, int, int] = (0, 0, 255)    # Red
    
    # Cooldown between alerts (seconds)
    ALERT_COOLDOWN: float = 3.0


@dataclass
class CameraConfig:
    """Camera and video settings"""
    # Default camera index (0 = built-in webcam)
    CAMERA_INDEX: int = 0
    
    # Frame dimensions
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    
    # Target FPS
    TARGET_FPS: int = 30
    
    # Face detection confidence threshold
    DETECTION_CONFIDENCE: float = 0.5
    
    # Tracking confidence threshold
    TRACKING_CONFIDENCE: float = 0.5


# Global configuration instances
EYE_CONFIG = EyeConfig()
MOUTH_CONFIG = MouthConfig()
HEAD_POSE_CONFIG = HeadPoseConfig()
ALERT_CONFIG = AlertConfig()
CAMERA_CONFIG = CameraConfig()
