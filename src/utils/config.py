"""
Configuration settings for Driver Monitoring System
All thresholds and parameters are centralized here for easy tuning.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Project layout (resolved relative to this file, never the working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = Path(os.environ.get("DMS_LOGS_DIR") or PROJECT_ROOT / "logs")
RECORDINGS_DIR = PROJECT_ROOT / "recordings"


def _env(name: str, default, cast=float):
    """Read a DMS_* environment override, falling back to the default."""
    raw = os.environ.get(f"DMS_{name}")
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def _tunable(name: str, default, cast=float):
    """Dataclass field whose default can be overridden with DMS_<name>."""
    return field(default_factory=lambda: _env(name, default, cast))


@dataclass
class EyeConfig:
    """Eye Aspect Ratio (EAR) configuration"""
    # EAR threshold - below this value, eye is considered closed
    EAR_THRESHOLD: float = _tunable("EAR_THRESHOLD", 0.21)

    # Seconds the eyes must stay closed to trigger a drowsiness alert
    CLOSED_SECONDS: float = _tunable("CLOSED_SECONDS", 1.0)

    # Calibration: threshold = CALIBRATION_RATIO * driver's open-eye EAR
    CALIBRATION_RATIO: float = 0.72
    
    # MediaPipe face mesh eye landmark indices
    # Left eye landmarks
    LEFT_EYE_INDICES: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    # Right eye landmarks  
    RIGHT_EYE_INDICES: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)


@dataclass
class MouthConfig:
    """Mouth Aspect Ratio (MAR) configuration"""
    # MAR threshold - above this value, mouth is considered open (yawning)
    MAR_THRESHOLD: float = _tunable("MAR_THRESHOLD", 0.6)
    
    # Duration in seconds mouth must be open to count as yawn
    YAWN_DURATION: float = _tunable("YAWN_DURATION", 2.0)
    
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
    YAW_THRESHOLD: float = _tunable("YAW_THRESHOLD", 30.0)
    
    # Pitch threshold (looking up/down) in degrees
    PITCH_THRESHOLD: float = _tunable("PITCH_THRESHOLD", 20.0)
    
    # Roll threshold (head tilt) in degrees
    ROLL_THRESHOLD: float = _tunable("ROLL_THRESHOLD", 25.0)
    
    # Seconds the head must stay turned away to trigger a distraction alert
    DISTRACTION_SECONDS: float = _tunable("DISTRACTION_SECONDS", 1.5)
    
    # Generic 3D face model in the camera convention (x right, y down, z away
    # from the camera). The identity rotation is a frontal face, so angles stay
    # near 0 (no +/-180 wrap-around). Units are arbitrary (roughly mm).
    # Order: nose tip, chin, outer corner of the eye on the image-left,
    # outer corner of the eye on the image-right, mouth corners (left, right).
    MODEL_POINTS_3D: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 330.0, 65.0),
        (-225.0, -170.0, 135.0),
        (225.0, -170.0, 135.0),
        (-150.0, 150.0, 125.0),
        (150.0, 150.0, 125.0),
    )

    # MediaPipe indices matching MODEL_POINTS_3D: nose, chin, eye 33 (image-left),
    # eye 263 (image-right), mouth corners 61 (image-left) and 291 (image-right)
    LANDMARK_INDICES: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)


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
    
    # Cooldown between repeated alerts of the same kind (seconds)
    ALERT_COOLDOWN: float = _tunable("ALERT_COOLDOWN", 5.0)

    # Seconds without a face before a 'driver not detected' alert
    NO_FACE_SECONDS: float = _tunable("NO_FACE_SECONDS", 3.0)


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
