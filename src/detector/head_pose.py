"""
Head Pose Estimation Module for Distraction Detection
Uses solvePnP to estimate 3D head orientation.

Angle convention (degrees, zero = looking straight at the camera):
    yaw   > 0 : face turned toward the right of the image
    pitch > 0 : head tilted up (looking up)
    roll  > 0 : head tilted clockwise in the image
"""

import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.config import HEAD_POSE_CONFIG


class HeadPoseEstimator:
    """
    Head pose estimation using OpenCV's solvePnP on six face landmarks.
    Distraction = head turned/tilted beyond a threshold for `distraction_seconds`.
    """

    LANDMARK_INDICES = HEAD_POSE_CONFIG.LANDMARK_INDICES

    def __init__(
        self,
        yaw_threshold: Optional[float] = None,
        pitch_threshold: Optional[float] = None,
        roll_threshold: Optional[float] = None,
        distraction_seconds: Optional[float] = None,
        history_size: int = 5,
    ):
        pick = lambda v, d: d if v is None else v
        self.yaw_threshold = pick(yaw_threshold, HEAD_POSE_CONFIG.YAW_THRESHOLD)
        self.pitch_threshold = pick(pitch_threshold, HEAD_POSE_CONFIG.PITCH_THRESHOLD)
        self.roll_threshold = pick(roll_threshold, HEAD_POSE_CONFIG.ROLL_THRESHOLD)
        self.distraction_seconds = pick(distraction_seconds, HEAD_POSE_CONFIG.DISTRACTION_SECONDS)

        self.model_points = np.array(HEAD_POSE_CONFIG.MODEL_POINTS_3D, dtype=np.float64)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._prev_pose = None  # (rvec, tvec) of the last successful solve

        # Per-driver neutral pose (set by calibrate)
        self.neutral = np.zeros(3)

        self.distraction_since: Optional[float] = None
        self.distraction_duration = 0.0
        self.is_distracted = False

        self.angle_history = {k: deque(maxlen=history_size) for k in ('yaw', 'pitch', 'roll')}

    def _get_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Approximate pinhole intrinsics (focal length ~ image width)."""
        f = float(frame_width)
        return np.array([[f, 0, frame_width / 2], [0, f, frame_height / 2], [0, 0, 1]], dtype=np.float64)

    def estimate_pose(self, face_data) -> Tuple[float, float, float]:
        """Return raw (pitch, yaw, roll) in degrees for a FaceData."""
        if self.camera_matrix is None:
            self.camera_matrix = self._get_camera_matrix(face_data.image_width, face_data.image_height)
        image_points = face_data.get_landmarks_float(self.LANDMARK_INDICES)
        return self.estimate_pose_from_points(image_points)

    def estimate_pose_from_points(self, image_points: np.ndarray) -> Tuple[float, float, float]:
        """solvePnP on 6 pixel points ordered like MODEL_POINTS_3D."""
        pts = np.ascontiguousarray(image_points, dtype=np.float64)
        success = False
        if self._prev_pose is not None:
            rvec, tvec = (a.copy() for a in self._prev_pose)
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, pts, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
            if success and tvec[2, 0] <= 0:
                success = False
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points, pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if success and tvec[2, 0] <= 0:
                success = False
        if not success:
            self._prev_pose = None
            return 0.0, 0.0, 0.0
        self._prev_pose = (rvec, tvec)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return self._rotation_matrix_to_euler(rotation_matrix)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Rotation matrix (model -> camera) to (pitch, yaw, roll) in degrees using
        the module's sign convention. Ranges are pitch/roll in (-180, 180] and
        yaw in [-90, 90]; for a face looking at the camera all are near zero.
        """
        sy = np.hypot(R[0, 0], R[1, 0])
        pitch_raw = np.arctan2(R[2, 1], R[2, 2])
        yaw_raw = np.arctan2(-R[2, 0], sy)
        roll_raw = np.arctan2(R[1, 0], R[0, 0])
        return (float(-np.degrees(pitch_raw)),
                float(-np.degrees(yaw_raw)),
                float(np.degrees(roll_raw)))

    def get_smoothed_angles(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """Moving average of the (neutral-corrected) angles."""
        for name, value in (('pitch', pitch), ('yaw', yaw), ('roll', roll)):
            self.angle_history[name].append(value)
        return (float(np.mean(self.angle_history['pitch'])),
                float(np.mean(self.angle_history['yaw'])),
                float(np.mean(self.angle_history['roll'])))

    def calibrate(self, poses: np.ndarray) -> Tuple[float, float, float]:
        """
        Set the driver's neutral pose from (pitch, yaw, roll) samples taken while
        looking at the road/camera. Returns the neutral (pitch, yaw, roll).
        """
        arr = np.asarray(poses, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 10:
            raise ValueError("need at least 10 (pitch, yaw, roll) samples")
        self.neutral = np.median(arr, axis=0)
        return tuple(float(v) for v in self.neutral)

    def update(self, face_data, timestamp: Optional[float] = None) -> dict:
        """Update state with one frame's landmarks (timestamp in seconds)."""
        if timestamp is None:
            timestamp = time.time()
        pitch, yaw, roll = self.estimate_pose(face_data)
        return self.update_angles(pitch, yaw, roll, timestamp)

    def update_angles(self, pitch: float, yaw: float, roll: float, timestamp: float) -> dict:
        """State machine core, driven directly by raw angles (used by tests)."""
        raw = (pitch, yaw, roll)
        pitch, yaw, roll = pitch - self.neutral[0], yaw - self.neutral[1], roll - self.neutral[2]
        sp, sy, sr = self.get_smoothed_angles(pitch, yaw, roll)

        looking_away = abs(sy) > self.yaw_threshold or abs(sp) > self.pitch_threshold
        head_tilted = abs(sr) > self.roll_threshold

        if looking_away or head_tilted:
            if self.distraction_since is None:
                self.distraction_since = timestamp
            self.distraction_duration = timestamp - self.distraction_since
        else:
            self.distraction_since = None
            self.distraction_duration = 0.0

        self.is_distracted = self.distraction_duration >= self.distraction_seconds and (
            looking_away or head_tilted)

        return {
            'raw_pitch': raw[0], 'raw_yaw': raw[1], 'raw_roll': raw[2],
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'smoothed_pitch': sp, 'smoothed_yaw': sy, 'smoothed_roll': sr,
            'looking_away': looking_away,
            'head_tilted': head_tilted,
            'is_distracted': self.is_distracted,
            'distraction_duration': self.distraction_duration,
            'distraction_progress': min(self.distraction_duration / self.distraction_seconds, 1.0),
        }

    def reset(self):
        """Reset tracking state (e.g. when the face is lost)."""
        self.distraction_since = None
        self.distraction_duration = 0.0
        self.is_distracted = False
        self._prev_pose = None
        for h in self.angle_history.values():
            h.clear()


def draw_head_pose(
    frame: np.ndarray,
    face_data,
    pitch: float,
    yaw: float,
    roll: float = 0.0,
    axis_length: int = 100,
) -> np.ndarray:
    """Draw the facing direction as a line from the nose tip (uses the module's angle convention)."""
    nose = face_data.get_landmark(1)
    end = (int(nose[0] + axis_length * np.sin(np.radians(yaw))),
           int(nose[1] - axis_length * np.sin(np.radians(pitch))))
    cv2.arrowedLine(frame, nose, end, (255, 0, 0), 2, tipLength=0.25)
    return frame
