"""
Head Pose Estimation Module for Distraction Detection
Uses solvePnP to estimate 3D head orientation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import HEAD_POSE_CONFIG


class HeadPoseEstimator:
    """
    Head pose estimation using OpenCV's solvePnP.
    Calculates pitch, yaw, and roll angles to detect distraction.
    """
    
    # MediaPipe Face Mesh landmarks for pose estimation
    # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
    LANDMARK_INDICES = (1, 152, 263, 33, 287, 57)
    
    def __init__(
        self,
        yaw_threshold: float = None,
        pitch_threshold: float = None,
        roll_threshold: float = None,
        distraction_frames: int = None
    ):
        """
        Initialize head pose estimator.
        
        Args:
            yaw_threshold: Yaw angle threshold for distraction (degrees)
            pitch_threshold: Pitch angle threshold for distraction (degrees)
            roll_threshold: Roll angle threshold for distraction (degrees)
            distraction_frames: Consecutive frames for distraction alert
        """
        self.yaw_threshold = yaw_threshold or HEAD_POSE_CONFIG.YAW_THRESHOLD
        self.pitch_threshold = pitch_threshold or HEAD_POSE_CONFIG.PITCH_THRESHOLD
        self.roll_threshold = roll_threshold or HEAD_POSE_CONFIG.ROLL_THRESHOLD
        self.distraction_frames = distraction_frames or HEAD_POSE_CONFIG.DISTRACTION_FRAMES
        
        # 3D model points (standard face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (will be updated based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # State tracking
        self.distraction_counter = 0
        self.is_distracted = False
        
        # History for smoothing
        self.angle_history = {'yaw': [], 'pitch': [], 'roll': []}
        self.history_size = 5
    
    def _get_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Get camera intrinsic matrix based on frame dimensions."""
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def estimate_pose(self, face_data) -> Tuple[float, float, float]:
        """
        Estimate head pose from face landmarks.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Update camera matrix if needed
        if self.camera_matrix is None:
            self.camera_matrix = self._get_camera_matrix(
                face_data.image_width, 
                face_data.image_height
            )
        
        # Get 2D image points
        image_points = []
        for idx in self.LANDMARK_INDICES:
            point = face_data.get_landmark(idx)
            image_points.append(point)
        
        image_points = np.array(image_points, dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
        
        return pitch, yaw, roll
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Decompose rotation matrix to get projection matrix
        proj_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]
        
        return pitch, yaw, roll
    
    def get_smoothed_angles(self, pitch: float, yaw: float, roll: float) -> Tuple[float, float, float]:
        """
        Get temporally smoothed angles.
        
        Args:
            pitch, yaw, roll: Current frame angles
            
        Returns:
            Smoothed (pitch, yaw, roll)
        """
        for name, value in [('pitch', pitch), ('yaw', yaw), ('roll', roll)]:
            self.angle_history[name].append(value)
            if len(self.angle_history[name]) > self.history_size:
                self.angle_history[name].pop(0)
        
        return (
            np.mean(self.angle_history['pitch']),
            np.mean(self.angle_history['yaw']),
            np.mean(self.angle_history['roll'])
        )
    
    def update(self, face_data) -> dict:
        """
        Update head pose tracking state with new frame data.
        
        Args:
            face_data: FaceData object with landmarks
            
        Returns:
            Dictionary with tracking results
        """
        pitch, yaw, roll = self.estimate_pose(face_data)
        smoothed_pitch, smoothed_yaw, smoothed_roll = self.get_smoothed_angles(pitch, yaw, roll)
        
        # Check if looking away (distracted)
        looking_away = (
            abs(smoothed_yaw) > self.yaw_threshold or
            abs(smoothed_pitch) > self.pitch_threshold
        )
        
        head_tilted = abs(smoothed_roll) > self.roll_threshold
        
        is_distracted_current = looking_away or head_tilted
        
        if is_distracted_current:
            self.distraction_counter += 1
        else:
            self.distraction_counter = 0
        
        self.is_distracted = self.distraction_counter >= self.distraction_frames
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'smoothed_pitch': smoothed_pitch,
            'smoothed_yaw': smoothed_yaw,
            'smoothed_roll': smoothed_roll,
            'looking_away': looking_away,
            'head_tilted': head_tilted,
            'is_distracted': self.is_distracted,
            'distraction_frames': self.distraction_counter,
            'distraction_progress': min(self.distraction_counter / self.distraction_frames, 1.0)
        }
    
    def reset(self):
        """Reset tracking state"""
        self.distraction_counter = 0
        self.is_distracted = False
        for key in self.angle_history:
            self.angle_history[key].clear()


def draw_head_pose(
    frame: np.ndarray,
    face_data,
    pitch: float,
    yaw: float,
    roll: float,
    axis_length: int = 100
) -> np.ndarray:
    """
    Draw head pose axes on frame.
    
    Args:
        frame: BGR image
        face_data: FaceData object
        pitch, yaw, roll: Head angles in degrees
        axis_length: Length of axes to draw
        
    Returns:
        Frame with pose axes drawn
    """
    # Get nose tip as origin
    nose_tip = face_data.get_landmark(1)
    
    # Create rotation matrix from angles
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    Ry = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    Rz = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Axis endpoints
    axes = np.array([
        [axis_length, 0, 0],   # X - Red
        [0, axis_length, 0],   # Y - Green
        [0, 0, axis_length]    # Z - Blue
    ])
    
    # Rotate axes
    rotated_axes = (R @ axes.T).T
    
    # Project to 2D and draw
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
    for axis, color in zip(rotated_axes, colors):
        end_point = (
            int(nose_tip[0] + axis[0]),
            int(nose_tip[1] - axis[1])  # Flip Y for image coordinates
        )
        cv2.line(frame, nose_tip, end_point, color, 2)
    
    return frame
