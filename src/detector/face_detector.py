"""
Face Detection Module using MediaPipe Face Landmarker (Tasks API)
Provides 478 facial landmarks for comprehensive face analysis.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os
import urllib.request


# Model download URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")


def download_model():
    """Download the face landmarker model if not present."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
    return MODEL_PATH


@dataclass
class FaceData:
    """Container for detected face data"""
    landmarks: np.ndarray          # (478, 3) array of landmark coordinates (normalized)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    confidence: float              # Detection confidence
    image_width: int
    image_height: int
    
    def get_landmark(self, idx: int) -> Tuple[int, int]:
        """Get pixel coordinates of a specific landmark"""
        lm = self.landmarks[idx]
        return int(lm[0] * self.image_width), int(lm[1] * self.image_height)
    
    def get_landmarks(self, indices: Tuple[int, ...]) -> np.ndarray:
        """Get pixel coordinates for multiple landmarks"""
        coords = []
        for idx in indices:
            lm = self.landmarks[idx]
            coords.append([int(lm[0] * self.image_width), int(lm[1] * self.image_height)])
        return np.array(coords)


class FaceDetector:
    """
    Face detector using MediaPipe Face Landmarker (Tasks API).
    Provides 478 facial landmarks including eyes, mouth, and face contour.
    """
    
    def __init__(
        self,
        max_faces: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True
    ):
        """
        Initialize the face detector.
        
        Args:
            max_faces: Maximum number of faces to detect
            detection_confidence: Minimum confidence for face detection
            tracking_confidence: Minimum confidence for landmark tracking
            refine_landmarks: Whether to output blendshapes (not used in basic mode)
        """
        # Download model if needed
        model_path = download_model()
        
        # Create FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Store for visualization
        self._last_result = None
    
    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Detect face and extract landmarks from a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            FaceData object if face detected, None otherwise
        """
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        result = self.detector.detect(mp_image)
        self._last_result = result
        
        if not result.face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = result.face_landmarks[0]
        
        # Convert to numpy array (normalized coordinates)
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks
        ])
        
        # Calculate bounding box from landmarks
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add some padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return FaceData(
            landmarks=landmarks,
            bbox=bbox,
            confidence=1.0,  # Tasks API doesn't expose per-face confidence directly
            image_width=w,
            image_height=h
        )
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        face_data: FaceData,
        draw_mesh: bool = False,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Draw detected landmarks on the frame.
        
        Args:
            frame: BGR image to draw on
            face_data: Detected face data
            draw_mesh: Whether to draw full mesh (draws points instead)
            draw_contours: Whether to draw key contour points
            
        Returns:
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        
        if draw_contours or draw_mesh:
            # Draw key landmarks
            key_indices = [
                # Face oval
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                # Eyes
                33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
                # Nose
                1, 2, 98, 327,
                # Mouth
                61, 291, 0, 17, 13, 14
            ]
            
            for idx in key_indices:
                if idx < len(face_data.landmarks):
                    point = face_data.get_landmark(idx)
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)
        
        # Draw bounding box
        x, y, bw, bh = face_data.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        return frame
    
    def close(self):
        """Release resources"""
        if self.detector:
            self.detector.close()
