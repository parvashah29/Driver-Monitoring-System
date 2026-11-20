"""
Visual Alert Module for Driver Monitoring System
Provides on-screen visual warnings and overlays.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import ALERT_CONFIG


class VisualAlert:
    """
    Visual alert system for displaying warnings on the video feed.
    Includes status overlays, progress bars, and flashing alerts.
    """
    
    def __init__(self):
        """Initialize visual alert system."""
        self.colors = {
            'normal': ALERT_CONFIG.COLOR_NORMAL,
            'warning': ALERT_CONFIG.COLOR_WARNING,
            'critical': ALERT_CONFIG.COLOR_CRITICAL,
            'text': (255, 255, 255),
            'background': (0, 0, 0),
        }
        
        self._flash_state = False
        self._last_flash_time = time.time()
        self._flash_interval = 0.2  # seconds
    
    def _get_flash_state(self) -> bool:
        """Get current flash state for blinking effects."""
        current_time = time.time()
        if current_time - self._last_flash_time >= self._flash_interval:
            self._flash_state = not self._flash_state
            self._last_flash_time = current_time
        return self._flash_state
    
    def draw_status_panel(
        self,
        frame: np.ndarray,
        status: Dict[str, Any],
        position: Tuple[int, int] = (10, 10)
    ) -> np.ndarray:
        """
        Draw status panel with current metrics.
        
        Args:
            frame: BGR image
            status: Dictionary with status values
            position: Top-left position of panel
            
        Returns:
            Frame with status panel
        """
        x, y = position
        line_height = 25
        padding = 10
        
        # Calculate panel size
        lines = []
        if 'ear' in status:
            lines.append(f"EAR: {status['ear']:.3f}")
        if 'mar' in status:
            lines.append(f"MAR: {status['mar']:.3f}")
        if 'yaw' in status:
            lines.append(f"Yaw: {status['yaw']:.1f}°")
        if 'pitch' in status:
            lines.append(f"Pitch: {status['pitch']:.1f}°")
        if 'fps' in status:
            lines.append(f"FPS: {status['fps']:.1f}")
        if 'state' in status:
            lines.append(f"State: {status['state']}")
        
        panel_width = 150
        panel_height = len(lines) * line_height + 2 * padding
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                     self.colors['normal'], 1)
        
        # Draw text
        for i, line in enumerate(lines):
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(frame, line, (x + padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return frame
    
    def draw_progress_bar(
        self,
        frame: np.ndarray,
        progress: float,
        label: str,
        position: Tuple[int, int],
        size: Tuple[int, int] = (200, 20),
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Draw a progress bar.
        
        Args:
            frame: BGR image
            progress: Progress value 0.0 to 1.0
            label: Label text
            position: Top-left position
            size: (width, height) of bar
            color: Bar color (auto-selects based on progress if None)
            
        Returns:
            Frame with progress bar
        """
        x, y = position
        width, height = size
        
        # Auto-select color based on progress
        if color is None:
            if progress < 0.5:
                color = self.colors['normal']
            elif progress < 0.8:
                color = self.colors['warning']
            else:
                color = self.colors['critical']
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     (100, 100, 100), 1)
        
        # Draw filled portion
        fill_width = int(width * min(progress, 1.0))
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Draw label
        cv2.putText(frame, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return frame
    
    def draw_alert_banner(
        self,
        frame: np.ndarray,
        message: str,
        severity: str = 'warning'
    ) -> np.ndarray:
        """
        Draw alert banner at top of frame.
        
        Args:
            frame: BGR image
            message: Alert message
            severity: 'warning' or 'critical'
            
        Returns:
            Frame with alert banner
        """
        h, w = frame.shape[:2]
        banner_height = 50
        
        # Get color based on severity
        if severity == 'critical':
            # Flashing effect for critical
            color = self.colors['critical'] if self._get_flash_state() else (100, 0, 0)
        else:
            color = self.colors['warning']
        
        # Draw banner background
        cv2.rectangle(frame, (0, 0), (w, banner_height), color, -1)
        
        # Draw message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2
        
        cv2.putText(frame, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def draw_face_indicator(
        self,
        frame: np.ndarray,
        face_detected: bool,
        bbox: Tuple[int, int, int, int] = None
    ) -> np.ndarray:
        """
        Draw face detection indicator.
        
        Args:
            frame: BGR image
            face_detected: Whether face is detected
            bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            Frame with indicator
        """
        if face_detected and bbox:
            x, y, bw, bh = bbox
            color = self.colors['normal']
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, "Driver Detected", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            h, w = frame.shape[:2]
            # Draw warning overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            message = "NO DRIVER DETECTED"
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            cv2.putText(frame, message, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['critical'], 2)
        
        return frame
    
    def draw_drowsiness_overlay(
        self,
        frame: np.ndarray,
        is_drowsy: bool,
        progress: float
    ) -> np.ndarray:
        """
        Draw drowsiness detection overlay.
        
        Args:
            frame: BGR image
            is_drowsy: Whether drowsiness is detected
            progress: Drowsiness progress 0.0 to 1.0
            
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        if is_drowsy:
            # Flash red overlay
            if self._get_flash_state():
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            self.draw_alert_banner(frame, "⚠️ DROWSINESS DETECTED - WAKE UP!", 'critical')
        elif progress > 0.3:
            # Warning progress bar
            self.draw_progress_bar(
                frame, progress, "Drowsiness",
                (w - 220, 10), (200, 15)
            )
        
        return frame


def create_dashboard_overlay(
    frame: np.ndarray,
    eye_data: Dict,
    mouth_data: Dict,
    head_data: Dict,
    fps: float = 0.0
) -> np.ndarray:
    """
    Create complete dashboard overlay on frame.
    
    Args:
        frame: BGR image
        eye_data: Eye tracking results
        mouth_data: Mouth tracking results
        head_data: Head pose results
        fps: Current FPS
        
    Returns:
        Frame with complete overlay
    """
    visual = VisualAlert()
    h, w = frame.shape[:2]
    
    # Determine overall state
    if eye_data.get('is_drowsy'):
        state = "DROWSY"
        frame = visual.draw_alert_banner(frame, "⚠️ DROWSINESS DETECTED!", 'critical')
    elif head_data.get('is_distracted'):
        state = "DISTRACTED"
        frame = visual.draw_alert_banner(frame, "⚠️ KEEP EYES ON ROAD!", 'warning')
    elif mouth_data.get('is_yawning'):
        state = "YAWNING"
        frame = visual.draw_alert_banner(frame, "Yawn detected - Take a break", 'warning')
    else:
        state = "NORMAL"
    
    # Status panel
    status = {
        'ear': eye_data.get('smoothed_ear', 0),
        'mar': mouth_data.get('smoothed_mar', 0),
        'yaw': head_data.get('smoothed_yaw', 0),
        'pitch': head_data.get('smoothed_pitch', 0),
        'fps': fps,
        'state': state
    }
    frame = visual.draw_status_panel(frame, status)
    
    # Progress bars
    if eye_data.get('drowsiness_progress', 0) > 0:
        frame = visual.draw_progress_bar(
            frame, eye_data['drowsiness_progress'], "Drowsiness",
            (w - 220, 60), (200, 15)
        )
    
    if mouth_data.get('yawn_progress', 0) > 0:
        frame = visual.draw_progress_bar(
            frame, mouth_data['yawn_progress'], "Yawn",
            (w - 220, 90), (200, 15)
        )
    
    if head_data.get('distraction_progress', 0) > 0:
        frame = visual.draw_progress_bar(
            frame, head_data['distraction_progress'], "Distraction",
            (w - 220, 120), (200, 15)
        )
    
    return frame
