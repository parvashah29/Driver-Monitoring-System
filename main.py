#!/usr/bin/env python3
"""
Driver Monitoring System - Main Application
Real-time driver fatigue and distraction detection.

Usage:
    python main.py                    # Run with webcam
    python main.py --video path/to/video.mp4  # Run with video file
    python main.py --no-audio         # Disable audio alerts
    python main.py --debug            # Enable debug visualization
"""

import argparse
import time
import cv2
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import (
    FaceDetector, 
    EyeTracker, 
    MouthTracker, 
    HeadPoseEstimator,
    draw_eye_landmarks,
    draw_mouth_landmarks,
    draw_head_pose
)
from src.alerts import AudioAlert, VisualAlert, create_dashboard_overlay
from src.utils import CAMERA_CONFIG, EYE_CONFIG, DMSLogger


class DriverMonitoringSystem:
    """
    Main Driver Monitoring System application.
    Combines all detection modules for real-time driver monitoring.
    """
    
    def __init__(
        self,
        video_source: int = 0,
        enable_audio: bool = True,
        enable_logging: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize the Driver Monitoring System.
        
        Args:
            video_source: Camera index or video file path
            enable_audio: Whether to enable audio alerts
            enable_logging: Whether to enable event logging
            debug_mode: Whether to show debug visualizations
        """
        self.video_source = video_source
        self.debug_mode = debug_mode
        
        # Initialize detectors
        print("Initializing face detector...")
        self.face_detector = FaceDetector(
            detection_confidence=CAMERA_CONFIG.DETECTION_CONFIDENCE,
            tracking_confidence=CAMERA_CONFIG.TRACKING_CONFIDENCE
        )
        
        print("Initializing eye tracker...")
        self.eye_tracker = EyeTracker()
        
        print("Initializing mouth tracker...")
        self.mouth_tracker = MouthTracker()
        
        print("Initializing head pose estimator...")
        self.head_pose = HeadPoseEstimator()
        
        # Initialize alerts
        self.audio_alert = AudioAlert(enable_voice=True) if enable_audio else None
        self.visual_alert = VisualAlert()
        
        # Initialize logger
        self.logger = DMSLogger() if enable_logging else None
        
        # Video capture and recording
        self.cap = None
        self.video_writer = None
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance tracking
        self.fps_history = []
        self.fps_history_size = 30
        
        # Alert state tracking (to prevent repeated alerts)
        self._last_alert_type = None
        self._alert_cooldown = 5.0  # seconds
        self._last_alert_time = 0
        
        print("DMS initialized successfully!")
    
    def _calculate_fps(self, frame_time: float) -> float:
        """Calculate smoothed FPS."""
        fps = 1.0 / max(frame_time, 0.001)
        self.fps_history.append(fps)
        if len(self.fps_history) > self.fps_history_size:
            self.fps_history.pop(0)
        return sum(self.fps_history) / len(self.fps_history)
    
    def _should_alert(self, alert_type: str) -> bool:
        """Check if we should trigger an alert (with cooldown)."""
        current_time = time.time()
        if (alert_type != self._last_alert_type or 
            current_time - self._last_alert_time >= self._alert_cooldown):
            self._last_alert_type = alert_type
            self._last_alert_time = current_time
            return True
        return False
    
    def process_frame(self, frame: cv2.Mat) -> tuple:
        """
        Process a single frame through all detection modules.
        
        Args:
            frame: BGR image from camera/video
            
        Returns:
            Tuple of (processed_frame, results_dict)
        """
        # Detect face
        face_data = self.face_detector.detect(frame)
        
        results = {
            'face_detected': face_data is not None,
            'eye': {},
            'mouth': {},
            'head': {},
            'alert_level': 0
        }
        
        if face_data is None:
            # No face detected
            frame = self.visual_alert.draw_face_indicator(frame, False)
            
            if self.audio_alert and self._should_alert('no_face'):
                self.audio_alert.alert_no_face()
            
            return frame, results
        
        # Update all trackers
        eye_results = self.eye_tracker.update(face_data)
        mouth_results = self.mouth_tracker.update(face_data)
        head_results = self.head_pose.update(face_data)
        
        results['eye'] = eye_results
        results['mouth'] = mouth_results
        results['head'] = head_results
        
        # Debug visualization
        if self.debug_mode:
            # Draw face landmarks
            self.face_detector.draw_landmarks(frame, face_data, draw_contours=True)
            
            # Draw eye landmarks
            draw_eye_landmarks(
                frame, face_data,
                EYE_CONFIG.LEFT_EYE_INDICES,
                EYE_CONFIG.RIGHT_EYE_INDICES
            )
            
            # Draw mouth landmarks
            draw_mouth_landmarks(frame, face_data)
            
            # Draw head pose axes
            draw_head_pose(
                frame, face_data,
                head_results['pitch'],
                head_results['yaw'],
                head_results['roll']
            )
        
        # Check for alerts
        if eye_results['is_drowsy']:
            results['alert_level'] = 2
            if self.audio_alert and self._should_alert('drowsy'):
                self.audio_alert.alert_drowsy('critical')
            if self.logger:
                self.logger.log_drowsiness(eye_results['smoothed_ear'], 'critical')
        
        elif head_results['is_distracted']:
            results['alert_level'] = 1
            if self.audio_alert and self._should_alert('distracted'):
                self.audio_alert.alert_distracted()
            if self.logger:
                self.logger.log_distraction(
                    head_results['smoothed_yaw'],
                    head_results['smoothed_pitch'],
                    'warning'
                )
        
        elif mouth_results['is_yawning']:
            results['alert_level'] = 1
            if self.audio_alert and self._should_alert('yawn'):
                self.audio_alert.alert_yawn()
            if self.logger:
                self.logger.log_yawn(
                    mouth_results['smoothed_mar'],
                    int(mouth_results['yawn_duration'] * 1000)
                )
        
        # Draw face bounding box
        frame = self.visual_alert.draw_face_indicator(frame, True, face_data.bbox)
        
        return frame, results
    
    def run(self):
        """Run the main detection loop."""
        # Open video source
        if isinstance(self.video_source, str):
            self.cap = cv2.VideoCapture(self.video_source)
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.video_source}")
            return
        
        # Get video properties for recording
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Initialize video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"dms_session_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print("\n" + "="*50)
        print("Driver Monitoring System - Running")
        print("="*50)
        print(f"Recording to: {output_path}")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug mode")
        print("Press 'r' to reset counters")
        print("="*50 + "\n")
        
        prev_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video file or camera error
                    if isinstance(self.video_source, str):
                        print("End of video file.")
                        break
                    else:
                        print("Error reading from camera.")
                        break
                
                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - prev_time
                prev_time = current_time
                fps = self._calculate_fps(frame_time)
                
                # Update logger frame count
                if self.logger:
                    self.logger.update_frame_count()
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Create dashboard overlay
                processed_frame = create_dashboard_overlay(
                    processed_frame,
                    results['eye'],
                    results['mouth'],
                    results['head'],
                    fps
                )
                
                # Write frame to video
                if self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Display
                cv2.imshow('Driver Monitoring System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quit requested.")
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('r'):
                    self.eye_tracker.reset_blink_counter()
                    self.mouth_tracker.reset_yawn_counter()
                    print("Counters reset.")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to: {self.output_dir}/")
        
        cv2.destroyAllWindows()
        
        if self.face_detector:
            self.face_detector.close()
        
        if self.audio_alert:
            self.audio_alert.close()
        
        if self.logger:
            summary = self.logger.close()
            print("\n" + "="*50)
            print("Session Summary")
            print("="*50)
            print(f"Duration: {summary['duration_seconds']:.1f}s")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Drowsiness events: {summary['drowsy_frames']}")
            print(f"Yawn count: {summary['yawn_count']}")
            print(f"Distraction events: {summary['distraction_count']}")
            print("="*50)
        
        print("Cleanup complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Driver Monitoring System - Real-time fatigue detection'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Path to video file (default: use webcam)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio alerts'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable event logging'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug visualization'
    )
    
    args = parser.parse_args()
    
    # Determine video source
    video_source = args.video if args.video else args.camera
    
    # Create and run DMS
    dms = DriverMonitoringSystem(
        video_source=video_source,
        enable_audio=not args.no_audio,
        enable_logging=not args.no_log,
        debug_mode=args.debug
    )
    
    dms.run()


if __name__ == '__main__':
    main()
