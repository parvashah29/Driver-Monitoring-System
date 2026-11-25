"""
Event logging system for Driver Monitoring System
Logs all fatigue events, alerts, and session statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import csv

from .config import LOGS_DIR


@dataclass
class FatigueEvent:
    """Represents a single fatigue detection event"""
    timestamp: str
    event_type: str  # 'drowsiness', 'yawn', 'distraction', 'no_face'
    severity: str    # 'warning', 'critical'
    ear_value: Optional[float] = None
    mar_value: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    duration_ms: Optional[int] = None


class DMSLogger:
    """Logger for Driver Monitoring System events"""
    
    def __init__(self, log_dir=None, session_id: Optional[str] = None, console: bool = True):
        # Default: <project>/logs (git-ignored), independent of the working directory
        self.log_dir = Path(log_dir) if log_dir else LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._console = console
        # Set to the media length (s) when processing a video file faster than real time
        self.duration_override: Optional[float] = None
        
        # Create session ID
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Initialize event list
        self.events: List[FatigueEvent] = []
        
        # Session statistics
        self.stats = {
            'total_frames': 0,
            'drowsy_frames': 0,   # frames spent in the drowsy state
            'drowsy_count': 0,    # distinct drowsiness events
            'no_face_count': 0,
            'yawn_count': 0,
            'distraction_count': 0,
            'warning_alerts': 0,
            'critical_alerts': 0,
        }
        
        # Setup Python logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Per-session logger writing to logs/dms_session_<id>.log (and the console)."""
        log_file = self.log_dir / f"dms_session_{self.session_id}.log"
        # Dedicated, non-propagating logger so sessions don't pollute (or depend on)
        # the root logger configuration.
        self.logger = logging.getLogger(f"dms.session.{self.session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        self._handlers = [logging.FileHandler(log_file)]
        if self._console:
            self._handlers.append(logging.StreamHandler(sys.stdout))
        for h in self._handlers:
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.info(f"DMS Session started: {self.session_id}")

    def log_event(self, event: FatigueEvent):
        """Log a fatigue event"""
        self.events.append(event)
        
        # Update statistics
        if event.event_type == 'drowsiness':
            self.stats['drowsy_count'] += 1
        elif event.event_type == 'no_face':
            self.stats['no_face_count'] += 1
        elif event.event_type == 'yawn':
            self.stats['yawn_count'] += 1
        elif event.event_type == 'distraction':
            self.stats['distraction_count'] += 1
        
        if event.severity == 'warning':
            self.stats['warning_alerts'] += 1
        elif event.severity == 'critical':
            self.stats['critical_alerts'] += 1
        
        # Log to file
        ear_str = f"{event.ear_value:.3f}" if event.ear_value is not None else "N/A"
        mar_str = f"{event.mar_value:.3f}" if event.mar_value is not None else "N/A"
        self.logger.warning(
            f"{event.event_type.upper()} [{event.severity}] - "
            f"EAR: {ear_str}, MAR: {mar_str}, "
            f"yaw: {event.head_yaw if event.head_yaw is not None else 'N/A'}"
        )
    
    def log_drowsiness(self, ear_value: float, severity: str = 'warning'):
        """Log a drowsiness event"""
        event = FatigueEvent(
            timestamp=datetime.now().isoformat(),
            event_type='drowsiness',
            severity=severity,
            ear_value=ear_value
        )
        self.log_event(event)
    
    def log_yawn(self, mar_value: float, duration_ms: int):
        """Log a yawn event"""
        event = FatigueEvent(
            timestamp=datetime.now().isoformat(),
            event_type='yawn',
            severity='warning',
            mar_value=mar_value,
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_distraction(self, yaw: float, pitch: float, severity: str = 'warning'):
        """Log a distraction event"""
        event = FatigueEvent(
            timestamp=datetime.now().isoformat(),
            event_type='distraction',
            severity=severity,
            head_yaw=yaw,
            head_pitch=pitch
        )
        self.log_event(event)
    
    def log_no_face(self, duration_ms: int):
        """Log a 'driver not detected' event"""
        self.log_event(FatigueEvent(
            timestamp=datetime.now().isoformat(),
            event_type='no_face',
            severity='warning',
            duration_ms=duration_ms,
        ))

    def update_frame_count(self, count: int = 1, drowsy: bool = False):
        """Update total frame count (and the drowsy-frame count if `drowsy`)"""
        self.stats['total_frames'] += count
        if drowsy:
            self.stats['drowsy_frames'] += count
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary statistics"""
        session_duration = self.duration_override if self.duration_override is not None else (
            datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'duration_seconds': session_duration,
            'total_events': len(self.events),
            **self.stats,
            'drowsiness_rate': self.stats['drowsy_frames'] / max(self.stats['total_frames'], 1),
            'events_per_minute': len(self.events) / max(session_duration / 60, 1),
        }
    
    def save_session(self):
        """Save session data to files"""
        # Save events to JSON
        events_file = self.log_dir / f"events_{self.session_id}.json"
        with open(events_file, 'w') as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
        
        # Save summary to JSON
        summary_file = self.log_dir / f"summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_session_summary(), f, indent=2)
        
        # Save events to CSV for easy analysis
        csv_file = self.log_dir / f"events_{self.session_id}.csv"
        if self.events:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.events[0]).keys())
                writer.writeheader()
                for event in self.events:
                    writer.writerow(asdict(event))
        
        self.logger.info(f"Session saved: {events_file}")
        return summary_file
    
    def close(self):
        """Close the session and save data"""
        self.save_session()
        summary = self.get_session_summary()
        self.logger.info(f"Session ended. Total events: {summary['total_events']}")
        for h in self._handlers:
            self.logger.removeHandler(h)
            h.close()
        return summary
