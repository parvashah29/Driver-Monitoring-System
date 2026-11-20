"""
Event logging system for Driver Monitoring System
Logs all fatigue events, alerts, and session statistics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import csv


@dataclass
class FatigueEvent:
    """Represents a single fatigue detection event"""
    timestamp: str
    event_type: str  # 'drowsiness', 'yawn', 'distraction'
    severity: str    # 'warning', 'critical'
    ear_value: Optional[float] = None
    mar_value: Optional[float] = None
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    duration_ms: Optional[int] = None


class DMSLogger:
    """Logger for Driver Monitoring System events"""
    
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session ID
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Initialize event list
        self.events: List[FatigueEvent] = []
        
        # Session statistics
        self.stats = {
            'total_frames': 0,
            'drowsy_frames': 0,
            'yawn_count': 0,
            'distraction_count': 0,
            'warning_alerts': 0,
            'critical_alerts': 0,
        }
        
        # Setup Python logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure Python logging"""
        log_file = self.log_dir / f"dms_session_{self.session_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DMS Session started: {self.session_id}")
    
    def log_event(self, event: FatigueEvent):
        """Log a fatigue event"""
        self.events.append(event)
        
        # Update statistics
        if event.event_type == 'drowsiness':
            self.stats['drowsy_frames'] += 1
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
            f"EAR: {ear_str}, MAR: {mar_str}"
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
    
    def update_frame_count(self, count: int = 1):
        """Update total frame count"""
        self.stats['total_frames'] += count
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary statistics"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
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
        return summary
