"""
Detection pipeline: frame in -> measurements, alert decisions and statistics out.

This module has no UI and no wall-clock dependency: every frame carries an
explicit timestamp (seconds), so recorded videos can be processed faster than real
time with exactly the behaviour a live camera would get.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .alerts.manager import AlertManager
from .detector import EyeTracker, FaceDetector, HeadPoseEstimator, MouthTracker
from .utils.config import ALERT_CONFIG

# Tracker state is cleared after this long without a face (brief detector
# dropouts do not wipe the state)
TRACKER_RESET_AFTER = 0.5  # seconds

_KIND_TO_STATE = {
    'drowsy': 'DROWSY', 'distracted': 'DISTRACTED', 'yawn': 'YAWNING', 'no_face': 'NO DRIVER',
}


@dataclass
class FrameResult:
    timestamp: float
    face_detected: bool
    face_data: object = None
    eye: Dict = field(default_factory=dict)
    mouth: Dict = field(default_factory=dict)
    head: Dict = field(default_factory=dict)
    state: str = 'NORMAL'
    alert_level: int = 0
    fired: Optional[str] = None          # alert kind announced on this frame
    new_episodes: List[str] = field(default_factory=list)
    calibrating: bool = False


def _summary(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    p1, p5, med, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    return {'min': float(np.min(values)), 'p1': float(p1), 'p5': float(p5),
            'median': float(med), 'p95': float(p95), 'p99': float(p99),
            'max': float(np.max(values))}


class DMSPipeline:
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        eye_tracker: Optional[EyeTracker] = None,
        mouth_tracker: Optional[MouthTracker] = None,
        head_pose: Optional[HeadPoseEstimator] = None,
        alert_manager: Optional[AlertManager] = None,
        audio=None,
        logger=None,
        no_face_seconds: Optional[float] = None,
    ):
        self.detector = detector or FaceDetector()
        self.eye = eye_tracker or EyeTracker()
        self.mouth = mouth_tracker or MouthTracker()
        self.head = head_pose or HeadPoseEstimator()
        self.alerts = alert_manager or AlertManager()
        self.audio = audio
        self.logger = logger
        self.no_face_seconds = (ALERT_CONFIG.NO_FACE_SECONDS
                                if no_face_seconds is None else no_face_seconds)

        self._no_face_since: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._first_ts: Optional[float] = None

        # calibration
        self._calib_until: Optional[float] = None
        self._calib_ears: List[float] = []
        self._calib_poses: List[tuple] = []
        self.calibration: Optional[Dict] = None

        # statistics (raw per-frame values for frames with a face)
        self.frames = 0
        self.face_frames = 0
        self.drowsy_frames = 0
        self._stats = {k: [] for k in ('ear', 'mar', 'yaw', 'pitch', 'roll')}

    # ------------------------------------------------------------ calibration
    def start_calibration(self, seconds: float, now: float):
        """Collect open-eye / neutral-pose samples for `seconds` starting at `now`."""
        self._calib_until = now + seconds
        self._calib_ears.clear()
        self._calib_poses.clear()
        self.calibration = None

    @property
    def calibrating(self) -> bool:
        return self._calib_until is not None

    def _finish_calibration(self):
        self._calib_until = None
        try:
            threshold = self.eye.calibrate(self._calib_ears)
            neutral = self.head.calibrate(np.array(self._calib_poses))
            self.calibration = {
                'ok': True, 'ear_baseline': self.eye.baseline_ear,
                'ear_threshold': threshold,
                'neutral_pitch_yaw_roll': list(neutral),
                'samples': len(self._calib_ears),
            }
        except ValueError as exc:
            self.calibration = {'ok': False, 'reason': f"{exc} (got {len(self._calib_ears)})",
                                'samples': len(self._calib_ears)}
        self.eye.reset()
        self.head.reset()
        self.mouth.reset()

    # -------------------------------------------------------------- per frame
    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        return self.process_face(self.detector.detect(frame), timestamp)

    def process_face(self, face, timestamp: float) -> FrameResult:
        """Run trackers and alert logic for an already detected face (or None)."""
        if self._first_ts is None:
            self._first_ts = timestamp
        self._last_ts = timestamp
        self.frames += 1
        result = FrameResult(timestamp=timestamp, face_detected=face is not None, face_data=face)

        if face is None:
            if self._no_face_since is None:
                self._no_face_since = timestamp
            absent = timestamp - self._no_face_since
            if absent >= TRACKER_RESET_AFTER:
                self.eye.reset()
                self.mouth.reset()
                self.head.reset()
            active = {'no_face': absent >= self.no_face_seconds and not self.calibrating}
            self._decide(result, active, timestamp, absent_ms=int(absent * 1000))
            return result

        self._no_face_since = None
        self.face_frames += 1
        result.eye = self.eye.update(face, timestamp)
        result.mouth = self.mouth.update(face, timestamp)
        result.head = self.head.update(face, timestamp)

        raw_pose = (result.head['raw_pitch'], result.head['raw_yaw'], result.head['raw_roll'])
        self._stats['ear'].append(result.eye['avg_ear'])
        self._stats['mar'].append(result.mouth['mar'])
        self._stats['pitch'].append(raw_pose[0])
        self._stats['yaw'].append(raw_pose[1])
        self._stats['roll'].append(raw_pose[2])

        if self.calibrating:
            self._calib_ears.append(result.eye['avg_ear'])
            self._calib_poses.append(raw_pose)
            result.calibrating = True
            result.state = 'CALIBRATING'
            if timestamp >= self._calib_until:
                self._finish_calibration()
            return result

        if result.eye['is_drowsy']:
            self.drowsy_frames += 1
        active = {
            'drowsy': result.eye['is_drowsy'],
            'distracted': result.head['is_distracted'],
            'yawn': result.mouth['is_yawning'],
        }
        self._decide(result, active, timestamp)
        return result

    def _decide(self, result: FrameResult, active: Dict[str, bool], now: float, absent_ms: int = 0):
        decision = self.alerts.update(active, now)
        result.new_episodes = decision.new_episodes
        result.fired = decision.fire
        if decision.active:
            result.state = _KIND_TO_STATE[decision.active]
            result.alert_level = 2 if decision.active == 'drowsy' else 1

        if self.logger:
            self.logger.update_frame_count(1, drowsy=bool(active.get('drowsy')))
            for kind in decision.new_episodes:
                if kind == 'drowsy':
                    self.logger.log_drowsiness(result.eye['smoothed_ear'], 'critical')
                elif kind == 'distracted':
                    self.logger.log_distraction(result.head['smoothed_yaw'],
                                                result.head['smoothed_pitch'], 'warning')
                elif kind == 'yawn':
                    self.logger.log_yawn(result.mouth['smoothed_mar'],
                                         int(result.mouth['yawn_duration'] * 1000))
                elif kind == 'no_face':
                    self.logger.log_no_face(absent_ms)

        if self.audio and decision.fire:
            {
                'drowsy': lambda: self.audio.alert_drowsy('critical'),
                'distracted': self.audio.alert_distracted,
                'yawn': self.audio.alert_yawn,
                'no_face': self.audio.alert_no_face,
            }[decision.fire]()

    # ---------------------------------------------------------------- report
    def report(self) -> Dict:
        """Aggregate statistics for the frames processed so far."""
        media_seconds = (self._last_ts - self._first_ts) if self.frames > 1 else 0.0
        return {
            'frames': self.frames,
            'media_seconds': round(media_seconds, 2),
            'face_frames': self.face_frames,
            'face_fraction': self.face_frames / self.frames if self.frames else 0.0,
            'blinks': self.eye.blink_counter,
            'drowsy_frames': self.drowsy_frames,
            'episodes': dict(self.alerts.episode_counts),
            'alerts_fired': dict(self.alerts.fire_counts),
            'ear': _summary(self._stats['ear']),
            'mar': _summary(self._stats['mar']),
            'yaw': _summary(self._stats['yaw']),
            'pitch': _summary(self._stats['pitch']),
            'roll': _summary(self._stats['roll']),
            'ear_threshold': self.eye.ear_threshold,
            'mar_threshold': self.mouth.mar_threshold,
            'calibration': self.calibration,
        }
