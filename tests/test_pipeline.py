import json

import pytest

from src.alerts import AlertManager
from src.detector import EyeTracker, HeadPoseEstimator, MouthTracker
from src.pipeline import DMSPipeline
from src.utils.logger import DMSLogger
from tests.helpers import make_face


class StubDetector:
    def detect(self, frame):
        return None

    def close(self):
        pass


class FakeAudio:
    def __init__(self):
        self.calls = []

    def alert_drowsy(self, severity='warning'): self.calls.append('drowsy')
    def alert_yawn(self): self.calls.append('yawn')
    def alert_distracted(self): self.calls.append('distracted')
    def alert_no_face(self): self.calls.append('no_face')


def make_pipeline(**kw):
    return DMSPipeline(
        detector=StubDetector(),
        eye_tracker=EyeTracker(ear_threshold=0.2, closed_seconds=1.0, history_size=1),
        mouth_tracker=MouthTracker(mar_threshold=0.6, yawn_duration=2.0, history_size=1),
        head_pose=HeadPoseEstimator(distraction_seconds=1.5, history_size=1),
        alert_manager=AlertManager(cooldown=5.0),
        **kw)


def run(p, face, seconds, start=0.0, fps=30):
    out = None
    for i in range(int(seconds * fps)):
        out = p.process_face(face, start + i / fps)
    return out


def test_no_face_never_triggers_fatigue_alerts_and_no_false_alarm_early():
    audio = FakeAudio()
    p = make_pipeline(audio=audio)
    run(p, None, 2.5)                       # shorter than NO_FACE_SECONDS (3 s)
    assert audio.calls == []
    run(p, None, 3.0, start=2.5)
    assert audio.calls == ['no_face']       # exactly one, then cooldown
    assert p.alerts.episode_counts['drowsy'] == 0


def test_normal_face_no_alerts():
    audio = FakeAudio()
    p = make_pipeline(audio=audio)
    r = run(p, make_face(ear=0.3, mar=0.02), 5)
    assert r.state == 'NORMAL' and audio.calls == []


def test_drowsy_alert_once_per_cooldown_and_logged_once(tmp_path):
    audio = FakeAudio()
    log = DMSLogger(log_dir=tmp_path, session_id="t", console=False)
    p = make_pipeline(audio=audio, logger=log)
    r = run(p, make_face(ear=0.05), 12)     # 12 s of closed eyes
    assert r.state == 'DROWSY' and r.alert_level == 2
    assert audio.calls == ['drowsy'] * 3    # t~1, ~6, ~11 s
    summary = log.close()
    assert summary['drowsy_count'] == 1     # one episode, not one per frame
    assert summary['drowsy_frames'] > 200   # frames in drowsy state are counted separately
    events = json.load(open(tmp_path / "events_t.json"))
    assert [e['event_type'] for e in events] == ['drowsiness']


def test_yawn_and_distraction_states():
    p = make_pipeline(audio=FakeAudio())
    assert run(p, make_face(mar=0.8), 3).state == 'YAWNING'
    p2 = make_pipeline()
    assert run(p2, make_face(pose=(0, 40, 0)), 3).state == 'DISTRACTED'


def test_trackers_reset_after_face_lost():
    p = make_pipeline()
    run(p, make_face(ear=0.05), 0.8)
    assert p.eye.closed_duration > 0.5
    run(p, None, 1.0, start=0.8)
    assert p.eye.closed_duration == 0 and not p.eye.is_drowsy


def test_calibration_adapts_threshold_and_suppresses_alerts():
    audio = FakeAudio()
    p = make_pipeline(audio=audio)
    p.start_calibration(2.0, 0.0)
    r = run(p, make_face(ear=0.40, pose=(-10, 5, 0)), 2.5)
    assert p.calibration['ok'] and not p.calibrating
    assert p.eye.ear_threshold == pytest.approx(0.72 * 0.40, abs=1e-3)
    assert p.head.neutral[0] == pytest.approx(-10, abs=1.5)
    assert audio.calls == []


def test_calibration_fails_gracefully_without_face():
    p = make_pipeline()
    p.start_calibration(1.0, 0.0)
    run(p, make_face(), 0.2)                 # only ~6 samples
    p._calib_until = 0.0
    p.process_face(make_face(), 0.3)
    assert p.calibration['ok'] is False


def test_report_contents():
    p = make_pipeline()
    run(p, make_face(ear=0.3), 2)
    run(p, None, 1, start=2)
    rep = p.report()
    assert rep['frames'] == 90 and rep['face_frames'] == 60
    assert rep['face_fraction'] == pytest.approx(60 / 90)
    assert rep['ear']['median'] == pytest.approx(0.3, abs=1e-6)
