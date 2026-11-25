import numpy as np
import pytest

from src.detector import EyeTracker, HeadPoseEstimator, MouthTracker
from tests.helpers import make_face, make_pose_face


# ---------------------------------------------------------------- EAR
def test_ear_formula_known_value():
    pts = np.array([[0, 0], [3, 2], [7, 2], [10, 0], [7, -2], [3, -2]], dtype=float)
    assert EyeTracker.calculate_ear(pts) == pytest.approx((4 + 4) / (2 * 10))


def test_ear_degenerate_eye_is_zero():
    assert EyeTracker.calculate_ear(np.zeros((6, 2))) == 0.0


@pytest.mark.parametrize("ear", [0.05, 0.2, 0.3, 0.4])
def test_ear_from_landmarks(ear):
    assert EyeTracker().get_ear(make_face(ear=ear))[2] == pytest.approx(ear, abs=1e-6)


def test_blink_counted_but_not_drowsy():
    t = EyeTracker(ear_threshold=0.2, closed_seconds=1.0, history_size=1)
    ts = 0.0
    for ear in [0.3] * 5 + [0.05] * 4 + [0.3] * 5:   # ~0.13 s closure at 30 fps
        r = t.update_ear(ear, ts)
        ts += 1 / 30
        assert not r['is_drowsy']
    assert t.blink_counter == 1


def test_single_frame_dip_is_not_a_blink():
    t = EyeTracker(ear_threshold=0.2, closed_seconds=1.0, history_size=1)
    for i, ear in enumerate([0.3, 0.05, 0.3, 0.3]):
        t.update_ear(ear, i / 30)
    assert t.blink_counter == 0


def test_drowsy_after_closed_seconds_independent_of_fps():
    for fps in (10, 30, 60):
        t = EyeTracker(ear_threshold=0.2, closed_seconds=1.0, history_size=1)
        first = None
        for i in range(fps * 3):
            r = t.update_ear(0.05, i / fps)
            if r['is_drowsy'] and first is None:
                first = i / fps
        assert first == pytest.approx(1.0, abs=1.0 / fps + 1e-9)


def test_drowsy_clears_when_eyes_open_and_long_closure_is_not_a_blink():
    t = EyeTracker(ear_threshold=0.2, closed_seconds=0.5, history_size=1)
    for i in range(60):
        t.update_ear(0.05, i / 30)
    assert t.is_drowsy
    r = t.update_ear(0.35, 2.0)
    assert not r['is_drowsy'] and t.blink_counter == 0


def test_eye_calibration_sets_threshold_from_baseline():
    t = EyeTracker()
    thr = t.calibrate([0.30] * 50)
    assert thr == pytest.approx(0.72 * 0.30)
    assert t.baseline_ear == pytest.approx(0.30)
    with pytest.raises(ValueError):
        t.calibrate([0.3] * 3)


def test_eye_calibration_is_clamped():
    assert EyeTracker().calibrate([0.9] * 20) <= 0.30
    assert EyeTracker().calibrate([0.05] * 20) >= 0.12


# ---------------------------------------------------------------- MAR
@pytest.mark.parametrize("mar", [0.0, 0.3, 0.7])
def test_mar_from_landmarks(mar):
    assert MouthTracker().calculate_mar(make_face(mar=mar)) == pytest.approx(mar, abs=1e-6)


def test_yawn_requires_sustained_opening_and_counts_once():
    t = MouthTracker(mar_threshold=0.6, yawn_duration=2.0, history_size=1)
    for i in range(30):                      # 1 s open: not a yawn
        assert not t.update_mar(0.8, i / 30)['is_yawning']
    t.update_mar(0.0, 1.0)
    assert t.yawn_counter == 0
    flags = [t.update_mar(0.8, 2.0 + i / 30)['is_yawning'] for i in range(120)]  # 4 s open
    assert flags[-1] and not flags[0]
    assert t.yawn_counter == 1
    t.update_mar(0.0, 7.0)
    assert not t.is_yawning


# ---------------------------------------------------------------- head pose
@pytest.mark.parametrize("pose", [(0, 0, 0), (0, 25, 0), (0, -25, 0), (15, 0, 0), (-15, 0, 0),
                                  (0, 0, 20), (0, 0, -20), (10, -20, 8)])
def test_head_pose_recovers_known_angles(pose):
    est = HeadPoseEstimator()
    pitch, yaw, roll = est.estimate_pose(make_pose_face(pose))
    assert (pitch, yaw, roll) == pytest.approx(pose, abs=1.5)


def test_head_pose_no_180_degree_wraparound():
    est = HeadPoseEstimator()
    for pose in [(0, 0, 0), (5, 5, 3), (-8, 10, -4)]:
        _, _, roll = est.estimate_pose(make_pose_face(pose))
        assert abs(roll) < 20


def test_distraction_needs_duration_and_clears():
    h = HeadPoseEstimator(yaw_threshold=30, pitch_threshold=20, roll_threshold=25,
                          distraction_seconds=1.5, history_size=1)
    for i in range(30):                         # 1 s looking away
        assert not h.update_angles(0, 45, 0, i / 30)['is_distracted']
    assert h.update_angles(0, 45, 0, 1.6)['is_distracted']
    assert not h.update_angles(0, 0, 0, 1.7)['is_distracted']
    # brief glance resets
    h.update_angles(0, 45, 0, 2.0)
    h.update_angles(0, 0, 0, 2.2)
    assert h.distraction_duration == 0


def test_head_tilt_and_pitch_count_as_distraction():
    h = HeadPoseEstimator(distraction_seconds=0.1, history_size=1)
    h.update_angles(0, 0, 40, 0.0)
    assert h.update_angles(0, 0, 40, 0.2)['is_distracted']
    h.reset()
    h.update_angles(30, 0, 0, 0.0)
    assert h.update_angles(30, 0, 0, 0.2)['is_distracted']


def test_head_calibration_offsets_neutral_pose():
    h = HeadPoseEstimator(history_size=1)
    neutral = h.calibrate(np.tile([-18.0, 6.0, 1.0], (20, 1)))   # camera mounted high
    assert neutral == pytest.approx((-18.0, 6.0, 1.0))
    r = h.update_angles(-18.0, 6.0, 1.0, 0.0)
    assert abs(r['smoothed_pitch']) < 1e-9 and not r['looking_away']
    with pytest.raises(ValueError):
        h.calibrate(np.zeros((3, 3)))
