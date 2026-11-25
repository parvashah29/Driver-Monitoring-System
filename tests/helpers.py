"""Synthetic face builders: landmark arrays with exactly known EAR / MAR / head pose."""

import cv2
import numpy as np

from src.detector.face_detector import FaceData
from src.utils.config import EYE_CONFIG, HEAD_POSE_CONFIG

W, H = 640, 480


def rotation_from_angles(pitch, yaw, roll):
    """Inverse of HeadPoseEstimator._rotation_matrix_to_euler (module sign convention)."""
    p, y, r = np.radians(-pitch), np.radians(-yaw), np.radians(roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_pose(pitch, yaw, roll, distance=2000.0):
    """Pixel positions of the six pose landmarks for a given head pose."""
    R = rotation_from_angles(pitch, yaw, roll)
    rvec, _ = cv2.Rodrigues(R)
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    pts, _ = cv2.projectPoints(np.array(HEAD_POSE_CONFIG.MODEL_POINTS_3D, dtype=np.float64),
                               rvec, np.array([0.0, 0.0, distance]), cam, np.zeros(4))
    return pts.reshape(-1, 2)


def _set_px(lm, idx, x, y):
    lm[idx, 0] = x / W
    lm[idx, 1] = y / H


def make_face(ear=0.30, mar=0.02, pose=(0.0, 0.0, 0.0)):
    """
    FaceData whose measured EAR/MAR equal the arguments and whose head pose is `pose`.
    Eye/mouth corners double as pose landmarks, so eyes and mouth are built around
    the projected pose points (axis-aligned, which keeps EAR/MAR exact).
    """
    lm = np.full((478, 3), 0.5)
    pts = dict(zip(HEAD_POSE_CONFIG.LANDMARK_INDICES, project_pose(*pose)))
    for idx, (x, y) in pts.items():
        _set_px(lm, idx, x, y)

    eye_w = 40.0
    h = ear * eye_w / 2.0
    # right eye (image-left): P1=33 is a pose landmark, P4=133 = P1 + width
    p1, p2, p3, p4, p5, p6 = EYE_CONFIG.RIGHT_EYE_INDICES
    x0, y0 = pts[33]
    _set_px(lm, p4, x0 + eye_w, y0)
    for a, b, dx, sgn in ((p2, p6, 0.2, 1), (p3, p5, 0.8, 1)):
        _set_px(lm, a, x0 + dx * eye_w, y0 - h)
        _set_px(lm, b, x0 + dx * eye_w, y0 + h)
    # left eye (image-right): P4=263 is a pose landmark, P1=362 = P4 - width
    p1, p2, p3, p4, p5, p6 = EYE_CONFIG.LEFT_EYE_INDICES
    x4, y4 = pts[263]
    _set_px(lm, p1, x4 - eye_w, y4)
    for a, b, dx in ((p2, p6, 0.2), (p3, p5, 0.8)):
        _set_px(lm, a, x4 - eye_w + dx * eye_w, y4 - h)
        _set_px(lm, b, x4 - eye_w + dx * eye_w, y4 + h)

    # mouth: corners 61/291 are pose landmarks; MAR = opening / corner distance
    (xl, yl), (xr, yr) = pts[61], pts[291]
    width = np.hypot(xr - xl, yr - yl)
    opening = mar * width
    cx, cy = (xl + xr) / 2, (yl + yr) / 2
    for top, bot, fx in ((13, 14, 0.5), (82, 87, 0.3)):
        x = xl + fx * (xr - xl)
        y = yl + fx * (yr - yl)
        _set_px(lm, top, x, y - opening / 2)
        _set_px(lm, bot, x, y + opening / 2)
    return FaceData(landmarks=lm, bbox=(0, 0, W, H), confidence=1.0, image_width=W, image_height=H)


def make_pose_face(pose):
    """FaceData carrying only the pose landmarks (EAR/MAR are meaningless)."""
    lm = np.full((478, 3), 0.5)
    pts = project_pose(*pose)
    for idx, (x, y) in zip(HEAD_POSE_CONFIG.LANDMARK_INDICES, pts):
        _set_px(lm, idx, x, y)
    return FaceData(landmarks=lm, bbox=(0, 0, W, H), confidence=1.0, image_width=W, image_height=H)
