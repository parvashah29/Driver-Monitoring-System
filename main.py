#!/usr/bin/env python3
"""
Driver Monitoring System - command line application.
Real-time driver drowsiness, yawn and distraction detection.

Usage:
    python main.py                              # webcam, live window
    python main.py --video clip.mp4             # analyse a video file
    python main.py --video clip.mp4 --headless --report out.json   # no window, JSON stats
    python main.py --video clip.mp4 --headless --save out.mp4      # write annotated video
    python main.py --calibrate 5                # 5 s per-driver calibration at start
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.alerts import AudioAlert, create_dashboard_overlay  # noqa: E402
from src.detector import (  # noqa: E402
    FaceDetector, draw_eye_landmarks, draw_head_pose, draw_mouth_landmarks,
)
from src.detector.face_detector import ModelUnavailableError  # noqa: E402
from src.pipeline import DMSPipeline, FrameResult  # noqa: E402
from src.utils import CAMERA_CONFIG, EYE_CONFIG, DMSLogger  # noqa: E402
from src.utils.config import RECORDINGS_DIR  # noqa: E402


def annotate(frame: np.ndarray, result: FrameResult, fps: float, debug: bool, privacy: bool) -> np.ndarray:
    """Draw landmarks (debug/privacy) and the HUD. Privacy mode replaces the image by black."""
    if privacy:
        frame = np.zeros_like(frame)
    face = result.face_data
    if face is not None and (debug or privacy):
        FaceDetector.draw_landmarks(frame, face, draw_contours=True)
        draw_eye_landmarks(frame, face, EYE_CONFIG.LEFT_EYE_INDICES, EYE_CONFIG.RIGHT_EYE_INDICES)
        draw_mouth_landmarks(frame, face)
        if result.head:
            draw_head_pose(frame, face, result.head['pitch'], result.head['yaw'])
    return create_dashboard_overlay(frame, result.eye, result.mouth, result.head, fps,
                                    state=result.state)


def open_source(args):
    """Open the video file or webcam; returns (capture, is_file) or exits with a clear message."""
    if args.video:
        if not os.path.isfile(args.video):
            sys.exit(f"Error: video file not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Error: could not decode video file: {args.video}")
        return cap, True
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"Error: could not open camera {args.camera}.\n"
            "  - Is a webcam connected and not used by another app?\n"
            "  - macOS: grant camera access to your terminal (System Settings > Privacy & Security > Camera).\n"
            "  - Try another index with --camera 1, or analyse a recording with --video PATH."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG.FRAME_HEIGHT)
    return cap, False


def run(args) -> int:
    cap, is_file = open_source(args)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if args.fps:
        src_fps = args.fps  # container metadata can be wrong (e.g. webcam recordings)
    elif not src_fps or src_fps != src_fps or src_fps < 1:
        src_fps = 30.0

    try:
        detector = FaceDetector(detection_confidence=CAMERA_CONFIG.DETECTION_CONFIDENCE,
                                tracking_confidence=CAMERA_CONFIG.TRACKING_CONFIDENCE)
    except ModelUnavailableError as exc:
        cap.release()
        sys.exit(f"Error: {exc}")

    audio = None
    if not args.no_audio and not args.headless:
        audio = AudioAlert(enable_voice=not args.no_voice)
        print(f"Audio backend: {audio.backend}")
    logger = DMSLogger(console=not args.headless) if not args.no_log else None
    pipeline = DMSPipeline(detector=detector, audio=audio, logger=logger)

    show = not args.headless
    writer: Optional[cv2.VideoWriter] = None
    save_path = None
    if args.save is not None:
        save_path = args.save or str(RECORDINGS_DIR / f"dms_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    print("\n" + "=" * 50)
    print("Driver Monitoring System - running")
    print(f"Source: {args.video or f'camera {args.camera}'}"
          f"{' (headless)' if args.headless else ''}")
    if show:
        print("Keys: q quit | d debug landmarks | r reset counters")
    print("=" * 50)

    debug = args.debug
    smoothed_fps = 0.0
    prev_wall = time.perf_counter()
    start_wall = prev_wall
    frame_idx = 0
    exit_code = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of video." if is_file else "Error: camera stopped delivering frames.")
                exit_code = 0 if is_file else 1
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            # Video files use media time (frame_idx / fps) so processing speed is irrelevant
            ts = frame_idx / src_fps if is_file else time.monotonic()
            if args.calibrate and frame_idx == 0:
                pipeline.start_calibration(args.calibrate, ts)
                print(f"Calibrating for {args.calibrate:.0f}s - look at the road with eyes open...")

            result = pipeline.process(frame, ts)
            frame_idx += 1

            now = time.perf_counter()
            inst = 1.0 / max(now - prev_wall, 1e-3)
            prev_wall = now
            smoothed_fps = inst if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * inst

            if show or save_path:
                out = annotate(frame, result, smoothed_fps, debug, args.privacy)
                if save_path and writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), src_fps, (w, h))
                    if not writer.isOpened():
                        print(f"Warning: could not open {save_path} for writing; not saving.")
                        save_path, writer = None, None
                if writer:
                    writer.write(out)
                if show:
                    try:
                        cv2.imshow('Driver Monitoring System', out)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        print("Warning: no display available (OpenCV GUI failed); continuing headless.")
                        show = False
                        key = 255
                    if key == ord('q'):
                        print("Quit requested.")
                        break
                    elif key == ord('d'):
                        debug = not debug
                    elif key == ord('r'):
                        pipeline.eye.reset_blink_counter()
                        pipeline.mouth.reset_yawn_counter()
                        print("Counters reset.")
            if args.headless and frame_idx % 500 == 0:
                print(f"  ... {frame_idx} frames ({frame_idx / (time.perf_counter() - start_wall):.0f} fps)")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        wall = time.perf_counter() - start_wall
        cap.release()
        if writer:
            writer.release()
            print(f"Annotated video saved to: {save_path}")
        if show:
            cv2.destroyAllWindows()
        detector.close()
        if audio:
            audio.close()
        report = pipeline.report()
        report['processing_fps'] = frame_idx / wall if wall > 0 else 0.0
        report['source'] = args.video or f"camera:{args.camera}"
        if logger:
            if is_file:
                logger.duration_override = report['media_seconds']
            logger.close()
        print_report(report)
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report written to {args.report}")
    return exit_code


def print_report(r: dict):
    print("\n" + "=" * 50)
    print("Session summary")
    print("=" * 50)
    print(f"Frames: {r['frames']}  ({r['media_seconds']:.1f}s of media, {r['processing_fps']:.0f} fps processing)")
    print(f"Face found in {r['face_fraction'] * 100:.1f}% of frames")
    if r['ear']:
        print(f"EAR  median {r['ear']['median']:.3f}  [p5 {r['ear']['p5']:.3f}, p95 {r['ear']['p95']:.3f}]"
              f"   threshold {r['ear_threshold']:.3f}")
        print(f"MAR  median {r['mar']['median']:.3f}  [p5 {r['mar']['p5']:.3f}, p95 {r['mar']['p95']:.3f}]"
              f"   threshold {r['mar_threshold']:.2f}")
    if r['calibration']:
        print(f"Calibration: {r['calibration']}")
    ep = r['episodes']
    print(f"Blinks {r['blinks']} | drowsy episodes {ep['drowsy']} | yawns {ep['yawn']} | "
          f"distraction episodes {ep['distracted']} | no-driver episodes {ep['no_face']}")
    print("=" * 50)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Driver Monitoring System - real-time fatigue detection')
    p.add_argument('--video', '-v', default=None, help='analyse this video file instead of the webcam')
    p.add_argument('--camera', '-c', type=int, default=0, help='camera index (default 0)')
    p.add_argument('--headless', '--no-display', dest='headless', action='store_true',
                   help='no window and no audio (for servers / batch analysis)')
    p.add_argument('--save', nargs='?', const='', default=None, metavar='PATH',
                   help='write the annotated video (default path: recordings/dms_<time>.mp4)')
    p.add_argument('--privacy', action='store_true',
                   help='in saved/displayed output replace the image with black and draw landmarks only')
    p.add_argument('--report', metavar='PATH', help='write a JSON statistics report')
    p.add_argument('--calibrate', type=float, default=0, metavar='SECONDS',
                   help='personalise EAR threshold and neutral head pose from the first SECONDS')
    p.add_argument('--fps', type=float, default=0,
                   help='override the video frame rate used for timing (if file metadata is wrong)')
    p.add_argument('--max-frames', type=int, default=0, help='stop after N frames')
    p.add_argument('--no-audio', action='store_true', help='disable audio alerts')
    p.add_argument('--no-voice', action='store_true', help='beeps only, no text-to-speech')
    p.add_argument('--no-log', action='store_true', help='do not write session logs to logs/')
    p.add_argument('--debug', '-d', action='store_true', help='draw landmarks and pose')
    return p


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
