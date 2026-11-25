import json
import os
import urllib.error

import pytest

from src.detector import face_detector as fd
from src.utils import config
from src.utils.logger import DMSLogger


# ------------------------------------------------------------ model download
def test_model_path_is_project_relative():
    assert fd.MODEL_PATH.startswith(str(config.PROJECT_ROOT))
    assert fd.MODEL_PATH.endswith(os.path.join("models", "face_landmarker.task"))


def test_download_skipped_when_present(tmp_path, monkeypatch):
    p = tmp_path / "m.task"
    p.write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(fd.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no network expected"))
    assert fd.download_model(str(p)) == str(p)


def test_download_offline_gives_clear_error_and_no_partial_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(fd.urllib.request, "urlopen", boom)
    p = tmp_path / "models" / "m.task"
    with pytest.raises(fd.ModelUnavailableError, match="internet connection"):
        fd.download_model(str(p))
    assert not p.exists() and not (tmp_path / "models" / "m.task.part").exists()


def test_download_from_local_url_is_atomic(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"y" * 1_500_000)
    dst = tmp_path / "out" / "m.task"
    fd.download_model(str(dst), url=src.as_uri())
    assert dst.stat().st_size == 1_500_000 and not (tmp_path / "out" / "m.task.part").exists()


def test_truncated_download_rejected(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"y" * 100)
    dst = tmp_path / "m.task"
    with pytest.raises(fd.ModelUnavailableError):
        fd.download_model(str(dst), url=src.as_uri())
    assert not dst.exists()


# ------------------------------------------------------------------- config
def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DMS_EAR_THRESHOLD", "0.17")
    monkeypatch.setenv("DMS_YAW_THRESHOLD", "not-a-number")
    assert config.EyeConfig().EAR_THRESHOLD == 0.17
    assert config.HeadPoseConfig().YAW_THRESHOLD == 30.0


# ------------------------------------------------------------------- logger
def test_logger_writes_files_to_given_dir(tmp_path):
    log = DMSLogger(log_dir=tmp_path / "nested", session_id="abc", console=False)
    log.log_drowsiness(0.1, 'critical')
    log.log_yawn(0.7, 2100)
    log.log_distraction(40.0, 3.0)
    log.log_no_face(3500)
    s = log.close()
    d = tmp_path / "nested"
    for name in ("dms_session_abc.log", "events_abc.json", "events_abc.csv", "summary_abc.json"):
        assert (d / name).exists()
    assert s['drowsy_count'] == 1 and s['yawn_count'] == 1 and s['distraction_count'] == 1
    assert s['no_face_count'] == 1 and len(json.load(open(d / "events_abc.json"))) == 4


def test_two_sessions_in_one_process_each_get_a_log_file(tmp_path):
    for sid in ("one", "two"):
        l = DMSLogger(log_dir=tmp_path, session_id=sid, console=False)
        l.log_yawn(0.7, 2000)
        l.close()
    assert "YAWN" in (tmp_path / "dms_session_one.log").read_text()
    assert "YAWN" in (tmp_path / "dms_session_two.log").read_text()


# ---------------------------------------------------------------------- CLI
def test_cli_missing_video_exits_with_message():
    import main
    with pytest.raises(SystemExit) as e:
        main.main(["--video", "/nonexistent/x.mp4", "--headless"])
    assert "not found" in str(e.value)


def test_cli_parser_options():
    import main
    a = main.build_parser().parse_args(["--video", "a.mp4", "--no-display", "--save", "--calibrate", "4"])
    assert a.headless and a.save == "" and a.calibrate == 4 and a.video == "a.mp4"
    assert main.build_parser().parse_args([]).video is None   # webcam is the default
