import json
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "dashboard", "app.py")


def test_dashboard_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DMS_LOGS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("No sessions" in i.value for i in at.info)


def test_dashboard_with_session_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DMS_LOGS_DIR", str(tmp_path))
    (tmp_path / "summary_s1.json").write_text(json.dumps({
        "session_id": "s1", "start_time": "2026-01-01T10:00:00", "duration_seconds": 120,
        "total_events": 2, "total_frames": 3000, "drowsy_count": 1, "drowsy_frames": 40,
        "yawn_count": 1, "distraction_count": 0, "critical_alerts": 1}))
    (tmp_path / "events_s1.json").write_text(json.dumps([
        {"timestamp": "2026-01-01T10:00:10", "event_type": "drowsiness", "severity": "critical",
         "ear_value": 0.1, "mar_value": None},
        {"timestamp": "2026-01-01T10:01:10", "event_type": "yawn", "severity": "warning",
         "ear_value": None, "mar_value": 0.7}]))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.metric[0].value == "1"
    at.sidebar.radio[0].set_value("📈 Session Analysis").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("⚙️ Settings").run()
    assert not at.exception
