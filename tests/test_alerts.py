import threading

import numpy as np
import pytest

from src.alerts import AlertManager, AudioAlert
from src.alerts import audio_alert


def test_first_activation_fires_then_cooldown_suppresses():
    m = AlertManager(cooldown=5.0)
    d = m.update({'drowsy': True}, 0.0)
    assert d.fire == 'drowsy' and d.new_episodes == ['drowsy']
    for t in (1, 2, 4.9):
        assert m.update({'drowsy': True}, t).fire is None
    assert m.update({'drowsy': True}, 5.0).fire == 'drowsy'
    assert m.fire_counts['drowsy'] == 2 and m.episode_counts['drowsy'] == 1


def test_flicker_cannot_spam_but_counts_episodes():
    m = AlertManager(cooldown=5.0)
    fired = 0
    for i in range(20):
        d = m.update({'yawn': i % 2 == 0}, i * 0.1)
        fired += d.fire == 'yawn'
    assert fired == 1
    assert m.episode_counts['yawn'] == 1          # flicker within `rearm` = one episode


def test_episode_rearms_after_inactive_gap():
    m = AlertManager(cooldown=5.0, rearm=1.0)
    m.update({'yawn': True}, 0.0)
    m.update({}, 0.5)
    m.update({'yawn': True}, 0.9)                 # gap < rearm: same episode
    assert m.episode_counts['yawn'] == 1
    m.update({}, 1.0)
    m.update({}, 2.5)                             # gap >= rearm
    m.update({'yawn': True}, 3.0)
    assert m.episode_counts['yawn'] == 2


def test_priority_and_no_alert_when_nothing_active():
    m = AlertManager(cooldown=5.0)
    assert m.update({}, 0).fire is None and m.update({}, 0).active is None
    d = m.update({'yawn': True, 'drowsy': True, 'distracted': True}, 1.0)
    assert d.fire == 'drowsy' and d.active == 'drowsy'
    assert set(d.new_episodes) == {'yawn', 'drowsy', 'distracted'}


def test_lower_priority_announced_once_higher_clears():
    m = AlertManager(cooldown=5.0)
    m.update({'drowsy': True, 'yawn': True}, 0)
    d = m.update({'yawn': True}, 1)
    assert d.fire == 'yawn'


# ------------------------------------------------------------------ audio
def test_beep_samples_shape_and_range():
    s = audio_alert.generate_beep_samples(800, 0.3)
    assert s.shape == (int(22050 * 0.3), 2) and s.dtype == np.int16
    assert abs(int(s[0, 0])) < 100 and abs(int(s.max())) <= 32767


def test_audio_survives_mixer_init_failure(monkeypatch, capsys):
    if audio_alert.PYGAME_AVAILABLE:
        monkeypatch.setattr(audio_alert.pygame.mixer, "init",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device")))
    a = AudioAlert(enable_voice=False)
    assert a.backend == "terminal-bell"
    assert a.alert_drowsy('critical') in (True, False)
    a.close()


def test_audio_no_packages_at_all(monkeypatch):
    monkeypatch.setattr(audio_alert, "PYGAME_AVAILABLE", False)
    monkeypatch.setattr(audio_alert, "PYTTSX3_AVAILABLE", False)
    a = AudioAlert(enable_voice=True)
    assert a.backend == "terminal-bell" and not a.enable_voice
    a.close()


def test_audio_is_thread_safe_and_close_is_idempotent():
    a = AudioAlert(enable_voice=False)
    errors = []

    def spam():
        try:
            for _ in range(50):
                a.alert_yawn(); a.alert_distracted(); a.alert_no_face(); a.alert_drowsy('critical')
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=spam) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    a.close(); a.close()
    assert not errors
    assert a.alert_yawn() is False        # closed: silently ignored
