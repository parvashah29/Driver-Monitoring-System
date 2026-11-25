"""
Audio Alert Module for Driver Monitoring System

Beeps (pygame) and optional voice warnings (pyttsx3). Everything degrades
gracefully: with no audio device (headless/CI), or if pygame/pyttsx3 are not
installed, alerts fall back to the terminal bell or are silently skipped and the
detection loop is never affected.

All sound is produced on ONE background worker thread, so the main video loop
never blocks and the (non thread-safe) audio engines are only touched from a
single thread. If the worker is busy, new alerts are dropped instead of queued,
which prevents a backlog of stale warnings.
"""

import queue
import sys
import threading
from typing import Optional, Tuple

import numpy as np

from ..utils.config import ALERT_CONFIG

# Optional imports - gracefully handle missing packages
try:
    import os as _os
    _os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

SAMPLE_RATE = 22050


def generate_beep_samples(frequency: int, duration: float) -> np.ndarray:
    """Stereo int16 sine beep with a 10 ms fade in/out (no clicks)."""
    n = max(int(SAMPLE_RATE * duration), 1)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    fade = min(int(SAMPLE_RATE * 0.01), n // 2)
    if fade > 0:
        envelope = np.ones(n)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        wave = wave * envelope
    mono = (wave * 32767 * 0.8).astype(np.int16)
    return np.column_stack((mono, mono))


class AudioAlert:
    """Thread-safe, failure-tolerant audio alerts."""

    messages = {
        'drowsy_warning': "Warning! You appear drowsy. Please take a break.",
        'drowsy_critical': "Alert! Severe drowsiness detected! Pull over immediately!",
        'yawn': "Yawning detected. Consider taking a break.",
        'distracted': "Please keep your eyes on the road.",
        'no_face': "Driver not detected. Please face the camera.",
    }

    def __init__(self, enable_voice: bool = True):
        self.enable_voice = enable_voice and PYTTSX3_AVAILABLE
        self.backend = "none"
        self._mixer_ok = False
        self._tts_engine = None
        self._closed = False

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
                self._mixer_ok = True
                self.backend = "pygame"
            except Exception as exc:  # no audio device, headless, driver error...
                print(f"Warning: audio device unavailable ({exc}); using terminal bell.")
        if not self._mixer_ok:
            self.backend = "terminal-bell"

        # Single worker thread owns the mixer playback and the TTS engine
        self._queue: "queue.Queue[Optional[Tuple[int, float, Optional[str]]]]" = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, name="dms-audio", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ worker
    def _init_tts(self):
        if not self.enable_voice:
            return
        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', 150)
            self._tts_engine.setProperty('volume', 1.0)
        except Exception as exc:
            print(f"Warning: text-to-speech unavailable ({exc}); voice alerts disabled.")
            self._tts_engine = None
            self.enable_voice = False

    def _run(self):
        self._init_tts()
        while True:
            item = self._queue.get()
            if item is None:
                break
            frequency, duration, speech = item
            try:
                self._play_beep(frequency, duration)
                if speech and self._tts_engine is not None:
                    self._tts_engine.say(speech)
                    self._tts_engine.runAndWait()
            except Exception:
                pass  # audio must never take the monitor down

    def _play_beep(self, frequency: int, duration: float):
        if self._mixer_ok:
            sound = pygame.sndarray.make_sound(generate_beep_samples(frequency, duration))
            channel = sound.play()
            if channel is not None:
                while channel.get_busy() and not self._closed:
                    pygame.time.wait(20)
        else:
            sys.stdout.write('\a')
            sys.stdout.flush()

    # --------------------------------------------------------------------- API
    def _submit(self, frequency: int, duration: float, speech: Optional[str]) -> bool:
        """Queue a sound; returns False (and drops it) if the worker is busy."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait((frequency, duration, speech if self.enable_voice else None))
            return True
        except queue.Full:
            return False

    def alert_drowsy(self, severity: str = 'warning') -> bool:
        if severity == 'critical':
            return self._submit(ALERT_CONFIG.CRITICAL_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION * 2,
                                self.messages['drowsy_critical'])
        return self._submit(ALERT_CONFIG.WARNING_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION,
                            self.messages['drowsy_warning'])

    def alert_yawn(self) -> bool:
        return self._submit(ALERT_CONFIG.WARNING_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION, self.messages['yawn'])

    def alert_distracted(self) -> bool:
        return self._submit(ALERT_CONFIG.WARNING_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION, self.messages['distracted'])

    def alert_no_face(self) -> bool:
        return self._submit(600, 0.2, self.messages['no_face'])

    def close(self):
        """Stop the worker and release audio resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._worker.join(timeout=2.0)
        if self._mixer_ok:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
