"""
Audio Alert Module for Driver Monitoring System
Provides audio warnings for fatigue detection.
"""

import os
import threading
import time
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import ALERT_CONFIG

# Optional imports - gracefully handle missing packages
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class AudioAlert:
    """
    Audio alert system using pygame for beeps and pyttsx3 for voice.
    Falls back to system beep if packages are not available.
    """
    
    def __init__(self, enable_voice: bool = True):
        """
        Initialize audio alert system.
        
        Args:
            enable_voice: Whether to enable text-to-speech alerts
        """
        self.enable_voice = enable_voice and PYTTSX3_AVAILABLE
        self._initialized = False
        self._last_alert_time = 0
        self._cooldown = ALERT_CONFIG.ALERT_COOLDOWN
        
        # Initialize pygame mixer
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self._initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize pygame mixer: {e}")
        
        # Initialize text-to-speech
        self._tts_engine = None
        if self.enable_voice:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 150)
                self._tts_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"Warning: Could not initialize TTS: {e}")
                self._tts_engine = None
        
        # Alert messages
        self.messages = {
            'drowsy_warning': "Warning! You appear drowsy. Please take a break.",
            'drowsy_critical': "Alert! Severe drowsiness detected! Pull over immediately!",
            'yawn': "Yawning detected. Consider taking a break.",
            'distracted': "Please keep your eyes on the road.",
            'no_face': "Driver not detected. Please face the camera."
        }
    
    def _can_alert(self) -> bool:
        """Check if enough time has passed since last alert."""
        current_time = time.time()
        if current_time - self._last_alert_time >= self._cooldown:
            self._last_alert_time = current_time
            return True
        return False
    
    def _generate_beep(self, frequency: int, duration: float) -> Optional[object]:
        """
        Generate a beep sound using pygame.
        
        Args:
            frequency: Beep frequency in Hz
            duration: Duration in seconds
            
        Returns:
            pygame Sound object or None
        """
        if not PYGAME_AVAILABLE or not self._initialized:
            return None
        
        import numpy as np
        
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        
        # Generate sine wave
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones(n_samples)
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave = wave * envelope
        
        # Convert to 16-bit audio
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo by duplicating
        stereo_wave = np.column_stack((wave, wave))
        
        # Create pygame sound
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound
    
    def beep(self, frequency: int = 800, duration: float = 0.3):
        """
        Play a beep sound.
        
        Args:
            frequency: Beep frequency in Hz
            duration: Duration in seconds
        """
        if not self._can_alert():
            return
        
        if PYGAME_AVAILABLE and self._initialized:
            sound = self._generate_beep(frequency, duration)
            if sound:
                sound.play()
        else:
            # Fallback: system beep (Mac/Linux)
            print('\a')
    
    def warning_beep(self):
        """Play warning beep."""
        self.beep(ALERT_CONFIG.WARNING_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION)
    
    def critical_beep(self):
        """Play critical/alarm beep."""
        self.beep(ALERT_CONFIG.CRITICAL_BEEP_FREQ, ALERT_CONFIG.BEEP_DURATION * 2)
    
    def speak(self, message: str, blocking: bool = False):
        """
        Speak a message using text-to-speech.
        
        Args:
            message: Text to speak
            blocking: Whether to wait for speech to complete
        """
        if not self._tts_engine:
            return
        
        if blocking:
            self._tts_engine.say(message)
            self._tts_engine.runAndWait()
        else:
            # Run in background thread
            thread = threading.Thread(target=self._speak_async, args=(message,))
            thread.daemon = True
            thread.start()
    
    def _speak_async(self, message: str):
        """Async speech helper."""
        try:
            self._tts_engine.say(message)
            self._tts_engine.runAndWait()
        except Exception:
            pass
    
    def alert_drowsy(self, severity: str = 'warning'):
        """
        Alert for drowsiness detection.
        
        Args:
            severity: 'warning' or 'critical'
        """
        if severity == 'critical':
            self.critical_beep()
            if self.enable_voice:
                self.speak(self.messages['drowsy_critical'])
        else:
            self.warning_beep()
            if self.enable_voice:
                self.speak(self.messages['drowsy_warning'])
    
    def alert_yawn(self):
        """Alert for yawn detection."""
        self.warning_beep()
        if self.enable_voice:
            self.speak(self.messages['yawn'])
    
    def alert_distracted(self):
        """Alert for distraction detection."""
        self.warning_beep()
        if self.enable_voice:
            self.speak(self.messages['distracted'])
    
    def alert_no_face(self):
        """Alert when driver face is not detected."""
        self.beep(600, 0.2)
        if self.enable_voice:
            self.speak(self.messages['no_face'])
    
    def close(self):
        """Clean up resources."""
        if PYGAME_AVAILABLE and self._initialized:
            pygame.mixer.quit()
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
