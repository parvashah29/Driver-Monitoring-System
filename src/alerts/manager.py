"""
Alert state machine: decides *when* an alert should fire.

Rules
-----
* Only the highest-priority active condition is announced
  (drowsy > distracted > yawn > no_face).
* Each kind has a cooldown: while a condition stays active it is re-announced at
  most once per `cooldown` seconds, and a flickering condition cannot spam either.
* A new *episode* is counted on an inactive -> active transition of a kind
  (independent of priority/cooldown) once the kind has been inactive for at least
  `rearm` seconds, so a value flickering around a threshold is one event, not many.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.config import ALERT_CONFIG

PRIORITY = ('drowsy', 'distracted', 'yawn', 'no_face')


@dataclass
class AlertDecision:
    new_episodes: List[str] = field(default_factory=list)  # kinds that just started
    fire: Optional[str] = None                             # kind to announce now (audio)
    active: Optional[str] = None                           # highest-priority active kind


class AlertManager:
    def __init__(self, cooldown: Optional[float] = None, rearm: float = 1.0):
        self.cooldown = ALERT_CONFIG.ALERT_COOLDOWN if cooldown is None else cooldown
        self.rearm = rearm
        self._was_active: Dict[str, bool] = {k: False for k in PRIORITY}
        self._inactive_since: Dict[str, float] = {}
        self._last_fire: Dict[str, float] = {}
        self.episode_counts: Dict[str, int] = {k: 0 for k in PRIORITY}
        self.fire_counts: Dict[str, int] = {k: 0 for k in PRIORITY}

    def update(self, active: Dict[str, bool], now: float) -> AlertDecision:
        decision = AlertDecision()
        for kind in PRIORITY:
            is_active = bool(active.get(kind, False))
            if is_active:
                if not self._was_active[kind]:
                    self.episode_counts[kind] += 1
                    decision.new_episodes.append(kind)
                self._was_active[kind] = True
                self._inactive_since.pop(kind, None)
            elif self._was_active[kind]:
                since = self._inactive_since.setdefault(kind, now)
                if now - since >= self.rearm:
                    self._was_active[kind] = False
                    del self._inactive_since[kind]

        for kind in PRIORITY:
            if active.get(kind, False):
                decision.active = kind
                last = self._last_fire.get(kind)
                if last is None or now - last >= self.cooldown:
                    self._last_fire[kind] = now
                    self.fire_counts[kind] += 1
                    decision.fire = kind
                break
        return decision

    def reset(self):
        self.__init__(self.cooldown, self.rearm)
