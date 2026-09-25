"""State machine for conservative ACT contact-recovery replanning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ContactPlateauRecovery:
    """Request a replan when partial task progress stops improving."""

    min_step: int = 60
    plateau_steps: int = 10
    min_press_depth_m: float = 0.0011
    press_epsilon_m: float = 0.0001
    cooldown_steps: int = 25
    max_replans: int = 2
    step: int = 0
    replan_count: int = 0
    last_replan_step: int | None = None
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_step < 1:
            raise ValueError("min_step must be positive")
        if self.plateau_steps < 1:
            raise ValueError("plateau_steps must be positive")
        if self.min_press_depth_m < 0:
            raise ValueError("min_press_depth_m must be non-negative")
        if self.press_epsilon_m < 0:
            raise ValueError("press_epsilon_m must be non-negative")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if self.max_replans < 1:
            raise ValueError("max_replans must be positive")
        self._history = deque(maxlen=self.plateau_steps + 1)

    def reset(self) -> None:
        self.step = 0
        self.replan_count = 0
        self.last_replan_step = None
        self.events.clear()
        self._history.clear()

    def observe(
        self,
        *,
        max_press_depth_m: float,
        success_depth_m: float,
        queued_actions: int,
    ) -> bool:
        """Record one environment step and return whether to clear the queue."""
        self.step += 1
        depth = float(max_press_depth_m)
        self._history.append((self.step, depth))

        if self.step < self.min_step or queued_actions <= 0:
            return False
        if self.replan_count >= self.max_replans:
            return False
        if not self.min_press_depth_m <= depth < float(success_depth_m):
            return False
        if self.last_replan_step is not None:
            if self.step - self.last_replan_step < self.cooldown_steps:
                return False
        if len(self._history) < self.plateau_steps + 1:
            return False

        oldest_step, oldest_depth = self._history[0]
        if self.step - oldest_step < self.plateau_steps:
            return False
        improvement = depth - oldest_depth
        if improvement >= self.press_epsilon_m:
            return False

        self.replan_count += 1
        self.last_replan_step = self.step
        self.events.append(
            {
                "step": self.step,
                "max_press_depth_m": depth,
                "window_improvement_m": improvement,
                "discarded_actions": int(queued_actions),
            }
        )
        self._history.clear()
        self._history.append((self.step, depth))
        return True

    def metrics(self) -> dict:
        return {
            "enabled": True,
            "replan_count": self.replan_count,
            "events": list(self.events),
        }
