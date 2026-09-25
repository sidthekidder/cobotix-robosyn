"""State machine for conservative ACT contact-recovery replanning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


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


@dataclass
class ProximityPressOracle:
    """Diagnostic oracle that completes the final press from a nearby ACT pose.

    The target comes from the task's scene-specific expert IK plan.  Only the
    right arm is changed; all other commanded joints are held at their current
    positions.  This is intentionally simulator-only and must never be treated
    as a deployable policy result.
    """

    min_step: int = 35
    target_plan_step: int = 49
    trigger_max_joint_error_rad: float = 0.45
    interpolation_steps: int = 12
    hold_steps: int = 8
    telemetry_stride: int = 2
    step: int = 0
    triggered: bool = False
    trigger_step: int | None = None
    plan_steps: int | None = None
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_step < 0:
            raise ValueError("min_step must be non-negative")
        if self.target_plan_step < 0:
            raise ValueError("target_plan_step must be non-negative")
        if self.trigger_max_joint_error_rad <= 0:
            raise ValueError("trigger_max_joint_error_rad must be positive")
        if self.interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        if self.telemetry_stride < 1:
            raise ValueError("telemetry_stride must be positive")
        self._target_action = None
        self._target_right_qpos = None
        self._target_eef_pose = None
        self._right_action_indices = None
        self._completion_actions = []

    @staticmethod
    def _scalar(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].cpu())
        return float(np.asarray(value).reshape(-1)[0])

    def reset(self) -> None:
        self.step = 0
        self.triggered = False
        self.trigger_step = None
        self.plan_steps = None
        self.trace.clear()
        self._target_action = None
        self._target_right_qpos = None
        self._target_eef_pose = None
        self._right_action_indices = None
        self._completion_actions.clear()

    def _prepare(self, task_env) -> None:
        actions = task_env.create_demo_action_list()
        if actions is None or len(actions) == 0:
            raise RuntimeError("The click-bell expert could not produce an oracle target.")

        self.plan_steps = len(actions)
        target_index = min(self.target_plan_step, self.plan_steps - 1)
        target = torch.as_tensor(actions[target_index], dtype=torch.float32)
        if target.ndim == 2:
            target = target[0]
        target = target.detach().to(task_env.device)

        right_joint_ids = task_env.robot.get_joint_ids(
            name="right_arm", remove_mimic=True
        )
        global_to_active = {
            joint_id: index
            for index, joint_id in enumerate(task_env.active_joint_ids)
        }
        right_action_indices = [global_to_active[joint_id] for joint_id in right_joint_ids]
        target_right = target[right_action_indices]
        target_eef = task_env.robot.compute_fk(
            name="right_arm", qpos=target_right.unsqueeze(0), to_matrix=True
        )[0]

        self._target_action = target
        self._target_right_qpos = target_right
        self._target_eef_pose = target_eef
        self._right_action_indices = right_action_indices

    def _current_action(self, task_env) -> torch.Tensor:
        return task_env.robot.get_qpos()[:, task_env.active_joint_ids][0].detach()

    def _measure(self, task_env) -> dict:
        current_action = self._current_action(task_env)
        current_right = current_action[self._right_action_indices]
        joint_error = torch.abs(current_right - self._target_right_qpos)
        current_pose = task_env.robot.get_link_pose("right_link6", to_matrix=True)[0]
        delta_rotation = current_pose[:3, :3].T @ self._target_eef_pose[:3, :3]
        cos_angle = torch.clamp((torch.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0)
        diagnostics = task_env.get_episode_diagnostics()
        return {
            "step": self.step,
            "max_joint_error_rad": float(torch.max(joint_error).cpu()),
            "rms_joint_error_rad": float(torch.sqrt(torch.mean(joint_error**2)).cpu()),
            "lateral_eef_error_m": float(
                torch.linalg.norm(
                    current_pose[:2, 3] - self._target_eef_pose[:2, 3]
                ).cpu()
            ),
            "vertical_eef_error_m": float(
                (current_pose[2, 3] - self._target_eef_pose[2, 3]).cpu()
            ),
            "orientation_error_rad": float(torch.arccos(cos_angle).cpu()),
            "max_press_depth_m": self._scalar(diagnostics["max_press_depth_m"]),
            "oracle_active": bool(self.triggered),
        }

    def next_action(self, task_env) -> torch.Tensor | None:
        """Observe the current pose and return an oracle command after triggering."""
        if self._target_action is None:
            self._prepare(task_env)

        self.step += 1
        measurement = self._measure(task_env)
        if self.step == 1 or self.step % self.telemetry_stride == 0:
            self.trace.append(measurement)

        if not self.triggered:
            if (
                self.step >= self.min_step
                and measurement["max_joint_error_rad"]
                <= self.trigger_max_joint_error_rad
            ):
                self.triggered = True
                self.trigger_step = self.step
                current = self._current_action(task_env)
                target = current.clone()
                target[self._right_action_indices] = self._target_right_qpos
                self._completion_actions = [
                    torch.lerp(current, target, amount).clone()
                    for amount in torch.linspace(
                        1.0 / self.interpolation_steps,
                        1.0,
                        self.interpolation_steps,
                        device=current.device,
                    )
                ]
                self._completion_actions.extend(
                    target.clone() for _ in range(self.hold_steps)
                )
                self.trace.append({**measurement, "event": "trigger"})

        if self.triggered and self._completion_actions:
            return self._completion_actions.pop(0).unsqueeze(0)
        return None

    def metrics(self) -> dict:
        return {
            "enabled": True,
            "diagnostic_only": True,
            "triggered": self.triggered,
            "trigger_step": self.trigger_step,
            "expert_plan_steps": self.plan_steps,
            "target_plan_step": self.target_plan_step,
            "trigger_max_joint_error_rad": self.trigger_max_joint_error_rad,
            "completion_actions_remaining": len(self._completion_actions),
            "trace": list(self.trace),
        }
