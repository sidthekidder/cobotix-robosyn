"""Deterministic perturbation plans for ClickBell corrective demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionSpec:
    """One controlled end-effector perturbation applied before recording."""

    episode: int
    anchor_step: int
    target_step: int
    recovery_steps: int
    offset_m: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def sample_correction_spec(
    rng: np.random.Generator,
    episode: int,
    *,
    anchor_min: int = 34,
    anchor_max: int = 44,
    target_step: int = 49,
    recovery_steps: int = 12,
    lateral_mm: tuple[float, float] = (6.0, 12.0),
    vertical_mm: tuple[float, float] = (3.0, 7.0),
) -> CorrectionSpec:
    """Sample a near-contact perturbation without touching simulator state.

    Two thirds of examples shift laterally in world x/y, where a small image
    localization error causes a miss. The remaining examples shift along world
    z, which covers early/late contact. A single axis is changed per episode so
    failures and recovery quality remain easy to interpret.
    """

    if not (0 <= anchor_min <= anchor_max < target_step):
        raise ValueError("anchor range must be non-negative and end before target_step")
    if recovery_steps < 2:
        raise ValueError("recovery_steps must be at least 2")

    anchor_step = int(rng.integers(anchor_min, anchor_max + 1))
    axis = int(rng.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3]))
    magnitude_range = lateral_mm if axis < 2 else vertical_mm
    magnitude_mm = float(rng.uniform(*magnitude_range))
    sign = float(rng.choice([-1.0, 1.0]))
    offset = [0.0, 0.0, 0.0]
    offset[axis] = sign * magnitude_mm / 1000.0

    return CorrectionSpec(
        episode=episode,
        anchor_step=anchor_step,
        target_step=target_step,
        recovery_steps=recovery_steps,
        offset_m=tuple(offset),
    )


def build_correction_specs(episodes: int, seed: int, **kwargs) -> list[CorrectionSpec]:
    """Return a reproducible perturbation schedule."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    return [sample_correction_spec(rng, i, **kwargs) for i in range(episodes)]
