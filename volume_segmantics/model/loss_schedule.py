"""Per-head loss-weight scheduling.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from volume_segmantics.data.pipeline_loader import (
    KNOWN_LOSS_SCHEDULES,
    LossScheduleEntryConfig,
)


@dataclass(frozen=True)
class LossSchedule:
    """Per-head schedule pulled from ``pipeline.yaml::loss_schedule.<head>``.

    """

    schedule: str = "constant"
    start_weight: float = 1.0
    end_weight: float = 1.0
    warmup_fraction: float = 0.0

    @classmethod
    def from_config(
        cls, cfg: Optional[LossScheduleEntryConfig],
    ) -> "LossSchedule":
        """Build from a parsed :class:`LossScheduleEntryConfig` (or default).

        ``None`` returns a flat ``schedule="constant"`` with weight
        ``1.0`` — the no-op schedule used when the head has no
        ``loss_schedule`` block.
        """
        if cfg is None:
            return cls()
        return cls(
            schedule=cfg.schedule,
            start_weight=cfg.start_weight,
            end_weight=cfg.end_weight,
            warmup_fraction=cfg.warmup_fraction,
        )


def apply_schedule(
    schedule: str,
    *,
    start_weight: float,
    end_weight: float,
    warmup_fraction: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Resolve the schedule's weight at ``current_step``.

    Parameters
    ----------
    schedule
        One of :data:`pipeline_loader.KNOWN_LOSS_SCHEDULES`.
    start_weight, end_weight
        Schedule endpoints.
    warmup_fraction
        Fraction of training during which the ramp is active. Outside
        the ramp window the weight holds at ``end_weight``.
    current_step
        Current optimiser step (0-indexed).
    total_steps
        Total optimiser steps for the run. Must be positive.

    Returns
    -------
    float
        Resolved weight at this step.
    """
    if schedule not in KNOWN_LOSS_SCHEDULES:
        raise ValueError(
            f"unknown schedule {schedule!r}; "
            f"known: {sorted(KNOWN_LOSS_SCHEDULES)}"
        )
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive; got {total_steps}")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError(
            f"warmup_fraction must be in [0, 1]; got {warmup_fraction}"
        )

    if schedule == "constant":
        return float(end_weight)

    # Both ramp schedules share the same ramp-window math.
    ramp_steps = int(round(warmup_fraction * total_steps))
    if ramp_steps <= 0:
        # Degenerate: no ramp, jump straight to end_weight.
        return float(end_weight)

    if current_step >= ramp_steps:
        return float(end_weight)

    # Both "linear_warmup" and "linear_decay" linearly interpolate
    # start_weight -> end_weight across the ramp window; the schedule name is
    # descriptive only (the caller picks the direction via the endpoints, e.g.
    # start>end for a decay). The math is therefore identical for both.
    t = float(current_step) / float(ramp_steps)
    return float(start_weight + t * (end_weight - start_weight))


def resolve_head_weights(
    schedules: Mapping[str, LossSchedule],
    head_static_weights: Mapping[str, float],
    *,
    current_step: int,
    total_steps: int,
) -> Mapping[str, float]:
    """Resolve every enabled head's effective weight at ``current_step``.

    Effective weight = ``head.loss_weight × schedule.value(step)``.
    Heads without a schedule get the static weight unchanged
    (equivalent to ``constant`` schedule with ``end_weight=1``).

    Returns a fresh dict keyed by head name.
    """
    out = {}
    for head_name, static_weight in head_static_weights.items():
        sched = schedules.get(head_name, LossSchedule())
        scale = apply_schedule(
            schedule=sched.schedule,
            start_weight=sched.start_weight,
            end_weight=sched.end_weight,
            warmup_fraction=sched.warmup_fraction,
            current_step=current_step,
            total_steps=total_steps,
        )
        out[head_name] = float(static_weight) * scale
    return out


__all__ = [
    "LossSchedule",
    "apply_schedule",
    "resolve_head_weights",
]
