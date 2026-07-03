"""Tests for per-head loss-weight scheduling.

Pins the semantics of ``apply_schedule``: both ``linear_warmup`` and
``linear_decay`` linearly interpolate ``start_weight -> end_weight`` over the
ramp window (the caller controls direction via the endpoints), then hold at
``end_weight``. Also covers the ramp-rounding-to-zero boundary and validation.
"""

import pytest

from volume_segmantics.model.loss_schedule import (
    LossSchedule,
    apply_schedule,
    resolve_head_weights,
)


def _apply(schedule, start, end, frac, step, total):
    return apply_schedule(
        schedule=schedule,
        start_weight=start,
        end_weight=end,
        warmup_fraction=frac,
        current_step=step,
        total_steps=total,
    )


def test_constant_ignores_step_and_returns_end_weight():
    for step in (0, 5, 100):
        assert _apply("constant", 0.2, 0.9, 0.5, step, 100) == 0.9


def test_linear_warmup_interpolates_start_to_end():
    # ramp window = 50 steps (0.5 * 100)
    assert _apply("linear_warmup", 0.0, 1.0, 0.5, 0, 100) == pytest.approx(0.0)
    assert _apply("linear_warmup", 0.0, 1.0, 0.5, 25, 100) == pytest.approx(0.5)
    # at/after ramp end -> holds at end_weight
    assert _apply("linear_warmup", 0.0, 1.0, 0.5, 50, 100) == pytest.approx(1.0)
    assert _apply("linear_warmup", 0.0, 1.0, 0.5, 80, 100) == pytest.approx(1.0)


def test_linear_decay_interpolates_start_to_end():
    # With start > end, "decay" genuinely decreases over the ramp.
    assert _apply("linear_decay", 1.0, 0.0, 0.5, 0, 100) == pytest.approx(1.0)
    assert _apply("linear_decay", 1.0, 0.0, 0.5, 25, 100) == pytest.approx(0.5)
    assert _apply("linear_decay", 1.0, 0.0, 0.5, 50, 100) == pytest.approx(0.0)


def test_warmup_and_decay_share_interpolation_for_equal_endpoints():
    # The two schedules use identical ramp math; given the same endpoints they
    # must agree at every step. (Documents the merged implementation.)
    for step in (0, 10, 25, 40, 50, 99):
        w = _apply("linear_warmup", 0.2, 0.8, 0.5, step, 100)
        d = _apply("linear_decay", 0.2, 0.8, 0.5, step, 100)
        assert w == pytest.approx(d)


def test_ramp_rounding_to_zero_jumps_to_end_weight():
    # warmup_fraction * total_steps rounds to 0 -> degenerate, jump to end.
    assert _apply("linear_warmup", 0.0, 1.0, 0.001, 0, 10) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(schedule="nope", start=0.0, end=1.0, frac=0.5, step=0, total=10),
         "unknown schedule"),
        (dict(schedule="linear_warmup", start=0.0, end=1.0, frac=0.5, step=0, total=0),
         "total_steps must be positive"),
        (dict(schedule="linear_warmup", start=0.0, end=1.0, frac=1.5, step=0, total=10),
         "warmup_fraction must be in"),
    ],
)
def test_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _apply(kwargs["schedule"], kwargs["start"], kwargs["end"],
               kwargs["frac"], kwargs["step"], kwargs["total"])


def test_resolve_head_weights_scales_static_weight():
    schedules = {
        "semantic": LossSchedule(schedule="linear_warmup", start_weight=0.0,
                                 end_weight=1.0, warmup_fraction=0.5),
    }
    static = {"semantic": 2.0, "boundary": 3.0}
    out = resolve_head_weights(
        schedules, static, current_step=25, total_steps=100
    )
    # semantic: scale 0.5 at midpoint * static 2.0 = 1.0
    assert out["semantic"] == pytest.approx(1.0)
    # boundary has no schedule -> constant scale 1.0 * static 3.0 = 3.0
    assert out["boundary"] == pytest.approx(3.0)
