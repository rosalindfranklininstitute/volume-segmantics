"""Tests for loss registry, loss schedule.


"""

from __future__ import annotations

import pytest
import torch

# Trigger registration of the loss factories.
import volume_segmantics.model.loss_registry  # noqa: F401
from volume_segmantics.data import pipeline_registry as reg
from volume_segmantics.data.pipeline_loader import (
    KNOWN_LOSS_NAMES,
    LossScheduleEntryConfig,
)
from volume_segmantics.model.loss_schedule import (
    LossSchedule,
    apply_schedule,
    resolve_head_weights,
)


# Registry coverage 


def test_all_known_losses_registered():
    registered = set(reg.list_losses())
    expected = set(KNOWN_LOSS_NAMES)
    assert registered >= expected, (
        f"missing losses: {sorted(expected - registered)}"
    )


# Per-loss forward 


def _semantic_inputs(num_classes: int = 4):
    pred = torch.randn(2, num_classes, 16, 16)
    target = torch.randint(0, num_classes, (2, 16, 16))
    return pred, target


def _boundary_inputs():
    pred = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    return pred, target


def _distance_inputs():
    pred = torch.randn(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16) * 5.0
    return pred, target


def _sdm_inputs():
    pred = torch.tanh(torch.randn(2, 1, 16, 16))
    target = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
    return pred, target


@pytest.mark.parametrize("name", [
    "dice_ce", "combined_ce_dice", "class_weighted_dice",
    "dice", "generalized_dice", "cross_entropy", "tversky",
])
def test_semantic_loss_forward(name):
    fn = reg.build_loss(name, head_name="semantic", num_classes=4)
    pred, target = _semantic_inputs(num_classes=4)
    out = fn(pred, target)
    assert torch.is_tensor(out)
    assert out.dim() == 0
    assert torch.isfinite(out)


@pytest.mark.parametrize("name", [
    "bce", "boundary_bce", "boundary_dice",
    "boundary_bce_dice", "boundary_loss", "boundary_dou",
])
def test_boundary_loss_forward(name):
    fn = reg.build_loss(name, head_name="boundary")
    pred, target = _boundary_inputs()
    out = fn(pred, target)
    assert torch.is_tensor(out)
    assert out.dim() == 0
    assert torch.isfinite(out)


@pytest.mark.parametrize("name", ["distance_l1", "distance_mse"])
def test_distance_loss_forward(name):
    fn = reg.build_loss(name, head_name="distance")
    pred, target = _distance_inputs()
    out = fn(pred, target)
    assert torch.is_tensor(out)
    assert out.dim() == 0
    assert torch.isfinite(out)
    assert float(out) >= 0.0


@pytest.mark.parametrize("name", ["sdm_l1", "sdm_mse"])
def test_sdm_loss_forward(name):
    fn = reg.build_loss(name, head_name="sdm")
    pred, target = _sdm_inputs()
    out = fn(pred, target)
    assert torch.is_tensor(out)
    assert out.dim() == 0
    assert torch.isfinite(out)
    assert float(out) >= 0.0


def test_dice_dispatches_by_head_name():
    """`dice` factory routes to multi-class wrapper for semantic, sigmoid
    wrapper for boundary."""
    fn_sem = reg.build_loss("dice", head_name="semantic", num_classes=4)
    fn_bnd = reg.build_loss("dice", head_name="boundary")
    pred_sem, tgt_sem = _semantic_inputs(num_classes=4)
    pred_bnd, tgt_bnd = _boundary_inputs()
    assert torch.isfinite(fn_sem(pred_sem, tgt_sem))
    assert torch.isfinite(fn_bnd(pred_bnd, tgt_bnd))


def test_generalized_dice_rejects_non_semantic():
    with pytest.raises(ValueError, match="generalized_dice"):
        reg.build_loss("generalized_dice", head_name="boundary")


def test_loss_zero_for_perfect_distance_prediction():
    pred = torch.zeros(2, 1, 16, 16)
    target = torch.zeros(2, 1, 16, 16)
    fn = reg.build_loss("distance_l1")
    assert float(fn(pred, target)) == 0.0


def test_loss_zero_for_perfect_sdm_prediction():
    target = torch.rand(2, 1, 16, 16) * 2 - 1
    pred = target.clone()
    fn = reg.build_loss("sdm_mse")
    assert float(fn(pred, target)) == pytest.approx(0.0, abs=1e-7)


def test_l1_decreases_when_pred_approaches_target():
    target = torch.rand(2, 1, 16, 16) * 5
    bad_pred = target + 3.0
    good_pred = target + 0.1
    fn = reg.build_loss("distance_l1")
    assert float(fn(good_pred, target)) < float(fn(bad_pred, target))


# Loss schedule 


def test_apply_schedule_constant_returns_end_weight():
    w = apply_schedule(
        "constant", start_weight=0.0, end_weight=0.7,
        warmup_fraction=0.5, current_step=0, total_steps=100,
    )
    assert w == pytest.approx(0.7)


def test_apply_schedule_linear_warmup_at_start_returns_start():
    w = apply_schedule(
        "linear_warmup", start_weight=0.0, end_weight=1.0,
        warmup_fraction=0.5, current_step=0, total_steps=100,
    )
    assert w == pytest.approx(0.0)


def test_apply_schedule_linear_warmup_mid_ramp():
    w = apply_schedule(
        "linear_warmup", start_weight=0.0, end_weight=1.0,
        warmup_fraction=0.5, current_step=25, total_steps=100,
    )
    # 25 of 50 ramp steps -> t=0.5 -> weight = 0.5
    assert w == pytest.approx(0.5)


def test_apply_schedule_linear_warmup_post_ramp_holds():
    w = apply_schedule(
        "linear_warmup", start_weight=0.0, end_weight=1.0,
        warmup_fraction=0.5, current_step=80, total_steps=100,
    )
    assert w == pytest.approx(1.0)


def test_apply_schedule_linear_decay():
    w_start = apply_schedule(
        "linear_decay", start_weight=1.0, end_weight=0.1,
        warmup_fraction=0.5, current_step=0, total_steps=100,
    )
    w_mid = apply_schedule(
        "linear_decay", start_weight=1.0, end_weight=0.1,
        warmup_fraction=0.5, current_step=25, total_steps=100,
    )
    w_late = apply_schedule(
        "linear_decay", start_weight=1.0, end_weight=0.1,
        warmup_fraction=0.5, current_step=80, total_steps=100,
    )
    assert w_start == pytest.approx(1.0)
    assert w_mid == pytest.approx(0.55)  # halfway between 1.0 and 0.1
    assert w_late == pytest.approx(0.1)


def test_apply_schedule_zero_warmup_fraction_jumps_to_end():
    w = apply_schedule(
        "linear_warmup", start_weight=0.0, end_weight=1.0,
        warmup_fraction=0.0, current_step=0, total_steps=100,
    )
    assert w == pytest.approx(1.0)


def test_apply_schedule_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown schedule"):
        apply_schedule(
            "exponential", start_weight=0.0, end_weight=1.0,
            warmup_fraction=0.1, current_step=0, total_steps=100,
        )


def test_apply_schedule_zero_total_steps_raises():
    with pytest.raises(ValueError, match="total_steps"):
        apply_schedule(
            "constant", start_weight=0.0, end_weight=1.0,
            warmup_fraction=0.1, current_step=0, total_steps=0,
        )


def test_apply_schedule_warmup_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match="warmup_fraction"):
        apply_schedule(
            "linear_warmup", start_weight=0.0, end_weight=1.0,
            warmup_fraction=1.5, current_step=0, total_steps=100,
        )


# LossSchedule.from_config 


def test_loss_schedule_from_config_none_is_identity():
    sched = LossSchedule.from_config(None)
    assert sched.schedule == "constant"
    assert sched.end_weight == 1.0


def test_loss_schedule_from_config_round_trip():
    cfg = LossScheduleEntryConfig(
        schedule="linear_warmup",
        start_weight=0.0, end_weight=1.0, warmup_fraction=0.1,
    )
    sched = LossSchedule.from_config(cfg)
    assert sched.schedule == "linear_warmup"
    assert sched.start_weight == 0.0
    assert sched.end_weight == 1.0
    assert sched.warmup_fraction == 0.1


# resolve_head_weights 


def test_resolve_head_weights_no_schedules_returns_static():
    static = {"semantic": 1.0, "boundary": 0.5}
    schedules = {}  # heads with no schedule = constant 1.0
    out = resolve_head_weights(
        schedules, static, current_step=0, total_steps=100,
    )
    assert out == {"semantic": 1.0, "boundary": 0.5}


def test_resolve_head_weights_applies_schedule():
    static = {"semantic": 1.0, "distance": 0.5}
    schedules = {
        "distance": LossSchedule(
            schedule="linear_warmup",
            start_weight=0.0, end_weight=1.0, warmup_fraction=0.5,
        ),
    }
    # At step 0: distance has weight 0.5 * 0.0 = 0.0.
    out0 = resolve_head_weights(
        schedules, static, current_step=0, total_steps=100,
    )
    assert out0["semantic"] == pytest.approx(1.0)
    assert out0["distance"] == pytest.approx(0.0)
    # At step 50 (post-ramp): distance has weight 0.5 * 1.0 = 0.5.
    out50 = resolve_head_weights(
        schedules, static, current_step=50, total_steps=100,
    )
    assert out50["distance"] == pytest.approx(0.5)
