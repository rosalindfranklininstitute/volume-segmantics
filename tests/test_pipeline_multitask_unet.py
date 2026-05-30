"""Tests for PipelineMultitaskUnet, PipelineMultiTaskLossCalculator.

"""

from __future__ import annotations

import pytest
import torch

import volume_segmantics.model.heads        # registers heads
import volume_segmantics.model.loss_registry  # registers losses
import volume_segmantics.data.targets       # registers target generators
import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.data.pipeline_loader import (
    HeadConfig, PipelineConfig, LossScheduleEntryConfig,
)
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.model_2d import create_model_on_device
from volume_segmantics.model.pipeline_multitask_unet import PipelineMultitaskUnet
from volume_segmantics.model.training.multitask_calculator import (
    MultiTaskTargetError, PipelineMultiTaskLossCalculator,
)


# Fixtures 


@pytest.fixture
def four_heads_cfg():
    return {
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
        "distance": HeadConfig(enabled=True, loss="distance_l1"),
        "sdm":      HeadConfig(
            enabled=True, loss="sdm_l1",
            extra={"variant": "binary", "d_clip": 5.0},
        ),
    }


@pytest.fixture
def four_head_modules(four_heads_cfg):
    return build_head_modules(four_heads_cfg, in_channels=16, num_classes=4)


@pytest.fixture
def four_head_targets():
    return {
        "semantic": torch.randint(0, 4, (2, 16, 16)),
        "boundary": (torch.rand(2, 1, 16, 16) > 0.7).float(),
        "distance": torch.rand(2, 1, 16, 16) * 5.0,
        "sdm":      torch.rand(2, 1, 16, 16) * 2.0 - 1.0,
    }


@pytest.fixture
def four_head_preds():
    return (
        torch.randn(2, 4, 16, 16),
        torch.randn(2, 1, 16, 16),
        torch.randn(2, 1, 16, 16),
        torch.tanh(torch.randn(2, 1, 16, 16)),
    )


# PipelineMultitaskUnet 


def test_model_construction_and_head_order(four_head_modules):
    model = PipelineMultitaskUnet(
        head_modules=four_head_modules,
        encoder_name="resnet34", encoder_weights=None,
        in_channels=1,
    )
    assert model.head_names == ("semantic", "boundary", "distance", "sdm")
    assert model.num_heads == 4


def test_model_forward_returns_tuple_in_head_order(four_head_modules):
    model = PipelineMultitaskUnet(
        head_modules=four_head_modules,
        encoder_name="resnet34", encoder_weights=None,
        in_channels=1,
    )
    x = torch.randn(2, 1, 64, 64)
    outs = model(x)
    assert isinstance(outs, tuple)
    assert len(outs) == 4
    # Per-head shape contracts.
    sem, bnd, dst, sdm = outs
    assert sem.shape == (2, 4, 64, 64)   # num_classes=4
    assert bnd.shape == (2, 1, 64, 64)
    assert dst.shape == (2, 1, 64, 64)
    assert sdm.shape == (2, 1, 64, 64)
    # SDM is tanh-bounded.
    assert float(sdm.max()) <= 1.0
    assert float(sdm.min()) >= -1.0


def test_model_decoder_shared_across_heads(four_head_modules):
    model = PipelineMultitaskUnet(
        head_modules=four_head_modules,
        encoder_name="resnet34", encoder_weights=None,
    )
    # Single shared decoder per b3 design.
    assert len(model.decoders) == 1
    assert model.head_to_decoder == [0, 0, 0, 0]


def test_model_rejects_empty_head_list():
    with pytest.raises(ValueError, match="at least one"):
        PipelineMultitaskUnet(head_modules=[], encoder_name="resnet34",
                              encoder_weights=None)


def test_model_rejects_head_with_wrong_in_channels(four_heads_cfg):
    # Build heads at in_channels=8 but the resnet34 decoder outputs 16.
    heads = build_head_modules(
        four_heads_cfg, in_channels=8, num_classes=4,
    )
    with pytest.raises(ValueError, match="in_channels="):
        PipelineMultitaskUnet(
            head_modules=heads, encoder_name="resnet34",
            encoder_weights=None,
        )


# PipelineMultiTaskLossCalculator 


def test_calculator_compute_returns_named_tuple(
    four_heads_cfg, four_head_preds, four_head_targets,
):
    calc = PipelineMultiTaskLossCalculator(
        head_configs=four_heads_cfg, num_classes=4,
    )
    out = calc(four_head_preds, four_head_targets,
               current_step=0, total_steps=100)
    assert hasattr(out, "total_loss")
    assert hasattr(out, "per_head_losses")
    assert hasattr(out, "per_head_weights")
    # All four heads contributed.
    assert set(out.per_head_losses) == {
        "semantic", "boundary", "distance", "sdm",
    }
    assert torch.is_tensor(out.total_loss)
    assert out.total_loss.dim() == 0


def test_calculator_total_loss_is_weighted_sum(
    four_heads_cfg, four_head_preds, four_head_targets,
):
    calc = PipelineMultiTaskLossCalculator(
        head_configs=four_heads_cfg, num_classes=4,
    )
    out = calc(four_head_preds, four_head_targets)
    expected = sum(
        out.per_head_losses[n] * out.per_head_weights[n]
        for n in calc.head_names
    )
    assert float(out.total_loss) == pytest.approx(float(expected), abs=1e-6)


def test_calculator_distance_zero_when_perfect(four_heads_cfg):
    # Config with only distance head enabled.
    cfg = {
        "distance": HeadConfig(enabled=True, loss="distance_l1"),
    }
    calc = PipelineMultiTaskLossCalculator(
        head_configs=cfg, num_classes=2,
    )
    pred = torch.zeros(2, 1, 16, 16)
    targets = {"distance": torch.zeros(2, 1, 16, 16)}
    out = calc((pred,), targets)
    assert float(out.total_loss) == 0.0
    assert float(out.per_head_losses["distance"]) == 0.0


def test_calculator_missing_target_raises(
    four_heads_cfg, four_head_preds, four_head_targets,
):
    calc = PipelineMultiTaskLossCalculator(
        head_configs=four_heads_cfg, num_classes=4,
    )
    # Drop sdm target.
    incomplete = dict(four_head_targets)
    incomplete.pop("sdm")
    with pytest.raises(MultiTaskTargetError, match="sdm"):
        calc(four_head_preds, incomplete)


def test_calculator_predictions_mismatch_raises(
    four_heads_cfg, four_head_targets,
):
    calc = PipelineMultiTaskLossCalculator(
        head_configs=four_heads_cfg, num_classes=4,
    )
    # Three predictions instead of four.
    bad_preds = (
        torch.randn(2, 4, 16, 16),
        torch.randn(2, 1, 16, 16),
        torch.randn(2, 1, 16, 16),
    )
    with pytest.raises(ValueError, match="expected 4"):
        calc(bad_preds, four_head_targets)


def test_calculator_no_enabled_heads_raises():
    cfg = {"semantic": HeadConfig(enabled=False)}
    with pytest.raises(ValueError, match="no enabled heads"):
        PipelineMultiTaskLossCalculator(head_configs=cfg, num_classes=2)


def test_calculator_resolves_default_loss_when_omitted():
    """``HeadConfig.loss=None`` falls back to the per-head default
    name from the calculator."""
    cfg = {"semantic": HeadConfig(enabled=True, loss=None)}
    calc = PipelineMultiTaskLossCalculator(head_configs=cfg, num_classes=4)
    assert calc.per_head_loss_names == {"semantic": "dice_ce"}


def test_calculator_schedule_resolves_distance_weight():
    cfg = {
        "semantic": HeadConfig(enabled=True),
        "distance": HeadConfig(enabled=True, loss="distance_l1",
                               loss_weight=0.5),
    }
    sched = {
        "distance": LossScheduleEntryConfig(
            schedule="linear_warmup",
            start_weight=0.0, end_weight=1.0, warmup_fraction=0.5,
        ),
    }
    calc = PipelineMultiTaskLossCalculator(
        head_configs=cfg, num_classes=4, loss_schedules=sched,
    )
    preds = (torch.randn(2, 4, 16, 16), torch.randn(2, 1, 16, 16))
    targets = {
        "semantic": torch.randint(0, 4, (2, 16, 16)),
        "distance": torch.rand(2, 1, 16, 16) * 5.0,
    }
    # At step 0, distance scaled to 0.0.
    out0 = calc(preds, targets, current_step=0, total_steps=100)
    assert out0.per_head_weights["distance"] == pytest.approx(0.0)
    # At step 50 (past mid-warmup), distance scaled to 0.5.
    out50 = calc(preds, targets, current_step=50, total_steps=100)
    assert out50.per_head_weights["distance"] == pytest.approx(0.5)


def test_calculator_from_pipeline_config_round_trip():
    cfg = PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True),
            "boundary": HeadConfig(enabled=True, loss="bce"),
        },
    )
    calc = PipelineMultiTaskLossCalculator.from_pipeline_config(
        cfg, num_classes=3,
    )
    assert calc.head_names == ("semantic", "boundary")


# create_model_on_device dispatch amd deprecation 


def test_create_model_dispatches_pipeline_multitask_unet(four_head_modules):
    struct = {
        "type": utils.ModelType.PIPELINE_MULTITASK_UNET,
        "head_modules": four_head_modules,
        "encoder_name": "resnet34",
        "encoder_weights": None,
        "encoder_depth": 5,
        "in_channels": 1,
    }
    model = create_model_on_device("cpu", struct)
    assert isinstance(model, PipelineMultitaskUnet)
    assert model.head_names == ("semantic", "boundary", "distance", "sdm")


def test_create_model_pipeline_multitask_requires_head_modules():
    struct = {
        "type": utils.ModelType.PIPELINE_MULTITASK_UNET,
        "encoder_name": "resnet34",
        "encoder_weights": None,
    }
    with pytest.raises(ValueError, match="head_modules"):
        create_model_on_device("cpu", struct)


def test_create_model_legacy_multitask_raises_deprecation():
    struct = {"type": utils.ModelType.MULTITASK_UNET}
    with pytest.raises(DeprecationWarning, match="pipeline.yaml"):
        create_model_on_device("cpu", struct)


# End-to-end: model, calculator, backprop 


def test_end_to_end_model_calculator_loss_decreases(
    four_heads_cfg, four_head_modules,
):
    """Component-level smoke: model, calculator, Adam -> loss decreases.

    """
    model = PipelineMultitaskUnet(
        head_modules=four_head_modules,
        encoder_name="resnet34", encoder_weights=None,
        in_channels=1,
    )
    calc = PipelineMultiTaskLossCalculator(
        head_configs=four_heads_cfg, num_classes=4,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # One fixed batch 
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    targets = {
        "semantic": torch.randint(0, 4, (2, 64, 64)),
        "boundary": (torch.rand(2, 1, 64, 64) > 0.7).float(),
        "distance": torch.rand(2, 1, 64, 64) * 5.0,
        "sdm":      torch.rand(2, 1, 64, 64) * 2.0 - 1.0,
    }
    initial_loss = None
    for step in range(5):
        optim.zero_grad()
        preds = model(x)
        out = calc(preds, targets, current_step=step, total_steps=5)
        if step == 0:
            initial_loss = float(out.total_loss)
        out.total_loss.backward()
        optim.step()
    final_loss = float(out.total_loss)
    assert final_loss < initial_loss, (
        f"loss should decrease: initial={initial_loss}, final={final_loss}"
    )
