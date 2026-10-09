"""The train settings' loss options reach the default trainer's semantic head loss.

``ce_weight``, ``dice_weight``, ``dice_weight_mode`` and
``exclude_background_from_dice`` were read by the legacy raw-torch trainer (`--legacy-trainer`)
only; the Lightning trainer resolved ``CombinedCEDiceLoss`` to a factory that
took ``alpha`` / ``beta`` alone and received neither, so it trained with the
loss's own 0.5 / 0.5 and default Dice weighting whatever the settings said.
"""

from types import SimpleNamespace

import pytest

import volume_segmantics.data.targets  # noqa: F401 - register targets
import volume_segmantics.model.heads  # noqa: F401 - register heads
import volume_segmantics.model.loss_registry  # noqa: F401 - register losses
from volume_segmantics.data import pipeline_registry
from volume_segmantics.data.pipeline_loader import (
    PipelineConfigError,
    legacy_settings_to_pipeline_config,
    parse_pipeline_dict,
    semantic_loss_options_from_settings,
    with_semantic_loss_settings,
)
from volume_segmantics.model.operations.trainer_losses import ClassWeightedDiceLoss, CombinedCEDiceLoss
from volume_segmantics.model.training.multitask_calculator import PipelineMultiTaskLossCalculator


def _settings(**over):
    base = dict(loss_criterion="CombinedCEDiceLoss", quality="medium")
    base.update(over)
    return SimpleNamespace(**base)


def test_the_dice_ce_factory_forwards_every_head_option():
    loss = pipeline_registry.build_loss(
        "dice_ce", num_classes=3, alpha=0.2, beta=0.8, dice_weight_mode="uniform", exclude_background=True,
    )
    assert isinstance(loss, CombinedCEDiceLoss) and (loss.alpha, loss.beta) == (0.2, 0.8)
    assert loss.dice_loss.weight_mode == "uniform" and loss.dice_loss.exclude_background is True
    weighted = pipeline_registry.build_loss(
        "class_weighted_dice", num_classes=3, dice_weight_mode="inverse_freq", exclude_background=True,
    )
    assert isinstance(weighted, ClassWeightedDiceLoss)
    assert weighted.weight_mode == "inverse_freq" and weighted.exclude_background is True


def test_the_legacy_mapping_carries_the_documented_options():
    cfg = legacy_settings_to_pipeline_config(_settings(
        ce_weight=0.2, dice_weight=0.8, dice_weight_mode="inverse_freq", exclude_background_from_dice=True,
    ))
    assert cfg.heads["semantic"].extra == {
        "alpha": 0.2, "beta": 0.8, "dice_weight_mode": "inverse_freq", "exclude_background": True,
    }
    weighted = legacy_settings_to_pipeline_config(_settings(
        loss_criterion="ClassWeightedDiceLoss", dice_weight_mode="uniform", exclude_background_from_dice=False,
    ))
    assert weighted.heads["semantic"].loss == "class_weighted_dice"
    assert weighted.heads["semantic"].extra == {"weight_mode": "uniform", "exclude_background": False}
    # The deprecated BCEDiceLoss spelling takes its weights from alpha / beta.
    bce = legacy_settings_to_pipeline_config(_settings(
        loss_criterion="BCEDiceLoss", alpha=0.3, beta=0.7, ce_weight=0.2, dice_weight=0.8,
    ))
    assert (bce.heads["semantic"].extra["alpha"], bce.heads["semantic"].extra["beta"]) == (0.3, 0.7)
    # Nothing set, or a loss that takes none of them: the loss keeps its defaults.
    assert legacy_settings_to_pipeline_config(_settings()).heads["semantic"].extra == {}
    assert legacy_settings_to_pipeline_config(_settings(loss_criterion="DiceLoss", ce_weight=0.2)).heads["semantic"].extra == {}


def test_bad_values_are_refused():
    with pytest.raises(PipelineConfigError, match="dice_weight_mode"):
        semantic_loss_options_from_settings(_settings(dice_weight_mode="cosine"), "dice_ce")
    with pytest.raises(PipelineConfigError, match="non-negative"):
        semantic_loss_options_from_settings(_settings(ce_weight=-1), "dice_ce")
    with pytest.raises(ValueError, match="weight_mode"):
        ClassWeightedDiceLoss(3, weight_mode="cosine")


def test_a_pipeline_yaml_heads_own_option_wins_over_the_settings():
    cfg = parse_pipeline_dict({"heads": {"semantic": {"enabled": True, "loss": "dice_ce", "alpha": 0.9}}})
    merged = with_semantic_loss_settings(cfg, _settings(ce_weight=0.2, dice_weight=0.8, dice_weight_mode="uniform"))
    assert merged.heads["semantic"].extra == {"alpha": 0.9, "beta": 0.8, "dice_weight_mode": "uniform"}
    untouched = parse_pipeline_dict({"heads": {"semantic": {"enabled": True, "loss": "tversky"}}})
    assert with_semantic_loss_settings(untouched, _settings(ce_weight=0.2)) is untouched
    assert with_semantic_loss_settings(cfg, _settings()) is cfg


def test_the_calculator_builds_the_semantic_loss_with_the_settings_values():
    cfg = legacy_settings_to_pipeline_config(_settings(
        ce_weight=0.2, dice_weight=0.8, dice_weight_mode="inverse_freq", exclude_background_from_dice=True,
    ))
    calc = PipelineMultiTaskLossCalculator.from_pipeline_config(cfg, num_classes=3)
    loss = calc.per_head_losses["semantic"]
    assert isinstance(loss, CombinedCEDiceLoss) and (loss.alpha, loss.beta) == (0.2, 0.8)
    assert loss.dice_loss.weight_mode == "inverse_freq" and loss.dice_loss.exclude_background is True
