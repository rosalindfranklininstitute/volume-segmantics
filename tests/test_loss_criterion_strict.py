"""The default trainer raises an error for a ``loss_criterion`` that it does
not have.

Before this change, an incorrect name used ``CombinedCEDiceLoss`` and gave no
message. A semantic head in ``pipeline.yaml`` that had no ``loss`` ignored the
``loss_criterion`` of the settings.
"""
from types import SimpleNamespace

import pytest

from volume_segmantics.data.pipeline_loader import (
    PipelineConfigError,
    legacy_settings_to_pipeline_config,
    parse_pipeline_dict,
    with_semantic_loss_settings,
)

SEMANTIC_ONLY = {"heads": {"semantic": {"enabled": True}}}


def test_a_misspelt_loss_criterion_is_refused():
    with pytest.raises(PipelineConfigError, match="TverskyLos"):
        legacy_settings_to_pipeline_config(SimpleNamespace(loss_criterion="TverskyLos"))


def test_a_semantic_head_without_a_loss_takes_the_settings_loss_criterion():
    cfg = parse_pipeline_dict(SEMANTIC_ONLY)
    assert cfg.heads["semantic"].loss is None
    merged = with_semantic_loss_settings(cfg, SimpleNamespace(loss_criterion="TverskyLoss"))
    assert merged.heads["semantic"].loss == "tversky"
    with pytest.raises(PipelineConfigError):
        with_semantic_loss_settings(cfg, SimpleNamespace(loss_criterion="TverskyLos"))


def test_a_loss_the_head_states_wins_over_the_settings():
    cfg = parse_pipeline_dict({"heads": {"semantic": {"enabled": True, "loss": "dice"}}})
    merged = with_semantic_loss_settings(cfg, SimpleNamespace(loss_criterion="TverskyLoss"))
    assert merged.heads["semantic"].loss == "dice"
