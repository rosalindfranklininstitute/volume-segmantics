"""Tests for the checked state_dict loader 

`load_state_dict(strict=False)` silently tolerates any key mismatch, so an
incompatible checkpoint would load a partially- or fully-random model with no
error. `_load_state_dict_checked` makes the fully-random case fatal and partial
mismatches visible.
"""

import logging

import pytest
import torch
import torch.nn as nn

from volume_segmantics.model.model_2d import _load_state_dict_checked


def test_matching_state_dict_loads_cleanly(caplog):
    src = nn.Linear(4, 3)
    dst = nn.Linear(4, 3)
    with caplog.at_level(logging.WARNING):
        _load_state_dict_checked(dst, src.state_dict(), context="ok.pth")
    # No warnings; weights actually copied across.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert torch.equal(dst.weight, src.weight)
    assert torch.equal(dst.bias, src.bias)


def test_completely_incompatible_checkpoint_raises():
    # A checkpoint whose keys do not overlap the model at all -> nothing loads
    # -> the model would stay fully random, so this must raise. (Shape mismatches
    # on overlapping keys are already raised loudly by torch, so this guard is
    # specifically for the silent no-overlap case.)
    foreign_state = {"some.other.layer.weight": torch.zeros(3, 4),
                     "some.other.layer.bias": torch.zeros(3)}
    dst = nn.Linear(4, 3)
    with pytest.raises(RuntimeError, match="none of"):
        _load_state_dict_checked(dst, foreign_state, context="wrong.pth")


def test_partial_mismatch_warns_but_loads(caplog):
    # Model has two sub-layers; checkpoint only carries the first -> the second
    # stays at init. That's a partial load: visible warning, no raise.
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    partial = {f"0.{k}": v for k, v in nn.Linear(4, 4).state_dict().items()}
    with caplog.at_level(logging.WARNING):
        incompatible = _load_state_dict_checked(model, partial, context="partial.pth")
    assert incompatible.missing_keys  # the second layer's params
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and "missing" in warnings[0].getMessage().lower()


def test_unexpected_keys_warn_but_load(caplog):
    src = nn.Linear(4, 3)
    state = dict(src.state_dict())
    state["bogus_extra_key"] = torch.zeros(2)  # unexpected, not in the model
    dst = nn.Linear(4, 3)
    with caplog.at_level(logging.WARNING):
        incompatible = _load_state_dict_checked(dst, state, context="extra.pth")
    assert "bogus_extra_key" in incompatible.unexpected_keys
    # Real params still matched, so no raise; warning emitted.
    assert torch.equal(dst.weight, src.weight)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]
