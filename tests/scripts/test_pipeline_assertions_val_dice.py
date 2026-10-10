"""The ``val-dice-within`` harness assertion compares the same checkpoint
key on both sides.

Before this, each side took the first key it found from a candidate list, so
a Lightning checkpoint (which records ``val_fg_dice``) was compared against a
legacy checkpoint's ``loss_val``: a Dice against a loss.
"""

from pathlib import Path

import pytest
import torch

from tests.scripts.pipeline_assertions import assert_val_dice_within


def _write_ckpt(path: Path, **scalars) -> Path:
    torch.save({"model_state_dict": {}, **scalars}, str(path))
    return path


def test_compares_loss_val_even_when_one_side_records_a_dice(tmp_path):
    lightning = _write_ckpt(tmp_path / "a.pytorch", val_fg_dice=0.84, loss_val=0.40)
    legacy = _write_ckpt(tmp_path / "b.pytorch", loss_val=0.39)
    assert_val_dice_within(lightning, legacy, max_drift=0.10)  # no raise


def test_fails_on_loss_val_drift_and_names_the_key(tmp_path):
    a = _write_ckpt(tmp_path / "a.pytorch", val_fg_dice=0.84, loss_val=0.22)
    b = _write_ckpt(tmp_path / "b.pytorch", loss_val=0.39)
    with pytest.raises(AssertionError, match=r"loss_val drift=0\.17"):
        assert_val_dice_within(a, b, max_drift=0.10)


def test_missing_key_is_reported_for_the_side_lacking_it(tmp_path):
    a = _write_ckpt(tmp_path / "a.pytorch", val_fg_dice=0.84, loss_val=0.22)
    b = _write_ckpt(tmp_path / "b.pytorch", val_fg_dice=0.80)
    with pytest.raises(AssertionError, match=r"does not expose 'loss_val'.*b="):
        assert_val_dice_within(a, b, max_drift=0.10)


def test_alternative_key_is_read_from_both_sides(tmp_path):
    a = _write_ckpt(tmp_path / "a.pytorch", val_fg_dice=0.84, loss_val=0.22)
    b = _write_ckpt(tmp_path / "b.pytorch", val_fg_dice=0.80, loss_val=0.90)
    assert_val_dice_within(a, b, max_drift=0.10, key="val_fg_dice")  # no raise
