"""The 2.5D input stack: recorded in the checkpoint at training, checked
exactly at prediction instead of silently adapting the model to the settings."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.model_2d import (
    check_input_stack,
    create_model_from_file,
    create_model_on_device,
)
from volume_segmantics.model.operations.lightning2d import VolSegCheckpointCallback
from volume_segmantics.utilities.early_stopping import EarlyStopping

pytest.importorskip("segmentation_models_pytorch")


def _checkpoint(tmp_path, in_channels, input_stack=None):
    struct = {"type": "U_NET", "encoder_name": "resnet18", "encoder_weights": None,
              "in_channels": in_channels, "classes": 2}
    model = create_model_on_device("cpu", dict(struct))
    payload = {"model_state_dict": model.state_dict(), "model_struc_dict": struct, "label_codes": {}}
    if input_stack is not None:
        payload["input_stack"] = input_stack
    path = tmp_path / f"m{in_channels}.pytorch"
    torch.save(payload, path)
    return path, model


def _first_conv(model):
    return next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))


def _predict_settings(use_2_5d, num_slices):
    return SimpleNamespace(use_2_5d_prediction=use_2_5d, num_slices=num_slices)


def test_the_stack_a_training_run_feeds_the_model():
    assert cfg.input_stack_from_settings(SimpleNamespace(use_2_5d_slicing=True, num_slices=5)) == {
        "use_2_5d": True, "num_slices": 5,
    }
    assert cfg.input_stack_from_settings(SimpleNamespace()) == {"use_2_5d": False, "num_slices": 3}


def test_a_recorded_stack_is_compared_exactly():
    stack = {"use_2_5d": True, "num_slices": 3}
    assert check_input_stack(use_2_5d=True, num_slices=3, model_stack=stack) is None
    assert "use_2_5d_prediction=False" in check_input_stack(use_2_5d=False, num_slices=3, model_stack=stack)
    problem = check_input_stack(use_2_5d=True, num_slices=5, model_stack=stack)
    assert "num_slices=3" in problem and "num_slices=5" in problem
    # A recorded 2D stack ignores num_slices, which means nothing in 2D.
    assert check_input_stack(
        use_2_5d=False, num_slices=5, model_stack={"use_2_5d": False, "num_slices": 3},
    ) is None


def test_an_older_checkpoint_is_compared_on_its_first_conv():
    assert check_input_stack(use_2_5d=True, num_slices=3, model_in_channels=3) is None
    assert check_input_stack(use_2_5d=False, num_slices=3, model_in_channels=1) is None
    assert check_input_stack(use_2_5d=False, num_slices=3) is None  # nothing known, nothing to refuse
    assert "takes 1 input channels" in check_input_stack(use_2_5d=True, num_slices=3, model_in_channels=1)


@pytest.mark.parametrize("num_slices", [1, 2, 4])
def test_an_even_or_too_small_stack_is_refused(num_slices):
    assert "odd num_slices" in check_input_stack(
        use_2_5d=True, num_slices=num_slices, model_in_channels=num_slices,
    )


def test_the_loader_refuses_a_recorded_stack_that_disagrees_with_the_settings(tmp_path):
    path, _ = _checkpoint(tmp_path, 3, input_stack={"use_2_5d": True, "num_slices": 3})
    with pytest.raises(ValueError, match="2.5D input stack"):
        create_model_from_file(path, gpu=False, settings=_predict_settings(False, 3))
    with pytest.raises(ValueError, match="num_slices=5"):
        create_model_from_file(path, gpu=False, settings=_predict_settings(True, 5))


def test_the_loader_accepts_the_stack_the_checkpoint_recorded(tmp_path):
    path, _ = _checkpoint(tmp_path, 3, input_stack={"use_2_5d": True, "num_slices": 3})
    model, classes, _, _ = create_model_from_file(path, gpu=False, settings=_predict_settings(True, 3))
    assert classes == 2 and _first_conv(model).in_channels == 3


def test_the_loader_refuses_a_2d_checkpoint_predicted_as_2_5d(tmp_path):
    path, _ = _checkpoint(tmp_path, 1)
    with pytest.raises(ValueError, match="takes 1 input channels"):
        create_model_from_file(path, gpu=False, settings=_predict_settings(True, 3))


def test_a_legacy_three_channel_checkpoint_is_folded_to_grayscale_exactly(tmp_path, caplog):
    """No record, three channels, 2D requested: trained on replicated grayscale,
    so the folded model must give the original's output on the replicated input."""
    path, original = _checkpoint(tmp_path, 3)
    with caplog.at_level(logging.WARNING):
        model, _, _, _ = create_model_from_file(path, gpu=False, settings=_predict_settings(False, 3))
    assert "replicated grayscale" in caplog.text
    assert _first_conv(model).in_channels == 1
    x = torch.rand(1, 1, 64, 64)
    original.eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), original(x.repeat(1, 3, 1, 1)), atol=1e-5)


def test_without_settings_the_loader_checks_nothing(tmp_path):
    path, _ = _checkpoint(tmp_path, 3)
    model, _, _, _ = create_model_from_file(path, gpu=False)
    assert _first_conv(model).in_channels == 3


def test_the_lightning_checkpoint_records_the_stack(tmp_path):
    cb = VolSegCheckpointCallback(
        output_path=tmp_path / "ckpt.pytorch", model_struc_dict={"type": "U_NET"},
        input_stack={"use_2_5d": True, "num_slices": 5},
    )
    trainer = MagicMock()
    trainer.callback_metrics = {}
    pl_module = SimpleNamespace(get_model_state_dict_for_volseg=lambda: {})
    cb._save_volseg_checkpoint(trainer, pl_module, cb.output_path)
    loaded = torch.load(tmp_path / "ckpt.pytorch", weights_only=False)
    assert loaded["input_stack"] == {"use_2_5d": True, "num_slices": 5}
    assert "input_stack" not in loaded["model_struc_dict"]


def test_the_legacy_trainer_checkpoint_records_the_stack(tmp_path):
    es = EarlyStopping(
        patience=2, path=tmp_path / "ckpt.pytorch", model_dict={},
        input_stack={"use_2_5d": False, "num_slices": 3},
    )
    model = torch.nn.Linear(2, 2)
    es(0.5, model, torch.optim.SGD(model.parameters(), lr=0.1), label_codes={})
    loaded = torch.load(tmp_path / "ckpt.pytorch", weights_only=False)
    assert loaded["input_stack"] == {"use_2_5d": False, "num_slices": 3}
