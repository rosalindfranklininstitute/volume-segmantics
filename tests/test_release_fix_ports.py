"""Additional fixes for release

Each test pins the defect it closes: the entropy estimator's four return
values on the medium and high paths, an unknown output format refused before
prediction, the Lightning devices resolved from ``cuda_device``, the training
``image_size`` recorded in the checkpoint and read back, and the batch size
taken from the memory the driver reports free.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.model_2d import create_model_from_file
from volume_segmantics.model.operations.vol_seg_prediction_manager import VolSeg2DPredictionManager
from volume_segmantics.scripts.train_2d_model import _resolve_lightning_devices

pl = pytest.importorskip("pytorch_lightning")
from volume_segmantics.model.operations.lightning2d import VolSegCheckpointCallback  # noqa: E402


class _EntropyOnlyPredictor:
    """Answers the entropy estimator with four values and nothing else."""

    def __init__(self):
        self.calls = []

    def _prediction_estimate_entropy(self, data_vol):
        self.calls.append("entropy")
        shape = data_vol.shape
        return (np.zeros(shape, np.uint8), np.zeros(shape, np.float16),
                np.zeros(shape, np.float32), np.zeros(shape, np.uint8))

    def get_additional_task_outputs(self):
        return None


def _manager(settings, predictor):
    m = VolSeg2DPredictionManager.__new__(VolSeg2DPredictionManager)
    m.settings = settings
    m.predictor = predictor
    m.data_vol = np.zeros((4, 8, 8), np.uint8)
    return m


@pytest.mark.parametrize("quality", [utils.Quality.MEDIUM, utils.Quality.HIGH])
def test_entropy_output_runs_on_the_medium_and_high_paths(prediction_settings, quality):
    """These paths unpacked three values from a four-value estimator and crashed."""
    prediction_settings.output_entropy = True
    prediction_settings.output_probs = False
    prediction_settings.one_hot = False
    predictor = _EntropyOnlyPredictor()
    prediction = _manager(prediction_settings, predictor).predict_volume_to_path(None, quality)
    assert prediction.shape == (4, 8, 8) and predictor.calls == ["entropy"]


def test_an_unknown_output_format_is_refused_before_prediction(prediction_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OUTPUT_FORMAT", "npy")
    predictor = _EntropyOnlyPredictor()
    with pytest.raises(ValueError, match="Unknown output format 'npy'"):
        _manager(prediction_settings, predictor).predict_volume_to_path(tmp_path / "out.npy", utils.Quality.LOW)
    assert predictor.calls == []


def test_devices_default_to_the_settings_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _resolve_lightning_devices(SimpleNamespace(cuda_device=1), "gpu") == ([1], "auto")


def test_multi_gpu_is_opt_in_and_selects_permissive_ddp(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    s = lambda **k: SimpleNamespace(cuda_device=0, **k)  # noqa: E731
    assert _resolve_lightning_devices(s(lightning_devices="auto"), "gpu") == ("auto", "ddp_find_unused_parameters_true")
    assert _resolve_lightning_devices(s(lightning_devices="2"), "gpu") == (2, "ddp_find_unused_parameters_true")
    assert _resolve_lightning_devices(s(lightning_devices="1"), "gpu") == (1, "auto")
    assert _resolve_lightning_devices(s(lightning_devices="auto", lightning_strategy="ddp"), "gpu") == ("auto", "ddp")
    assert _resolve_lightning_devices(s(), "cpu") == ("auto", "auto")


def test_checkpoint_records_the_training_image_size_and_the_loader_returns_it(tmp_path):
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=1, classes=2)
    struct = {"type": "U_NET", "encoder_name": "resnet18", "encoder_weights": None,
              "in_channels": 1, "classes": 2, "encoder_depth": 5}
    trainer = SimpleNamespace(callbacks=[], global_rank=0, sanity_checking=False)
    pl_module = SimpleNamespace(get_model_state_dict_for_volseg=model.state_dict)
    sized = VolSegCheckpointCallback(tmp_path / "m.pytorch", struct, label_codes={0: "bg", 1: "fg"}, image_size=96)
    sized._save_volseg_checkpoint(trainer, pl_module, 0.25)
    assert torch.load(tmp_path / "m.pytorch", weights_only=False)["image_size"] == 96
    _, classes, _, image_size = create_model_from_file(tmp_path / "m.pytorch", gpu=False)
    assert (classes, image_size) == (2, 96)
    unsized = VolSegCheckpointCallback(tmp_path / "n.pytorch", struct)
    unsized._save_volseg_checkpoint(trainer, pl_module, 0.25)
    assert create_model_from_file(tmp_path / "n.pytorch", gpu=False)[3] is None


def test_batch_size_uses_the_memory_the_driver_has_free(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: (_ for _ in ()).throw(AssertionError("total memory must not be consulted")))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: ((cfg.BIG_CUDA_THRESHOLD - 1) * gib, 64 * gib))
    assert utils.get_batch_size(SimpleNamespace(cuda_device=0)) == cfg.SMALL_CUDA_BATCH
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: ((cfg.BIG_CUDA_THRESHOLD + 1) * gib, 64 * gib))
    assert utils.get_batch_size(SimpleNamespace(cuda_device=0)) == cfg.BIG_CUDA_TRAIN_BATCH
