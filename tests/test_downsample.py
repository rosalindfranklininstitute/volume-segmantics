"""``downsample``: the labels follow the image at training, and every
prediction output goes back to the input volume's shape."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from volume_segmantics.api import _at_input_scale
from volume_segmantics.data.base_data_manager import BaseDataManager
from volume_segmantics.data.slicers import TrainingDataSlicer
from volume_segmantics.model.operations import vol_seg_prediction_manager as vspm
from volume_segmantics.model.operations.vol_seg_prediction_manager import VolSeg2DPredictionManager
from volume_segmantics.utilities import Quality

VOL_SHAPE = (3, 5, 6)  # not a cube, so no axis can be mistaken for another
NUM_CLASSES = 3


def test_downsample_labels_matches_downsample_data_shape():
    vol = np.arange(7 * 9 * 10, dtype=np.float32).reshape(7, 9, 10)
    labels = (vol % 3).astype(np.uint8)
    small = utils.downsample_labels(labels)
    assert small.shape == utils.downsample_data(vol).shape == (4, 5, 5)
    assert small[1, 2, 3] == labels[2, 4, 6]  # nearest: every second voxel
    assert set(np.unique(small)) <= set(np.unique(labels))  # picked, never averaged


@pytest.mark.parametrize("shape", [(8, 8, 8), (7, 9, 10)])
def test_upsample_to_restores_labels_probs_and_channel_first_arrays(shape):
    small = tuple(-(-n // 2) for n in shape)
    labels = np.random.default_rng(0).integers(0, 3, size=small).astype(np.uint8)
    up = utils.upsample_to(labels, shape)
    assert up.shape == shape and up.dtype == labels.dtype
    assert np.array_equal(up[::2, ::2, ::2], labels)  # the voxel itself
    odd = up[1::2, 1::2, 1::2]  # and its nearest neighbour
    assert np.array_equal(odd, labels[: odd.shape[0], : odd.shape[1], : odd.shape[2]])
    assert utils.upsample_to(np.zeros((*small, 2), np.float16), shape).shape == (*shape, 2)
    assert utils.upsample_to(np.zeros((2, *small), np.uint8), shape).shape == (2, *shape)
    with pytest.raises(ValueError, match="not"):
        utils.upsample_to(np.zeros((3, 3, 3)), shape)


def test_data_manager_records_the_input_shape_before_downsampling(training_settings):
    training_settings.downsample = True
    vol = np.random.default_rng(0).integers(0, 256, size=(7, 9, 10))
    manager = BaseDataManager(vol, training_settings)
    assert manager.input_shape == (7, 9, 10) and manager.data_vol.shape == (4, 5, 5)


def test_slicer_downsamples_the_labels_with_the_image(rand_int_volume, rand_label_volume, training_settings):
    training_settings.downsample = True
    slicer = TrainingDataSlicer(rand_int_volume, rand_label_volume, training_settings)
    assert slicer.data_vol.shape == slicer.seg_vol.shape == tuple(-(-n // 2) for n in rand_int_volume.shape)
    assert slicer.input_shape == rand_int_volume.shape


def test_api_arrays_return_to_the_input_shape():
    small, shape = (4, 4, 4), (8, 8, 8)
    arrays = {
        "teacher_argmax": np.ones(small, np.uint8),
        "teacher_probs": np.ones((2, *small), np.float32),
        "note": "kept",
    }
    out = _at_input_scale(arrays, SimpleNamespace(downsample=True), shape)
    assert out["teacher_argmax"].shape == shape and out["teacher_probs"].shape == (2, *shape)
    assert out["note"] == "kept"
    assert _at_input_scale(arrays, SimpleNamespace(downsample=False), shape) is arrays


class _StubPredictor:
    """A single-axis predictor over whatever volume it is handed, with a boundary head."""

    def __init__(self, *_args, **_kwargs):
        self.label_codes = {0: "background", 1: "feature", 2: "other"}
        self._tasks = {}

    def _predict_single_axis(self, vol, axis=None):
        labels = np.full(vol.shape, 1, dtype=np.uint8)
        probs = np.zeros((*vol.shape, NUM_CLASSES), dtype=np.float32)
        probs[..., 1] = 1.0
        logits = np.zeros((*vol.shape, NUM_CLASSES), dtype=np.float32)
        self._tasks = {"boundary": {
            "labels": np.ones(vol.shape, np.uint8),
            "probs": np.full(vol.shape, 0.5, np.float32),
            "logits": None,
        }}
        return labels, probs, logits

    def get_additional_task_outputs(self):
        return self._tasks


def _read(path):
    if path.suffix in (".h5", ".hdf5", ".hdf"):
        with h5py.File(path, "r") as f:
            return f[next(iter(f.keys()))][()]
    return tifffile.imread(path)


def test_prediction_manager_writes_every_output_at_the_input_shape(monkeypatch, prediction_settings, tmp_path):
    """The model sees the volume at half scale; what is returned and written is the input's size."""
    monkeypatch.setattr(vspm, "VolSeg2dPredictor", _StubPredictor)
    prediction_settings.downsample = True
    prediction_settings.output_probs = True
    prediction_settings.output_entropy = False
    prediction_settings.one_hot = False
    big = np.random.default_rng(0).integers(0, 256, size=tuple(2 * n for n in VOL_SHAPE))
    manager = VolSeg2DPredictionManager("unused-model-path", big, prediction_settings)
    assert manager.data_vol.shape == VOL_SHAPE and manager.input_shape == big.shape
    ext = {"hdf": "h5"}.get(cfg.OUTPUT_FORMAT, cfg.OUTPUT_FORMAT)
    out = tmp_path / f"pred.{ext}"
    prediction = manager.predict_volume_to_path(out, Quality.LOW)
    assert prediction.shape == big.shape
    assert _read(out).shape == big.shape
    assert _read(tmp_path / f"pred_probs.{ext}").shape == (*big.shape, NUM_CLASSES)
    assert _read(tmp_path / f"pred_BND.{ext}").shape == big.shape


def test_restore_input_scale_leaves_non_arrays_alone():
    manager = VolSeg2DPredictionManager.__new__(VolSeg2DPredictionManager)
    manager.input_shape = (8, 8, 8)
    arrays = {"prediction": np.zeros((4, 4, 4), np.uint8), "probs": None}
    tasks = {"boundary": {"labels": np.zeros((4, 4, 4), np.uint8), "logits": None}}
    manager._restore_input_scale(arrays, tasks)
    assert arrays["prediction"].shape == (8, 8, 8) and arrays["probs"] is None
    assert tasks["boundary"]["labels"].shape == (8, 8, 8) and tasks["boundary"]["logits"] is None
