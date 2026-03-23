import numpy as np
import pytest
import torch
from types import SimpleNamespace

from volume_segmantics.data.datasets import (
    VolSeg2dDataset,
    VolSeg2dPredictionDataset,
    get_2d_prediction_dataset,
    get_2d_training_dataset,
    get_2d_validation_dataset,
    get_augmentation_module,
)

@pytest.fixture()
def training_dataset(image_dir, label_dir, training_settings):
    return get_2d_training_dataset(image_dir, label_dir, training_settings)

@pytest.fixture()
def validation_dataset(image_dir, label_dir, training_settings):
    return get_2d_validation_dataset(image_dir, label_dir, training_settings)

@pytest.fixture()
def prediction_dataset(rand_int_volume):
    return get_2d_prediction_dataset(rand_int_volume)

class Test2dDataset:
    def test_get_2d_training_dataset_type(self, training_dataset):
        assert isinstance(training_dataset, VolSeg2dDataset)

    def test_get_2d_training_dataset_length(self, training_dataset):
        assert len(training_dataset) == 20

    def test_get_2d_training_dataset_get_item(self, training_dataset):
        im, mask = training_dataset[2]
        assert isinstance(im, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_get_2d_validation_dataset_type(self, validation_dataset):
        assert isinstance(validation_dataset, VolSeg2dDataset)

    def test_get_2d_validation_dataset_length(self, validation_dataset):
        assert len(validation_dataset) == 20

    def test_get_2d_validation_dataset_get_item(self, validation_dataset):
        im, mask = validation_dataset[2]
        assert isinstance(im, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_training_dataset_getitem_shapes_and_dtypes(self, training_dataset):
        """__getitem__ returns (image, mask) with correct ndim, dtypes and matching spatial dims."""
        im, mask = training_dataset[0]
        assert im.dim() == 3  # (C, H, W)
        assert mask.dim() == 2  # (H, W)
        assert im.dtype in (torch.float32, torch.float64)
        assert mask.dtype in (torch.long, torch.int64, torch.uint8)
        assert im.shape[1] == mask.shape[0] and im.shape[2] == mask.shape[1]

    def test_validation_dataset_getitem_label_values_in_range(self, validation_dataset):
        """Validation __getitem__ mask has integer labels (e.g. in [0, num_classes-1])."""
        _, mask = validation_dataset[0]
        assert mask.dtype in (torch.long, torch.int64, torch.uint8)
        unique = torch.unique(mask)
        assert unique.min().item() >= 0
        assert unique.max().item() <= 255  # reasonable label range

class Test2dPredictionDataset:
    def test_get_2d_prediction_dataset_type(self, prediction_dataset):
        assert isinstance(prediction_dataset, VolSeg2dPredictionDataset)

    def test_get_2d_prediction_dataset_length(self, prediction_dataset, rand_int_volume):
        assert len(prediction_dataset) == rand_int_volume.shape[0]

    def test_get_2d_prediction_dataset_get_item(self, prediction_dataset):
        im = prediction_dataset[2]
        assert isinstance(im, torch.Tensor)

    def test_get_augmentation_module_monai(self):
        pytest.importorskip("monai")
        from volume_segmantics.data import augmentations_monai as augs_monai

        settings = SimpleNamespace(augmentation_library="monai")
        mod = get_augmentation_module(settings)
        assert mod is augs_monai

    def test_prediction_dataset_2_5d_shape_and_border_handling(self, rand_int_volume):
        # Configure 2.5D prediction with 3 slices
        settings = SimpleNamespace(
            use_2_5d_prediction=True,
            num_slices=3,
            use_imagenet_norm=False,
            augmentation_library="albumentations",
        )
        ds = get_2d_prediction_dataset(rand_int_volume, settings)
        assert isinstance(ds, VolSeg2dPredictionDataset)
        assert len(ds) == rand_int_volume.shape[0]

        # After postprocessing, __getitem__ returns tensor (C, H, W); preprocessing may resize/pad
        im0 = ds[0]
        im_last = ds[len(ds) - 1]
        assert im0.dim() == 3 and im_last.dim() == 3
        assert im0.shape[0] == settings.num_slices  # channels first
        assert im_last.shape[0] == settings.num_slices
        # Spatial dims are determined by preprocessing (e.g. padding to mult of 32), not raw volume
        assert im0.shape[1] > 0 and im0.shape[2] > 0
        assert im_last.shape[1] == im0.shape[1] and im_last.shape[2] == im0.shape[2]
