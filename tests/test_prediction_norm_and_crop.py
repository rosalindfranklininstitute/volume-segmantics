"""Prediction gives the model the same data range as training. The output
position is the same as the input position.

Before this change, with ``use_imagenet_norm: False``, prediction sent 0-255
data to a model that had 0-1 training data. Also, the crop that removes the
prediction padding started one voxel too far when a slice dimension was a
multiple of 4 plus 1.
"""
import numpy as np
import pytest
import torch

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.data.augmentations import get_pred_preprocess_augs
from volume_segmantics.data.datasets import VolSeg2dPredictionDataset


def test_prediction_without_imagenet_norm_scales_integers_to_unit_range():
    volume = np.random.randint(0, 256, size=(3, 32, 32)).astype(np.uint8)
    dataset = VolSeg2dPredictionDataset(volume, imagenet_norm=False)
    image = dataset[1]
    assert image.dtype == np.float32
    assert np.allclose(image, volume[1] / 255.0)
    floats = np.random.rand(3, 32, 32).astype(np.float32)
    assert np.array_equal(VolSeg2dPredictionDataset(floats, imagenet_norm=False)[0], floats[0])


@pytest.mark.parametrize("height,width", [(33, 64), (37, 41), (61, 45), (501, 30), (64, 64), (35, 35)])
def test_crop_undoes_the_prediction_padding_exactly(height, width):
    image = np.random.randint(1, 255, size=(height, width)).astype(np.uint8)
    padded = get_pred_preprocess_augs(height, width)(image=image)["image"]
    cropped = utils.crop_tensor_to_array(torch.from_numpy(padded)[None], [height, width])
    assert np.array_equal(cropped[0], image)


@pytest.mark.parametrize("imagenet_norm", [True, False])
@pytest.mark.parametrize("use_2_5d", [False, True])
def test_monai_prediction_slices_match_albumentations(imagenet_norm, use_2_5d):
    """``augmentation_library: monai`` in the predict settings prepares the same
    slice as albumentations inside the original image area. Before this, the
    ImageNet normalisation raised "Cannot interpret 'torch.uint8' as a data
    type" on the MONAI tensor, and a 2.5D stack failed in the padding."""
    from types import SimpleNamespace

    from volume_segmantics.data.datasets import get_2d_prediction_dataset

    height, width = 65, 70
    vol = np.random.default_rng(0).integers(0, 256, (5, height, width), dtype=np.uint8)
    slices = {}
    for library in ("albumentations", "monai"):
        settings = SimpleNamespace(
            augmentation_library=library, use_imagenet_norm=imagenet_norm,
            use_2_5d_prediction=use_2_5d, num_slices=3,
        )
        slices[library] = torch.as_tensor(get_2d_prediction_dataset(vol, settings)[2])
    ours, ref = slices["monai"], slices["albumentations"]
    assert ours.shape == ref.shape == (3 if use_2_5d else 1, 96, 96)
    assert ours.dtype == torch.float32
    top, left = (96 - height) // 2, (96 - width) // 2
    window = (slice(None), slice(top, top + height), slice(left, left + width))
    assert torch.allclose(ours[window], ref[window])
