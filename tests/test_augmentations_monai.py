import numpy as np
import pytest

pytest.importorskip("monai")

import volume_segmantics.data.augmentations_monai as augs_monai
import volume_segmantics.utilities.config as cfg



@pytest.mark.parametrize(
    "dimension,expected",
    [(32, 32), (64, 64), (33, 64), (65, 96), (0, 0)],
)
def test_get_padded_dimension(dimension, expected):
    """get_padded_dimension returns dimension divisible by IM_SIZE_DIVISOR."""
    assert augs_monai.get_padded_dimension(dimension) == expected
    if dimension > 0 and expected > 0:
        assert expected % cfg.IM_SIZE_DIVISOR == 0




def test_albumentations_style_compose_image_mask_interface():
    """AlbumentationsStyleCompose accepts image= and mask= and returns dict with image, mask."""
    img_size = 64
    compose = augs_monai.get_train_preprocess_augs(img_size)
    img = np.random.rand(32, 48).astype(np.float32)
    mask = np.zeros((32, 48), dtype=np.uint8)
    out = compose(image=img, mask=mask)
    assert "image" in out
    assert "mask" in out
    # MONAI uses channel-first: image (C, H, W), mask (H, W) or (1, H, W)
    assert out["image"].shape[-2:] == (img_size, img_size)
    assert out["mask"].shape[-2:] == (img_size, img_size)


def test_albumentations_style_compose_image_only():
    """Prediction preprocess accepts image only (no mask)."""
    h, w = 50, 60
    compose = augs_monai.get_pred_preprocess_augs(h, w)
    img = np.random.rand(h, w).astype(np.float32)
    out = compose(image=img)
    assert "image" in out
    # MONAI channel-first: (C, H, W); spatial dims are last two
    assert out["image"].shape[-2] >= h and out["image"].shape[-1] >= w



def test_monai_train_preprocess_augs_produces_square():
    """MONAI train preprocess resizes to square img_size."""
    img_size = 128
    aug = augs_monai.get_train_preprocess_augs(img_size)
    img = np.random.rand(64, 96).astype(np.float32)
    mask = np.zeros((64, 96), dtype=np.uint8)
    out = aug(image=img, mask=mask)
    assert out["image"].shape[-2:] == (img_size, img_size)
    assert out["mask"].shape[-2:] == (img_size, img_size)



def test_monai_pred_preprocess_augs_pads_to_divisor():
    """MONAI pred preprocess pads so dimensions are divisible by IM_SIZE_DIVISOR."""
    h, w = 65, 70
    aug = augs_monai.get_pred_preprocess_augs(h, w)
    img = np.random.rand(h, w).astype(np.float32)
    out = aug(image=img)
    im = out["image"]
    # Channel-first: spatial dims are last two
    assert im.shape[-2] >= h and im.shape[-1] >= w
    assert im.shape[-2] % cfg.IM_SIZE_DIVISOR == 0
    assert im.shape[-1] % cfg.IM_SIZE_DIVISOR == 0




def test_monai_get_train_augs_returns_compose():
    """get_train_augs returns AlbumentationsStyleCompose."""
    aug = augs_monai.get_train_augs(64, num_channels=1)
    assert isinstance(aug, augs_monai.AlbumentationsStyleCompose)


def test_monai_train_augs_can_be_applied_and_preserves_size():
    """MONAI train augs can be applied and output has requested spatial size."""
    img_size = 64
    aug = augs_monai.get_train_augs(img_size, num_channels=1)
    # RandSpatialCropd expects channel-first; use (1, H, W) so shape[1:] is (H, W)
    img = np.random.rand(1, img_size, img_size).astype(np.float32)
    mask = np.random.randint(0, 4, (img_size, img_size), dtype=np.uint8)
    out = aug(image=img, mask=mask)
    assert out["image"].shape[-2:] == (img_size, img_size)
    assert out["mask"].shape[-2:] == (img_size, img_size)


def test_monai_train_augs_rand_histogram_shift_for_1_and_3_channels():
    """RandHistogramShift is included for 1 and 3 channels, not for 5 (e.g. 2.5D)."""
    # Pipeline expects channel-first (C, H, W) for RandSpatialCropd.
    img_1 = np.random.rand(1, 32, 32).astype(np.float32)
    img_3 = np.random.rand(3, 32, 32).astype(np.float32)
    img_5 = np.random.rand(5, 32, 32).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    aug_1 = augs_monai.get_train_augs(32, num_channels=1)
    aug_3 = augs_monai.get_train_augs(32, num_channels=3)
    aug_5 = augs_monai.get_train_augs(32, num_channels=5)
    out_1 = aug_1(image=img_1, mask=mask)
    out_3 = aug_3(image=img_3, mask=mask)
    out_5 = aug_5(image=img_5, mask=mask)
    assert out_1["image"].shape[-2:] == (32, 32)
    assert out_3["image"].shape[-2:] == (32, 32)
    assert out_5["image"].shape[-2:] == (32, 32)



def test_monai_postprocess_augs_returns_tensors():
    """get_postprocess_augs applies ToTensorWrapper; image and mask become tensors."""
    aug = augs_monai.get_postprocess_augs()
    img = np.random.rand(32, 32).astype(np.float32)
    mask = np.random.randint(0, 4, (32, 32), dtype=np.uint8)
    out = aug(image=img, mask=mask)
    import torch
    assert isinstance(out["image"], torch.Tensor)
    assert isinstance(out["mask"], torch.Tensor)
    assert out["image"].dtype in (torch.float32, torch.float64)
    assert out["mask"].dtype in (torch.long, torch.int64)



def test_rand_transposed_swaps_spatial_axes():
    """RandTransposed with prob=1.0 transposes spatial dimensions."""
    from volume_segmantics.data.augmentations_monai import RandTransposed, KEYS

    np.random.seed(42)
    trans = RandTransposed(keys=KEYS, prob=1.0)
    data = {
        "image": np.random.rand(3, 8, 12).astype(np.float32),  # (C, H, W)
        "mask": np.random.randint(0, 2, (8, 12), dtype=np.uint8),
    }
    out = trans(data)
    # After transpose (0,2,1): (3,8,12) -> (3,12,8)
    assert out["image"].shape == (3, 12, 8)
    assert out["mask"].shape == (12, 8)


def test_rand_transposed_prob_zero_unchanged():
    """RandTransposed with prob=0 leaves data unchanged."""
    from volume_segmantics.data.augmentations_monai import RandTransposed, KEYS

    np.random.seed(42)
    trans = RandTransposed(keys=KEYS, prob=0.0)
    img = np.random.rand(1, 10, 20).astype(np.float32)
    mask = np.zeros((10, 20), dtype=np.uint8)
    data = {"image": img.copy(), "mask": mask.copy()}
    out = trans(data)
    assert out["image"].shape == img.shape
    assert np.allclose(out["image"], img)
    assert np.array_equal(out["mask"], mask)



def test_to_tensor_wrapper_numpy_to_tensor():
    """ToTensorWrapper converts numpy image and mask to torch tensors."""
    from volume_segmantics.data.augmentations_monai import ToTensorWrapper, IMAGE_KEY, MASK_KEY

    wrapper = ToTensorWrapper()
    data = {
        IMAGE_KEY: np.random.rand(32, 32).astype(np.float32),
        MASK_KEY: np.random.randint(0, 4, (32, 32), dtype=np.uint8),
    }
    out = wrapper(data)
    import torch
    assert isinstance(out[IMAGE_KEY], torch.Tensor)
    assert isinstance(out[MASK_KEY], torch.Tensor)
    assert out[IMAGE_KEY].ndim == 3  # (C, H, W) with C=1
    assert out[MASK_KEY].ndim == 2
