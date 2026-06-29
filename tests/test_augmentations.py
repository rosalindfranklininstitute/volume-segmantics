import numpy as np
import pytest
import albumentations as A

import volume_segmantics.data.augmentations as augs
import volume_segmantics.utilities.config as cfg


@pytest.mark.parametrize(
    "test_input,expected", [(32, 32), (64, 64), (33, 64), (13, 32), (0, 0)]
)
def test_get_padded_dimension_some_vals(test_input, expected):
    assert augs.get_padded_dimension(test_input) == expected


def test_other_funcs_return_compose():
    assert isinstance(augs.get_train_preprocess_augs(256), A.Compose)
    assert isinstance(augs.get_pred_preprocess_augs(32, 64), A.Compose)
    assert isinstance(augs.get_train_augs(128), A.Compose)
    assert isinstance(augs.get_postprocess_augs(), A.Compose)


def test_train_preprocess_augs_produces_square_image():
    img_size = 128
    aug = augs.get_train_preprocess_augs(img_size)
    # smaller rectangular image
    img = np.zeros((64, 96), dtype=np.uint8)
    mask = np.zeros_like(img)
    out = aug(image=img, mask=mask)
    im_out, mask_out = out["image"], out["mask"]
    assert im_out.shape == (img_size, img_size)
    assert mask_out.shape == (img_size, img_size)


def test_pred_preprocess_augs_pads_to_divisor():
    h, w = 65, 70
    aug = augs.get_pred_preprocess_augs(h, w)
    img = np.zeros((h, w), dtype=np.uint8)
    out = aug(image=img)
    im_out = out["image"]
    assert im_out.shape[0] >= h and im_out.shape[1] >= w
    assert im_out.shape[0] % cfg.IM_SIZE_DIVISOR == 0
    assert im_out.shape[1] % cfg.IM_SIZE_DIVISOR == 0


def test_train_augs_respects_num_channels_for_clahe():
    img_size = 64
    # 1-channel: CLAHE should be present
    comp_1 = augs.get_train_augs(img_size, num_channels=1)
    assert any(isinstance(t, A.CLAHE) for t in comp_1.transforms)

    # 3-channel: CLAHE should be present
    comp_3 = augs.get_train_augs(img_size, num_channels=3)
    assert any(isinstance(t, A.CLAHE) for t in comp_3.transforms)

    # 5-channel (e.g. 2.5D with 5 slices): CLAHE should be skipped
    comp_5 = augs.get_train_augs(img_size, num_channels=5)
    assert not any(isinstance(t, A.CLAHE) for t in comp_5.transforms)


def test_train_augs_can_be_applied_and_preserves_size():
    img_size = 64
    aug = augs.get_train_augs(img_size, num_channels=1)
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    mask = np.zeros_like(img)
    out = aug(image=img, mask=mask)
    im_out, mask_out = out["image"], out["mask"]
    assert im_out.shape == (img_size, img_size)
    assert mask_out.shape == (img_size, img_size)
