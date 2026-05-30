"""Tests for v0.4.0b3 pipeline-mode dataset and dataloader.


"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

# Triggers head, target-generator, loss registrations.
import volume_segmantics.model.heads        # noqa: F401
import volume_segmantics.model.loss_registry  # noqa: F401
import volume_segmantics.data.targets       # noqa: F401

import volume_segmantics.utilities.base_data_utils as bdu
import volume_segmantics.utilities.config as cfg_mod
from volume_segmantics.data.dataloaders import (
    _pipeline_mode_required,
    get_pipeline_training_dataloaders,
)
from volume_segmantics.data.pipeline_dataset import (
    PHOTOMETRIC_TRANSFORMS,
    PipelineMultiTaskDataset,
    build_pipeline_augmentations,
    compute_pipeline_targets,
)
from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    HeadConfig,
    InstanceAssemblyConfig,
    PipelineConfig,
    TransformSpec,
)
from volume_segmantics.utilities.base_data_utils import prepare_training_batch


# Fixtures 


@pytest.fixture
def label_slice():
    """20×20 slice with a 10×10 class-1 square + 4×4 class-2 patch."""
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[5:15, 5:15] = 1
    arr[7:11, 7:11] = 2
    return arr


@pytest.fixture
def four_heads_cfg():
    return {
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
        "distance": HeadConfig(enabled=True, loss="distance_l1"),
        "sdm":      HeadConfig(
            enabled=True, loss="sdm_l1",
            extra={"variant": "binary", "d_clip": 5.0},
        ),
    }


@pytest.fixture
def four_heads_per_class_sdm_cfg():
    return {
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "sdm": HeadConfig(
            enabled=True, loss="sdm_l1",
            extra={"variant": "per_class", "d_clip": 5.0},
        ),
    }


@pytest.fixture
def fixture_dir(tmp_path):
    """20-slice fixture directory tree (data + seg)."""
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(20):
        cv2.imwrite(
            str(img_dir / f"data_{i:03d}.png"),
            (rng.random((64, 64)) * 255).astype(np.uint8),
        )
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


@pytest.fixture
def basic_pipeline_config(four_heads_cfg):
    return PipelineConfig(
        heads=four_heads_cfg,
        augmentations=AugmentationsConfig(
            backend="albumentations",
            train_transforms=[
                TransformSpec(name="HorizontalFlip", params={"p": 1.0}),
                TransformSpec(name="RandomBrightnessContrast",
                              params={"p": 0.0}),
            ],
        ),
    )


# compute_pipeline_targets 


def test_compute_pipeline_targets_keys(label_slice, four_heads_cfg):
    out = compute_pipeline_targets(
        label_slice, heads_cfg=four_heads_cfg, num_classes=3,
    )
    # Semantic is skipped by default (handled by dataset).
    assert set(out) == {"boundary", "distance", "sdm"}


def test_compute_pipeline_targets_includes_semantic_when_asked(
    label_slice, four_heads_cfg,
):
    out = compute_pipeline_targets(
        label_slice, heads_cfg=four_heads_cfg, num_classes=3,
        skip_semantic=False,
    )
    assert "semantic" in out
    np.testing.assert_array_equal(out["semantic"], label_slice.astype(np.int64))


def test_compute_pipeline_targets_skips_disabled():
    cfg = {
        "semantic": HeadConfig(enabled=True),
        "boundary": HeadConfig(enabled=False, loss="bce"),
        "distance": HeadConfig(enabled=True, loss="distance_l1"),
    }
    arr = np.zeros((10, 10), dtype=np.int32)
    arr[3:7, 3:7] = 1
    out = compute_pipeline_targets(arr, heads_cfg=cfg, num_classes=2)
    assert "distance" in out
    assert "boundary" not in out


def test_compute_pipeline_targets_per_class_sdm_shape(
    label_slice, four_heads_per_class_sdm_cfg,
):
    out = compute_pipeline_targets(
        label_slice, heads_cfg=four_heads_per_class_sdm_cfg, num_classes=3,
    )
    assert out["sdm"].shape == (2, 20, 20)  # num_classes - 1


def test_compute_pipeline_targets_3d_input_raises(four_heads_cfg):
    arr = np.zeros((4, 20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="2D label slice"):
        compute_pipeline_targets(arr, heads_cfg=four_heads_cfg, num_classes=3)


# build_pipeline_augmentations 


def test_photometric_transforms_set_includes_brightness_and_clahe():
    assert "RandomBrightnessContrast" in PHOTOMETRIC_TRANSFORMS
    assert "CLAHE" in PHOTOMETRIC_TRANSFORMS
    assert "GaussNoise" in PHOTOMETRIC_TRANSFORMS


def test_build_aug_splits_spatial_and_photometric():
    aug_cfg = AugmentationsConfig(
        train_transforms=[
            TransformSpec(name="HorizontalFlip", params={"p": 1.0}),
            TransformSpec(name="RandomBrightnessContrast", params={"p": 0.5}),
            TransformSpec(name="VerticalFlip", params={"p": 1.0}),
        ],
    )
    spatial, photo, addl = build_pipeline_augmentations(
        aug_cfg,
        enabled_head_names={"semantic": True, "boundary": True,
                            "distance": True, "sdm": True},
        img_size=64,
    )
    # Spatial Compose has the 2 flips + LongestMaxSize + PadIfNeeded.
    assert spatial is not None
    spatial_names = [t.__class__.__name__ for t in spatial.transforms]
    assert "HorizontalFlip" in spatial_names
    assert "VerticalFlip" in spatial_names
    assert "RandomBrightnessContrast" not in spatial_names
    # Photometric Compose has only the brightness transform.
    assert photo is not None
    photo_names = [t.__class__.__name__ for t in photo.transforms]
    assert photo_names == ["RandomBrightnessContrast"]


def test_build_aug_additional_targets_match_enabled_heads():
    aug_cfg = AugmentationsConfig(train_transforms=[])
    _, _, addl = build_pipeline_augmentations(
        aug_cfg,
        enabled_head_names={"semantic": True, "boundary": True,
                            "distance": False, "sdm": True},
    )
    # Semantic uses Albumentations's built-in 'mask' key; not in addl.
    assert "semantic" not in addl
    assert addl == {"boundary": "mask", "sdm": "image"}


def test_build_aug_unknown_transform_raises():
    aug_cfg = AugmentationsConfig(
        train_transforms=[
            TransformSpec(name="GalaxyBrainAugmentation", params={}),
        ],
    )
    with pytest.raises(ValueError, match="Unknown Albumentations"):
        build_pipeline_augmentations(
            aug_cfg, enabled_head_names={"semantic": True},
        )


def test_build_aug_returns_none_when_no_transforms():
    aug_cfg = AugmentationsConfig(train_transforms=[])
    spatial, photo, addl = build_pipeline_augmentations(
        aug_cfg, enabled_head_names={"semantic": True, "boundary": True},
    )
    # No img_size + no transforms -> no spatial Compose.
    assert spatial is None
    assert photo is None
    assert addl == {"boundary": "mask"}


# PipelineMultiTaskDataset 


def test_dataset_basic_sample_shape(fixture_dir, basic_pipeline_config):
    img_dir, msk_dir = fixture_dir
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64,
    )
    assert len(ds) == 20
    sample = ds[0]
    assert set(sample) == {"image", "semantic", "boundary", "distance", "sdm"}
    assert sample["image"].shape == (1, 64, 64)
    assert sample["semantic"].shape == (64, 64)
    assert sample["boundary"].shape == (1, 64, 64)
    assert sample["distance"].shape == (1, 64, 64)
    assert sample["sdm"].shape == (1, 64, 64)


def test_dataset_dtypes(fixture_dir, basic_pipeline_config):
    img_dir, msk_dir = fixture_dir
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64,
    )
    s = ds[0]
    assert s["image"].dtype == torch.float32
    assert s["semantic"].dtype == torch.int64
    assert s["boundary"].dtype == torch.float32
    assert s["distance"].dtype == torch.float32
    assert s["sdm"].dtype == torch.float32


def test_dataset_sdm_in_tanh_range(fixture_dir, basic_pipeline_config):
    img_dir, msk_dir = fixture_dir
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64,
    )
    s = ds[0]
    assert float(s["sdm"].max()) <= 1.0
    assert float(s["sdm"].min()) >= -1.0


def test_dataset_per_class_sdm_emits_multi_channel(
    fixture_dir, four_heads_per_class_sdm_cfg,
):
    img_dir, msk_dir = fixture_dir
    cfg = PipelineConfig(
        heads=four_heads_per_class_sdm_cfg,
        augmentations=AugmentationsConfig(train_transforms=[]),
    )
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=cfg, num_classes=3, img_size=64,
    )
    sample = ds[0]
    # K = num_classes - 1 = 2 for per_class.
    assert sample["sdm"].shape == (2, 64, 64)


def test_dataset_imagenet_norm_changes_image_range(
    fixture_dir, basic_pipeline_config,
):
    img_dir, msk_dir = fixture_dir
    ds_norm = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64, imagenet_norm=True,
    )
    ds_raw = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64, imagenet_norm=False,
    )
    s_norm = ds_norm[0]
    s_raw = ds_raw[0]
    # Raw mode: divided by 255, range [0, 1].
    assert float(s_raw["image"].min()) >= 0.0
    assert float(s_raw["image"].max()) <= 1.0
    # ImageNet-norm mode: shifts down (image - mean), so min < 0.
    assert float(s_norm["image"].min()) < 0.0


def test_dataset_normalisation_uses_config_function(
    fixture_dir, basic_pipeline_config, monkeypatch,
):
    """The v0.5 Bug 1 lesson: train + predict must agree on
    normalisation. The dataset reads the same constant the prediction
    path reads. Verify by patching the constant + observing the
    dataset's per-pixel mean shifts accordingly.
    """
    img_dir, msk_dir = fixture_dir
    sentinel_mean = 0.7
    sentinel_std = 0.1
    monkeypatch.setattr(cfg_mod, "IMAGENET_MEAN", sentinel_mean)
    monkeypatch.setattr(cfg_mod, "IMAGENET_STD", sentinel_std)
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64, imagenet_norm=True,
    )
    s = ds[0]
    # Normalised pixel = (pixel/255 - 0.7) / 0.1. For random uint8
    # inputs the post-norm mean should sit near
    # (mean(pixel/255) - 0.7) / 0.1 ≈ -2 (random uniform mean ~ 0.5).
    assert float(s["image"].mean()) < -1.0


def test_dataset_horizontal_flip_consistent_across_targets(
    fixture_dir, four_heads_cfg,
):
    """The load-bearing augmentation parity test: a deterministic
    HorizontalFlip(p=1.0) produces images and per-head targets where
    flipping the image right-to-left also flips every target. v0.5
    Bug 1 was about a different contract drift but the lesson — that
    train-time spatial augmentation must be in lockstep across all
    heads — is the same.
    """
    img_dir, msk_dir = fixture_dir

    cfg_no_flip = PipelineConfig(
        heads=four_heads_cfg,
        augmentations=AugmentationsConfig(train_transforms=[]),
    )
    cfg_flip = PipelineConfig(
        heads=four_heads_cfg,
        augmentations=AugmentationsConfig(train_transforms=[
            TransformSpec(name="HorizontalFlip", params={"p": 1.0}),
        ]),
    )
    ds_a = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=cfg_no_flip, num_classes=3, img_size=64,
        imagenet_norm=False,
    )
    ds_b = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=cfg_flip, num_classes=3, img_size=64,
        imagenet_norm=False,
    )
    a = ds_a[0]
    b = ds_b[0]
    # Image flipped horizontally.
    np.testing.assert_array_equal(
        b["image"].numpy(),
        np.flip(a["image"].numpy(), axis=-1),
    )
    # Every head's target flipped horizontally too.
    np.testing.assert_array_equal(
        b["semantic"].numpy(),
        np.flip(a["semantic"].numpy(), axis=-1),
    )
    np.testing.assert_array_equal(
        b["boundary"].numpy(),
        np.flip(a["boundary"].numpy(), axis=-1),
    )
    np.testing.assert_array_equal(
        b["distance"].numpy(),
        np.flip(a["distance"].numpy(), axis=-1),
    )
    np.testing.assert_array_equal(
        b["sdm"].numpy(),
        np.flip(a["sdm"].numpy(), axis=-1),
    )


def test_dataset_validation_skips_augmentation(
    fixture_dir, basic_pipeline_config,
):
    img_dir, msk_dir = fixture_dir
    ds = PipelineMultiTaskDataset(
        img_dir, msk_dir,
        pipeline_config=basic_pipeline_config,
        num_classes=3, img_size=64, validation=True,
    )
    # Two reads of the same index produce the same output (no random aug).
    a = ds[0]
    b = ds[0]
    np.testing.assert_array_equal(a["image"].numpy(), b["image"].numpy())


def test_dataset_missing_dir_raises(tmp_path, basic_pipeline_config):
    with pytest.raises(FileNotFoundError, match="images_dir"):
        PipelineMultiTaskDataset(
            tmp_path / "no_such", tmp_path,
            pipeline_config=basic_pipeline_config,
            num_classes=3,
        )


def test_dataset_image_count_mismatch_raises(tmp_path, basic_pipeline_config):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    cv2.imwrite(str(img_dir / "data_0.png"), np.zeros((10, 10), dtype=np.uint8))
    # No matching mask.
    with pytest.raises(ValueError, match="image / mask count"):
        PipelineMultiTaskDataset(
            img_dir, msk_dir,
            pipeline_config=basic_pipeline_config,
            num_classes=3,
        )


#  _pipeline_mode_required 


def test_pipeline_mode_not_required_for_semantic_only():
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True)},
    )
    assert _pipeline_mode_required(cfg) is False


def test_pipeline_mode_required_for_boundary_head():
    cfg = PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True),
            "boundary": HeadConfig(enabled=True, loss="bce"),
        },
    )
    assert _pipeline_mode_required(cfg) is True


def test_pipeline_mode_required_when_instance_assembly_set():
    # uSegment3D backend forces pipeline-mode dataloader even if heads
    # are semantic-only (the assembly step needs per-axis instances
    # which need the multi-target plumbing).
    cfg = PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True),
            "distance": HeadConfig(enabled=True, loss="distance_l1"),
        },
        instance_assembly=InstanceAssemblyConfig(backend="usegment3d"),
    )
    assert _pipeline_mode_required(cfg) is True


# get_pipeline_training_dataloaders 


def test_get_pipeline_dataloaders_split(
    fixture_dir, basic_pipeline_config, monkeypatch,
):
    img_dir, msk_dir = fixture_dir
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)

    settings = SimpleNamespace(
        image_size=64, training_set_proportion=0.85, cuda_device=0,
        use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        max_label_no=3,
    )
    train_dl, val_dl = get_pipeline_training_dataloaders(
        img_dir, msk_dir, settings, basic_pipeline_config,
    )
    # 20 samples; 0.85 train -> 17 train + 3 val.
    # train_dl batch_size=4, drop_last=True -> 4 batches.
    assert len(train_dl) == 4
    # val_dl batch_size=4, no drop_last -> 1 batch.
    assert len(val_dl) == 1


def test_get_pipeline_dataloaders_yields_dict_batch(
    fixture_dir, basic_pipeline_config, monkeypatch,
):
    img_dir, msk_dir = fixture_dir
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = SimpleNamespace(
        image_size=64, training_set_proportion=0.85, cuda_device=0,
        use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        max_label_no=3,
    )
    train_dl, _ = get_pipeline_training_dataloaders(
        img_dir, msk_dir, settings, basic_pipeline_config,
    )
    batch = next(iter(train_dl))
    assert isinstance(batch, dict)
    assert {"image", "semantic", "boundary", "distance", "sdm"} <= set(batch)
    assert batch["image"].shape == (4, 1, 64, 64)
    assert batch["semantic"].shape == (4, 64, 64)


# prepare_training_batch 


def test_prepare_training_batch_routes_pipeline_dict():
    """The b3 pipeline dict shape (`'image'` + head names) takes the new
    helper path. MONAI dict (`'img'` + `'seg'`) keeps the legacy path.
    """
    batch = {
        "image": torch.zeros(2, 1, 16, 16),
        "semantic": torch.zeros(2, 16, 16, dtype=torch.int64),
        "boundary": torch.zeros(2, 1, 16, 16),
    }
    inputs, targets = prepare_training_batch(batch, device="cpu", num_labels=2)
    assert inputs.shape == (2, 1, 16, 16)
    assert set(targets) == {"semantic", "boundary"}
    assert targets["semantic"].dtype == torch.int64
    assert targets["boundary"].dtype == torch.float32


def test_prepare_training_batch_legacy_monai_dict_unchanged():
    """The legacy 'img'/'seg' dict path still routes to the v0.4 logic."""
    batch = {
        "img": torch.zeros(2, 1, 16, 16),
        "seg": torch.zeros(2, 1, 16, 16, dtype=torch.int64),
    }
    inputs, targets = prepare_training_batch(batch, device="cpu", num_labels=2)
    # Legacy path one-hot encodes seg -> (B, num_labels, H, W) uint8.
    assert "seg" in targets
    assert targets["seg"].shape == (2, 2, 16, 16)
    assert targets["seg"].dtype == torch.uint8


def test_prepare_training_batch_legacy_tuple_unchanged():
    """Legacy tuple `[inputs, targets]` path still works"""
    batch = [
        torch.zeros(2, 1, 16, 16),
        torch.zeros(2, 16, 16, dtype=torch.int64),
    ]
    inputs, targets = prepare_training_batch(batch, device="cpu", num_labels=3)
    assert inputs.shape == (2, 1, 16, 16)
    # Legacy path -> (B, num_labels, H, W) uint8 one-hot.
    assert targets.shape == (2, 3, 16, 16)
    assert targets.dtype == torch.uint8
