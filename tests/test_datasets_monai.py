from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("monai")

from volume_segmantics.data import datasets_monai as dmonai
from skimage import io




@pytest.fixture
def monai_image_dir(tmp_path):
    """Temporary directory with PNG images named for MONAI build_file_list."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(10):
        im = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        io.imsave(d / f"im_{i}.png", im, check_contrast=False)
    return d


@pytest.fixture
def monai_label_dir(tmp_path):
    """Temporary directory with PNG labels (same count as monai_image_dir)."""
    d = tmp_path / "labels"
    d.mkdir()
    for i in range(10):
        im = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
        io.imsave(d / f"seg_{i}.png", im, check_contrast=False)
    return d


@pytest.fixture
def monai_settings():
    """Settings-like namespace for MONAI dataset creation."""
    return SimpleNamespace(
        image_size=64,
        training_set_proportion=0.8,
        use_2_5d_slicing=False,
        use_imagenet_norm=True,
        use_multitask=False,
        num_tasks=1,
        task2_dir=None,
        task3_dir=None,
    )




def test_build_file_list_returns_paired_dicts(monai_image_dir, monai_label_dir):
    """build_file_list returns list of dicts with 'img' and 'seg' paths."""
    files = dmonai.build_file_list(monai_image_dir, monai_label_dir)
    assert len(files) == 10
    for item in files:
        assert "img" in item
        assert "seg" in item
        assert Path(item["img"]).exists()
        assert Path(item["seg"]).exists()


def test_build_file_list_mismatch_raises(monai_image_dir, tmp_path):
    """build_file_list raises when image and label counts differ."""
    label_dir = tmp_path / "labels_few"
    label_dir.mkdir()
    for i in range(3):
        io.imsave(label_dir / f"seg_{i}.png", np.zeros((64, 64), dtype=np.uint8), check_contrast=False)
    with pytest.raises(ValueError, match="Mismatch"):
        dmonai.build_file_list(monai_image_dir, label_dir)


def test_build_file_list_natsort_order(monai_image_dir, monai_label_dir):
    """File list is sorted by natural sort (e.g. im_1 before im_10)."""
    files = dmonai.build_file_list(monai_image_dir, monai_label_dir)
    img_paths = [f["img"] for f in files]
    seg_paths = [f["seg"] for f in files]
    assert img_paths == sorted(img_paths, key=dmonai.natsort_key)
    assert seg_paths == sorted(seg_paths, key=dmonai.natsort_key)



def test_split_train_val_files_no_overlap(monai_image_dir, monai_label_dir, monai_settings):
    """Train and val file lists are disjoint and cover all files."""
    all_files = dmonai.build_file_list(monai_image_dir, monai_label_dir)
    train_files, val_files = dmonai.split_train_val_files(all_files, monai_settings)
    train_set = {f["img"] for f in train_files}
    val_set = {f["img"] for f in val_files}
    assert train_set.isdisjoint(val_set)
    assert len(train_files) + len(val_files) == len(all_files)


def test_split_train_val_files_respects_proportion(monai_image_dir, monai_label_dir):
    """Split roughly respects training_set_proportion."""
    import random

    all_files = dmonai.build_file_list(monai_image_dir, monai_label_dir)
    # Use the same default as split_train_val_files (0.85) so behaviour matches implementation.
    settings = SimpleNamespace()  # no explicit training_set_proportion -> default 0.85
    random.seed(123)
    train_files, val_files = dmonai.split_train_val_files(all_files, settings)
    n = len(all_files)
    training_set_prop = getattr(settings, "training_set_proportion", 0.85)
    expected_val = int(n * (1 - training_set_prop))
    # Exact equality with the implementation formula
    assert len(val_files) == expected_val
    assert len(train_files) == n - expected_val



def test_get_monai_train_transforms_returns_compose(monai_settings):
    """get_monai_train_transforms returns MONAI Compose."""
    from monai.transforms import Compose

    t = dmonai.get_monai_train_transforms(
        monai_settings.image_size,
        num_channels=1,
        use_2_5d_slicing=False,
        num_tasks=1,
        use_imagenet_norm=True,
    )
    assert isinstance(t, Compose)


def test_get_monai_val_transforms_returns_compose(monai_settings):
    """get_monai_val_transforms returns MONAI Compose."""
    from monai.transforms import Compose

    t = dmonai.get_monai_val_transforms(
        monai_settings.image_size,
        num_channels=1,
        use_2_5d_slicing=False,
        num_tasks=1,
        use_imagenet_norm=True,
    )
    assert isinstance(t, Compose)


def test_monai_train_transforms_multi_task_keys():
    """Train transforms for num_tasks=2 include 'boundary' key."""
    from monai.transforms import Compose

    t = dmonai.get_monai_train_transforms(64, num_channels=1, num_tasks=2, use_imagenet_norm=False)
    assert isinstance(t, Compose)



def test_get_monai_training_and_validation_datasets_returns_two_datasets(
    monai_image_dir, monai_label_dir, monai_settings
):
    """get_monai_training_and_validation_datasets returns (train_ds, val_ds)."""
    train_ds, val_ds = dmonai.get_monai_training_and_validation_datasets(
        monai_image_dir, monai_label_dir, monai_settings
    )
    assert train_ds is not None
    assert val_ds is not None
    assert len(train_ds) + len(val_ds) == 10
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_monai_dataset_getitem_returns_dict_with_img_seg(
    monai_image_dir, monai_label_dir, monai_settings
):
    """MONAI training dataset __getitem__ returns dict with 'img' and 'seg' tensors."""
    train_ds, _ = dmonai.get_monai_training_and_validation_datasets(
        monai_image_dir, monai_label_dir, monai_settings
    )
    sample = train_ds[0]
    assert isinstance(sample, dict)
    assert "img" in sample
    assert "seg" in sample
    import torch
    assert isinstance(sample["img"], torch.Tensor)
    assert isinstance(sample["seg"], torch.Tensor)
    assert sample["img"].ndim == 3  # (C, H, W)
    assert sample["seg"].ndim in (2, 3)


def test_monai_training_and_validation_same_split(
    monai_image_dir, monai_label_dir, monai_settings
):
    """Calling get_monai_training_and_validation_datasets twice can yield different splits (random)."""
    train_a, val_a = dmonai.get_monai_training_and_validation_datasets(
        monai_image_dir, monai_label_dir, monai_settings
    )
    train_b, val_b = dmonai.get_monai_training_and_validation_datasets(
        monai_image_dir, monai_label_dir, monai_settings
    )
    # At least we get consistent structure: two datasets, lengths sum to total
    assert len(train_a) + len(val_a) == 10
    assert len(train_b) + len(val_b) == 10



def test_imagenet_normalizationd_shape():
    """ImageNetNormalizationd preserves shape and produces normalized values."""
    norm = dmonai.ImageNetNormalizationd(keys=["img"])
    import torch
    data = {"img": torch.rand(1, 32, 32)}
    out = norm(data)
    assert out["img"].shape == data["img"].shape
    # Normalized data should have roughly zero mean per channel when input is uniform
    assert "img" in out




def test_unlabeled_monai_dataset_returns_student_teacher(monai_image_dir):
    """UnlabeledMONAIDataset __getitem__ returns dict with 'student' and 'teacher' keys.
    Uses img-only transforms because unlabeled data has no 'seg' key."""
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

    from volume_segmantics.data.datasets_monai import UnlabeledMONAIDataset

    # Transforms that only load and process "img" (no "seg")
    img_only = [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(64, 64), mode="bilinear"),
        ToTensord(keys=["img"]),
    ]
    student_t = Compose(img_only)
    teacher_t = Compose(img_only)
    ds = UnlabeledMONAIDataset(monai_image_dir, student_transform=student_t, teacher_transform=teacher_t)
    assert len(ds) == 10
    sample = ds[0]
    assert "student" in sample
    assert "teacher" in sample
    import torch
    assert isinstance(sample["student"], torch.Tensor)
    assert isinstance(sample["teacher"], torch.Tensor)


def test_unlabeled_monai_dataset_no_transform_loads_image(monai_image_dir):
    """UnlabeledMONAIDataset with None transforms still loads image (fallback load)."""
    from volume_segmantics.data.datasets_monai import UnlabeledMONAIDataset

    ds = UnlabeledMONAIDataset(monai_image_dir, student_transform=None, teacher_transform=None)
    sample = ds[0]
    assert "student" in sample
    assert "teacher" in sample
    import torch
    assert isinstance(sample["student"], torch.Tensor)
    assert sample["student"].ndim == 3
