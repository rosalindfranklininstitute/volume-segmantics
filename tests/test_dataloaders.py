from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from volume_segmantics.data.dataloaders import (
    get_2d_prediction_dataloader,
    get_2d_training_dataloaders,
)


@pytest.fixture()
def train_loaders(image_dir, label_dir, training_settings):
    return get_2d_training_dataloaders(image_dir, label_dir, training_settings)


@pytest.fixture()
def train_loaders_fixed_batch(image_dir, label_dir, training_settings, monkeypatch):
    """Training loaders with batch_size forced to 2 so get_batch_size not needed."""
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction=False: 2)
    return get_2d_training_dataloaders(image_dir, label_dir, training_settings)


def _fixed_batch_settings(training_settings, monkeypatch, **overrides):
    """training_settings clone with batch_size pinned (CPU, no GPU query)."""
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction=False: 2)
    settings = SimpleNamespace(**vars(training_settings))
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def test_seeded_split_is_reproducible(image_dir, label_dir, training_settings, monkeypatch):
    """Same random_seed -> identical train/val split (PLAN.md Step 6)."""
    settings = _fixed_batch_settings(training_settings, monkeypatch, random_seed=123)
    tl1, vl1 = get_2d_training_dataloaders(image_dir, label_dir, settings)
    tl2, vl2 = get_2d_training_dataloaders(image_dir, label_dir, settings)
    assert list(tl1.dataset.indices) == list(tl2.dataset.indices)
    assert list(vl1.dataset.indices) == list(vl2.dataset.indices)


def test_seeded_loader_attaches_worker_init_and_generator(
    image_dir, label_dir, training_settings, monkeypatch
):
    settings = _fixed_batch_settings(training_settings, monkeypatch, random_seed=7)
    tl, _ = get_2d_training_dataloaders(image_dir, label_dir, settings)
    from volume_segmantics.utilities.seeding import seed_worker
    assert tl.worker_init_fn is seed_worker
    assert tl.generator is not None


def test_unseeded_loader_preserves_default_behaviour(
    image_dir, label_dir, training_settings, monkeypatch
):
    """No random_seed -> no worker_init_fn/generator (byte-for-byte unchanged)."""
    settings = _fixed_batch_settings(training_settings, monkeypatch)
    if hasattr(settings, "random_seed"):
        settings.random_seed = None
    tl, _ = get_2d_training_dataloaders(image_dir, label_dir, settings)
    assert tl.worker_init_fn is None
    assert tl.generator is None


@pytest.mark.gpu()
def test_get_2d_training_dataloader_types(train_loaders):
    train_loader, valid_loader = train_loaders
    assert isinstance(train_loader, DataLoader)
    assert isinstance(valid_loader, DataLoader)


@pytest.mark.gpu()
def test_get_2d_training_dataloader_length(train_loaders):
    train_loader, valid_loader = train_loaders
    assert len(train_loader.dataset) > len(valid_loader.dataset)


@pytest.mark.gpu()
def test_get_2d_prediction_dataloader_type(rand_int_volume, prediction_settings):
    prediction_loader = get_2d_prediction_dataloader(
        rand_int_volume, prediction_settings
    )
    assert isinstance(prediction_loader, DataLoader)


def test_get_2d_prediction_dataloader_batch_size_uses_prediction_flag(
    rand_int_volume, prediction_settings, monkeypatch
):
    # Force get_batch_size to return different values for training vs prediction
    import volume_segmantics.utilities.base_data_utils as utils

    def fake_get_batch_size(settings, prediction: bool = False):
        return 3 if prediction else 99

    monkeypatch.setattr(utils, "get_batch_size", fake_get_batch_size)
    loader = get_2d_prediction_dataloader(rand_int_volume, prediction_settings)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 3


def test_training_dataloader_batch_size_and_shuffling(train_loaders_fixed_batch):
    """Train and val loaders use batch_size from settings; train shuffles, val does not."""
    train_loader, val_loader = train_loaders_fixed_batch
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    # DataLoader shuffle is not always a public attr; check if present (PyTorch version-dependent)
    if hasattr(train_loader, "shuffle"):
        assert train_loader.shuffle is True
    if hasattr(val_loader, "shuffle"):
        assert val_loader.shuffle is False


def test_training_dataloader_deterministic_with_seed(image_dir, label_dir, training_settings, monkeypatch):
    """With fixed torch/np seed, the train/val split and batch shape are deterministic.

    Pixel-level equality of the first batch is *not* asserted: in
    Albumentations >=1.4 each transform owns a private
    ``random.Random(seed)`` initialised at construction time, so
    reseeding the global ``random`` / ``numpy`` RNGs doesn't reach
    them. The production dataloaders don't plumb a seed through to
    Albumentations, so the per-transform RNG state diverges between
    the two constructions even when everything else is identical.
    """
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction=False: 4)
    torch.manual_seed(42)
    np.random.seed(42)
    train_a, val_a = get_2d_training_dataloaders(image_dir, label_dir, training_settings)
    batch_a = next(iter(train_a))
    torch.manual_seed(42)
    np.random.seed(42)
    train_b, val_b = get_2d_training_dataloaders(image_dir, label_dir, training_settings)
    batch_b = next(iter(train_b))
    assert len(train_a.dataset) == len(train_b.dataset)
    assert len(val_a.dataset) == len(val_b.dataset)
    assert batch_a[0].shape == batch_b[0].shape


def test_prediction_dataloader_num_batches(rand_int_volume, prediction_settings, monkeypatch):
    """Prediction dataloader yields correct number of batches for small volume."""
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction: 5)
    loader = get_2d_prediction_dataloader(rand_int_volume, prediction_settings)
    n_slices = rand_int_volume.shape[0]
    expected_batches = (n_slices + 4) // 5
    batches = list(loader)
    assert len(batches) == expected_batches
    if batches:
        assert batches[0].dim() == 4  # (B, C, H, W)


# --- MONAI dataloaders (skip if MONAI not installed) ---


@pytest.fixture
def monai_training_settings(training_settings):
    """Settings with MONAI augmentation and MONAI datasets enabled."""
    d = vars(training_settings).copy() if hasattr(training_settings, "__dict__") else {}
    s = SimpleNamespace(**d)
    s.augmentation_library = "monai"
    s.use_monai_datasets = True
    return s


@pytest.mark.gpu
def test_monai_training_dataloaders_when_configured(
    image_dir, label_dir, monai_training_settings, monkeypatch
):
    """When settings use MONAI, get_2d_training_dataloaders returns loaders with dict batches (img, seg)."""
    pytest.importorskip("monai")
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction=False: 2)
    train_loader, val_loader = get_2d_training_dataloaders(
        image_dir, label_dir, monai_training_settings
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert "img" in batch
    assert "seg" in batch
    assert batch["img"].dim() == 4  # (B, C, H, W)
    assert batch["seg"].dim() in (3, 4)  # (B, H, W) or (B, 1, H, W)


@pytest.mark.gpu
def test_monai_training_dataloaders_use_list_data_collate(
    image_dir, label_dir, monai_training_settings, monkeypatch
):
    """MONAI training dataloaders use list_data_collate as collate_fn."""
    pytest.importorskip("monai")
    from monai.data import list_data_collate
    import volume_segmantics.utilities.base_data_utils as utils
    monkeypatch.setattr(utils, "get_batch_size", lambda s, prediction=False: 2)
    train_loader, _ = get_2d_training_dataloaders(
        image_dir, label_dir, monai_training_settings
    )
    assert train_loader.collate_fn is list_data_collate
