"""The default trainer applies augmentations by default.

Before this change, the default trainer applied only resize and pad when
there was no ``pipeline.yaml``, or when the file gave no
``augmentations.train_transforms``. Also, the packaged default
``GaussNoise var_limit: [10, 50]`` was approximately ten times too strong
with Albumentations releases that have ``std_range``.
"""
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from volume_segmantics.data.pipeline_dataset import _spec_to_albumentations
from volume_segmantics.data.pipeline_loader import (
    TransformSpec,
    default_train_transforms,
    legacy_settings_to_pipeline_config,
    load_pipeline_yaml,
    parse_pipeline_dict,
)

REPO = Path(__file__).resolve().parents[1]
SEMANTIC_ONLY = {"heads": {"semantic": {"enabled": True}}}


def test_the_packaged_default_has_training_augmentations():
    names = [spec.name for spec in default_train_transforms()]
    assert "HorizontalFlip" in names and "GaussNoise" in names


def test_a_pipeline_without_an_augmentations_block_keeps_the_default_list():
    default = [spec.name for spec in default_train_transforms()]
    for data in (SEMANTIC_ONLY, {**SEMANTIC_ONLY, "augmentations": {"backend": "albumentations"}}):
        cfg = parse_pipeline_dict(data)
        assert [spec.name for spec in cfg.augmentations.train_transforms] == default


def test_an_explicit_empty_list_switches_augmentation_off():
    cfg = parse_pipeline_dict({**SEMANTIC_ONLY, "augmentations": {"train_transforms": []}})
    assert cfg.augmentations.train_transforms == []


def test_a_stated_list_replaces_the_default():
    cfg = parse_pipeline_dict({
        **SEMANTIC_ONLY,
        "augmentations": {"train_transforms": [{"name": "VerticalFlip", "p": 1.0}]},
    })
    assert [spec.name for spec in cfg.augmentations.train_transforms] == ["VerticalFlip"]


def test_training_without_a_pipeline_yaml_is_augmented():
    cfg = legacy_settings_to_pipeline_config(SimpleNamespace(loss_criterion="CombinedCEDiceLoss"))
    assert [s.name for s in cfg.augmentations.train_transforms] == [s.name for s in default_train_transforms()]


def test_the_shipped_multitask_example_trains_with_augmentation():
    cfg = load_pipeline_yaml(REPO / "volseg-settings" / "pipeline_multitask_example.yaml")
    assert cfg.augmentations.train_transforms

# Noise strength


def test_default_gauss_noise_is_a_few_grey_levels_under_the_installed_albumentations():
    spec = TransformSpec("GaussNoise", {"var_limit": [10.0, 50.0], "p": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        transform = _spec_to_albumentations(spec)
    image = np.full((256, 256, 1), 128, dtype=np.uint8)
    sigmas = [float((transform(image=image)["image"].astype(float) - 128).std()) for _ in range(10)]
    # sqrt(10) .. sqrt(50) is 3.2 .. 7.1 grey levels.
    assert 2.5 < min(sigmas) and max(sigmas) < 8.0, sigmas
