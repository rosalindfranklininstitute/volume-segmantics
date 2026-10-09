"""Tests for the optional MONAI augmentation path in the pipeline dataset.

The pipeline/Lightning dataset is albumentations-based by default; setting
``augmentation_library: monai`` swaps the training spatial augmentation for a
MONAI multi-key Compose (behind the same call interface). These tests cover
the builder branch, the adapter's layout/consistency behaviour, graceful
fallback, and an end-to-end ``PipelineMultiTaskDataset.__getitem__``.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

# Trigger head / target-generator registration.
import volume_segmantics.data.targets  # noqa: F401

import albumentations as A
from volume_segmantics.data.pipeline_dataset import (
    PipelineMultiTaskDataset,
    UnlabeledPipelineDataset,
    _MonaiPipelineAug,
    build_pipeline_augmentations,
    resolve_augmentation_backend,
)
from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    HeadConfig,
    PipelineConfig,
    TransformSpec,
)

pytest.importorskip("monai", reason="MONAI augmentation path requires monai")

from volume_segmantics.data.augmentations_monai import (  # noqa: E402
    _spec_to_monai,
    build_pipeline_monai_transforms,
)

IMG = 64
HEADS = {"semantic": True, "boundary": True, "distance": True}


@pytest.fixture
def aug_cfg():
    # Empty list exercises the built-in default MONAI recipe.
    return AugmentationsConfig(train_transforms=[])


class TestBuilderBranch:
    def test_monai_returns_adapter_and_no_photometric(self, aug_cfg):
        spatial, photo, addt = build_pipeline_augmentations(
            aug_cfg, enabled_head_names=HEADS, img_size=IMG,
            augmentation_library="monai",
        )
        assert isinstance(spatial, _MonaiPipelineAug)
        assert photo is None  # intensity augs are folded into the MONAI compose
        assert addt == {"boundary": "mask", "distance": "image"}

    def test_default_is_albumentations(self, aug_cfg):
        spatial, _, _ = build_pipeline_augmentations(
            aug_cfg, enabled_head_names=HEADS, img_size=IMG,
        )
        assert isinstance(spatial, A.Compose)
        assert not isinstance(spatial, _MonaiPipelineAug)

    def test_falls_back_when_img_size_unset(self, aug_cfg):
        spatial, _, _ = build_pipeline_augmentations(
            aug_cfg, enabled_head_names=HEADS, img_size=None,
            augmentation_library="monai",
        )
        assert not isinstance(spatial, _MonaiPipelineAug)


class TestAdapterBehaviour:
    @pytest.fixture
    def adapter(self, aug_cfg):
        spatial, _, _ = build_pipeline_augmentations(
            aug_cfg, enabled_head_names=HEADS, img_size=IMG,
            augmentation_library="monai",
        )
        return spatial

    @pytest.fixture
    def sample(self):
        H, W = 40, 48  # non-square -> exercises resize + pad to square
        rng = np.random.default_rng(0)
        return {
            "image": rng.integers(0, 256, size=(H, W), dtype=np.uint8),
            "mask": rng.integers(0, 4, size=(H, W)).astype(np.int64),
            "boundary": (rng.random((H, W)) > 0.5).astype(np.float32),
            "distance": (rng.random((H, W)) * 5.0).astype(np.float32),
        }

    def test_outputs_are_square_and_keys_preserved(self, adapter, sample):
        out = adapter(**sample)
        for k in ("image", "mask", "boundary", "distance"):
            assert out[k].shape[:2] == (IMG, IMG)

    def test_image_scaled_to_unit_range(self, adapter, sample):
        out = adapter(**sample)
        assert out["image"].min() >= 0.0
        assert out["image"].max() <= 1.0

    def test_nearest_mode_preserves_labels(self, adapter, sample):
        out = adapter(**sample)
        # Semantic mask uses nearest interp -> no new fractional label values.
        mask_vals = set(np.unique(np.rint(out["mask"]).astype(int)).tolist())
        assert mask_vals.issubset({0, 1, 2, 3})
        # Boundary uses nearest interp -> stays binary.
        bnd_vals = set(np.unique(np.rint(out["boundary"]).astype(int)).tolist())
        assert bnd_vals.issubset({0, 1})

    def test_geometric_consistency_across_keys(self, adapter, sample):
        """The same geometric transform must be applied to image and mask."""
        marker_img = np.zeros((40, 48), np.uint8)
        marker_img[:6, :6] = 255
        marker_msk = np.zeros((40, 48), np.int64)
        marker_msk[:6, :6] = 3
        out = adapter(
            image=marker_img, mask=marker_msk,
            boundary=sample["boundary"], distance=sample["distance"],
        )
        img_hot = out["image"] > 0.5
        msk_hot = np.rint(out["mask"]).astype(int) == 3
        union = np.logical_or(img_hot, msk_hot).sum()
        inter = np.logical_and(img_hot, msk_hot).sum()
        assert union > 0 and inter / union > 0.5


class TestSpecTranslator:
    """pipeline.yaml transform specs are translated to MONAI equivalents."""

    KEYS = ["image", "mask", "boundary"]
    MODES = ["bilinear", "nearest", "nearest"]

    @pytest.mark.parametrize(
        "name,params",
        [
            ("HorizontalFlip", {"p": 1.0}),
            ("VerticalFlip", {"p": 1.0}),
            ("Flip", {"p": 1.0}),
            ("RandomRotate90", {"p": 1.0}),
            ("Transpose", {"p": 1.0}),
            ("Rotate", {"limit": 30, "p": 1.0}),
            ("ShiftScaleRotate", {"shift_limit": 0.1, "scale_limit": 0.2,
                                  "rotate_limit": 15, "p": 1.0}),
            ("ElasticTransform", {"p": 1.0}),
            ("GridDistortion", {"distort_limit": 0.2, "p": 1.0}),
            ("RandomBrightnessContrast", {"brightness_limit": 0.2,
                                          "contrast_limit": 0.2, "p": 1.0}),
            ("RandomGamma", {"gamma_limit": [80, 120], "p": 1.0}),
            ("GaussNoise", {"var_limit": [10.0, 50.0], "p": 1.0}),
            ("GaussianBlur", {"p": 1.0}),
        ],
    )
    def test_known_transforms_map(self, name, params):
        t = _spec_to_monai(
            TransformSpec(name=name, params=params),
            keys=self.KEYS, modes=self.MODES, image_only=["image"], img_size=IMG,
        )
        assert t is not None

    def test_unknown_transform_returns_none(self):
        t = _spec_to_monai(
            TransformSpec(name="FancyPCA", params={}),
            keys=self.KEYS, modes=self.MODES, image_only=["image"], img_size=IMG,
        )
        assert t is None

    def test_shipped_default_config_fully_translates(self):
        """Every transform in the shipped pipeline.default.yaml maps to MONAI."""
        from volume_segmantics.data.pipeline_loader import load_pipeline_yaml
        from volume_segmantics.settings import PIPELINE_DEFAULT_PATH

        cfg = load_pipeline_yaml(PIPELINE_DEFAULT_PATH)
        for spec in cfg.augmentations.train_transforms:
            assert _spec_to_monai(
                spec, keys=self.KEYS, modes=self.MODES,
                image_only=["image"], img_size=IMG,
            ) is not None, spec.name

    def test_config_specs_drive_the_compose(self):
        """Only the configured (translatable) transforms end up in the compose."""
        specs = [
            TransformSpec(name="HorizontalFlip", params={"p": 0.5}),
            TransformSpec(name="FancyPCA", params={}),  # unsupported -> skipped
        ]
        comp = build_pipeline_monai_transforms(
            IMG, self.KEYS, self.MODES, train_transforms=specs,
        )
        names = [type(t).__name__ for t in comp.transforms]
        # Sizing + the one supported transform; the unknown is dropped.
        assert names == ["Resized", "SpatialPadd", "RandFlipd"], names

    def test_empty_list_uses_default_recipe(self):
        comp = build_pipeline_monai_transforms(
            IMG, self.KEYS, self.MODES, train_transforms=[],
        )
        names = [type(t).__name__ for t in comp.transforms]
        assert "RandFlipd" in names and "OneOf" in names


class TestMonaiNativePassthrough:
    """MONAI-only transforms (no Albumentations equivalent) resolve natively."""

    KEYS = ["image", "mask", "boundary"]
    MODES = ["bilinear", "nearest", "nearest"]

    def _translate(self, name, **params):
        return _spec_to_monai(
            TransformSpec(name=name, params=params),
            keys=self.KEYS, modes=self.MODES, image_only=["image"], img_size=IMG,
        )

    @pytest.mark.parametrize(
        "name",
        ["RandBiasField", "RandGibbsNoise", "RandHistogramShift",
         "RandKSpaceSpikeNoise", "RandRicianNoise"],
    )
    def test_monai_only_intensity_resolves_image_only(self, name):
        t = self._translate(name, p=1.0)
        assert t is not None
        # dict variant resolved despite base names that end in "d".
        assert type(t).__name__ == name + "d"
        # image-only by default (targets untouched).
        assert list(t.keys) == ["image"]

    def test_dict_suffixed_name_and_params_pass_through(self):
        t = self._translate("RandBiasFieldd", p=1.0, coeff_range=(0.0, 0.2), degree=3)
        assert type(t).__name__ == "RandBiasFieldd"

    def test_spatial_flag_applies_across_all_keys(self):
        t = self._translate("RandZoom", p=1.0, spatial=True,
                            min_zoom=0.9, max_zoom=1.1)
        assert list(t.keys) == self.KEYS

    def test_truly_unknown_returns_none(self):
        assert self._translate("TotallyFakeTransform", p=1.0) is None

    def test_bad_params_are_caught_and_skipped(self):
        # Real transform, invalid kwarg -> construction fails -> None (skipped).
        assert self._translate("RandGibbsNoise", p=1.0, nonexistent_kw=5) is None

    def test_end_to_end_native_in_dataset_builder(self):
        cfg = AugmentationsConfig(train_transforms=[
            TransformSpec(name="HorizontalFlip", params={"p": 1.0}),
            TransformSpec(name="RandBiasField", params={"p": 1.0}),
            TransformSpec(name="RandGibbsNoise", params={"p": 1.0, "alpha": (0.0, 0.5)}),
        ])
        spatial, photo, _ = build_pipeline_augmentations(
            cfg, enabled_head_names={"semantic": True, "boundary": True},
            img_size=IMG, augmentation_library="monai",
        )
        names = [type(t).__name__ for t in spatial._compose.transforms]
        assert "RandBiasFieldd" in names
        assert "RandGibbsNoised" in names
        assert "RandFlipd" in names  # albumentations-mapped alongside native


class TestEndToEndDataset:
    """PipelineMultiTaskDataset with monai augmentation yields valid batches."""

    def _write_fixture(self, tmp_path):
        img_dir = tmp_path / "data"
        msk_dir = tmp_path / "seg"
        img_dir.mkdir()
        msk_dir.mkdir()
        rng = np.random.default_rng(1)
        for i in range(4):
            cv2.imwrite(
                str(img_dir / f"data_{i:03d}.png"),
                (rng.random((50, 60)) * 255).astype(np.uint8),
            )
            msk = np.zeros((50, 60), dtype=np.uint8)
            msk[16:40, 16:40] = 1
            msk[24:32, 24:32] = 2
            cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
        return img_dir, msk_dir

    def test_getitem_shapes_with_monai(self, tmp_path):
        img_dir, msk_dir = self._write_fixture(tmp_path)
        pipeline_config = PipelineConfig(
            heads={
                "semantic": HeadConfig(enabled=True, loss="dice_ce"),
                "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
                "distance": HeadConfig(enabled=True, loss="distance_l1"),
            },
            # Real config transforms -> exercises the spec->MONAI translator
            # through the full dataset path.
            augmentations=AugmentationsConfig(
                train_transforms=[
                    TransformSpec(name="HorizontalFlip", params={"p": 0.5}),
                    TransformSpec(name="RandomRotate90", params={"p": 0.5}),
                    TransformSpec(
                        name="RandomBrightnessContrast",
                        params={"brightness_limit": 0.2, "contrast_limit": 0.2,
                                "p": 0.3},
                    ),
                ],
            ),
        )
        settings = SimpleNamespace(augmentation_library="monai")
        dset = PipelineMultiTaskDataset(
            img_dir, msk_dir,
            pipeline_config=pipeline_config,
            num_classes=3,
            settings=settings,
            validation=False,
            img_size=IMG,
            imagenet_norm=True,
        )
        # Building the dataset must have selected the MONAI adapter.
        assert isinstance(dset.spatial_aug, _MonaiPipelineAug)
        assert dset.photometric_aug is None

        item = dset[0]
        assert item["image"].shape == (1, IMG, IMG)
        assert item["semantic"].shape == (IMG, IMG)
        assert item["semantic"].dtype.is_floating_point is False  # long labels
        assert item["boundary"].shape == (1, IMG, IMG)
        assert item["distance"].shape == (1, IMG, IMG)
        assert bool(np.isfinite(item["image"].numpy()).all())


class TestBackendResolution:
    """pipeline.yaml's augmentations.backend wins; auto follows the train settings."""

    def test_explicit_backend_beats_the_settings(self):
        assert resolve_augmentation_backend(AugmentationsConfig(backend="monai"), SimpleNamespace(augmentation_library="albumentations")) == "monai"
        assert resolve_augmentation_backend(AugmentationsConfig(backend="albumentations"), SimpleNamespace(augmentation_library="monai")) == "albumentations"

    def test_auto_follows_the_settings_and_defaults_to_albumentations(self):
        assert resolve_augmentation_backend(AugmentationsConfig(), SimpleNamespace(augmentation_library="monai")) == "monai"
        assert resolve_augmentation_backend(AugmentationsConfig(), SimpleNamespace()) == "albumentations"
        assert resolve_augmentation_backend(AugmentationsConfig(), None) == "albumentations"

    def test_the_dataset_honours_backend_monai_from_pipeline_yaml(self, tmp_path):
        img_dir, msk_dir = TestEndToEndDataset()._write_fixture(tmp_path)
        pipeline_config = PipelineConfig(
            heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
            augmentations=AugmentationsConfig(backend="monai", train_transforms=[
                TransformSpec(name="HorizontalFlip", params={"p": 0.5}),
            ]),
        )
        dset = PipelineMultiTaskDataset(
            img_dir, msk_dir, pipeline_config=pipeline_config, num_classes=3,
            settings=SimpleNamespace(augmentation_library="albumentations"),
            validation=False, img_size=IMG,
        )
        assert isinstance(dset.spatial_aug, _MonaiPipelineAug)
        item = dset[0]
        assert item["image"].shape == (1, IMG, IMG) and item["semantic"].shape == (IMG, IMG)


class TestUnlabeledDatasetWithMonai:
    def test_two_views_under_the_monai_backend(self, tmp_path):
        unl = tmp_path / "unl"
        unl.mkdir()
        rng = np.random.default_rng(2)
        for i in range(3):
            cv2.imwrite(str(unl / f"u_{i:03d}.png"), (rng.random((50, 60)) * 255).astype(np.uint8))
        pipeline_config = PipelineConfig(
            heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
            augmentations=AugmentationsConfig(backend="monai", train_transforms=[
                TransformSpec(name="HorizontalFlip", params={"p": 0.5}),
                TransformSpec(name="RandomBrightnessContrast", params={"p": 0.5}),
            ]),
        )
        dset = UnlabeledPipelineDataset(unl, pipeline_config=pipeline_config, img_size=IMG)
        assert isinstance(dset.spatial_aug, _MonaiPipelineAug) and dset.photometric_aug is None
        item = dset[0]
        assert item["image"].shape == (1, IMG, IMG) and item["image_teacher"].shape == (1, IMG, IMG)
        assert bool(np.isfinite(item["image"].numpy()).all())
