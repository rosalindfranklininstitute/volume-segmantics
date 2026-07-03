"""Pipeline-mode dataset and augmentation builder.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import albumentations as A
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset

import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import pipeline_registry as _registry
from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    HeadConfig,
    KNOWN_HEAD_NAMES,
    PipelineConfig,
    TransformSpec,
)
# Triggers target-generator registration. Importing for side effect.
import volume_segmantics.data.targets  # noqa: F401


logger = logging.getLogger(__name__)



PHOTOMETRIC_TRANSFORMS: frozenset = frozenset({
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomBrightness",
    "RandomContrast",
    "GaussNoise",
    "GaussianBlur",
    "Blur",
    "MotionBlur",
    "MedianBlur",
    "Sharpen",
    "CLAHE",
    "ColorJitter",
    "ToGray",
    "ChannelShuffle",
    "InvertImg",
    "Solarize",
    "Posterize",
    "Equalize",
    "FancyPCA",
    "HueSaturationValue",
})


def _is_photometric(name: str) -> bool:
    return name in PHOTOMETRIC_TRANSFORMS


#  Target computation helper 


def compute_pipeline_targets(
    label_slice: np.ndarray,
    *,
    heads_cfg: Mapping[str, HeadConfig],
    num_classes: int,
    skip_semantic: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute per-head targets from a 2D label slice.

    Parameters
    ----------
    label_slice
        ``(H, W)`` integer label array (background = 0).
    heads_cfg
        Mapping from head name to :class:`HeadConfig`. Only heads with
        ``enabled=True`` get a target.
    num_classes
        Number of classes including background. Forwarded to target
        generators that need it (per-class SDM).
    skip_semantic
        When ``True`` (default) the semantic head's target — which is
        the label slice itself — is omitted from the result. The
        dataset adds it back under the ``"semantic"`` key directly.
        Set to ``False`` to include it as a copy of the input array.

    Returns
    -------
    dict
        ``{head_name: ndarray}`` for every enabled head. Shapes:

        * ``boundary``: ``(H, W) float32`` in ``{0, 1}``.
        * ``distance``: ``(H, W) float32`` non-negative.
        * ``sdm``: ``(H, W) float32`` (binary variant) or
          ``(K, H, W) float32`` (per_class with K = num_classes - 1).
        * ``semantic`` (only when ``skip_semantic=False``):
          ``(H, W)`` integer indices.
    """
    if label_slice.ndim != 2:
        raise ValueError(
            f"compute_pipeline_targets expects a 2D label slice; got "
            f"shape {label_slice.shape}"
        )
    out: Dict[str, np.ndarray] = {}
    for head_name, head_cfg in heads_cfg.items():
        if not head_cfg.enabled:
            continue
        if head_name == "semantic":
            if not skip_semantic:
                out["semantic"] = label_slice.astype(np.int64)
            continue
        if head_name not in KNOWN_HEAD_NAMES:
            raise ValueError(
                f"compute_pipeline_targets: unknown head {head_name!r}"
            )
        # Forward num_classes + the head's extra knobs to the generator.
        gen_kwargs: Dict[str, Any] = dict(num_classes=num_classes)
        gen_kwargs.update(head_cfg.extra)
        gen = _registry.build_target_generator(head_name, **gen_kwargs)
        out[head_name] = gen(label_slice)
    return out


#  Augmentation builder 


def _spec_to_albumentations(spec: TransformSpec) -> A.BasicTransform:
    """Resolve a :class:`TransformSpec` to a concrete Albumentations object.

    pipeline version takes a pragmatic stance: any name available as an attribute of
    :mod:`albumentations` is accepted, instantiated with the spec's
    ``params``. The Albumentations name space is large but stable;
    explicit registration would just duplicate it.
    """
    cls = getattr(A, spec.name, None)
    if cls is None:
        raise ValueError(
            f"Unknown Albumentations transform: {spec.name!r}. Names are "
            f"resolved against the `albumentations` namespace."
        )
    try:
        return cls(**spec.params)
    except TypeError as exc:
        raise ValueError(
            f"Failed to construct {spec.name}({spec.params}): {exc}"
        ) from exc


def build_pipeline_augmentations(
    aug_cfg: AugmentationsConfig,
    *,
    enabled_head_names: Mapping[str, bool],
    img_size: Optional[int] = None,
) -> Tuple[Optional[A.Compose], Optional[A.Compose], Dict[str, str]]:
    """Build spatial + photometric Albumentations Compose objects.

    Splits the transform list:

    * **spatial**: anything not in :data:`PHOTOMETRIC_TRANSFORMS`. Runs
      across the multi-target sample (image + every enabled head's
      target) so spatial flips / rotations / crops stay consistent.
    * **photometric**: the rest. Runs on the image only.

    Returns ``(spatial_aug, photometric_aug, additional_targets)`` —
    the dict maps Albumentations target keys to "mask" or "image" so
    each head's target gets the right interpolation. The dataset uses
    ``additional_targets`` to wire the spatial Compose to all head
    targets in one call.

    Parameters
    ----------
    aug_cfg
        Parsed ``pipeline.yaml::augmentations`` block.
    enabled_head_names
        ``{head_name: bool}`` from the pipeline config. Only enabled
        heads get an entry in ``additional_targets``.
    img_size
        Optional pre-pad size; when set, a leading
        :class:`A.LongestMaxSize` + :class:`A.PadIfNeeded` 

    Notes
    -----
    Albumentations + multi-channel images: Albumentations expects
    ``(H, W, C)``. The dataset transposes per_class SDM
    ``(C, H, W) -> (H, W, C)`` before feeding and back after.
    """
    additional_targets: Dict[str, str] = {}
    for head_name, enabled in enabled_head_names.items():
        if not enabled:
            continue
        if head_name == "semantic":
            # Semantic uses Albumentations's built-in "mask" key.
            continue
        if head_name == "boundary":
            additional_targets["boundary"] = "mask"
        elif head_name == "distance":
            additional_targets["distance"] = "image"
        elif head_name == "sdm":
            additional_targets["sdm"] = "image"
        # Future heads land here with the right kind.

    # Convert TransformSpec list to Albumentations transforms.
    raw_transforms: List[Tuple[A.BasicTransform, bool]] = [
        (_spec_to_albumentations(spec), _is_photometric(spec.name))
        for spec in aug_cfg.train_transforms
    ]

    spatial_transforms: List[A.BasicTransform] = []
    if img_size is not None:
        #  resize-to-longest and pad to square.
        spatial_transforms.append(A.LongestMaxSize(max_size=img_size, p=1.0))
        spatial_transforms.append(
            A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0)
        )
    spatial_transforms.extend(t for t, is_photo in raw_transforms if not is_photo)

    photometric_transforms: List[A.BasicTransform] = [
        t for t, is_photo in raw_transforms if is_photo
    ]

    spatial_aug: Optional[A.Compose] = None
    if spatial_transforms:
        spatial_aug = A.Compose(
            spatial_transforms, additional_targets=additional_targets,
        )

    photometric_aug: Optional[A.Compose] = (
        A.Compose(photometric_transforms) if photometric_transforms else None
    )

    return spatial_aug, photometric_aug, additional_targets


#  Dataset 


class PipelineMultiTaskDataset(BaseDataset):
    """Per-sample multi-task dataset for pipeline mode.

    Reads pre-sliced ``data*.png|.tiff`` images and ``seg*.png`` label
    slices from disk ( produced by :class:`TrainingDataSlicer`), computes per-head targets on-the-fly
    via the registered target generators, applies augmentations across
    all targets consistently, normalises the image, and returns a dict
    keyed by ``"image"`` plus every enabled head's name.

    Output shapes (per sample):

    * ``image``: ``(C, H, W) float32``. ``C=1`` for grayscale,
      ``C=3`` for 2.5D. ImageNet-normalised when
      ``imagenet_norm=True``.
    * ``semantic``: ``(H, W) int64`` class indices. Present iff
      ``heads.semantic.enabled``.
    * ``boundary``: ``(1, H, W) float32`` in ``{0, 1}``.
    * ``distance``: ``(1, H, W) float32`` non-negative.
    * ``sdm``: ``(K, H, W) float32`` in ``[-1, 1]`` (K=1 for binary,
      ``num_classes - 1`` for per_class).
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        *,
        pipeline_config: PipelineConfig,
        num_classes: int,
        settings: Optional[SimpleNamespace] = None,
        validation: bool = False,
        img_size: Optional[int] = None,
        imagenet_norm: bool = True,
        use_2_5d_slicing: bool = False,
        num_2_5d_slices: int = 1,
    ) -> None:
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        if not images_dir.exists():
            raise FileNotFoundError(
                f"PipelineMultiTaskDataset: images_dir not found: {images_dir}"
            )
        if not masks_dir.exists():
            raise FileNotFoundError(
                f"PipelineMultiTaskDataset: masks_dir not found: {masks_dir}"
            )

        self.images_fps: List[Path] = sorted(
            list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.tiff"))
            + list(images_dir.glob("*.tif")),
            key=self._natsort,
        )
        self.masks_fps: List[Path] = sorted(
            list(masks_dir.glob("*.png")),
            key=self._natsort,
        )
        if len(self.images_fps) == 0:
            raise ValueError(
                f"PipelineMultiTaskDataset: no images found in {images_dir}"
            )
        if len(self.images_fps) != len(self.masks_fps):
            raise ValueError(
                f"PipelineMultiTaskDataset: image / mask count mismatch — "
                f"{len(self.images_fps)} vs {len(self.masks_fps)}"
            )

        self.pipeline_config = pipeline_config
        self.heads_cfg: Dict[str, HeadConfig] = pipeline_config.heads
        self.num_classes = int(num_classes)
        self.settings = settings
        self.validation = bool(validation)
        self.use_2_5d_slicing = bool(use_2_5d_slicing)
        self.num_2_5d_slices = int(num_2_5d_slices)

        # Augmentations: skip for validation; also feeds val via
        # only the preprocessing pad-to-square step.
        if validation:
            self.spatial_aug = None
            self.photometric_aug = None
            self.additional_targets: Dict[str, str] = {
                k: v for k, v in (
                    ("boundary", "mask") if heads_enabled(pipeline_config, "boundary") else (None, None),
                    ("distance", "image") if heads_enabled(pipeline_config, "distance") else (None, None),
                    ("sdm", "image") if heads_enabled(pipeline_config, "sdm") else (None, None),
                ) if k is not None
            }
            if img_size is not None:
                # Validation pad-to-square only.
                self.preprocess = A.Compose(
                    [
                        A.LongestMaxSize(max_size=img_size, p=1.0),
                        A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
                    ],
                    additional_targets=self.additional_targets,
                )
            else:
                self.preprocess = None
        else:
            self.spatial_aug, self.photometric_aug, self.additional_targets = (
                build_pipeline_augmentations(
                    pipeline_config.augmentations,
                    enabled_head_names={
                        n: c.enabled for n, c in self.heads_cfg.items()
                    },
                    img_size=img_size,
                )
            )
            self.preprocess = None

        # ImageNet normalisation
        self.imagenet_norm = bool(imagenet_norm)
        if use_2_5d_slicing:
            mean, std = cfg.get_imagenet_normalization(settings)
            self.imagenet_mean = mean
            self.imagenet_std = std
        else:
            self.imagenet_mean = cfg.IMAGENET_MEAN
            self.imagenet_std = cfg.IMAGENET_STD

    #  Sample IO 

    def __len__(self) -> int:
        return len(self.images_fps)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self._read_image(self.images_fps[idx])
        label_slice = self._read_label(self.masks_fps[idx])

        # Compute per-head targets from the label slice (semantic is
        # the label slice itself — added below).
        head_targets = compute_pipeline_targets(
            label_slice,
            heads_cfg=self.heads_cfg,
            num_classes=self.num_classes,
            skip_semantic=True,
        )

        # Albumentations input dict.
        sample: Dict[str, Any] = {"image": image}
        # Semantic mask uses the built-in 'mask' key.
        if "semantic" in self.heads_cfg and self.heads_cfg["semantic"].enabled:
            sample["mask"] = label_slice.astype(np.int64)
        if "boundary" in head_targets:
            sample["boundary"] = head_targets["boundary"].astype(np.float32)
        if "distance" in head_targets:
            sample["distance"] = head_targets["distance"].astype(np.float32)
        if "sdm" in head_targets:
            sdm = head_targets["sdm"]
            # per_class SDM is (K, H, W); Albumentations wants (H, W, K).
            if sdm.ndim == 3:
                sdm = np.transpose(sdm, (1, 2, 0))
            sample["sdm"] = sdm.astype(np.float32)

        # Spatial augmentations (multi-target, consistent).
        if self.preprocess is not None:
            sample = self.preprocess(**sample)
        if self.spatial_aug is not None:
            sample = self.spatial_aug(**sample)

        # Photometric augmentations: image only.
        if self.photometric_aug is not None:
            sample["image"] = self.photometric_aug(image=sample["image"])["image"]

        # ImageNet normalisation
        sample["image"] = self._apply_imagenet_norm(sample["image"])

        # Convert to tensors with the right shape.
        return self._sample_to_tensors(sample)

    #  Helpers 

    @staticmethod
    def _natsort(item: Path):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(item))
        ]

    def _read_image(self, image_path: Path) -> np.ndarray:
        suffix = image_path.suffix.lower()
        if suffix in (".tiff", ".tif"):
            image = imageio.imread(str(image_path))
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)
        elif self.use_2_5d_slicing:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return image

    def _read_label(self, mask_path: Path) -> np.ndarray:
        return cv2.imread(str(mask_path), 0)

    def _apply_imagenet_norm(self, image: np.ndarray) -> np.ndarray:
        if not self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float32) / 255.0
            return image
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32) / 255.0
        if isinstance(self.imagenet_mean, (list, tuple)):
            # 2.5D / RGB: per-channel mean/std arrays.
            mean = np.asarray(self.imagenet_mean, dtype=np.float32)
            std = np.asarray(self.imagenet_std, dtype=np.float32)
            if image.ndim == 3:
                image = (image - mean) / std
            else:
                image = (image - mean[0]) / std[0]
        else:
            image = (image - float(self.imagenet_mean)) / float(self.imagenet_std)
        return image.astype(np.float32)

    def _sample_to_tensors(
        self, sample: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        # Image: (H,W) or (H,W,C) -> (C,H,W).
        image = sample["image"]
        if image.ndim == 2:
            image = image[..., np.newaxis]
        out["image"] = torch.from_numpy(
            np.transpose(image, (2, 0, 1)).astype(np.float32),
        )

        # Semantic mask (long).
        if "mask" in sample and self.heads_cfg.get(
            "semantic", HeadConfig(),
        ).enabled:
            out["semantic"] = torch.from_numpy(
                np.asarray(sample["mask"]).astype(np.int64),
            )

        # Boundary (1, H, W).
        if "boundary" in sample:
            arr = np.asarray(sample["boundary"]).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            out["boundary"] = torch.from_numpy(arr)

        # Distance (1, H, W).
        if "distance" in sample:
            arr = np.asarray(sample["distance"]).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            out["distance"] = torch.from_numpy(arr)

        # SDM (K, H, W). Was (H, W, K) for per_class or (H, W) for binary.
        if "sdm" in sample:
            arr = np.asarray(sample["sdm"]).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = np.transpose(arr, (2, 0, 1))
            out["sdm"] = torch.from_numpy(arr)

        return out


def heads_enabled(config: PipelineConfig, head_name: str) -> bool:
    """``True`` iff ``config.heads[head_name].enabled``."""
    h = config.heads.get(head_name)
    return bool(h is not None and h.enabled)


__all__ = [
    "PHOTOMETRIC_TRANSFORMS",
    "PipelineMultiTaskDataset",
    "build_pipeline_augmentations",
    "compute_pipeline_targets",
    "heads_enabled",
]
