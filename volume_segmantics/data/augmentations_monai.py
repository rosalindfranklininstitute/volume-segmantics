"""
Provides MONAI-based image augmentations for training and prediction.
Uses native MONAI transforms with an albumentations-compatible interface.
"""

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Identityd,
    OneOf,
    Rand2DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandAxisFlipd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGridDistortiond,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    ToTensord,
)
import volume_segmantics.utilities.config as cfg

logger = logging.getLogger(__name__)


# Keys used in the transform pipeline
IMAGE_KEY = "image"
MASK_KEY = "mask"
KEYS = [IMAGE_KEY, MASK_KEY]
IMAGE_ONLY_KEYS = [IMAGE_KEY]


class RandTransposed:
    """Random transpose transform that swaps spatial axes with a probability."""
    
    def __init__(self, keys: List[str], prob: float = 0.5):
        self.keys = keys
        self.prob = prob
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() > self.prob:
            return data
        
        result = dict(data)
        for key in self.keys:
            if key in result:
                arr = result[key]
                # Transpose spatial dimensions (last two axes)
                # Handle (C, H, W) -> (C, W, H) format
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3:
                        result[key] = np.transpose(arr, (0, 2, 1))
                    elif arr.ndim == 2:
                        result[key] = arr.T
                elif torch.is_tensor(arr):
                    if arr.ndim == 3:
                        result[key] = arr.permute(0, 2, 1)
                    elif arr.ndim == 2:
                        result[key] = arr.T
        return result


class ToTensorWrapper:
    """Convert numpy arrays to PyTorch tensors, handling both MONAI format and raw arrays."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data)
        
        if IMAGE_KEY in data:
            image = data[IMAGE_KEY]
            if torch.is_tensor(image):
                result[IMAGE_KEY] = image.float()
            else:
                image = np.asarray(image)
                # Handle different array formats
                if image.ndim == 2:
                    # (H, W) -> (1, H, W)
                    image = np.expand_dims(image, 0)
                elif image.ndim == 3 and image.shape[-1] in (1, 3):
                    # (H, W, C) -> (C, H, W)
                    image = np.transpose(image, (2, 0, 1))
                result[IMAGE_KEY] = torch.from_numpy(image.copy()).float()
        
        if MASK_KEY in data:
            mask = data[MASK_KEY]
            if torch.is_tensor(mask):
                # Remove channel dimension if present for mask
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                result[MASK_KEY] = mask.long()
            else:
                mask = np.asarray(mask)
                # Remove channel dimension if present for mask
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                result[MASK_KEY] = torch.from_numpy(mask.copy()).long()
        
        return result


class AlbumentationsStyleCompose:
    """Wrapper around MONAI Compose that provides albumentations-style interface.
    
    Allows calling transforms with keyword arguments:
        result = transform(image=image, mask=mask)
    
    Instead of MONAI's dictionary interface:
        result = transform({"image": image, "mask": mask})
    """
    
    def __init__(self, transforms: List, keys: List[str] = None):
        """Initialize the wrapper.
        
        Args:
            transforms: List of MONAI transforms
            keys: Keys to use for the data dict (default: ["image", "mask"])
        """
        self.compose = Compose(transforms)
        self.keys = keys if keys is not None else [IMAGE_KEY, MASK_KEY]
    
    def __call__(self, image=None, mask=None, **kwargs) -> Dict[str, Any]:
        """Apply transforms with albumentations-style interface.
        
        Args:
            image: Input image array (H, W) or (H, W, C)
            mask: Optional mask array (H, W)
            **kwargs: Additional keyword arguments (ignored for compatibility)
        
        Returns:
            Dictionary with "image" and optionally "mask" keys
        """
        # Build input dictionary
        data = {}
        if image is not None:
            data[IMAGE_KEY] = image
        if mask is not None:
            data[MASK_KEY] = mask
        
        # Apply MONAI transforms
        result = self.compose(data)
        
        # Return in same format as input
        output = {}
        if IMAGE_KEY in result:
            output["image"] = result[IMAGE_KEY]
        if MASK_KEY in result:
            output["mask"] = result[MASK_KEY]
        
        return output


def get_train_preprocess_augs(img_size: int) -> AlbumentationsStyleCompose:
    """Returns the augmentations required to pad images to the correct square size
    for training a network.

    Args:
        img_size (int): Length of square image edge.

    Returns:
        AlbumentationsStyleCompose: An augmentation pipeline to resize the image if needed.
    """
    return AlbumentationsStyleCompose([
        EnsureChannelFirstd(keys=KEYS, channel_dim="no_channel"),
        Resized(
            keys=KEYS,
            spatial_size=(img_size, img_size),
            mode=("bilinear", "nearest"),
        ),
    ])


def get_padded_dimension(dimension: int) -> int:
    """Returns the closest image dimension that can be divided by the divisor
    specified in the config.

    Args:
        dimension (int): input dimension.

    Returns:
        int: Dimension that can divided by divisor
    """
    image_divisor = cfg.IM_SIZE_DIVISOR
    if dimension % image_divisor == 0:
        return dimension
    return (math.floor(dimension / image_divisor) + 1) * image_divisor


def get_pred_preprocess_augs(
    img_size_y: int, img_size_x: int
) -> AlbumentationsStyleCompose:
    """Returns the augmentations required to pad images to the correct size for
    prediction.

    Args:
        img_size_y (int): Image size in y
        img_size_x (int): Image size in x

    Returns:
        AlbumentationsStyleCompose: An augmentation pipeline to pad the image if needed.
    """
    padded_y_dim = get_padded_dimension(img_size_y)
    padded_x_dim = get_padded_dimension(img_size_x)
    return AlbumentationsStyleCompose(
        [
            EnsureChannelFirstd(keys=[IMAGE_KEY], channel_dim="no_channel"),
            SpatialPadd(
                keys=[IMAGE_KEY],
                spatial_size=(padded_y_dim, padded_x_dim),
                mode="constant",
            ),
        ],
        keys=[IMAGE_KEY],
    )


def get_train_augs(img_size: int, num_channels: int = 1) -> AlbumentationsStyleCompose:
    """Returns the augmentations used for training a network.

    Args:
        img_size (int): The square image size required.
        num_channels (int): Number of channels in the image.

    Returns:
        AlbumentationsStyleCompose: Augmentations for training.
    """
    transforms = [
        # Random spatial crop with resize back to target size
        RandSpatialCropd(
            keys=KEYS,
            roi_size=(img_size // 2, img_size // 2),
            random_size=True,
            max_roi_size=(img_size, img_size),
            random_center=True,
            lazy=False,
        ),
        Resized(
            keys=KEYS,
            spatial_size=(img_size, img_size),
            mode=("bilinear", "nearest"),
        ),
        
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
        
        RandRotate90d(keys=KEYS, prob=0.5, spatial_axes=(0, 1)),
        RandTransposed(keys=KEYS, prob=0.5),
        OneOf(
            transforms=[
                Rand2DElasticd(
                    keys=KEYS,
                    spacing=(20, 20),
                    magnitude_range=(1, 3),
                    prob=1.0,
                    spatial_size=(img_size, img_size),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                ),
                RandGridDistortiond(
                    keys=KEYS,
                    prob=1.0,
                    distort_limit=(-0.03, 0.03),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                ),
            ],
            weights=[0.5, 0.5],
        ),
    ]

    if num_channels in (1, 3):
        transforms.append(
            RandHistogramShiftd(
                keys=IMAGE_ONLY_KEYS,
                num_control_points=(5, 15),
                prob=0.5,
            )
        )

    transforms.append(
        OneOf(
            transforms=[
                Compose([
                    RandShiftIntensityd(
                        keys=IMAGE_ONLY_KEYS,
                        offsets=(-0.2, 0.2),
                        prob=1.0,
                    ),
                    RandAdjustContrastd(
                        keys=IMAGE_ONLY_KEYS,
                        gamma=(0.8, 1.2),
                        prob=1.0,
                    ),
                ]),
                # Random gamma/intensity scale
                RandScaleIntensityd(
                    keys=IMAGE_ONLY_KEYS,
                    factors=(-0.3, 0.3),
                    prob=1.0,
                ),
            ],
            weights=[0.5, 0.5],
        )
    )

    return AlbumentationsStyleCompose(transforms)


def get_postprocess_augs() -> AlbumentationsStyleCompose:
    """Returns the final augmentations applied to the images.

    Returns:
        AlbumentationsStyleCompose: Postprocessing augmentations.
    """
    return AlbumentationsStyleCompose([
        ToTensorWrapper(),
    ])


#  Pipeline (multi-task) MONAI backend: pipeline.yaml transform specs -> MONAI
#  dictionary transforms applied consistently across image + every head target.


def _sym_limit(value, default):
    """Collapse an Albumentations limit (float or (lo, hi)) to a magnitude."""
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return max(abs(float(value[0])), abs(float(value[1])))
    return abs(float(value))


@dataclass(frozen=True)
class _SpecCtx:
    """Everything a builder in :data:`_MONAI_BUILDERS` needs.

    Bundled so the table stays data: one argument in, one transform out.
    """

    prob: float
    params: Dict[str, Any]
    keys: List[str]
    modes: List[str]
    image_only: List[str]
    img_size: int


def _build_affine(c: "_SpecCtx"):
    rot = _sym_limit(c.params.get("rotate_limit", c.params.get("limit")), 0)
    scale = _sym_limit(c.params.get("scale_limit"), 0)
    shift = _sym_limit(c.params.get("shift_limit"), 0)
    return RandAffined(
        keys=c.keys, prob=c.prob,
        rotate_range=(math.radians(rot),) if rot else None,
        scale_range=(scale, scale) if scale else None,
        translate_range=(shift * c.img_size, shift * c.img_size) if shift else None,
        mode=c.modes, padding_mode="zeros",
    )


def _build_grid_distortion(c: "_SpecCtx"):
    mag = _sym_limit(c.params.get("distort_limit"), 0.03)
    return RandGridDistortiond(
        keys=c.keys, prob=c.prob, distort_limit=(-mag, mag),
        mode=c.modes, padding_mode="zeros",
    )


def _build_gauss_noise(c: "_SpecCtx"):
    vl = c.params.get("var_limit", (10.0, 50.0))
    mean_var = (sum(vl) / len(vl)) if isinstance(vl, (list, tuple)) else float(vl)
    return RandGaussianNoised(
        keys=c.image_only, prob=c.prob, mean=0.0,
        std=math.sqrt(mean_var) / 255.0,
    )


def _build_gamma(c: "_SpecCtx"):
    gl = c.params.get("gamma_limit", (80, 120))
    gamma = (
        tuple(float(g) / 100.0 for g in gl)
        if isinstance(gl, (list, tuple)) else (0.8, 1.2)
    )
    return RandAdjustContrastd(keys=c.image_only, prob=c.prob, gamma=gamma)


# Albumentations transform name -> MONAI builder. Geometric entries apply to
# every key (with per-key ``modes``); photometric ones touch the image only and
# run on the ``[0, 1]``-scaled image. A name absent here is tried as a
# MONAI-native transform instead.
_MONAI_BUILDERS: Dict[str, Any] = {
    #  Geometric (all keys, per-key interpolation)
    "HorizontalFlip": lambda c: RandFlipd(keys=c.keys, prob=c.prob, spatial_axis=1),
    "VerticalFlip": lambda c: RandFlipd(keys=c.keys, prob=c.prob, spatial_axis=0),
    "Flip": lambda c: RandAxisFlipd(keys=c.keys, prob=c.prob),
    "RandomRotate90": lambda c: RandRotate90d(
        keys=c.keys, prob=c.prob, spatial_axes=(0, 1),
    ),
    "Transpose": lambda c: RandTransposed(keys=c.keys, prob=c.prob),
    "Rotate": lambda c: RandRotated(
        keys=c.keys, prob=c.prob,
        range_x=math.radians(_sym_limit(c.params.get("limit"), 90)),
        mode=c.modes, padding_mode="zeros", keep_size=True,
    ),
    "ShiftScaleRotate": _build_affine,
    "ElasticTransform": lambda c: Rand2DElasticd(
        keys=c.keys, prob=c.prob, spacing=(20, 20), magnitude_range=(1, 3),
        spatial_size=(c.img_size, c.img_size), mode=c.modes, padding_mode="zeros",
    ),
    "GridDistortion": _build_grid_distortion,

    #  Photometric (image only; image is [0, 1] at this point)
    "RandomBrightnessContrast": lambda c: Compose([
        RandShiftIntensityd(
            keys=c.image_only, prob=c.prob,
            offsets=_sym_limit(c.params.get("brightness_limit"), 0.2),
        ),
        RandScaleIntensityd(
            keys=c.image_only, prob=c.prob,
            factors=_sym_limit(c.params.get("contrast_limit"), 0.2),
        ),
    ]),
    "RandomBrightness": lambda c: RandShiftIntensityd(
        keys=c.image_only, prob=c.prob,
        offsets=_sym_limit(c.params.get("brightness_limit"), 0.2),
    ),
    "RandomContrast": lambda c: RandScaleIntensityd(
        keys=c.image_only, prob=c.prob,
        factors=_sym_limit(c.params.get("contrast_limit"), 0.2),
    ),
    "RandomGamma": _build_gamma,
    "GaussNoise": _build_gauss_noise,
    "GaussianBlur": lambda c: RandGaussianSmoothd(keys=c.image_only, prob=c.prob),
}
# Aliases that map to the same builder as an entry above.
_MONAI_BUILDERS.update({
    "RandRotate90": _MONAI_BUILDERS["RandomRotate90"],
    "Affine": _MONAI_BUILDERS["ShiftScaleRotate"],
    "Blur": _MONAI_BUILDERS["GaussianBlur"],
    "MotionBlur": _MONAI_BUILDERS["GaussianBlur"],
    "MedianBlur": _MONAI_BUILDERS["GaussianBlur"],
})


def _spec_to_monai(
    spec,
    *,
    keys: List[str],
    modes: List[str],
    image_only: List[str],
    img_size: int,
):
    """Translate one Albumentations-style ``TransformSpec`` to a MONAI dict
    transform, or return ``None`` if it cannot be mapped.

    Dispatches through :data:`_MONAI_BUILDERS`. Names with no Albumentations
    equivalent fall through to :func:`_spec_to_monai_native`, which resolves
    them directly against the MONAI namespace (e.g. ``RandBiasField``,
    ``RandGibbsNoise``) so MONAI-only augmentations can be requested from
    ``pipeline.yaml`` too.
    """
    builder = _MONAI_BUILDERS.get(spec.name)
    if builder is None:
        return _spec_to_monai_native(
            spec, keys=keys, modes=modes, image_only=image_only,
        )
    params = dict(spec.params or {})
    return builder(_SpecCtx(
        prob=float(params.get("p", 0.5)),
        params=params,
        keys=keys,
        modes=modes,
        image_only=image_only,
        img_size=img_size,
    ))


def _spec_to_monai_native(
    spec,
    *,
    keys: List[str],
    modes: List[str],
    image_only: List[str],
):
    """Resolve a MONAI-native transform by name, for MONAI-only augmentations
    that have no Albumentations equivalent.

    The name is resolved against the MONAI dictionary-transform namespace
    (the ``d`` suffix is added if missing, so ``RandBiasField`` finds
    ``RandBiasFieldd``). The transform is applied to the image only by
    default; set ``spatial: true`` in the spec's params to apply it across
    every key (masks/targets included) with per-key interpolation modes where
    the transform accepts them. ``p`` is mapped to MONAI's ``prob``; all other
    params pass straight through. Returns ``None`` if the name does not
    resolve to a MONAI dict transform.
    """
    import monai.transforms as _mt
    from monai.transforms import MapTransform

    name = spec.name
    cls = None
    for candidate in (name, name + "d"):
        found = getattr(_mt, candidate, None)
        if isinstance(found, type) and issubclass(found, MapTransform):
            cls = found
            break
    if cls is None:
        return None

    params = dict(spec.params or {})
    spatial = bool(params.pop("spatial", False))
    if "prob" not in params and "p" in params:
        params["prob"] = float(params.pop("p"))
    params.pop("p", None)
    params["keys"] = list(keys) if spatial else list(image_only)
    if spatial:
        try:
            accepted = set(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            accepted = set()
        if "mode" in accepted:
            params.setdefault("mode", list(modes))
        if "padding_mode" in accepted:
            params.setdefault("padding_mode", "zeros")
    try:
        return cls(**params)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        logger.warning(
            "MONAI augmentation: could not build native transform %r "
            "(%s); skipping it.", name, exc,
        )
        return None


def _default_monai_aug_transforms(
    img_size: int, keys: List[str], modes: List[str], image_only: List[str],
) -> List:
    """Fallback MONAI augmentation recipe used when the pipeline config lists
    no transforms: flips, rot90, elastic/grid distortion, and gentle
    intensity jitter."""
    return [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
        OneOf(
            transforms=[
                Rand2DElasticd(
                    keys=keys, spacing=(20, 20), magnitude_range=(1, 3), prob=1.0,
                    spatial_size=(img_size, img_size), mode=modes, padding_mode="zeros",
                ),
                RandGridDistortiond(
                    keys=keys, prob=1.0, distort_limit=(-0.03, 0.03),
                    mode=modes, padding_mode="zeros",
                ),
            ],
            weights=[0.5, 0.5],
        ),
        RandAdjustContrastd(keys=image_only, gamma=(0.8, 1.2), prob=0.5),
        RandScaleIntensityd(keys=image_only, factors=(-0.3, 0.3), prob=0.5),
    ]


def build_pipeline_monai_transforms(
    img_size: int,
    keys: List[str],
    modes: List[str],
    train_transforms: Optional[List] = None,
    image_key: str = "image",
) -> Compose:
    """Build a MONAI ``Compose`` for the pipeline (multi-task) dataset.

    Unlike :func:`get_train_augs` (hard-wired to image+mask), this applies the
    geometric transforms consistently across *every* key in ``keys``, the
    image plus each enabled head's target (``mask``, ``boundary``,
    ``distance``, ``sdm``), using the per-key interpolation ``modes``
    (``"bilinear"`` for the image and continuous targets, ``"nearest"`` for
    label targets). ``keys`` and ``modes`` are aligned lists.

    When ``train_transforms`` (the parsed ``pipeline.yaml::augmentations``
    list) is given, each spec is translated to its MONAI equivalent via
    :func:`_spec_to_monai`; unsupported names are warned about and skipped.
    When it is empty, a default recipe is used instead.
    """
    image_only = [image_key]
    transforms: List = [
        # Resize longest edge then pad to square: the MONAI equivalent of the
        # Albumentations LongestMaxSize + PadIfNeeded sizing step.
        Resized(keys=keys, spatial_size=img_size, size_mode="longest", mode=modes),
        SpatialPadd(keys=keys, spatial_size=(img_size, img_size), mode="constant"),
    ]
    if train_transforms:
        for spec in train_transforms:
            monai_t = _spec_to_monai(
                spec, keys=keys, modes=modes, image_only=image_only,
                img_size=img_size,
            )
            if monai_t is None:
                logger.warning(
                    "MONAI augmentation: no equivalent for transform %r; "
                    "skipping it.", getattr(spec, "name", spec),
                )
            else:
                transforms.append(monai_t)
    else:
        transforms.extend(
            _default_monai_aug_transforms(img_size, keys, modes, image_only)
        )
    return Compose(transforms)
