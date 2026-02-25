"""
Provides MONAI-based image augmentations for training and prediction.
Uses native MONAI transforms with an albumentations-compatible interface.
"""

import math
from typing import Dict, Any, List

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    Rand2DElasticd,
    RandGridDistortiond,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandHistogramShiftd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    ToTensord,
    OneOf,
)
import volume_segmantics.utilities.config as cfg


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
