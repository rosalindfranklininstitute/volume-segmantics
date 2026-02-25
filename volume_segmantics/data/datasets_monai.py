"""
MONAI-based datasets and transforms for 2D segmentation with multi-task support.
This module provides MONAI dataset implementations when using MONAI augmentations.
"""

import re
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Optional, Tuple
import random

try:
    from monai.data import Dataset as MONAIDataset
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Resized,
        ScaleIntensityd,
        RandFlipd,
        RandRotate90d,
        ToTensord,
        Transform,
    )
    # Try to import optional transforms
    try:
        from monai.transforms import Rand2DElasticD
    except ImportError:
        Rand2DElasticD = None
    
    try:
        from monai.transforms import RandSpatialCropd
    except ImportError:
        RandSpatialCropd = None
    
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    MONAIDataset = None

import volume_segmantics.utilities.config as cfg
import numpy as np
import torch
import logging


class ImageNetNormalizationd(Transform):
    """Apply ImageNet mean/std normalization to images."""
    
    def __init__(self, keys, mean=None, std=None):
        """Initialize ImageNet normalization transform.
        
        Args:
            keys: Keys to apply normalization to (typically ["img"])
            mean: Mean values (defaults to cfg.IMAGENET_MEAN)
            std: Std values (defaults to cfg.IMAGENET_STD)
        """
        self.keys = keys if isinstance(keys, list) else [keys]
        self.mean = mean if mean is not None else cfg.IMAGENET_MEAN
        self.std = std if std is not None else cfg.IMAGENET_STD
        
        # Convert to tensors if needed
        if isinstance(self.mean, (int, float)):
            self.mean = torch.tensor([self.mean])
        elif isinstance(self.mean, list):
            self.mean = torch.tensor(self.mean)
        
        if isinstance(self.std, (int, float)):
            self.std = torch.tensor([self.std])
        elif isinstance(self.std, list):
            self.std = torch.tensor(self.std)
    
    def __call__(self, data):
        """Apply normalization to specified keys."""
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                if not isinstance(img, torch.Tensor):
                    img = torch.as_tensor(img)
                
                # Normalize: (img - mean) / std
                # Handle channel dimension
                if len(img.shape) == 3:  # (C, H, W)
                    mean = self.mean.view(-1, 1, 1).to(img.device)
                    std = self.std.view(-1, 1, 1).to(img.device)
                elif len(img.shape) == 4:  # (B, C, H, W)
                    mean = self.mean.view(1, -1, 1, 1).to(img.device)
                    std = self.std.view(1, -1, 1, 1).to(img.device)
                else:
                    mean = self.mean.to(img.device)
                    std = self.std.to(img.device)
                
                d[key] = (img - mean) / std
        return d


def natsort_key(item):
    """Natural sort key function for file paths."""
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item))
    ]


def build_file_list(
    image_dir: Path,
    label_dir: Path,
    task2_dir: Optional[Path] = None,
    task3_dir: Optional[Path] = None,
    use_2_5d_slicing: bool = False,
) -> List[Dict[str, str]]:
    """Build a list of file dictionaries for MONAI dataset.
    
    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing segmentation label files
        task2_dir: Optional directory for second task labels (e.g., boundary)
        task3_dir: Optional directory for third task labels
        use_2_5d_slicing: Whether using 2.5D slicing (affects file format)
    
    Returns:
        List of dictionaries with file paths: [{"img": path, "seg": path, ...}]
    """
    image_extensions = ["*.png", "*.tiff", "*.tif"]
    images_fps = []
    for ext in image_extensions:
        images_fps.extend(list(image_dir.glob(ext)))
    
    images_fps = sorted(images_fps, key=natsort_key)
    
    segs_fps = sorted(list(label_dir.glob("*.png")), key=natsort_key)
    
    if len(images_fps) != len(segs_fps):
        raise ValueError(
            f"Mismatch between number of images ({len(images_fps)}) "
            f"and segmentation labels ({len(segs_fps)})"
        )
    
    all_files = []
    for img_path, seg_path in zip(images_fps, segs_fps):
        file_dict = {
            "img": str(img_path),
            "seg": str(seg_path),
        }
        
        if task2_dir is not None:
            # Find corresponding task2 file
            # Task2 files use prefix "task2_{count}" while seg files use "seg{count}"
            # Filename format: {prefix}_{axis}_stack_{index}.png
            # Convert: seg0_x_stack_0.png -> task2_0_x_stack_0.png
            seg_name = seg_path.name
            # Match pattern: seg{number} at start of filename, replace with task2_{number}
            task2_name = re.sub(r'^seg(\d+)', r'task2_\1', seg_name)
            task2_path = task2_dir / task2_name
            if task2_path.exists():
                file_dict["boundary"] = str(task2_path)
            else:
                # Try alternative: maybe task2 files use same prefix as seg?
                alt_task2_path = task2_dir / seg_name
                if alt_task2_path.exists():
                    file_dict["boundary"] = str(alt_task2_path)
                else:
                    raise FileNotFoundError(
                        f"Task2 file not found: {task2_path} or {alt_task2_path} "
                        f"for segmentation file {seg_path.name}. "
                        f"Expected task2 file with name: {task2_name} (or {seg_name})"
                    )
        
        # Add task3 if provided
        if task3_dir is not None:
            # Task3 files use prefix "task3_{count}" while seg files use "seg{count}"
            # Filename format: {prefix}_{axis}_stack_{index}.png
            # Convert: seg0_x_stack_0.png -> task3_0_x_stack_0.png
            seg_name = seg_path.name
            task3_name = re.sub(r'^seg(\d+)', r'task3_\1', seg_name)
            task3_path = task3_dir / task3_name
            if task3_path.exists():
                file_dict["task3"] = str(task3_path)
            else:
                alt_task3_path = task3_dir / seg_name
                if alt_task3_path.exists():
                    file_dict["task3"] = str(alt_task3_path)
                else:
                    raise FileNotFoundError(
                        f"Task3 file not found: {task3_path} or {alt_task3_path} "
                        f"for segmentation file {seg_path.name}. "
                        f"Expected task3 file with name: {task3_name} (or {seg_name})"
                    )
        
        all_files.append(file_dict)
    
    return all_files


def split_train_val_files(
    all_files: List[Dict[str, str]], settings: SimpleNamespace
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split file list into training and validation sets.
    
    Args:
        all_files: List of file dictionaries
        settings: Settings object with training_set_proportion
    
    Returns:
        Tuple of (train_files, val_files)
    """
    training_set_prop = getattr(settings, "training_set_proportion", 0.85)
    #random.seed(42)
    num_val = int(len(all_files) * (1 - training_set_prop))
    val_indices = random.sample(range(len(all_files)), num_val)
    
    val_files = [all_files[i] for i in val_indices]
    train_files = [all_files[i] for i in range(len(all_files)) if i not in val_indices]
    
    return train_files, val_files


def get_monai_train_transforms(
    img_size: int,
    num_channels: int = 1,
    use_2_5d_slicing: bool = False,
    num_tasks: int = 1,
    use_imagenet_norm: bool = True,
) -> Compose:
    """Build MONAI training transform pipeline.
    
    Args:
        img_size: Target image size (square)
        num_channels: Number of channels in images
        use_2_5d_slicing: Whether using 2.5D slicing
        num_tasks: Number of tasks (1=seg only, 2=seg+boundary, 3=seg+boundary+task3)
        use_imagenet_norm: Whether to use ImageNet normalization
    
    Returns:
        MONAI Compose transform pipeline
    """
    keys = ["img", "seg"]
    if num_tasks >= 2:
        keys.append("boundary")
    if num_tasks >= 3:
        keys.append("task3")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    
    resize_modes = ["bilinear"] + ["nearest"] * (len(keys) - 1)
    transforms.append(
        Resized(
            keys=keys,
            spatial_size=(img_size, img_size),
            mode=tuple(resize_modes),
        )
    )
    
    if use_imagenet_norm:
        transforms.append(ScaleIntensityd(keys=["img"]))
        transforms.append(ImageNetNormalizationd(keys=["img"]))
    else:
        transforms.append(ScaleIntensityd(keys=["img"]))
    
    # Augmentations
    transforms.extend([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # Vertical flip
        RandRotate90d(keys=keys, prob=0.5),
        RandRotate90d(keys=keys, prob=0.3, spatial_axes=[0, 1]),
    ])
    
    if Rand2DElasticD is not None:
        transforms.append(
            Rand2DElasticD(
                keys=keys,
                spacing=(20, 20),
                magnitude_range=(1, 2),
                prob=0.5,
                spatial_size=(img_size, img_size),
                mode=tuple(["bilinear"] + ["nearest"] * (len(keys) - 1)),
                padding_mode="zeros",
            )
        )
    
    transforms.append(ToTensord(keys=keys))
    
    return Compose(transforms)


def get_monai_val_transforms(
    img_size: int,
    num_channels: int = 1,
    use_2_5d_slicing: bool = False,
    num_tasks: int = 1,
    use_imagenet_norm: bool = True,
) -> Compose:
    """Build MONAI validation transform pipeline.
    
    Args:
        img_size: Target image size (square)
        num_channels: Number of channels in images
        use_2_5d_slicing: Whether using 2.5D slicing
        num_tasks: Number of tasks
        use_imagenet_norm: Whether to use ImageNet normalization
    
    Returns:
        MONAI Compose transform pipeline
    """
    keys = ["img", "seg"]
    if num_tasks >= 2:
        keys.append("boundary")
    if num_tasks >= 3:
        keys.append("task3")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]
    
    resize_modes = ["bilinear"] + ["nearest"] * (len(keys) - 1)
    transforms.append(
        Resized(
            keys=keys,
            spatial_size=(img_size, img_size),
            mode=tuple(resize_modes),
        )
    )
    
    if use_imagenet_norm:
        transforms.append(ScaleIntensityd(keys=["img"]))
        transforms.append(ImageNetNormalizationd(keys=["img"]))
    else:
        transforms.append(ScaleIntensityd(keys=["img"]))
    
    transforms.append(ToTensord(keys=keys))
    
    return Compose(transforms)


def get_monai_training_and_validation_datasets(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[MONAIDataset, MONAIDataset]:
    """Create MONAI training and validation datasets with a shared train/val split.
    
    This function ensures that training and validation datasets use the same split,
    preventing overlap between the two sets.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        settings: Settings object
    
    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is not available. Install MONAI to use MONAI datasets.")
    
    img_size = settings.image_size
    use_2_5d_slicing = getattr(settings, "use_2_5d_slicing", False)
    num_channels = getattr(settings, "num_slices", 3) if use_2_5d_slicing else 1
    use_imagenet_norm = getattr(settings, "use_imagenet_norm", True)
    
    # Determine number of tasks
    use_multitask = getattr(settings, "use_multitask", False)
    
    # Auto-enable multitask if task2_dir or task3_dir is set
    task2_dir_str = getattr(settings, "task2_dir", None)
    task3_dir_str = getattr(settings, "task3_dir", None)
    if (task2_dir_str is not None or task3_dir_str is not None) and not use_multitask:
        use_multitask = True
        logging.info(
            "Auto-enabling multitask mode because task2_dir or task3_dir is set in settings"
        )
    
    num_tasks = getattr(settings, "num_tasks", 1) if use_multitask else 1
    
    task2_dir = None
    task3_dir = None
    if use_multitask:
        if task2_dir_str:
            task2_dir = Path(task2_dir_str)
        
        if num_tasks >= 3:
            if task3_dir_str:
                task3_dir = Path(task3_dir_str)
    
    # Build file list once
    all_files = build_file_list(
        image_dir, label_dir, task2_dir, task3_dir, use_2_5d_slicing
    )
    
    # Split train/val once - this ensures both datasets use the same split
    train_files, val_files = split_train_val_files(all_files, settings)
    
    train_transforms = get_monai_train_transforms(
        img_size, num_channels, use_2_5d_slicing, num_tasks, use_imagenet_norm
    )
    val_transforms = get_monai_val_transforms(
        img_size, num_channels, use_2_5d_slicing, num_tasks, use_imagenet_norm
    )
    

    training_dataset = MONAIDataset(data=train_files, transform=train_transforms)
    validation_dataset = MONAIDataset(data=val_files, transform=val_transforms)
    
    return training_dataset, validation_dataset


def get_monai_training_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> MONAIDataset:
    training_dataset, _ = get_monai_training_and_validation_datasets(
        image_dir, label_dir, settings
    )
    return training_dataset


def get_monai_validation_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> MONAIDataset:
    _, validation_dataset = get_monai_training_and_validation_datasets(
        image_dir, label_dir, settings
    )
    return validation_dataset


class UnlabeledMONAIDataset(MONAIDataset):
    """
    MONAI dataset for unlabeled images with separate augmentations for student and teacher.
    Used for consistency regularization in semi-supervised learning.
    """
    
    def __init__(
        self,
        image_dir: Path,
        student_transform: Optional[Compose] = None,
        teacher_transform: Optional[Compose] = None,
    ):
        """
        Initialize unlabeled MONAI dataset.
        
        Args:
            image_dir: Directory containing unlabeled images
            student_transform: Strong augmentation pipeline for student model
            teacher_transform: Weak augmentation pipeline for teacher model
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is not available. Install MONAI to use MONAI datasets.")
        
        # Build file list (images only, no labels)
        image_extensions = ["*.png", "*.tiff", "*.tif"]
        images_fps = []
        for ext in image_extensions:
            images_fps.extend(list(image_dir.glob(ext)))
        
        images_fps = sorted(images_fps, key=natsort_key)
        
        data_dicts = [{"img": str(img_path)} for img_path in images_fps]
        
        super().__init__(data=data_dicts, transform=None)  # No default transform
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
    
    def __getitem__(self, index):
        """Return both student and teacher versions of the image."""
        data = self.data[index].copy()
        
        # Apply student transform (strong augmentation)
        if self.student_transform is not None:
            student_data = self.student_transform(data.copy())
        else:
            # Load image without transform
            from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord
            load_transform = Compose([
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
                ToTensord(keys=["img"])
            ])
            student_data = load_transform(data.copy())
        
        # Apply teacher transform (weak augmentation)
        if self.teacher_transform is not None:
            teacher_data = self.teacher_transform(data.copy())
        else:
            # Load image without transform (same as student fallback)
            from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord
            load_transform = Compose([
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
                ToTensord(keys=["img"])
            ])
            teacher_data = load_transform(data.copy())
        
        return {
            "student": student_data["img"],
            "teacher": teacher_data["img"],
        }


def get_monai_unlabeled_student_transforms(
    img_size: int,
    num_channels: int = 1,
    use_2_5d_slicing: bool = False,
    use_imagenet_norm: bool = True,
) -> Compose:
    """
    Get strong augmentation pipeline for student model (unlabeled data).
    Uses same strong augmentations as training data.
    
    Args:
        img_size: Target image size (square)
        num_channels: Number of channels in images
        use_2_5d_slicing: Whether using 2.5D slicing
        use_imagenet_norm: Whether to use ImageNet normalization
    
    Returns:
        MONAI Compose transform pipeline with strong augmentations
    """
    return get_monai_train_transforms(
        img_size, num_channels, use_2_5d_slicing, num_tasks=1, use_imagenet_norm=use_imagenet_norm
    )


def get_monai_unlabeled_teacher_transforms(
    img_size: int,
    num_channels: int = 1,
    use_2_5d_slicing: bool = False,
    use_imagenet_norm: bool = True,
) -> Compose:
    """
    Get weak augmentation pipeline for teacher model (unlabeled data).
    Uses validation transforms (minimal augmentation).
    
    Args:
        img_size: Target image size (square)
        num_channels: Number of channels in images
        use_2_5d_slicing: Whether using 2.5D slicing
        use_imagenet_norm: Whether to use ImageNet normalization
    
    Returns:
        MONAI Compose transform pipeline with weak augmentations
    """
    return get_monai_val_transforms(
        img_size, num_channels, use_2_5d_slicing, num_tasks=1, use_imagenet_norm=use_imagenet_norm
    )
