from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch.utils.data import DataLoader, Subset
from volume_segmantics.data.datasets import (get_2d_prediction_dataset,
                                             get_2d_training_dataset,
                                             get_2d_validation_dataset,
                                             get_2d_image_dir_prediction_dataset)


try:
    from monai.data import list_data_collate
    from volume_segmantics.data.datasets_monai import (
        get_monai_training_and_validation_datasets,
    )
    MONAI_DATASETS_AVAILABLE = True
except ImportError:
    MONAI_DATASETS_AVAILABLE = False
    list_data_collate = None



def get_2d_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    """Returns 2d training and validation dataloaders with indices split at random
    according to the percentage split specified in settings.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SimpleNamespace): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: 2d training and validation dataloaders
    """
    use_monai = (
        getattr(settings, "augmentation_library", "albumentations") == "monai"
        and getattr(settings, "use_monai_datasets", True)
        and MONAI_DATASETS_AVAILABLE
    )

    if use_monai:
        return get_monai_training_dataloaders(image_dir, label_dir, settings)

    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)

    full_training_dset = get_2d_training_dataset(image_dir, label_dir, settings)
    full_validation_dset = get_2d_validation_dataset(image_dir, label_dir, settings)
    # split the dataset into train and test
    dset_length = len(full_training_dset)
    indices = torch.randperm(dset_length).tolist()
    train_idx, validate_idx = np.split(indices, [int(dset_length * training_set_prop)])
    training_dataset = Subset(full_training_dset, train_idx)
    validation_dataset = Subset(full_validation_dset, validate_idx)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_monai_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    """Returns MONAI-based training and validation dataloaders.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SimpleNamespace): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: MONAI training and validation dataloaders
    """
    if not MONAI_DATASETS_AVAILABLE:
        raise ImportError(
            "MONAI datasets are not available. Install MONAI to use MONAI datasets."
        )

    batch_size = utils.get_batch_size(settings)

    training_dataset, validation_dataset = get_monai_training_and_validation_datasets(
        image_dir, label_dir, settings
    )

    # Create dataloaders with MONAI collate function
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_2d_prediction_dataloader(
    data_vol: np.array, settings: SimpleNamespace
) -> DataLoader:
    pred_dataset = get_2d_prediction_dataset(data_vol, settings)
    batch_size = utils.get_batch_size(settings, prediction=True)
    return DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )


def get_2d_image_dir_prediction_dataloader(
    image_dir: Path, settings: SimpleNamespace
) -> DataLoader:
    pred_dataset = get_2d_image_dir_prediction_dataset(image_dir, settings)
    images_fps = pred_dataset.images_fps
    #batch_size = utils.get_batch_size(settings, prediction=True)
    return DataLoader(
        pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    ), images_fps


def get_semi_supervised_dataloaders(
    labeled_image_dir: Path,
    labeled_label_dir: Path,
    unlabeled_image_dir: Path,
    settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for semi-supervised learning.

    Args:
        labeled_image_dir: Directory with labeled images
        labeled_label_dir: Directory with labeled segmentation masks
        unlabeled_image_dir: Directory with unlabeled images (no labels)
        settings: Settings object

    Returns:
        Tuple of (labeled_train_loader, unlabeled_train_loader, validation_loader)
    """
    # Get labeled training and validation loaders (existing functionality)
    labeled_train_loader, validation_loader = get_2d_training_dataloaders(
        labeled_image_dir, labeled_label_dir, settings
    )

    # Create unlabeled dataset
    use_monai = (
        getattr(settings, "augmentation_library", "albumentations") == "monai"
        and getattr(settings, "use_monai_datasets", True)
        and MONAI_DATASETS_AVAILABLE
    )

    if use_monai:
        # MONAI unlabeled dataset
        from volume_segmantics.data.datasets_monai import (
            get_monai_unlabeled_student_transforms,
            get_monai_unlabeled_teacher_transforms,
            UnlabeledMONAIDataset,
        )

        img_size = settings.image_size
        use_2_5d_slicing = getattr(settings, "use_2_5d_slicing", False)
        num_channels = getattr(settings, "num_slices", 3) if use_2_5d_slicing else 1
        use_imagenet_norm = getattr(settings, "use_imagenet_norm", True)

        # Strong augmentations for student
        student_transforms = get_monai_unlabeled_student_transforms(
            img_size, num_channels, use_2_5d_slicing, use_imagenet_norm
        )

        # Weak augmentations for teacher
        teacher_transforms = get_monai_unlabeled_teacher_transforms(
            img_size, num_channels, use_2_5d_slicing, use_imagenet_norm
        )

        unlabeled_dataset = UnlabeledMONAIDataset(
            unlabeled_image_dir,
            student_transform=student_transforms,
            teacher_transform=teacher_transforms
        )

        unlabeled_batch_size = getattr(settings, "unlabeled_batch_size",
                                      utils.get_batch_size(settings))

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=cfg.PIN_CUDA_MEMORY,
            drop_last=True,
        )
    else:
        # Non-MONAI unlabeled dataset
        from volume_segmantics.data.datasets import UnlabeledDataset
        from volume_segmantics.data.datasets import get_augmentation_module

        aug_module = get_augmentation_module(settings)
        img_size = settings.image_size
        use_2_5d_slicing = getattr(settings, "use_2_5d_slicing", False)
        num_channels = getattr(settings, "num_slices", 3) if use_2_5d_slicing else 1

        # Get training augmentations (strong for student)
        # Note: For non-MONAI, we use the same augmentation for both student and teacher
        # The dataset will return the same image, and we can apply different augmentations
        # in the training loop if needed
        augmentation = aug_module.get_train_augs(img_size, num_channels=num_channels) if hasattr(aug_module, 'get_train_augs') else None

        unlabeled_dataset = UnlabeledDataset(
            images_dir=unlabeled_image_dir,
            preprocessing=aug_module.get_train_preprocess_augs(img_size) if hasattr(aug_module, 'get_train_preprocess_augs') else None,
            augmentation=augmentation,
            imagenet_norm=getattr(settings, "use_imagenet_norm", True),
            postprocessing=aug_module.get_postprocess_augs() if hasattr(aug_module, 'get_postprocess_augs') else None,
            use_2_5d_slicing=use_2_5d_slicing,
        )

        unlabeled_batch_size = getattr(settings, "unlabeled_batch_size",
                                      utils.get_batch_size(settings))

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_CUDA_MEMORY,
            drop_last=True,
        )

    return labeled_train_loader, unlabeled_loader, validation_loader
