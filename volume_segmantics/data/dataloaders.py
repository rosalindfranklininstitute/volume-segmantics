from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch.utils.data import DataLoader, Subset
from volume_segmantics.data.datasets import (get_2d_prediction_dataset,
                                             get_2d_training_dataset,
                                             get_2d_validation_dataset,
                                             get_2d_image_dir_prediction_dataset)
from volume_segmantics.data.pipeline_dataset import PipelineMultiTaskDataset
from volume_segmantics.data.pipeline_loader import PipelineConfig
from volume_segmantics.utilities.seeding import make_generator, seed_worker


def _get_seed(settings: SimpleNamespace) -> Optional[int]:
    """Return the configured reproducibility seed, or ``None`` if unset.

    When ``None`` (the default), data loading keeps its existing
    non-deterministic behaviour. When set, the train/val split, shuffle order,
    and worker RNG are all made reproducible.
    """
    seed = getattr(settings, "random_seed", None)
    return None if seed is None else int(seed)


def _seeded_loader_kwargs(seed: Optional[int]) -> dict:
    """DataLoader kwargs that make shuffling/workers reproducible when seeded.

    Returns an empty dict when ``seed is None`` so the DataLoader is constructed
    exactly as before (no behaviour change on the default path).
    """
    if seed is None:
        return {}
    return {"worker_init_fn": seed_worker, "generator": make_generator(seed)}


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

    pipeline_cfg = getattr(settings, "pipeline_config", None)
    if pipeline_cfg is not None and _pipeline_mode_required(pipeline_cfg):
        return get_pipeline_training_dataloaders(
            image_dir, label_dir, settings, pipeline_cfg,
        )

    # Multitask requires MONAI datasets (they return dicts with boundary/task3 keys)
    use_multitask = getattr(settings, "use_multitask", False)
    use_monai = (
        (
            getattr(settings, "augmentation_library", "albumentations") == "monai"
            and getattr(settings, "use_monai_datasets", True)
        )
        or use_multitask  # Force MONAI datasets for multitask
    ) and MONAI_DATASETS_AVAILABLE

    if use_monai:
        return get_monai_training_dataloaders(image_dir, label_dir, settings)

    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)

    seed = _get_seed(settings)
    full_training_dset = get_2d_training_dataset(image_dir, label_dir, settings)
    full_validation_dset = get_2d_validation_dataset(image_dir, label_dir, settings)
    # split the dataset into train and test
    dset_length = len(full_training_dset)
    indices = torch.randperm(dset_length, generator=make_generator(seed)).tolist()
    train_idx, validate_idx = np.split(indices, [int(dset_length * training_set_prop)])
    training_dataset = Subset(full_training_dset, train_idx)
    validation_dataset = Subset(full_validation_dset, validate_idx)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
        **_seeded_loader_kwargs(seed),
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
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

    seed = _get_seed(settings)
    training_dataset, validation_dataset = get_monai_training_and_validation_datasets(
        image_dir, label_dir, settings
    )

    # Create dataloaders with MONAI collate function
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
        **_seeded_loader_kwargs(seed),
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
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
    seed = _get_seed(settings)
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
            persistent_workers=cfg.PERSISTENT_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=cfg.PIN_CUDA_MEMORY,
            drop_last=True,
            **_seeded_loader_kwargs(seed),
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
            persistent_workers=cfg.PERSISTENT_WORKERS,
            pin_memory=cfg.PIN_CUDA_MEMORY,
            drop_last=True,
            **_seeded_loader_kwargs(seed),
        )

    return labeled_train_loader, unlabeled_loader, validation_loader


#  pipeline-mode dataloaders 


def _pipeline_mode_required(pipeline_config: PipelineConfig) -> bool:
    """Returns True iff the parsed config needs the pipeline-mode path.
    """
    for head_name, head_cfg in pipeline_config.heads.items():
        if head_cfg.enabled and head_name != "semantic":
            return True
    if pipeline_config.instance_assembly.backend is not None:
        return True
    return False


def get_pipeline_training_dataloaders(
    image_dir: Path,
    label_dir: Path,
    settings: SimpleNamespace,
    pipeline_config: PipelineConfig,
    *,
    num_classes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build pipeline-mode train and val dataloaders 

    """
    if num_classes is None:
        num_classes = int(getattr(settings, "max_label_no", 2) or 2)

    img_size = int(getattr(settings, "image_size", 512) or 512)
    use_imagenet_norm = bool(getattr(settings, "use_imagenet_norm", True))
    use_2_5d_slicing = bool(getattr(settings, "use_2_5d_slicing", False))
    num_2_5d_slices = int(getattr(settings, "num_slices", 3) or 3)
    training_set_prop = float(
        getattr(settings, "training_set_proportion", 0.85) or 0.85
    )
    batch_size = utils.get_batch_size(settings)

    train_dset = PipelineMultiTaskDataset(
        image_dir, label_dir,
        pipeline_config=pipeline_config,
        num_classes=num_classes,
        settings=settings,
        validation=False,
        img_size=img_size,
        imagenet_norm=use_imagenet_norm,
        use_2_5d_slicing=use_2_5d_slicing,
        num_2_5d_slices=num_2_5d_slices,
    )
    val_dset = PipelineMultiTaskDataset(
        image_dir, label_dir,
        pipeline_config=pipeline_config,
        num_classes=num_classes,
        settings=settings,
        validation=True,
        img_size=img_size,
        imagenet_norm=use_imagenet_norm,
        use_2_5d_slicing=use_2_5d_slicing,
        num_2_5d_slices=num_2_5d_slices,
    )

    seed = _get_seed(settings)
    n = len(train_dset)
    indices = torch.randperm(n, generator=make_generator(seed)).tolist()
    train_idx, val_idx = np.split(indices, [int(n * training_set_prop)])
    training_dataset = Subset(train_dset, train_idx)
    validation_dataset = Subset(val_dset, val_idx)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
        **_seeded_loader_kwargs(seed),
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader
