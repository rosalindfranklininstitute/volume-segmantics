import re
from pathlib import Path
from types import SimpleNamespace
import imageio
import cv2
import numpy as np
import volume_segmantics.data.augmentations as augs
import volume_segmantics.data.augmentations_monai as augs_monai
import volume_segmantics.utilities.config as cfg
from torch.utils.data import Dataset as BaseDataset


def get_augmentation_module(settings: SimpleNamespace):
    """Returns the appropriate augmentation module based on settings.

    Args:
        settings (SimpleNamespace): Settings object containing augmentation_library setting.

    Returns:
        Module: Either augmentations (albumentations) or augmentations_monai module.
    """
    aug_lib = getattr(settings, 'augmentation_library', 'albumentations')
    if aug_lib == 'monai':
        return augs_monai
    else:
        return augs


class VolSeg2dDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (pathlib.Path): path to images folder
        masks_dir (pathlib.Path): path to segmentation masks folder
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, contrast adjustments)
        imagenet_norm (bool): Whether to normalise according to imagenet stats
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)
        use_2_5d_slicing (bool): Whether images are multi-channel (2.5D) or grayscale (2D)


    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        preprocessing=None,
        augmentation=None,
        imagenet_norm=True,
        postprocessing=None,
        use_2_5d_slicing=False,
    ):

        # Support both PNG and TIFF files
        self.images_fps = sorted(
            list(images_dir.glob("*.png")) + list(images_dir.glob("*.tiff")) + list(images_dir.glob("*.tif")),
            key=self.natsort
        )
        self.masks_fps = sorted(list(masks_dir.glob("*.png")), key=self.natsort)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing
        self.use_2_5d_slicing = use_2_5d_slicing

        if self.use_2_5d_slicing:
            # Use single channel normalization repeated for all channels
            self.imagenet_mean, self.imagenet_std = cfg.get_imagenet_normalization()
        else:
            self.imagenet_mean, self.imagenet_std = cfg.IMAGENET_MEAN, cfg.IMAGENET_STD

    def __getitem__(self, i):

        # read data - handle grayscale, RGB, and multi-channel images
        image_path = self.images_fps[i]
        file_extension = image_path.suffix.lower()

        if file_extension in ['.tiff', '.tif']:
            # Read TIFF files (can have multiple channels)
            image = imageio.imread(str(image_path))
            # Ensure image is in the correct format (H, W, C)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
        else:
            if self.use_2_5d_slicing:
                # Read as color (RGB-equivalent) when using 2.5D PNG slices
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Read as grayscale for 2D slicing
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(str(self.masks_fps[i]), 0)

        # apply pre-processing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]


        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std
        else:
            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float32) / 255.0  # [0, 1]

        # apply post-processing
        if self.postprocessing:
            sample = self.postprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)

    @staticmethod
    def natsort(item):
        return [
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item))
        ]


class VolSeg2dPredictionDataset(BaseDataset):
    """Splits 3D data volume into 2D images for inference.

    Args:
        images_dir (pathlib.Path): path to images folder
        masks_dir (pathlib.Path): path to segmentation masks folder
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        imagenet_norm (bool): Whether to normalise according to imagenet stats
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)
        use_2_5d_prediction (bool): Whether to create 2.5D representations (multi-channel from adjacent slices)


    """

    imagenet_mean = cfg.IMAGENET_MEAN
    imagenet_std = cfg.IMAGENET_STD

    def __init__(
        self,
        data_vol,
        preprocessing=None,
        imagenet_norm=True,
        postprocessing=None,
        use_2_5d_prediction=False,
        num_slices=3,
    ):
        self.data_vol = data_vol
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing
        self.use_2_5d_prediction = use_2_5d_prediction
        self.num_slices = num_slices

        if self.use_2_5d_prediction:
            # Use single channel normalization repeated for all channels
            self.imagenet_mean, self.imagenet_std = cfg.get_imagenet_normalization()
        else:
            self.imagenet_mean, self.imagenet_std = cfg.IMAGENET_MEAN, cfg.IMAGENET_STD

    def __getitem__(self, i):
        if self.use_2_5d_prediction:
            current_slice = self.data_vol[i]
            depth = len(self.data_vol)
            center_idx = self.num_slices // 2
            slices = []

            for j in range(self.num_slices):
                slice_idx = i - center_idx + j

                # Handle border cases by duplicating edge slices
                if slice_idx < 0:
                    slice_idx = 0
                elif slice_idx >= depth:
                    slice_idx = depth - 1

                slices.append(self.data_vol[slice_idx])


            image = np.stack(slices, axis=-1)
        else:
            # Standard 2D prediction
            image = self.data_vol[i]

        # apply pre-processing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std

        # apply post-processing
        if self.postprocessing:
            sample = self.postprocessing(image=image)
            image = sample["image"]

        return image

    def __len__(self):
        return self.data_vol.shape[0]



class VolSeg2dImageDirDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (pathlib.Path): path to images folder
        masks_dir (pathlib.Path): path to segmentation masks folder
        preprocessing (albumentations.Compose): data pre-processing
            (e.g. padding, resizing)
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, contrast adjustments)
        imagenet_norm (bool): Whether to normalise according to imagenet stats
        postprocessing (albumentations.Compose): data post-processing
            (e.g. Convert to Tensor)


    """

    imagenet_mean = cfg.IMAGENET_MEAN
    imagenet_std = cfg.IMAGENET_STD

    def __init__(
        self,
        images_dir,
        preprocessing=None,
        augmentation=None,
        imagenet_norm=True,
        postprocessing=None,
    ):

        self.images_fps = sorted(list(images_dir.glob("*.png")), key=self.natsort)

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(str(self.images_fps[i]), cv2.IMREAD_GRAYSCALE)


        # apply pre-processing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std

        # apply post-processing
        if self.postprocessing:
            sample = self.postprocessing(image=image)
            image = sample["image"]

        return image

    def __len__(self):
        return len(self.images_fps)

    @staticmethod
    def natsort(item):
        return [
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item))
        ]


def get_2d_training_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> VolSeg2dDataset:

    img_size = settings.image_size
    use_2_5d_slicing = getattr(settings, 'use_2_5d_slicing', False)
    num_channels = settings.num_slices if use_2_5d_slicing else 1
    use_imagenet_norm = getattr(settings, 'use_imagenet_norm', False)
    aug_module = get_augmentation_module(settings)
    return VolSeg2dDataset(
        image_dir,
        label_dir,
        preprocessing=aug_module.get_train_preprocess_augs(img_size),
        augmentation=aug_module.get_train_augs(img_size, num_channels=num_channels),
        postprocessing=aug_module.get_postprocess_augs(),
        use_2_5d_slicing=use_2_5d_slicing,
        imagenet_norm=use_imagenet_norm,
    )


def get_2d_validation_dataset(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> VolSeg2dDataset:

    img_size = settings.image_size
    use_2_5d_slicing = getattr(settings, 'use_2_5d_slicing', False)
    num_channels = settings.num_slices if use_2_5d_slicing else 1
    use_imagenet_norm = getattr(settings, 'use_imagenet_norm', False)
    aug_module = get_augmentation_module(settings)
    return VolSeg2dDataset(
        image_dir,
        label_dir,
        preprocessing=aug_module.get_train_preprocess_augs(img_size),
        postprocessing=aug_module.get_postprocess_augs(),
        use_2_5d_slicing=use_2_5d_slicing,
        imagenet_norm=use_imagenet_norm,
    )


def get_2d_prediction_dataset(data_vol: np.array, settings: SimpleNamespace = None) -> VolSeg2dPredictionDataset:
    y_dim, x_dim = data_vol.shape[1:]
    use_2_5d_prediction = getattr(settings, 'use_2_5d_prediction', False) if settings else False
    num_slices = getattr(settings, 'num_slices', 3) if settings else 3
    use_imagenet_norm = getattr(settings, 'use_imagenet_norm', True) if settings else True
    aug_module = get_augmentation_module(settings) if settings else augs
    return VolSeg2dPredictionDataset(
        data_vol,
        preprocessing=aug_module.get_pred_preprocess_augs(y_dim, x_dim),
        postprocessing=aug_module.get_postprocess_augs(),
        use_2_5d_prediction=use_2_5d_prediction,
        num_slices=num_slices,
        imagenet_norm=use_imagenet_norm,
    )

def get_2d_image_dir_prediction_dataset(image_dir: Path, settings: SimpleNamespace) -> VolSeg2dImageDirDataset:
    img_size = settings.output_size
    aug_module = get_augmentation_module(settings)

    return VolSeg2dImageDirDataset(
        image_dir,
        preprocessing=aug_module.get_pred_preprocess_augs(img_size, img_size),
        postprocessing=aug_module.get_postprocess_augs(),
    )


class UnlabeledDataset(BaseDataset):
    """
    Dataset for unlabeled images (no labels required).
    Used for consistency regularization in semi-supervised learning.
    """

    imagenet_mean = cfg.IMAGENET_MEAN
    imagenet_std = cfg.IMAGENET_STD

    def __init__(
        self,
        images_dir: Path,
        preprocessing=None,
        augmentation=None,
        imagenet_norm=True,
        postprocessing=None,
        use_2_5d_slicing=False,
    ):
        """
        Initialize unlabeled dataset.

        Args:
            images_dir: Directory containing unlabeled images
            preprocessing: Preprocessing transforms
            augmentation: Augmentation transforms (strong for student)
            imagenet_norm: Whether to apply ImageNet normalization
            postprocessing: Postprocessing transforms
            use_2_5d_slicing: Whether using 2.5D slicing
        """
        # Support both PNG and TIFF files
        self.images_fps = sorted(
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.tiff")) +
            list(images_dir.glob("*.tif")),
            key=self.natsort
        )
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.imagenet_norm = imagenet_norm
        self.postprocessing = postprocessing
        self.use_2_5d_slicing = use_2_5d_slicing

        if self.use_2_5d_slicing:
            # Use single channel normalization repeated for all channels
            self.imagenet_mean, self.imagenet_std = cfg.get_imagenet_normalization()
        else:
            self.imagenet_mean, self.imagenet_std = cfg.IMAGENET_MEAN, cfg.IMAGENET_STD

    def __getitem__(self, i):
        """Return unlabeled image (no mask)."""
        # Read image - handle grayscale, RGB, and multi-channel images
        image_path = self.images_fps[i]
        file_extension = image_path.suffix.lower()

        if file_extension in ['.tiff', '.tif']:
            # Read TIFF files (can have multiple channels)
            image = imageio.imread(str(image_path))
            # Ensure image is in the correct format (H, W, C)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
        else:
            if self.use_2_5d_slicing:
                # Read as color (RGB-equivalent) when using 2.5D PNG slices
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Read as grayscale for 2D slicing
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # Apply ImageNet normalization
        if self.imagenet_norm:
            if np.issubdtype(image.dtype, np.integer):
                # Convert to float
                image = image.astype(np.float32)
                image = image / 255
            image = image - self.imagenet_mean
            image = image / self.imagenet_std
        else:
            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float32) / 255.0  # [0, 1]

        # Apply postprocessing
        if self.postprocessing:
            sample = self.postprocessing(image=image)
            image = sample["image"]

        return image

    def __len__(self):
        return len(self.images_fps)

    @staticmethod
    def natsort(item):
        return [
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(item))
        ]
