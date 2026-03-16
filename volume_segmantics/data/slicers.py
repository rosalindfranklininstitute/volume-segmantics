import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import volume_segmantics.utilities.base_data_utils as utils
from skimage import img_as_ubyte, io
from tqdm import tqdm
from volume_segmantics.data.base_data_manager import BaseDataManager
from typing import Union


class TrainingDataSlicer(BaseDataManager):
    """
    Class that performs image preprocessing and provides methods to
    convert 3d data volumes into 2d image slices on disk for model training.
    Slicing is carried in all of the xy (z), xz (y) and yz (x) planes.
    Supports both 2D (single channel) and 2.5D (RGB) slicing modes.
    """

    def __init__(
        self,
        data_vol: Union[str, np.ndarray],
        label_vol: Union[str, np.ndarray, None] = None,
        settings: SimpleNamespace = None,
        label_type: str = "segmentation",
    ):
        if settings is None:
            raise ValueError("settings parameter is required")
        """Inits TrainingDataSlicer.

        Args:
            data_vol(Union[str, np.ndarray]): Either a path to an image data volume or a numpy array of 3D image data
            label_vol(Union[str, np.ndarray, None]): Either a path to a label data volume, a numpy array of 3D label data, or None for unlabeled data
            settings(SimpleNamespace): An object containing the training settings
            label_type(str): Type of label being processed (e.g., "segmentation", "task2", "task3")
        """
        super().__init__(data_vol, settings)
        self.data_im_out_dir = None
        self.seg_im_out_dir = None
        self.multilabel = False
        self.settings = settings
        self.label_type = label_type
        self.has_labels = label_vol is not None

        self.use_2_5d_slicing = getattr(settings, 'use_2_5d_slicing', False)
        self.num_slices = getattr(settings, 'num_slices', 3)
        self.slice_file_format = getattr(settings, 'slice_file_format', 'tiff')
        self.skip_border_slices = getattr(settings, 'skip_border_slices', False)

        # Validate 2.5D settings
        if self.use_2_5d_slicing:
            if self.num_slices % 2 == 0:
                raise ValueError(f"num_slices must be odd, got {self.num_slices}")
            if self.num_slices < 3:
                raise ValueError(f"num_slices must be >= 3, got {self.num_slices}")

            # Validate file format
            if self.slice_file_format.lower() == 'png' and self.num_slices > 3:
                raise ValueError(f"PNG format only supports up to 3 channels, but {self.num_slices} slices were requested. Use 'tiff' format for {self.num_slices} channels.")

            logging.info(f"2.5D slicing mode enabled - creating {self.num_slices}-channel images from adjacent slices")
            logging.info(f"Using {self.slice_file_format.upper()} format for multi-channel storage")
            if self.skip_border_slices:
                logging.info("Border slices will be skipped in 2.5D mode")

        # Handle labels (optional for unlabeled data)
        if label_vol is not None:
            self.label_vol_path = utils.setup_path_if_exists(label_vol)
            if self.label_vol_path is not None:
                self.seg_vol, _ = utils.get_numpy_from_path(
                    self.label_vol_path, internal_path=settings.seg_hdf5_path
                )
            elif isinstance(label_vol, np.ndarray):
                self.seg_vol = label_vol
            self._preprocess_labels()
        else:
            self.label_vol_path = None
            self.seg_vol = None
            self.num_seg_classes = 0
            self.codes = []
            logging.info("No labels provided - slicing unlabeled data only")

    def _preprocess_labels(self):
        seg_classes = np.unique(self.seg_vol)
        self.num_seg_classes = len(seg_classes)
        if self.num_seg_classes > 2:
            self.multilabel = True

        # Use appropriate label type name in logging
        label_name = self.label_type.capitalize() if self.label_type != "segmentation" else "segmentation"
        logging.info(
            f"Number of classes in {label_name} dataset: {self.num_seg_classes}"
        )
        logging.info(f"These classes are: {seg_classes}")
        if seg_classes[0] != 0 or not utils.sequential_labels(seg_classes):
            logging.info("Fixing label classes.")
            self._fix_label_classes(seg_classes)
        self.codes = [f"label_val_{i}" for i in seg_classes]

    def _fix_label_classes(self, seg_classes):
        """Changes the data values of classes in a segmented volume so that
        they start from zero.

        Args:
            seg_classes(list): An ascending list of the labels in the volume.
        """
        for idx, current in enumerate(seg_classes):
            self.seg_vol[self.seg_vol == current] = idx

    def output_data_slices(self, data_dir: Path, prefix: str) -> None:
        """
        Method that triggers slicing image data volume to disk in the
        xy (z), xz (y) and yz (x) planes.

        Args:
            data_dir (Path): Path to the directory for image output
            prefix (str): String to prepend to image filename
        """
        self.data_im_out_dir = data_dir
        logging.info("Slicing data volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self._output_slices_to_disk(self.data_vol, data_dir, prefix)

    def output_label_slices(self, data_dir: Path, prefix: str) -> None:
        """
        Method that triggers slicing label data volume to disk in the
        xy (z), xz (y) and yz (x) planes.

        Args:
            data_dir (Path): Path to the directory for label image output
            prefix (str): String to prepend to image filename
        """
        self.seg_im_out_dir = data_dir
        label_name = self.label_type.capitalize() if self.label_type != "segmentation" else "label"
        logging.info(f"Slicing {label_name} volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self._output_slices_to_disk(self.seg_vol, data_dir, prefix, label=True)

    def _output_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in the three orthogonal
        planes to images on disk.

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
            label (bool): Whether this is a label volume.
        """
        # Labels are always processed in 2D mode
        if label or not self.use_2_5d_slicing:
            self._output_2d_slices_to_disk(data_arr, output_path, name_prefix, label)
        else:
            self._output_2_5d_slices_to_disk(data_arr, output_path, name_prefix)

    def _output_2d_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in 2D mode (single channel).

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
            label (bool): Whether this is a label volume.
        """
        shape_tup = data_arr.shape
        axis_enum = utils.get_training_axis(self.settings)
        ax_idx_pairs = utils.get_axis_index_pairs(shape_tup, axis_enum)
        num_ims = utils.get_num_of_ims(shape_tup, axis_enum)
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path / f"{name_prefix}_{axis}_stack_{index}"
            self._output_im(
                utils.axis_index_to_slice(data_arr, axis, index), out_path, label
            )

    def _output_2_5d_slices_to_disk(self, data_arr, output_path, name_prefix):
        """Coordinates the slicing of an image volume in 2.5D mode (configurable channels).

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
        """
        shape_tup = data_arr.shape
        axis_enum = utils.get_training_axis(self.settings)
        ax_idx_pairs = utils.get_axis_index_pairs(shape_tup, axis_enum)
        num_ims = utils.get_num_of_ims(shape_tup, axis_enum)

        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path / f"{name_prefix}_{axis}_stack_{index:06d}"
            multi_channel_slice = self._create_2_5d_slice(data_arr, axis, index)
            self._output_im(multi_channel_slice, out_path, label=False, is_multi_channel=True)


    def _create_2_5d_slice(self, data_arr, axis, index):
        """Creates a 2.5D multi-channel slice from adjacent slices along the specified axis.

        Args:
            data_arr (array): The data volume to be sliced.
            axis (str): The axis along which to slice ('z', 'y', 'x').
            index (int): The slice index.

        Returns:
            array: Multi-channel image with shape (height, width, num_slices).
        """
        current_slice = utils.axis_index_to_slice(data_arr, axis, index)

        # Get depth along the specified axis
        if axis == "z":
            depth = data_arr.shape[0]
        elif axis == "y":
            depth = data_arr.shape[1]
        elif axis == "x":
            depth = data_arr.shape[2]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        center_idx = self.num_slices // 2
        slices = []

        for i in range(self.num_slices):
            slice_idx = index - center_idx + i

            # Handle border cases by duplicating edge slices
            if slice_idx < 0:
                slice_idx = 0
            elif slice_idx >= depth:
                slice_idx = depth - 1

            slice_data = utils.axis_index_to_slice(data_arr, axis, slice_idx)
            slices.append(slice_data)

        multi_channel_image = np.stack(slices, axis=2)

        return multi_channel_image

    def _output_im(self, data, path, label=False, is_multi_channel=False):
        """Converts a slice of data into an image on disk.

        Args:
            data (numpy.array): The data slice to be converted.
            path (str): The path of the image file including the filename prefix.
            label (bool): Whether to convert values >1 to 1 for binary segmentation.
            is_multi_channel (bool): Whether the data has multiple channels (2.5D).
        """
        if is_multi_channel:
            # Multi-channel data is already normalized to 0-1, convert to uint8
            if data.dtype != np.uint8:
                data = (data * 255).astype(np.uint8)

            file_extension = self.slice_file_format.lower()
            if file_extension == 'tiff':
                io.imsave(f"{path}.tiff", data, check_contrast=False)
            elif file_extension == 'png':
                # PNG only supports up to 3 channels
                io.imsave(f"{path}.png", data, check_contrast=False)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Use 'tiff' or 'png'.")
        else:
            # Single channel data
            if data.dtype != np.uint8:
                data = img_as_ubyte(data)
            io.imsave(f"{path}.png", data, check_contrast=False)

        if label and not self.multilabel:
            data[data > 1] = 1

    def _delete_image_dir(self, im_dir_path):
        if im_dir_path.exists():
            png_ims = list(im_dir_path.glob("*.png"))
            tiff_ims = list(im_dir_path.glob("*.tiff")) + list(im_dir_path.glob("*.tif"))
            all_ims = png_ims + tiff_ims

            logging.info(f"Deleting {len(png_ims)} PNG images and {len(tiff_ims)} TIFF images.")
            for im in all_ims:
                im.unlink()
            logging.info(f"Deleting the empty directory.")
            im_dir_path.rmdir()

    def clean_up_slices(self) -> None:
        """
        Deletes data and label image slices created by Slicer.
        Also cleans up task2 and task3 directories if they exist.
        """
        self._delete_image_dir(self.data_im_out_dir)
        self._delete_image_dir(self.seg_im_out_dir)
        # Clean up task2 and task3 directories if they were created
        if hasattr(self, 'task2_im_out_dir') and self.task2_im_out_dir is not None:
            self._delete_image_dir(self.task2_im_out_dir)
        if hasattr(self, 'task3_im_out_dir') and self.task3_im_out_dir is not None:
            self._delete_image_dir(self.task3_im_out_dir)
