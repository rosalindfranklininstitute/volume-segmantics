from pathlib import Path
from types import SimpleNamespace
from typing import Union

import logging
import os
import numpy as np

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from volume_segmantics.data.base_data_manager import BaseDataManager
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dImageDirPredictor

import cv2

class VolSeg2DPredictionManager(BaseDataManager):
    """Class that manages prediction of data volumes to disk using a
    2d deep learning network.
    """

    def __init__(
        self,
        model_file_path: str,
        data_vol: Union[str, np.ndarray],
        settings: SimpleNamespace,
    ) -> None:
        """Inits VolSeg2DPredictionManager.

        Args:
            model_file_path (str): String of filepath to trained model to use for prediction.
            data_vol (Union[str, np.ndarray]): String of filepath to data volume or numpy array of data to predict segmentation of.
            settings (SimpleNamespace): A prediction settings object.
        """
        super().__init__(data_vol, settings)
        self.predictor = VolSeg2dPredictor(model_file_path, settings)
        self.settings = settings

    def get_label_codes(self) -> dict:
        """Returns a dictionary of label codes, retrieved from the saved model.

        Returns:
            dict: Label codes. These provide information on the labels that were used
            when training the model along with any associated metadata.
        """
        return self.predictor.label_codes

    def predict_volume_to_path(
        self, output_path: Union[Path, None], quality: Union[utils.Quality, None] = None
    ) -> np.ndarray:
        """Method which triggers prediction of a 3D segmentation to disk at a specified quality.

        Here 'quality' refers to the number of axes/rotations that the segmentation is predicted
        in. e.g. Low quality, single axis (x, y) prediction; medium quality, three axis (x, y),
        (x, z), (y, z) prediction; high quality 12 way (3 axis and 4 rotations) prediction.
        Multi-axis predictions are combined into a final output volume by using maximum probabilities.

        Args:
            output_path (Union[Path, None]): Path to predict volume to.
            quality (Union[utils.Quality, None], optional): A quality to predict the segmentation to. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        probs = None
        logits = None
        entropy = None

        one_hot = self.settings.one_hot
        output_probs = self.settings.output_probs
        output_entropy = self.settings.output_entropy
        preferred_axis = utils.get_prediction_axis(
            self.settings
        )  # Specify single axis for prediction
        if quality is None:
            quality = utils.get_prediction_quality(self.settings)
        if quality == utils.Quality.LOW:
            if one_hot:
                prediction = self.predictor._predict_single_axis_to_one_hot(
                    self.data_vol, axis=preferred_axis
                )
            else:
                prediction, probs, logits = self.predictor._predict_single_axis(
                    self.data_vol, axis=preferred_axis
                )
        if quality == utils.Quality.MEDIUM:
            if one_hot:
                prediction = self.predictor._predict_3_ways_one_hot(self.data_vol)
            elif output_probs and not output_entropy:
                prediction, probs = self.predictor._predict_3_ways_max_probs(
                    self.data_vol
                )
            elif output_entropy and not output_probs:
                prediction, _, entropy = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )
            elif output_entropy and output_probs:
                prediction, probs, entropy = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )

        if quality == utils.Quality.HIGH:
            if one_hot:
                prediction = self.predictor._predict_12_ways_one_hot(self.data_vol)
            elif output_probs and not output_entropy:
                prediction, probs = self.predictor._predict_12_ways_max_probs(
                    self.data_vol
                )
            elif output_entropy and not output_probs:
                prediction, _, entropy = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )
            elif output_entropy and output_probs:
                prediction, probs, entropy = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )

        if quality == utils.Quality.Z_ONLY:
            if one_hot:
                raise NotImplementedError("One hot for Z-ONLY not implemented.")
            elif output_probs and not output_entropy:
                prediction, probs = self.predictor._predict_12_ways_max_probs(
                    self.data_vol
                )
            elif output_entropy and not output_probs:
                prediction, _, entropy, votes = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )
            elif output_entropy and output_probs:
                prediction, probs, entropy, votes = self.predictor._prediction_estimate_entropy(
                    self.data_vol
                )

        # Get additional task outputs if multi-task model
        additional_tasks = self.predictor.get_additional_task_outputs()

        if output_path is not None and cfg.OUTPUT_FORMAT == "hdf":
            # Save primary segmentation output
            utils.save_data_to_hdf5(
                prediction, output_path, chunking=self.input_data_chunking
            )
            if probs is not None and self.settings.output_probs:
                utils.save_data_to_hdf5(
                    probs,
                    f"{output_path.parent / output_path.stem}_probs.h5",
                    chunking=self.input_data_chunking,
                )

            if logits is not None and self.settings.output_probs:
                utils.save_data_to_hdf5(
                    logits,
                    f"{output_path.parent / output_path.stem}_logits.h5",
                    chunking=self.input_data_chunking,
                )

            # Save additional task outputs
            if additional_tasks:
                for task_name, task_data in additional_tasks.items():
                    task_suffix = self._get_task_suffix(task_name)
                    task_labels = task_data.get('labels')
                    task_probs = task_data.get('probs')
                    task_logits = task_data.get('logits')

                    if task_labels is not None:
                        task_output_path = f"{output_path.parent / output_path.stem}{task_suffix}.h5"
                        utils.save_data_to_hdf5(
                            task_labels, task_output_path, chunking=self.input_data_chunking
                        )
                        logging.info(f"Saved {task_name} labels to {task_output_path}")

                    if task_probs is not None and self.settings.output_probs:
                        task_probs_path = f"{output_path.parent / output_path.stem}{task_suffix}_probs.h5"
                        utils.save_data_to_hdf5(
                            task_probs, task_probs_path, chunking=self.input_data_chunking
                        )
                        logging.info(f"Saved {task_name} probabilities to {task_probs_path}")

                    if task_logits is not None and self.settings.output_probs:
                        task_logits_path = f"{output_path.parent / output_path.stem}{task_suffix}_logits.h5"
                        utils.save_data_to_hdf5(
                            task_logits, task_logits_path, chunking=self.input_data_chunking
                        )
                        logging.info(f"Saved {task_name} logits to {task_logits_path}")

        if output_path is not None and cfg.OUTPUT_FORMAT == "tif":
            utils.save_data_to_tif(
                prediction, output_path, compress=True
            )
            if probs is not None and self.settings.output_probs:
                utils.save_data_to_tif(
                    probs,
                    f"{output_path.parent / output_path.stem}_probs.tif",
                    compress=True,
                )
            if logits is not None and self.settings.output_probs:
                utils.save_data_to_tif(
                    logits,
                    f"{output_path.parent / output_path.stem}_logits.tif",
                    compress=False,
                )

            if additional_tasks:
                for task_name, task_data in additional_tasks.items():
                    task_suffix = self._get_task_suffix(task_name)
                    task_labels = task_data.get('labels')
                    task_probs = task_data.get('probs')
                    task_logits = task_data.get('logits')

                    if task_labels is not None:
                        task_output_path = f"{output_path.parent / output_path.stem}{task_suffix}.tif"
                        utils.save_data_to_tif(
                            task_labels, task_output_path, compress=True
                        )
                        logging.info(f"Saved {task_name} labels to {task_output_path}")

                    if task_probs is not None and self.settings.output_probs:
                        task_probs_path = f"{output_path.parent / output_path.stem}{task_suffix}_probs.tif"
                        utils.save_data_to_tif(
                            task_probs, task_probs_path, compress=True
                        )
                        logging.info(f"Saved {task_name} probabilities to {task_probs_path}")

                    if task_logits is not None and self.settings.output_probs:
                        task_logits_path = f"{output_path.parent / output_path.stem}{task_suffix}_logits.tif"
                        utils.save_data_to_tif(
                            task_logits, task_logits_path, compress=False
                        )
                        logging.info(f"Saved {task_name} logits to {task_logits_path}")


            if entropy is not None and self.settings.output_entropy:
                if cfg.OUTPUT_FORMAT == "hdf":
                    utils.save_data_to_hdf5(
                        entropy,
                        f"{output_path.parent / output_path.stem}_entropy.h5",
                        chunking=self.input_data_chunking,
                    )
                    utils.save_data_to_hdf5(
                        votes,
                        f"{output_path.parent / output_path.stem}_votes.h5",
                        chunking=self.input_data_chunking,
                    )
                else:
                    utils.save_data_to_tif(
                        entropy,
                        f"{output_path.parent / output_path.stem}_entropy.tif",
                        compress=True
                    )
                    utils.save_data_to_tif(
                        votes,
                        f"{output_path.parent / output_path.stem}_votes.tif",
                        compress=True
                    )

        return prediction

    def _get_task_suffix(self, task_name):
        """
        Map task name to file suffix.

        Args:
            task_name: Task name like 'task1', 'task2', etc.

        Returns:
            str: Suffix like '_BND', '_DIST', etc.
        """
        # Map task indices to suffixes
        # Primary segmentation is saved without suffix (or with _SEG if needed)
        # task1 (first additional task, typically boundary): append _BND
        # task2 (second additional task, typically distance map): append _DIST
        task_suffix_map = {
            'task1': '_BND',  # Boundary (first additional task)
            'task2': '_DIST',  # Distance map (second additional task)
        }
        return task_suffix_map.get(task_name, f'_{task_name.upper()}')




class VolSeg2DImageDirPredictionManager():
    """Class that manages prediction of a directory of images to disk using a
    2d deep learning network.
    """

    def __init__(
        self,
        model_file_path: str,
        image_dir: str,
        settings: SimpleNamespace,
    ) -> None:
        """Inits VolSeg2DImageDirPredictionManager.

        Args:
            model_file_path (str): String of filepath to trained model to use for prediction.
            image_dir (str): String of filepath to directory containing images to predict segmentation of.
            settings (SimpleNamespace): A prediction settings object.
        """
        self.image_dir = image_dir
        self.predictor = VolSeg2dImageDirPredictor(model_file_path, settings)
        self.settings = settings

    def get_label_codes(self) -> dict:
        """Returns a dictionary of label codes, retrieved from the saved model.

        Returns:
            dict: Label codes. These provide information on the labels that were used
            when training the model along with any associated metadata.
        """
        return self.predictor.label_codes

    def predict_image_dir_to_path(
        self, output_path: Union[Path, None]) -> np.ndarray:
        """Method which triggers prediction of a 3D segmentation to disk at a specified quality.

        Here 'quality' refers to the number of axes/rotations that the segmentation is predicted
        in. e.g. Low quality, single axis (x, y) prediction; medium quality, three axis (x, y),
        (x, z), (y, z) prediction; high quality 12 way (3 axis and 4 rotations) prediction.
        Multi-axis predictions are combined into a final output volume by using maximum probabilities.

        Args:
            output_path (Union[Path, None]): Path to predict volume to.
            quality (Union[utils.Quality, None], optional): A quality to predict the segmentation to. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        probs = None
        one_hot = self.settings.one_hot

        if one_hot:
            prediction, images_fps = self.predictor._predict_image_dir_to_one_hot(
                self.image_dir
            )
        else:
            prediction, probs, images_fps = self.predictor._predict_image_dir(
                self.image_dir
            )

        if output_path is not None:
            fnames = [os.path.basename(fp) for fp in images_fps]
            save_images(fnames, prediction, output_path)

            if probs is not None and self.settings.output_probs:
                pass
                # utils.save_data_to_hdf5(
                #     probs,
                #     f"{output_path.parent / output_path.stem}_probs.h5",
                #     chunking=self.input_data_chunking,
                # )
        return prediction



def save_images(filenames, image_arrays, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over filenames and corresponding image arrays
    for filename, img_array in zip(filenames, image_arrays):
        # Construct the full path for the output image
        output_path = os.path.join(output_dir, filename)
        # Save the image array as a PNG file
        cv2.imwrite(output_path, img_array.reshape((img_array.shape[1], img_array.shape[2])))
