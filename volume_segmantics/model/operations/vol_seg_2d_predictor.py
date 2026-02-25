import logging
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

from itertools import permutations as perm

import numpy as np
from scipy.stats import entropy
import sparse

import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch import nn as nn
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from volume_segmantics.data.dataloaders import get_2d_prediction_dataloader, get_2d_image_dir_prediction_dataloader
from volume_segmantics.model.model_2d import create_model_from_file, MultitaskSegmentationModel
from volume_segmantics.utilities.base_data_utils import Axis
from volume_segmantics.data import get_settings_data
import logging
from pathlib import Path
from types import SimpleNamespace


class VolSeg2dPredictor:
    """Class that performs U-Net prediction operations with optional sliding window inference."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, device_num=self.model_device_num
        )
        self.model, self.num_labels, self.label_codes = model_tuple

        self.use_2_5d_prediction = getattr(settings, 'use_2_5d_prediction', False)

        self.use_2_5d_prediction = getattr(settings, 'use_2_5d_prediction', False)

        self.use_sliding_window = getattr(settings, 'use_sliding_window', False)
        self.sw_roi_size = getattr(settings, 'sw_roi_size', (512, 512))
        self.sw_overlap = getattr(settings, 'sw_overlap', 0.25)
        self.sw_batch_size = getattr(settings, 'sw_batch_size', 4)
        self.sw_mode = getattr(settings, 'sw_mode', 'gaussian')

        self._last_additional_tasks = None

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def get_additional_task_outputs(self):
        """
        Get additional task outputs from the last prediction (for multi-task models).

        Returns:
            dict or None: Dictionary with task outputs (keys like 'task1', 'task2', etc.)
                Each task contains 'labels', 'probs', and 'logits' arrays.
                Returns None if not a multi-task model or no additional tasks.
        """
        return getattr(self, '_last_additional_tasks', None)

    def _get_task_suffix(self, task_name):
        """
        Map task name to file suffix.

        Args:
            task_name: Task name like 'task1', 'task2', etc.

        Returns:
            str: Suffix like '_BND', '_DIST', etc.
        """
        # Map task indices to suffixes
        # Primary segmentation is saved without suffix
        task_suffix_map = {
            'task1': '_BND',  # Boundary
            'task2': '_DIST',  # Distance map
        }
        return task_suffix_map.get(task_name, f'_{task_name.upper()}')

    def _should_use_multi_axis_for_task(self, task_name):
        """
        Determine if multi-axis prediction makes sense for a given task.

        Args:
            task_name: Task name like 'task1', 'task2', etc.

        Returns:
            bool: True if multi-axis prediction is appropriate, False otherwise
        """
        # Segmentation and boundary (label images) can use multi-axis
        if task_name == 'task1':  # Segmentation
            return True
        elif task_name == 'task2':  # Boundary (label image)
            return True
        elif task_name == 'task3':  # Distance map (directional, no multi-axis)
            return False
        else:
            return False

    def _is_multitask_model(self):
        """Check if the model is a multi-task model."""
        # Check if model has num_heads property (MultitaskSegmentationModel)
        if hasattr(self.model, 'num_heads'):
            return self.model.num_heads > 1
        # Check if model is wrapped in DataParallel
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'num_heads'):
            return self.model.module.num_heads > 1
        return False

    def _get_model_outputs(self, model_output):
        """
        Extract outputs from model, handling both single-task and multi-task models.

        Args:
            model_output: Output from model forward pass

        Returns:
            tuple: (primary_output, additional_outputs_dict)
                - primary_output: The first output (segmentation) for backward compatibility
                - additional_outputs_dict: Dict with keys like 'task1', 'task2', etc. for additional tasks
        """
        if isinstance(model_output, (list, tuple)) and len(model_output) > 1:
            # Multi-task model - return all outputs
            primary_output = model_output[0]
            additional_outputs = {}
            for idx, output in enumerate(model_output[1:], start=1):
                additional_outputs[f'task{idx}'] = output
            return primary_output, additional_outputs
        elif isinstance(model_output, (list, tuple)):
            # Single output in tuple/list
            return model_output[0], {}
        else:
            # Single output tensor
            return model_output, {}

    def _apply_2_5d_padding(self, data_vol):
        """Apply padding to data volume for 2.5D prediction to reduce border artifacts."""
        if not self.use_2_5d_prediction:
            return data_vol, None

        num_slices = getattr(self.settings, 'num_slices', 3)
        padding_factor = getattr(self.settings, 'prediction_padding_factor', 1.0)
        pad_width = int((num_slices // 2) * padding_factor)

        padded_vol = np.pad(data_vol, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
        logging.info(f"Padded volume from {data_vol.shape} to {padded_vol.shape} for 2.5D prediction")

        return padded_vol, pad_width

    def _crop_2_5d_output(self, output_array, pad_width):
        """Crop output array back to original size after 2.5D padding."""
        if not self.use_2_5d_prediction or pad_width is None:
            return output_array

        cropped = output_array[pad_width:pad_width + output_array.shape[0] - 2 * pad_width]
        logging.info(f"Cropped output from {output_array.shape} back to {cropped.shape}")

        return cropped

    def _predict_slice_sliding_window(self, slice_tensor, yx_dims):
        """
        Predict a single slice using sliding window inference.

        Args:
            slice_tensor: Input tensor [B, C, H, W]
            yx_dims: Original (H, W) dimensions for output cropping

        Returns:
            labels, probs, logits for this slice (primary segmentation)
            If multi-task model, also returns additional_task_outputs dict
        """
        B, C, H, W = slice_tensor.shape
        is_multitask = self._is_multitask_model()
        additional_outputs = {}

        # Check if sliding window is needed
        roi_h, roi_w = self.sw_roi_size
        use_sw = self.use_sliding_window and (H > roi_h or W > roi_w)

        if use_sw:
            def predictor(x):
                model_output = self.model(x)
                primary_output, _ = self._get_model_outputs(model_output)
                return primary_output

            output = sliding_window_inference(
                inputs=slice_tensor,
                roi_size=self.sw_roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=predictor,
                overlap=self.sw_overlap,
                mode=self.sw_mode,
                device=self.model_device_num
            )
            # For sliding window, we only get primary output from sliding window
            # Get additional outputs separately if needed (using full forward pass)
            if is_multitask:
                with torch.no_grad():
                    full_output = self.model(slice_tensor)
                    _, additional_outputs = self._get_model_outputs(full_output)
        else:
            model_output = self.model(slice_tensor)
            output, additional_outputs = self._get_model_outputs(model_output)

        # Process primary segmentation output
        s_max = nn.Softmax(dim=1)
        probs = s_max(output)
        labels = torch.argmax(probs, dim=1)

        # Crop to original dimensions
        labels_np = utils.crop_tensor_to_array(labels, yx_dims).astype(np.uint8)

        # Get max probs
        max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
        max_probs = torch.gather(probs, 1, max_prob_idx).squeeze(1)
        probs_np = utils.crop_tensor_to_array(probs, yx_dims).astype(np.float16)

        logits_np = utils.crop_tensor_to_array(output, yx_dims)

        # Process additional task outputs if multi-task model
        additional_task_outputs = {}
        if is_multitask and additional_outputs:
            for task_name, task_output in additional_outputs.items():
                # For binary tasks (boundary, etc.), apply sigmoid
                if task_output.shape[1] == 1:
                    task_probs = torch.sigmoid(task_output)
                    task_labels = (task_probs > 0.5).squeeze(1).long()
                else:
                    # For multi-class tasks, use softmax
                    task_probs = s_max(task_output)
                    task_labels = torch.argmax(task_probs, dim=1)

                task_labels_np = utils.crop_tensor_to_array(task_labels, yx_dims).astype(np.uint8)
                if task_output.shape[1] == 1:
                    task_probs_np = utils.crop_tensor_to_array(task_probs.squeeze(1), yx_dims).astype(np.float16)
                else:
                    max_task_prob_idx = torch.argmax(task_probs, dim=1, keepdim=True)
                    max_task_probs = torch.gather(task_probs, 1, max_task_prob_idx).squeeze(1)
                    task_probs_np = utils.crop_tensor_to_array(max_task_probs, yx_dims).astype(np.float16)

                task_logits_np = utils.crop_tensor_to_array(task_output, yx_dims)

                additional_task_outputs[task_name] = {
                    'labels': task_labels_np,
                    'probs': task_probs_np,
                    'logits': task_logits_np
                }

        if additional_task_outputs:
            return labels_np, probs_np, logits_np, additional_task_outputs
        return labels_np, probs_np, logits_np

    def _predict_single_axis(self, data_vol, output_probs=True, axis=Axis.Z):
        output_vol_list = []
        output_prob_list = []
        output_logits_list = []
        is_multitask = self._is_multitask_model()

        # For multi-task models, collect additional task outputs
        additional_task_vols = {} if is_multitask else None
        additional_task_probs = {} if (is_multitask and output_probs) else None
        additional_task_logits = {} if (is_multitask and output_probs) else None

        data_vol = utils.rotate_array_to_axis(data_vol, axis)

        # Apply padding for 2.5D prediction
        data_vol, pad_width = self._apply_2_5d_padding(data_vol)
        yx_dims = list(data_vol.shape[1:])

        data_loader = get_2d_prediction_dataloader(data_vol, self.settings)

        self.model.eval()
        logging.info(f"Predicting segmentation for volume of shape {data_vol.shape}.")
        if is_multitask:
            logging.info(f"Multi-task model detected with {self.model.num_heads if hasattr(self.model, 'num_heads') else 'unknown'} heads")
        if self.use_sliding_window:
            logging.info(f"Using sliding window: roi={self.sw_roi_size}, overlap={self.sw_overlap}")

        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Prediction batch", bar_format=cfg.TQDM_BAR_FORMAT
            ):
                batch = batch.to(torch.float32).to(self.model_device_num)

                result = self._predict_slice_sliding_window(batch, yx_dims)

                if is_multitask and len(result) == 4:
                    labels, probs, logits, additional_outputs = result
                    # Collect additional task outputs
                    for task_name, task_data in additional_outputs.items():
                        if task_name not in additional_task_vols:
                            additional_task_vols[task_name] = []
                            if output_probs:
                                additional_task_probs[task_name] = []
                                additional_task_logits[task_name] = []
                        additional_task_vols[task_name].append(task_data['labels'])
                        if output_probs:
                            additional_task_probs[task_name].append(task_data['probs'])
                            additional_task_logits[task_name].append(task_data['logits'])
                else:
                    labels, probs, logits = result

                output_vol_list.append(labels)
                if output_probs:
                    probs = utils.crop_tensor_to_array(probs, yx_dims)
                    output_prob_list.append(probs.astype(np.float16))
                    logits = utils.crop_tensor_to_array(logits, yx_dims)
                    output_logits_list.append(logits)

        labels = np.concatenate(output_vol_list)
        labels = self._crop_2_5d_output(labels, pad_width)
        labels = utils.rotate_array_to_axis(labels, axis)

        probs = np.concatenate(output_prob_list) if output_prob_list else None
        logits = np.concatenate(output_logits_list) if output_logits_list else None

        if probs is not None:
            probs = self._crop_2_5d_output(probs, pad_width)
            probs = utils.rotate_array_to_axis(probs, axis)
            probs = np.moveaxis(probs, 1, -1)

        if logits is not None:
            logits = self._crop_2_5d_output(logits, pad_width)
            logits = utils.rotate_array_to_axis(logits, axis)

        if is_multitask and additional_task_vols:
            processed_additional_tasks = {}
            for task_name in additional_task_vols:
                task_labels = np.concatenate(additional_task_vols[task_name])
                task_labels = self._crop_2_5d_output(task_labels, pad_width)
                task_labels = utils.rotate_array_to_axis(task_labels, axis)

                task_probs = None
                task_logits = None
                if output_probs and task_name in additional_task_probs:
                    task_probs = np.concatenate(additional_task_probs[task_name])
                    task_probs = self._crop_2_5d_output(task_probs, pad_width)
                    task_probs = utils.rotate_array_to_axis(task_probs, axis)

                    task_logits = np.concatenate(additional_task_logits[task_name])
                    task_logits = self._crop_2_5d_output(task_logits, pad_width)
                    task_logits = utils.rotate_array_to_axis(task_logits, axis)

                processed_additional_tasks[task_name] = {
                    'labels': task_labels,
                    'probs': task_probs,
                    'logits': task_logits
                }

            self._last_additional_tasks = processed_additional_tasks

        return labels, probs, logits

    def _predict_3_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        is_multitask = self._is_multitask_model()
        logging.info("Creating empty data volumes in RAM to combine 3 axis prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        logging.info("Predicting YX slices:")
        label_container[0], prob_container[0], _ = self._predict_single_axis(
            data_vol, output_probs=True
        )
        task_outputs_axis0 = self.get_additional_task_outputs() if is_multitask else None

        logging.info("Predicting ZX slices:")
        label_container[1], prob_container[1], _ = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.Y
        )
        task_outputs_axis1 = self.get_additional_task_outputs() if is_multitask else None

        logging.info("Merging XY and ZX volumes.")
        self._merge_vols_in_mem(prob_container, label_container)

        logging.info("Predicting ZY slices:")
        label_container[1], prob_container[1], _ = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.X
        )
        task_outputs_axis2 = self.get_additional_task_outputs() if is_multitask else None

        logging.info("Merging max of XY and ZX volumes with ZY volume.")
        self._merge_vols_in_mem(prob_container, label_container)

        additional_tasks_3way = {}
        if is_multitask and task_outputs_axis0:
            for task_name in task_outputs_axis0.keys():
                if self._should_use_multi_axis_for_task(task_name):
                    logging.info(f"Merging {task_name} predictions from 3 axes:")
                    task_label_container = np.empty((2, *shape_tup), dtype=np.uint8)
                    task_prob_container = np.empty((2, *shape_tup), dtype=np.float16)

                    task_label_container[0] = task_outputs_axis0[task_name]['labels']
                    task_prob_container[0] = task_outputs_axis0[task_name]['probs']
                    task_label_container[1] = task_outputs_axis1[task_name]['labels']
                    task_prob_container[1] = task_outputs_axis1[task_name]['probs']
                    self._merge_vols_in_mem(task_prob_container, task_label_container)

                    task_label_container[1] = task_outputs_axis2[task_name]['labels']
                    task_prob_container[1] = task_outputs_axis2[task_name]['probs']
                    self._merge_vols_in_mem(task_prob_container, task_label_container)

                    additional_tasks_3way[task_name] = {
                        'labels': task_label_container[0],
                        'probs': task_prob_container[0]
                    }
                else:
                    # Single axis only (e.g., distance maps) - use first axis
                    logging.info(f"Using single-axis prediction for {task_name} (multi-axis not applicable):")
                    additional_tasks_3way[task_name] = {
                        'labels': task_outputs_axis0[task_name]['labels'],
                        'probs': task_outputs_axis0[task_name]['probs']
                    }
            self._last_additional_tasks = additional_tasks_3way

        return label_container[0], prob_container[0]

    def _merge_vols_in_mem(self, prob_container, label_container):
        max_prob_idx = np.argmax(prob_container, axis=0)
        max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
        prob_container[0] = np.squeeze(
            np.take_along_axis(prob_container, max_prob_idx, axis=0)
        )
        label_container[0] = np.squeeze(
            np.take_along_axis(label_container, max_prob_idx, axis=0)
        )

    def _predict_12_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        is_multitask = self._is_multitask_model()
        logging.info("Creating empty data volumes in RAM to combine 12 way prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        label_container[0], prob_container[0] = self._predict_3_ways_max_probs(data_vol)
        task_outputs_3way = self.get_additional_task_outputs() if is_multitask else None

        task_containers = {}
        if is_multitask and task_outputs_3way:
            for task_name in task_outputs_3way.keys():
                if self._should_use_multi_axis_for_task(task_name):
                    task_containers[task_name] = {
                        'label_container': np.empty((2, *shape_tup), dtype=np.uint8),
                        'prob_container': np.empty((2, *shape_tup), dtype=np.float16)
                    }
                    task_containers[task_name]['label_container'][0] = task_outputs_3way[task_name]['labels']
                    task_containers[task_name]['prob_container'][0] = task_outputs_3way[task_name]['probs']

        original_data_vol = data_vol.copy() if is_multitask and task_outputs_3way else None

        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            labels, probs = self._predict_3_ways_max_probs(data_vol)
            label_container[1] = np.rot90(labels, -k)
            prob_container[1] = np.rot90(probs, -k)
            logging.info(f"Merging rot {k * 90} deg volume with rot {(k-1) * 90} deg volume.")
            self._merge_vols_in_mem(prob_container, label_container)

            # Merge task outputs if multi-task and task supports 12-way
            if is_multitask and task_outputs_3way:
                current_task_outputs = self.get_additional_task_outputs()
                if current_task_outputs:
                    for task_name in task_outputs_3way.keys():
                        if self._should_use_multi_axis_for_task(task_name) and task_name in task_containers:
                            # Merge rotated task outputs
                            task_containers[task_name]['label_container'][1] = np.rot90(
                                current_task_outputs[task_name]['labels'], -k
                            )
                            task_containers[task_name]['prob_container'][1] = np.rot90(
                                current_task_outputs[task_name]['probs'], -k
                            )
                            self._merge_vols_in_mem(
                                task_containers[task_name]['prob_container'],
                                task_containers[task_name]['label_container']
                            )

        if is_multitask and task_outputs_3way:
            final_task_outputs = {}
            for task_name in task_outputs_3way.keys():
                if self._should_use_multi_axis_for_task(task_name) and task_name in task_containers:
                    final_task_outputs[task_name] = {
                        'labels': task_containers[task_name]['label_container'][0],
                        'probs': task_containers[task_name]['prob_container'][0]
                    }
                else:
                    # Single axis only - use first 3-way result
                    final_task_outputs[task_name] = {
                        'labels': task_outputs_3way[task_name]['labels'],
                        'probs': task_outputs_3way[task_name]['probs']
                    }
            self._last_additional_tasks = final_task_outputs

        return label_container[0], prob_container[0]

    def _predict_Zonly_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 12 way prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        label_container[0], prob_container[0] = self._predict_single_axis(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            labels, probs = self._predict_single_axis(data_vol)
            label_container[1] = np.rot90(labels, -k)
            prob_container[1] = np.rot90(probs, -k)
            logging.info(
                f"Merging rot {k * 90} deg volume with rot {(k-1) * 90} deg volume."
            )
            self._merge_vols_in_mem(prob_container, label_container)
        return label_container[0], prob_container[0]

    def _predict_single_axis_to_one_hot(self, data_vol, axis=Axis.Z):
        prediction, _, _ = self._predict_single_axis(data_vol, axis=axis)
        return utils.one_hot_encode_array(prediction, self.num_labels)

    def _predict_3_ways_one_hot(self, data_vol):
        one_hot_out = self._predict_single_axis_to_one_hot(data_vol)
        one_hot_out += self._predict_single_axis_to_one_hot(data_vol, Axis.Y)
        one_hot_out += self._predict_single_axis_to_one_hot(data_vol, Axis.X)
        return one_hot_out

    def _predict_12_ways_one_hot(self, data_vol):
        one_hot_out = self._predict_3_ways_one_hot(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            one_hot_out += np.rot90(
                self._predict_3_ways_one_hot(data_vol), -k, axes=(-3, -2)
            )
        return one_hot_out

    def _predict_3_axis_generator(self, data_vol):
        for curr_axis in [Axis.Z, Axis.Y, Axis.X]:
            labels, _, _ = self._predict_single_axis(data_vol, output_probs=False, axis=curr_axis)
            yield labels

    def _predict_12_ways_generator(self, data_vol):
        for curr_axis in [Axis.Z, Axis.Y, Axis.X]:
            rotation_axes = {
                Axis.Z: (1, 2),
                Axis.X: (0, 2),
                Axis.Y: (0, 1)
            }
            for k in range(4):
                labels, probs, _ = self._predict_single_axis(np.ascontiguousarray(np.rot90(data_vol, k, axes=rotation_axes[curr_axis])), output_probs=False, axis=curr_axis)
                yield np.rot90(labels, -k, axes=rotation_axes[curr_axis])

    def _predict_Zonly_generator(self, data_vol):
            for k in range(4):
                no_flip = np.ascontiguousarray(np.rot90(data_vol, k, axes=(1, 2)))
                for flip_axis in range(-1, 3):
                    # No flips
                    if flip_axis == -1:
                        labels, probs, _ = self._predict_single_axis(data_vol=no_flip, output_probs=True, axis=Axis.Z)
                        yield np.rot90(labels, -k, axes=(1, 2)), np.rot90(probs, -k, axes=(1, 2))
                    else:
                        labels, probs, _ = self._predict_single_axis(data_vol=np.flip(no_flip, flip_axis), output_probs=True, axis=Axis.Z)
                        yield np.rot90(np.flip(labels, flip_axis), -k, axes=(1, 2)), np.rot90(np.flip(probs, flip_axis), -k, axes=(1, 2))


    def _convert_labels_map_to_count(self, labels_vol):
        volume_size = labels_vol.shape
        logging.info(f"Label volume shape = {volume_size}")

        label_vol_contig = np.ascontiguousarray(labels_vol)
        label_counts = np.bincount(label_vol_contig.ravel())
        label_unique = np.nonzero(label_counts)[0]
        label_sorted = label_unique[np.argsort([i for i in label_counts if i>0])]

        logging.info(f"Unique labels: {label_sorted}")
        logging.info(f"Label counts: {np.sort([i for i in label_counts if i>0])}")

        label_flattened = labels_vol.flatten()
        counts_matrix = np.zeros((len(label_sorted), *volume_size), dtype=np.uint8)
        for idx, curr_label in tqdm(enumerate(label_sorted[:-1]), total=len(label_sorted[:-1])):
            np.put(counts_matrix[idx],
                   np.argwhere(label_flattened==curr_label),
                   1)
        counts_matrix[-1] = 1 - np.any(counts_matrix, axis=0)

        return counts_matrix, label_sorted

    def _prediction_estimate_entropy(self, data_vol):
        if (self.settings.quality not in ["medium", "high", "z_only"]):
            raise ValueError("Error in vol_seg_2d_predictor._prediction_estimate_entropy: Entropy calculation must be done with a minimum prediction quality of medium.")

        logging.info("Collecting voting distributions:")
        counts_matrix = np.zeros((self.num_labels, *data_vol.shape), dtype=np.uint8)
        probs_matrix = np.zeros((*data_vol.shape, self.num_labels))
        if self.settings.quality=="medium":
            g = self._predict_3_ways_generator(data_vol)
            curr_counts, labels_list = self._convert_labels_map_to_count(data_vol)
            for i in range(3):
                logging.info(f"Voter {i+1} of 3 voting...")
                labels = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    probs_matrix[labels_list[idx]] += curr_label

        elif self.settings.quality=="medium":
            g = self._predict_12_ways_generator(data_vol)
            for i in range(12):
                logging.info(f"Voter {i+1} of 12 voting...")
                labels = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    probs_matrix[labels_list[idx]] += curr_label

        elif self.settings.quality=="z_only":
            g = self._predict_Zonly_generator(data_vol)
            for i in range(16):
                logging.info(f"Voter {i+1} of 16 voting...")
                labels, probs = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    counts_matrix[labels_list[idx]] += curr_label
                probs_matrix += probs
            probs_matrix = np.moveaxis(probs_matrix, -1, 0)

        logging.info("Aggregating prediction votes:")
        counts_matrix_contig = np.ascontiguousarray(counts_matrix)

        full_prediction_labels = np.argmax(probs_matrix, axis=0)
        full_prediction_probs = np.squeeze(
            np.take_along_axis(counts_matrix_contig, full_prediction_labels[np.newaxis, ...], axis=0)
        )

        if self.settings.quality=="medium":
            full_prediction_probs = full_prediction_probs.astype(float) / 3
        elif self.settings.quality=="z_only":
            full_prediction_probs = probs_matrix * .0625
        logging.info("Calculating prediction entropy (regularised) from voting distributions:")
        entropy_matrix = np.empty(data_vol.shape)
        for curr_slice in range(len(data_vol)):
            entropy_matrix[curr_slice] = entropy(counts_matrix_contig[:, curr_slice, ...], axis=0)
        entropy_matrix /= entropy(np.full((len(np.unique(full_prediction_labels)),),
                                          1/len(np.unique(full_prediction_labels))))

        return full_prediction_labels, full_prediction_probs, entropy_matrix, counts_matrix_contig


class VolSeg2dImageDirPredictor:
    """Class that performs U-Net prediction operations on image directories."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, device_num=self.model_device_num, settings=settings
        )
        self.model, self.num_labels, self.label_codes = model_tuple

        self._last_additional_tasks = None

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def get_additional_task_outputs(self):
        """
        Get additional task outputs from the last prediction (for multi-task models).

        Returns:
            dict or None: Dictionary with task outputs (keys like 'task1', 'task2', etc.)
                Each task contains 'labels' and optionally 'probs' arrays.
                Returns None if not a multi-task model or no additional tasks.
        """
        return getattr(self, '_last_additional_tasks', None)

    def _is_multitask_model(self):
        """Check if the model is a multi-task model."""
        if hasattr(self.model, 'num_heads'):
            return self.model.num_heads > 1
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'num_heads'):
            return self.model.module.num_heads > 1
        return False

    def _get_model_outputs(self, model_output):
        """
        Extract outputs from model, handling both single-task and multi-task models.

        Args:
            model_output: Output from model forward pass

        Returns:
            tuple: (primary_output, additional_outputs_dict)
                - primary_output: The first output (segmentation) for backward compatibility
                - additional_outputs_dict: Dict with keys like 'task1', 'task2', etc. for additional tasks
        """
        if isinstance(model_output, (list, tuple)) and len(model_output) > 1:
            primary_output = model_output[0]
            additional_outputs = {}
            for idx, output in enumerate(model_output[1:], start=1):
                additional_outputs[f'task{idx}'] = output
            return primary_output, additional_outputs
        elif isinstance(model_output, (list, tuple)):
            # Single output in tuple/list
            return model_output[0], {}
        else:
            # Single output tensor
            return model_output, {}

    def _predict_image_dir(self, image_dir, output_probs=False):
        output_vol_list = []
        output_prob_list = []
        is_multitask = self._is_multitask_model()

        additional_task_vols = {} if is_multitask else None
        additional_task_probs = {} if (is_multitask and output_probs) else None

        #data_vol = utils.rotate_array_to_axis(data_vol, axis)
        #yx_dims = list(data_vol.shape[1:])
        print(self.settings)
        yx_dims = (self.settings.output_size, self.settings.output_size)

        s_max = nn.Softmax(dim=1)
        data_loader, images_fps = get_2d_image_dir_prediction_dataloader(image_dir, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for image dir.")

        if is_multitask:
            logging.info(f"Multi-task model detected with {self.model.num_heads if hasattr(self.model, 'num_heads') else 'unknown'} heads")

        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Prediction batch", bar_format=cfg.TQDM_BAR_FORMAT
            ):
                model_output = self.model(batch.to(self.model_device_num))  # Forward pass
                output, additional_outputs = self._get_model_outputs(model_output)

                probs = s_max(output)  # Convert the logits to probs
                # TODO: Don't flatten channels if one-hot output is needed
                labels = torch.argmax(probs, dim=1)  # flatten channels
                labels = utils.crop_tensor_to_array(labels, yx_dims)
                output_vol_list.append(labels.astype(np.uint8))
                if output_probs:
                    # Get indices of max probs
                    max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
                    # Extract along axis from outputs
                    probs = torch.gather(probs, 1, max_prob_idx)
                    # Remove the label dimension
                    probs = torch.squeeze(probs, dim=1)
                    probs = utils.crop_tensor_to_array(probs, yx_dims)
                    output_prob_list.append(probs.astype(np.float16))


                if is_multitask and additional_outputs:
                    for task_name, task_output in additional_outputs.items():
                        if task_name not in additional_task_vols:
                            additional_task_vols[task_name] = []
                            if output_probs:
                                additional_task_probs[task_name] = []

                        if task_output.shape[1] == 1:
                            task_probs = torch.sigmoid(task_output)
                            task_labels = (task_probs > 0.5).squeeze(1).long()
                        else:
                            task_probs = s_max(task_output)
                            task_labels = torch.argmax(task_probs, dim=1)

                        task_labels = utils.crop_tensor_to_array(task_labels, yx_dims)
                        additional_task_vols[task_name].append(task_labels.astype(np.uint8))

                        if output_probs:
                            if task_output.shape[1] == 1:
                                task_probs_np = utils.crop_tensor_to_array(task_probs.squeeze(1), yx_dims)
                            else:
                                max_task_prob_idx = torch.argmax(task_probs, dim=1, keepdim=True)
                                task_probs_np = utils.crop_tensor_to_array(
                                    torch.gather(task_probs, 1, max_task_prob_idx).squeeze(1),
                                    yx_dims
                                )
                            additional_task_probs[task_name].append(task_probs_np.astype(np.float16))

        processed_additional_tasks = None
        if is_multitask and additional_task_vols:
            processed_additional_tasks = {}
            for task_name in additional_task_vols:
                task_data = {
                    'labels': additional_task_vols[task_name],
                    'probs': additional_task_probs.get(task_name) if output_probs else None
                }
                processed_additional_tasks[task_name] = task_data

        if processed_additional_tasks:
            self._last_additional_tasks = processed_additional_tasks

        return output_vol_list, output_prob_list, images_fps

    # TODO FIX
    def _predict_image_dir_to_one_hot(self, data_vol):
        prediction, _, images_fps = self._predict_single_axis(data_vol)
        return utils.one_hot_encode_array(prediction, self.num_labels), images_fps
