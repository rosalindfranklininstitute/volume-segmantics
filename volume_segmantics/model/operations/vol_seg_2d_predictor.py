import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch import nn as nn
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_prediction_dataloader
from volume_segmantics.model.model_2d import create_model_from_file
from volume_segmantics.utilities.base_data_utils import Axis
from dask import array as da


class VolSeg2dPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace, use_dask=False) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, self.model_device_num
        )
        self.model, self.num_labels, self.label_codes = model_tuple

        self.use_dask=use_dask #For multi-predictions

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def _predict_single_axis(self, data_vol, output_probs=True, axis=Axis.Z):
        '''
        Make 2D predictions using the current model along the axis specified

        Parameters:
            data_vol: data volume
            output_probs: True if want to return probabilities
            axis: Can be Axis.Z, Axis.Y or Axis.X

        Returns:
            volume data in in a tuple
            (labels, probs)
            If output_probs was set to False, then probs will return None
        '''
        output_vol_list = []
        output_prob_list = []
        data_vol = utils.rotate_array_to_axis(data_vol, axis)
        yx_dims = list(data_vol.shape[1:])
        s_max = nn.Softmax(dim=1)
        data_loader = get_2d_prediction_dataloader(data_vol, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for volume of shape {data_vol.shape}.")
        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Prediction batch", bar_format=cfg.TQDM_BAR_FORMAT
            ):
                output = self.model(batch.to(self.model_device_num))  # Forward pass
                probs = s_max(output)  # Convert the logits to probs
                # TODO: Don't flatten channels if one-hot output is needed
                labels = torch.argmax(probs, dim=1)  # flatten channels
                labels = utils.crop_tensor_to_array(labels, yx_dims)
                output_vol_list.append(labels.astype(np.uint8))
                # Collects only the probability of the label that gives maximum probability!!
                if output_probs:
                    # Get indices of max probs
                    max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
                    # Extract along axis from outputs
                    probs = torch.gather(probs, 1, max_prob_idx)
                    # Remove the label dimension
                    probs = torch.squeeze(probs, dim=1)
                    probs = utils.crop_tensor_to_array(probs, yx_dims)
                    output_prob_list.append(probs.astype(np.float16))

        labels = np.concatenate(output_vol_list)
        labels = utils.rotate_array_to_axis(labels, axis)
        probs = np.concatenate(output_prob_list) if output_prob_list else None
        if probs is not None:
            probs = utils.rotate_array_to_axis(probs, axis)
        return labels, probs


    def _predict_single_axis_all_probs(self, data_vol, axis=Axis.Z):
        '''
        Make 2D predictions using the current model along the axis specified

        Parameters:
            data_vol: data volume
            axis: Can be Axis.Z, Axis.Y or Axis.X

        Returns:
            volume data in in a tuple
            (labels, probs)
            If output_probs was set to False, then probs will return None
        '''
        output_vol_list = []
        output_prob_list = []
        data_vol = utils.rotate_array_to_axis(data_vol, axis)
        yx_dims = list(data_vol.shape[1:])
        s_max = nn.Softmax(dim=1)
        data_loader = get_2d_prediction_dataloader(data_vol, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for volume of shape {data_vol.shape}.")
        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Prediction batch", bar_format=cfg.TQDM_BAR_FORMAT
            ):
                output = self.model(batch.to(self.model_device_num))  # Forward pass
                probs = s_max(output)  # Convert the logits to probs

                labels = torch.argmax(probs, dim=1)  # flatten channels
                labels = utils.crop_tensor_to_array(labels, yx_dims)
                output_vol_list.append(labels.astype(np.uint8))
                # Collects only the probability of the label that gives maximum probability!!
                
                #By default, collect all the probabilities
                # # Get indices of max probs
                # max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
                # # Extract along axis from outputs
                # probs = torch.gather(probs, 1, max_prob_idx)
                # # Remove the label dimension
                # probs = torch.squeeze(probs, dim=1)
                #print(f"1. probs.shape:{probs.shape}")
                probs = utils.crop_tensor_to_array(probs, yx_dims)
                #print(f"2. probs.shape:{probs.shape}")
                output_prob_list.append(probs.astype(np.float16))

        labels = np.concatenate(output_vol_list)
        labels = utils.rotate_array_to_axis(labels, axis)
        probs = np.concatenate(output_prob_list) if output_prob_list else None
        #print(f"3. probs.shape:{probs.shape}")

        # Don't use rotate_array_to_axis, because probs has one extra dimension for class label
        if probs is not None:
            probs=np.transpose(probs,(0,2,3,1))
            #probs= probs.swapaxes(0,1)
            if axis == Axis.Z:
                pass
            if axis == Axis.Y:
                probs=probs.swapaxes(0, 1)
            if axis == Axis.X:
                probs=probs.swapaxes(0, 2)
            
        #print(f"4. probs.shape:{probs.shape}")

        return labels, probs
    

    # def _predict_3_ways_max_probs(self, data_vol):
    #     logging.info("_predict_3_ways_max_probs")
    #     try:
    #         return self._predict_3_ways_max_probs_cont(data_vol)
    #     except:
    #         logging.info("Failed to run prediction. Trying now with dask.")
    #         self.use_dask=True
    #         label_container0_da, prob_container0_da = self._predict_3_ways_max_probs_cont(data_vol)
    #         return label_container0_da.compute(), prob_container0_da.compute()
    
    def _predict_3_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 3 axis prediction.")
        if not self.use_dask:
            label_container = np.empty((2, *shape_tup), dtype=np.uint8)
            prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        else:
            label_container = da.empty((2, *shape_tup), dtype=np.uint8)
            prob_container = da.empty((2, *shape_tup), dtype=np.float16)

        logging.info("Predicting YX slices:")
        label_container[0], prob_container[0] = self._predict_single_axis(
            data_vol, output_probs=True
        )
        logging.info("Predicting ZX slices:")
        label_container[1], prob_container[1] = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.Y
        )
        logging.info("Merging XY and ZX volumes.")
        self._merge_vols_in_mem(prob_container, label_container)
        logging.info("Predicting ZY slices:")
        label_container[1], prob_container[1] = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.X
        )
        logging.info("Merging max of XY and ZX volumes with ZY volume.")
        self._merge_vols_in_mem(prob_container, label_container)
        if not self.use_dask:
            return label_container[0], prob_container[0]
        else:
            label_container0_da =  label_container[0]
            prob_container0_da = prob_container[0]
            return label_container0_da.compute(), prob_container0_da.compute()

    def _merge_vols_in_mem(self, prob_container, label_container):
        logging.info("_merge_vols_in_mem")
        if not self.use_dask:
            max_prob_idx = np.argmax(prob_container, axis=0)
            max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
            prob_container[0] = np.squeeze(
                np.take_along_axis(prob_container, max_prob_idx, axis=0)
            )
            label_container[0] = np.squeeze(
                np.take_along_axis(label_container, max_prob_idx, axis=0)
            )
        else:
            max_prob_idx = da.argmax(prob_container, axis=0)
            max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
            prob_container[0] = da.squeeze(
                da.take_along_axis(prob_container, max_prob_idx, axis=0)
            )
            label_container[0] = da.squeeze(
                da.take_along_axis(label_container, max_prob_idx, axis=0)
            )

    # def _predict_12_ways_max_probs(self, data_vol):
    #     logging.info("_predict_12_ways_max_probs")
    #     try:
    #         return self._predict_12_ways_max_probs_cont(data_vol)
    #     except:
    #         logging.info("Failed to run prediction. Trying now with dask.")
    #         self.use_dask=True
    #         label_container0_da, prob_container0_da = self._predict_12_ways_max_probs_cont(data_vol)
    #         return label_container0_da.compute(), prob_container0_da.compute()
        
    def _predict_12_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 12 way prediction.")
        if not self.use_dask:
            label_container = np.empty((2, *shape_tup), dtype=np.uint8)
            prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        else:
            label_container = da.empty((2, *shape_tup), dtype=np.uint8)
            prob_container = da.empty((2, *shape_tup), dtype=np.float16)
        
        label_container[0], prob_container[0] = self._predict_3_ways_max_probs(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            labels, probs = self._predict_3_ways_max_probs(data_vol)
            label_container[1] = np.rot90(labels, -k)
            prob_container[1] = np.rot90(probs, -k)
            logging.info(
                f"Merging rot {k * 90} deg volume with rot {(k-1) * 90} deg volume."
            )
            self._merge_vols_in_mem(prob_container, label_container)
        
        if not self.use_dask:
            return label_container[0], prob_container[0]
        else:
            label_container0_da =  label_container[0]
            prob_container0_da = prob_container[0]
            return label_container0_da.compute(), prob_container0_da.compute()

    def _predict_single_axis_to_one_hot(self, data_vol, axis=Axis.Z):
        prediction, _ = self._predict_single_axis(data_vol, axis=axis)
        return utils.one_hot_encode_array(prediction, self.num_labels)

    #TODO: implement dask for large volumes that fail to predict
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
