import logging
from pathlib import Path
from typing import Tuple, Union

import segmentation_models_pytorch as smp
import torch
import volume_segmantics.utilities.base_data_utils as utils


def create_model_on_device(device: Union[int, str], model_struc_dict: dict) -> torch.nn.Module:
    if device == "cpu":
        logging.warn("Model is being loaded onto CPU.")
    struct_dict_copy = model_struc_dict.copy()
    model_type = struct_dict_copy.pop("type")
    
    if model_type == utils.ModelType.U_NET:
        model = smp.Unet(**struct_dict_copy)
        logging.info(f"Sending the U-Net model to device {device}")
    elif model_type == utils.ModelType.U_NET_PLUS_PLUS:
        model = smp.UnetPlusPlus(**struct_dict_copy)
        logging.info(f"Sending the U-Net++ model to device {device}")
    elif model_type == utils.ModelType.FPN:
        model = smp.FPN(**struct_dict_copy)
        logging.info(f"Sending the FPN model to device {device}")
    elif model_type == utils.ModelType.DEEPLABV3:
        model = smp.DeepLabV3(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3 model to device {device}")
    elif model_type == utils.ModelType.DEEPLABV3_PLUS:
        model = smp.DeepLabV3Plus(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3+ model to device {device}")
    elif model_type == utils.ModelType.MA_NET:
        model = smp.MAnet(**struct_dict_copy)
        logging.info(f"Sending the MA-Net model to device {device}")
    elif model_type == utils.ModelType.LINKNET:
        model = smp.Linknet(**struct_dict_copy)
        logging.info(f"Sending the Linknet model to device {device}")
    elif model_type == utils.ModelType.PAN:
        model = smp.PAN(**struct_dict_copy)
        logging.info(f"Sending the PAN model to device {device}")
    return model.to(device)


def create_model_from_file(
    weights_fn: Path, gpu: bool = True, device_num: int = 0,
) -> Tuple[torch.nn.Module, int, dict]:
    """Creates and returns a model and the number of segmentation labels
    that are predicted by the model."""
    if gpu:
        device_type = utils.get_available_device_type()
        if device_type == "cuda":
            map_location = f"cuda:{device_num}"
        else:
            map_location = device_type
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading model dictionary from file.")
    model_dict = torch.load(weights_fn, map_location=map_location)
    model = create_model_on_device(map_location, model_dict["model_struc_dict"])
    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict["model_state_dict"])
    return model, model_dict["model_struc_dict"]["classes"], model_dict["label_codes"]
