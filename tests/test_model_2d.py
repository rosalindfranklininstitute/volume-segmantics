from pathlib import Path

import pytest
import torch
from torch.nn import DataParallel

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.model_2d import (
    create_model_from_file,
    create_model_from_file_full_weights,
    create_model_on_device,
)


# --- Model type lists for parametrization ---
SMP_MODEL_TYPES = [
    utils.ModelType.U_NET,
    utils.ModelType.U_NET_PLUS_PLUS,
    utils.ModelType.FPN,
    utils.ModelType.DEEPLABV3,
    utils.ModelType.DEEPLABV3_PLUS,
    utils.ModelType.MA_NET,
    utils.ModelType.LINKNET,
    utils.ModelType.PAN,
]

ALL_SMP_MODEL_TYPES = SMP_MODEL_TYPES + [utils.ModelType.SEGFORMER]

ALL_MODEL_TYPES = ALL_SMP_MODEL_TYPES + [
    utils.ModelType.VANILLA_UNET,
    utils.ModelType.MULTITASK_UNET,
]


@pytest.fixture
def model_struc_resnet34(binary_model_struc_dict):
    """Model structure dict with resnet34 encoder for broad SMP compatibility."""
    d = binary_model_struc_dict.copy()
    d["encoder_name"] = "resnet34"
    d["encoder_weights"] = "imagenet"
    return d


@pytest.fixture
def model_struc_segformer():
    """Minimal structure for Segformer (uses mit_b0 encoder supported by smp)."""
    return {
        "type": utils.ModelType.SEGFORMER,
        "encoder_name": "mit_b0",
        "encoder_weights": "imagenet",
        "in_channels": cfg.MODEL_INPUT_CHANNELS,
        "classes": 2,
    }


@pytest.fixture
def model_struc_vanilla_unet():
    """Minimal structure for Vanilla U-Net (no encoder)."""
    return {
        "type": utils.ModelType.VANILLA_UNET,
        "in_channels": cfg.MODEL_INPUT_CHANNELS,
        "classes": 2,
    }


@pytest.fixture
def model_struc_multitask_unet():
    """Minimal structure for Multitask U-Net."""
    return {
        "type": utils.ModelType.MULTITASK_UNET,
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": cfg.MODEL_INPUT_CHANNELS,
        "classes": 2,
    }


# --- create_model_on_device: all model types ---


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", SMP_MODEL_TYPES)
def test_create_model_on_device(binary_model_struc_dict, model_type):
    """Create each SMP model type using default YAML encoder."""
    binary_model_struc_dict["type"] = model_type
    model = create_model_on_device(0, binary_model_struc_dict)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", ALL_SMP_MODEL_TYPES)
def test_create_model_on_device_resnet34(model_struc_resnet34, model_type):
    """Create each encoder-based model with resnet34 for compatibility."""
    model_struc_resnet34["type"] = model_type
    model = create_model_on_device(0, model_struc_resnet34)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
def test_create_model_on_device_segformer(model_struc_segformer):
    """Create Segformer with mit_b0 encoder."""
    model = create_model_on_device(0, model_struc_segformer)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
def test_create_model_on_device_vanilla_unet(model_struc_vanilla_unet):
    """Create Vanilla U-Net (no pretrained encoder)."""
    model = create_model_on_device(0, model_struc_vanilla_unet)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
def test_create_model_on_device_multitask_unet(model_struc_multitask_unet):
    """Create Multitask U-Net with single head."""
    model = create_model_on_device(0, model_struc_multitask_unet)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "encoder_type",
    [
        "resnet34",
        "resnet50",
        "resnext50_32x4d",
        "efficientnet-b3",
        "efficientnet-b4",
        "timm-resnest50d",
        "timm-resnest101e",
    ],
)
def test_create_model_on_device_encoders(binary_model_struc_dict, encoder_type):
    binary_model_struc_dict["encoder_name"] = encoder_type
    model = create_model_on_device(0, binary_model_struc_dict)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0


# --- create_model_from_file ---


@pytest.mark.gpu
def test_create_model_from_file(model_path):
    """Load model from volseg .pytorch checkpoint; check model, device, classes, codes."""
    model, classes, codes = create_model_from_file(model_path)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    assert device.index == 0
    assert isinstance(classes, int)
    assert isinstance(codes, dict)


@pytest.mark.gpu
def test_create_model_from_file_returns_label_codes(model_path):
    """create_model_from_file returns (model, num_classes, label_codes)."""
    model, classes, codes = create_model_from_file(model_path)
    assert classes == 4  # from model_path fixture
    assert codes == {}   # fixture saves empty label_codes


@pytest.mark.gpu
def test_create_model_from_file_full_weights(tmp_path, model_struc_vanilla_unet):
    """create_model_from_file_full_weights loads raw state_dict and places model on device."""
    model = create_model_on_device(0, model_struc_vanilla_unet)
    path = tmp_path / "weights_only.pytorch"
    torch.save(model.state_dict(), path)
    loaded, num_classes, label_codes = create_model_from_file_full_weights(
        path, model_struc_vanilla_unet, device_num=0, gpu=True
    )
    assert isinstance(loaded, torch.nn.Module)
    assert next(loaded.parameters()).device.type == "cuda"


# --- Forward pass and failure cases ---


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_type,encoder_name",
    [
        (utils.ModelType.U_NET, "resnet34"),
        (utils.ModelType.VANILLA_UNET, None),
    ],
)
def test_model_forward_output_shape(model_type, encoder_name):
    """Forward pass produces correct output shape (B, classes, H, W)."""
    if encoder_name:
        struct = {
            "type": model_type,
            "encoder_name": encoder_name,
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "classes": 3,
        }
    else:
        struct = {
            "type": model_type,
            "in_channels": 1,
            "classes": 3,
        }
    model = create_model_on_device(0, struct)
    model.eval()
    x = torch.rand(2, 1, 64, 64, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 3, 64, 64)


@pytest.mark.gpu
def test_multitask_unet_forward_output_shape(model_struc_multitask_unet):
    """Multitask U-Net forward returns tuple of masks with correct shape."""
    model = create_model_on_device(0, model_struc_multitask_unet)
    model.eval()
    in_channels = model_struc_multitask_unet["in_channels"]
    x = torch.rand(1, in_channels, 64, 64, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple)
    assert len(out) >= 1
    assert out[0].shape == (1, 2, 64, 64)


def test_create_model_unknown_type_raises(binary_model_struc_dict):
    """Passing an invalid model type leaves model undefined and raises."""
    struct = binary_model_struc_dict.copy()
    struct["type"] = 999  # not a valid ModelType
    with pytest.raises((NameError, ValueError, TypeError)):
        create_model_on_device(0, struct)


def test_create_model_on_device_cpu(model_struc_vanilla_unet):
    """create_model_on_device with device 'cpu' places model on CPU (no GPU required)."""
    model = create_model_on_device("cpu", model_struc_vanilla_unet)
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    assert device.type == "cpu"


def test_create_model_from_file_cpu(tmp_path, model_struc_vanilla_unet):
    """create_model_from_file with gpu=False loads model on CPU."""
    model = create_model_on_device("cpu", model_struc_vanilla_unet)
    path = tmp_path / "cpu_model.pytorch"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_struc_dict": model_struc_vanilla_unet,
            "label_codes": {},
        },
        path,
    )
    loaded, classes, codes = create_model_from_file(path, gpu=False, device_num=0)
    assert isinstance(loaded, torch.nn.Module)
    assert next(loaded.parameters()).device.type == "cpu"
    assert classes == model_struc_vanilla_unet["classes"]
    assert isinstance(codes, dict)


@pytest.mark.gpu
def test_create_model_on_device_use_all_gpus_wraps_data_parallel(
    model_struc_vanilla_unet, monkeypatch
):
    """When device_count > 1 and USE_ALL_GPUS, create_model_on_device returns DataParallel model.
    Skips on single-GPU machines because DataParallel would touch device 1."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2+ GPUs to test DataParallel wrapping")
    import volume_segmantics.utilities.config as cfg

    monkeypatch.setattr(cfg, "USE_ALL_GPUS", True)
    model = create_model_on_device(0, model_struc_vanilla_unet)
    assert isinstance(model, DataParallel)
