"""The encoder that was asked for is the encoder that is built and saved.

"""

import pytest
import segmentation_models_pytorch as smp
import torch

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model import model_2d
from volume_segmantics.model.model_2d import (
    check_checkpoint_encoder_family,
    create_model_from_file,
    create_model_from_file_full_weights,
    create_model_on_device,
    verify_built_encoder,
)

MT = utils.ModelType


def _resnet(in_channels=1):
    return smp.Unet("resnet18", encoder_weights=None, in_channels=in_channels, classes=2)


def _struct(encoder_name, **over):
    d = {"type": MT.U_NET, "encoder_name": encoder_name, "encoder_weights": None,
         "in_channels": 1, "classes": 2, "encoder_depth": 5}
    d.update(over)
    return d


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    """None of these tests may reach torch.hub: a refusal must come first."""
    def _forbidden(*a, **k):
        raise AssertionError("torch.hub.load was called")
    monkeypatch.setattr(torch.hub, "load", _forbidden)


# --- verify_built_encoder


def test_a_cnn_built_for_a_dino_name_is_refused():
    with pytest.raises(RuntimeError, match="ResNetEncoder"):
        verify_built_encoder(_resnet(), "dinov2_vits14")


def test_a_matching_cnn_passes_and_a_model_without_an_encoder_is_ignored():
    verify_built_encoder(_resnet(), "resnet18")
    verify_built_encoder(torch.nn.Linear(2, 2), "dinov2_vits14")


def test_a_dino_encoder_for_a_dino_name_passes_and_for_a_cnn_name_is_refused():
    class DINOEncoder(torch.nn.Module):  # the class name is what the guard reads
        pass
    model = torch.nn.Module(); model.encoder = DINOEncoder()
    verify_built_encoder(model, "dinov2_vits14")
    with pytest.raises(RuntimeError, match="DINOEncoder"):
        verify_built_encoder(model, "resnet34")


def test_create_model_on_device_refuses_a_swallowed_dino_request(monkeypatch):
    """The original failure: smp returns its default ResNet for a DINO name."""
    real_unet = smp.Unet  # the stand-in must not recurse into itself
    monkeypatch.setattr(model_2d.smp, "Unet", lambda *a, **k: real_unet(
        "resnet18", encoder_weights=None, in_channels=k.get("in_channels", 1), classes=2))
    with pytest.raises(RuntimeError, match="ResNetEncoder"):
        create_model_on_device("cpu", _struct("dinov2_vits14", encoder_depth=4))


# --- check_checkpoint_encoder_family


def test_dino_claim_without_vit_weights_is_refused():
    with pytest.raises(RuntimeError, match="carries no DINO"):
        check_checkpoint_encoder_family(_resnet().state_dict(), "dinov2_vits14")


def test_cnn_claim_with_vit_weights_is_refused():
    sd = {"encoder.dino_model.patch_embed.proj.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="carries DINO"):
        check_checkpoint_encoder_family(sd, "resnet34")


def test_mean_teacher_and_dataparallel_layouts_are_normalised():
    sd = {"module." + k: v for k, v in _resnet().state_dict().items()}
    check_checkpoint_encoder_family({"student": sd, "teacher": sd}, "resnet18")
    with pytest.raises(RuntimeError, match="mislabelled"):
        check_checkpoint_encoder_family({"student": sd, "teacher": sd}, "dinov2_vits14")


def test_consistent_claims_pass():
    check_checkpoint_encoder_family(_resnet().state_dict(), "resnet18")
    check_checkpoint_encoder_family({"encoder.dino_model.blocks.0.norm1.weight": torch.zeros(1)}, "dinov3_vitl16")
    check_checkpoint_encoder_family({}, None)  # no encoder declared, nothing to corroborate


# --- the loaders refuse before building anything


def _write(tmp_path, struct, state_dict):
    path = tmp_path / "model.pytorch"
    torch.save({"model_struc_dict": struct, "model_state_dict": state_dict, "label_codes": {0: "bg", 1: "fg"}}, path)
    return path


def test_create_model_from_file_refuses_a_mislabelled_0_4_0_checkpoint(tmp_path):
    path = _write(tmp_path, _struct("dinov2_vits14", encoder_depth=4), _resnet().state_dict())
    with pytest.raises(RuntimeError, match="mislabelled CNN checkpoint"):
        create_model_from_file(path, gpu=False)


def test_create_model_from_file_full_weights_refuses_a_mislabelled_checkpoint(tmp_path):
    path = tmp_path / "weights.pytorch"
    torch.save(_resnet().state_dict(), path)
    with pytest.raises(RuntimeError, match="mislabelled CNN checkpoint"):
        create_model_from_file_full_weights(path, _struct("dinov2_vits14", encoder_depth=4), gpu=False)


def test_create_model_from_file_still_loads_a_consistent_cnn_checkpoint(tmp_path):
    path = _write(tmp_path, _struct("resnet18"), _resnet().state_dict())
    model, classes, codes, _ = create_model_from_file(path, gpu=False)
    assert type(model.encoder).__name__ == "ResNetEncoder" and classes == 2
