"""The default trainer builds the ``model.type`` the settings ask for.

Each type is built on its own decoder, the checkpoint records it, and prediction rebuilds it.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import volume_segmantics.model.heads  # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets  # noqa: F401 — register targets
import volume_segmantics.utilities.base_data_utils as bdu
from volume_segmantics.model.model_2d import create_model_from_file, create_model_on_device
from volume_segmantics.model.pipeline_multitask_model import PipelineMultitaskModel
from volume_segmantics.model.pipeline_multitask_unet import PipelineMultitaskUnet

pl = pytest.importorskip("pytorch_lightning")

from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule  # noqa: E402
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    EpochHistoryCallback,
    UnfreezeEncoderCallback,
    VolSeg2dLightningModule,
    VolSegCheckpointCallback,
    checkpoint_model_type_name,
    model_struct_dict_for_checkpoint,
    model_type_from_settings,
)

OTHER_TYPES = [
    "FPN", "DeepLabV3", "DeepLabV3_Plus", "MA_Net", "Linknet", "PAN",
    "Segformer", "U_Net_Plus_plus", "Vanilla_Unet",
]


def _settings(model_type="U_Net", **overrides):
    base = dict(
        model={"type": model_type, "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=128, training_set_proportion=0.85,
        cuda_device=0, use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=0, patience=1, lr_policy="constant",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _semantic():
    return PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})


def _two_heads():
    return PipelineConfig(heads={
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
    })


@pytest.mark.parametrize("model_type", OTHER_TYPES)
def test_the_requested_architecture_is_built_and_rebuilt_from_its_checkpoint(model_type):
    settings = _settings(model_type)
    module = VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=settings, num_classes=3)
    model = module._underlying_student
    assert isinstance(model, PipelineMultitaskModel)
    assert model.architecture.name == model_type.upper()
    assert model.head_names == ("semantic",)
    with torch.no_grad():
        out = module(torch.zeros(2, 1, 128, 128))
    assert tuple(out[0].shape) == (2, 3, 128, 128)

    struct = model_struct_dict_for_checkpoint(settings, module, 3)
    assert struct["type"] == model_type.upper()
    assert struct["heads"] == [{"name": "semantic", "out_channels": 3}]
    assert struct["decoder_sharing"] == "shared"
    rebuilt = create_model_on_device("cpu", struct)
    assert isinstance(rebuilt, PipelineMultitaskModel)
    assert rebuilt.architecture.name == model_type.upper()
    rebuilt.load_state_dict(module.get_model_state_dict_for_volseg())  # strict: same layout


def test_the_unet_family_still_builds_the_pipeline_unet():
    for name in ("U_Net", "Multitask_Unet", "PIPELINE_MULTITASK_UNET"):
        module = VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=_settings(name), num_classes=2)
        assert isinstance(module._underlying_student, PipelineMultitaskUnet), name
        assert model_struct_dict_for_checkpoint(_settings(name), module, 2)["type"] == "PIPELINE_MULTITASK_UNET"


def test_the_checkpoint_name_carries_the_model_type():
    assert checkpoint_model_type_name(_settings("U_Net"), ["semantic"]) == "U_NET"
    assert checkpoint_model_type_name(_settings("U_Net"), ["boundary", "semantic"]) == "PIPELINE_MULTITASK_UNET"
    assert checkpoint_model_type_name(_settings("Multitask_Unet"), ["boundary", "semantic"]) == "PIPELINE_MULTITASK_UNET"
    assert checkpoint_model_type_name(_settings("FPN"), ["semantic"]) == "FPN"
    assert checkpoint_model_type_name(_settings("Segformer"), ["boundary", "semantic"]) == "PIPELINE_MULTITASK_SEGFORMER"


def test_model_type_defaults_to_unet_when_unset():
    assert model_type_from_settings(SimpleNamespace(model={"encoder_name": "resnet18"})) is bdu.ModelType.U_NET
    assert model_type_from_settings(SimpleNamespace()) is bdu.ModelType.U_NET


def test_an_unknown_model_type_is_refused():
    with pytest.raises(ValueError, match="unknown model type"):
        VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=_settings("ResNet"), num_classes=2)


def test_a_multi_head_fpn_runs_every_head_on_the_fpn_decoder():
    module = VolSeg2dLightningModule.from_pipeline_config(_two_heads(), settings=_settings("FPN"), num_classes=2)
    model = module._underlying_student
    assert model.head_names == ("semantic", "boundary")
    assert type(model.decoders[0]).__name__ == "FPNDecoder"
    with torch.no_grad():
        sem, bnd = module(torch.zeros(1, 1, 128, 128))
    assert tuple(sem.shape) == (1, 2, 128, 128) and tuple(bnd.shape) == (1, 1, 128, 128)


def test_the_frozen_phase_freezes_the_architectures_own_encoder():
    module = VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=_settings("DeepLabV3_Plus"), num_classes=2)
    module.freeze_encoder()
    model = module._underlying_student
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoders.parameters())
    module.unfreeze_encoder()
    assert all(p.requires_grad for p in model.encoder.parameters())


@pytest.fixture
def fixture_data(tmp_path):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(20):
        cv2.imwrite(str(img_dir / f"data_{i:03d}.png"), (rng.random((64, 64)) * 255).astype(np.uint8))
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


def test_an_fpn_fit_writes_a_checkpoint_that_prediction_rebuilds_as_an_fpn(fixture_data, tmp_path, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = _settings("FPN", image_size=64)
    config = _semantic()
    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir, settings=settings, pipeline_config=config, num_classes=3,
    )
    data_module.setup()
    module = VolSeg2dLightningModule.from_pipeline_config(
        config, settings=settings, num_classes=3, total_steps=len(data_module.train_dataloader()),
    )
    ckpt = tmp_path / f"{checkpoint_model_type_name(settings, ['semantic'])}_fit.pytorch"
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[
            UnfreezeEncoderCallback(num_frozen_epochs=1),
            EpochHistoryCallback(),
            VolSegCheckpointCallback(
                output_path=ckpt,
                model_struc_dict=model_struct_dict_for_checkpoint(settings, module, 3),
                image_size=64,
            ),
        ],
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=data_module)

    assert ckpt.name.startswith("FPN_")
    saved = torch.load(ckpt, weights_only=False)
    assert saved["model_struc_dict"]["type"] == "FPN"
    model, num_labels, _, image_size = create_model_from_file(ckpt, gpu=False)
    assert isinstance(model, PipelineMultitaskModel) and model.architecture is bdu.ModelType.FPN
    assert (num_labels, image_size) == (3, 64)
    trained = module.get_model_state_dict_for_volseg()
    assert all(torch.equal(trained[k], v) for k, v in model.state_dict().items())
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 64, 64))
    assert tuple(out[0].shape) == (1, 3, 64, 64)
