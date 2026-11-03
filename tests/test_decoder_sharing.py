"""``decoder_sharing: separate`` gives every head its own decoder.

The pipeline U-Net ignored the setting and always shared one decoder, so
``separate`` changed nothing, not even the parameter count (reviewer issue 44).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import volume_segmantics.model.heads  # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets  # noqa: F401 — register targets
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.model_2d import create_model_on_device
from volume_segmantics.model.pipeline_multitask_model import PipelineMultitaskModel
from volume_segmantics.model.pipeline_multitask_unet import (
    PipelineMultitaskUnet,
    check_decoder_sharing,
)


def _heads_cfg():
    return {
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
    }


def _unet(decoder_sharing):
    return PipelineMultitaskUnet(
        head_modules=build_head_modules(_heads_cfg(), in_channels=16, num_classes=3),
        encoder_name="resnet18", encoder_weights=None, decoder_sharing=decoder_sharing,
    )


def _params(model):
    return sum(p.numel() for p in model.parameters())


def test_separate_gives_each_head_its_own_unet_decoder():
    shared, separate = _unet("shared"), _unet("separate")
    assert shared.num_decoders == 1 and shared.head_to_decoder == [0, 0]
    assert separate.num_decoders == 2 and separate.head_to_decoder == [0, 1]
    assert _params(separate) > _params(shared)
    assert _params(separate) - _params(shared) == _params(shared.decoders[0])
    with torch.no_grad():
        sem, bnd = separate(torch.zeros(1, 1, 64, 64))
    assert tuple(sem.shape) == (1, 3, 64, 64) and tuple(bnd.shape) == (1, 1, 64, 64)


def test_separate_decoders_on_another_architecture():
    shared = PipelineMultitaskModel(_heads_cfg(), 3, architecture="FPN", encoder_name="resnet18", encoder_weights=None)
    separate = PipelineMultitaskModel(
        _heads_cfg(), 3, architecture="FPN", encoder_name="resnet18", encoder_weights=None, decoder_sharing="separate",
    )
    assert shared.num_decoders == 1 and separate.num_decoders == 2
    assert separate.head_to_decoder == [0, 1]
    assert _params(separate) - _params(shared) == _params(shared.decoders[0])


def test_a_single_head_has_one_decoder_either_way():
    only_semantic = {"semantic": HeadConfig(enabled=True, loss="dice_ce")}
    model = PipelineMultitaskUnet(
        head_modules=build_head_modules(only_semantic, in_channels=16, num_classes=2),
        encoder_name="resnet18", encoder_weights=None, decoder_sharing="separate",
    )
    assert model.num_decoders == 1


@pytest.mark.parametrize("raw, expected", [(None, "shared"), ("", "shared"), ("shared", "shared"), ("Separate", "separate")])
def test_decoder_sharing_values_are_normalised(raw, expected):
    assert check_decoder_sharing(raw) == expected


def test_an_unknown_decoder_sharing_is_refused():
    with pytest.raises(ValueError, match="decoder_sharing"):
        _unet("independent")


def test_separate_decoders_survive_the_checkpoint_round_trip():
    separate = _unet("separate")
    struct = {
        "type": "PIPELINE_MULTITASK_UNET", "encoder_name": "resnet18", "encoder_weights": None,
        "encoder_depth": 5, "in_channels": 1, "classes": 3, "decoder_sharing": "separate",
        "heads": [{"name": n, "out_channels": h.out_channels} for n, h in zip(separate.head_names, separate.heads)],
    }
    rebuilt = create_model_on_device("cpu", struct)
    assert rebuilt.num_decoders == 2 and rebuilt.decoder_sharing == "separate"
    rebuilt.load_state_dict(separate.state_dict())
    # A checkpoint written before the key existed rebuilds the shared layout.
    del struct["decoder_sharing"]
    assert create_model_on_device("cpu", struct).num_decoders == 1


pl = pytest.importorskip("pytorch_lightning")
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    VolSeg2dLightningModule,
    model_struct_dict_for_checkpoint,
)


def _settings(**overrides):
    base = dict(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=64, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=0, patience=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_trainer_reads_decoder_sharing_from_the_settings_and_records_it():
    config = PipelineConfig(heads=_heads_cfg())
    settings = _settings(decoder_sharing="separate")
    module = VolSeg2dLightningModule.from_pipeline_config(config, settings=settings, num_classes=3)
    model = module._underlying_student
    assert model.num_decoders == 2 and model.decoder_sharing == "separate"
    assert model_struct_dict_for_checkpoint(settings, module, 3)["decoder_sharing"] == "separate"
    default = VolSeg2dLightningModule.from_pipeline_config(config, settings=_settings(), num_classes=3)
    assert default._underlying_student.num_decoders == 1


def test_the_trainer_refuses_an_unknown_decoder_sharing():
    with pytest.raises(ValueError, match="decoder_sharing"):
        VolSeg2dLightningModule.from_pipeline_config(
            PipelineConfig(heads=_heads_cfg()), settings=_settings(decoder_sharing="both"), num_classes=3,
        )
