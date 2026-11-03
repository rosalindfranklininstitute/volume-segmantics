"""``full_weights_path`` and ``encoder_weights_path`` reach the default trainer.


"""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import segmentation_models_pytorch as smp
import torch

from volume_segmantics.model.model_2d import (
    apply_weight_path_settings,
    load_encoder_weights_into,
    load_full_weights_into,
    weight_path_setting,
)


def _unet(in_channels=1, seed=0):
    torch.manual_seed(seed)
    return smp.Unet("resnet18", encoder_weights=None, in_channels=in_channels, classes=2)


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize("value", [None, False, "False", "false", "None", "null", "", "  "])
def test_unset_values_mean_no_path(value):
    assert weight_path_setting(SimpleNamespace(full_weights_path=value), "full_weights_path") is None


def test_a_set_value_is_a_path():
    got = weight_path_setting(SimpleNamespace(encoder_weights_path="~/w.pt"), "encoder_weights_path")
    assert got == Path("~/w.pt").expanduser()


def test_full_weights_load_from_a_volseg_checkpoint(tmp_path):
    source, target = _unet(seed=1), _unet(seed=2)
    assert not _same(source, target)
    ckpt = tmp_path / "model.pytorch"
    torch.save({"model_state_dict": source.state_dict(), "model_struc_dict": {"encoder_name": "resnet18"}}, ckpt)
    load_full_weights_into(target, ckpt, encoder_name="resnet18")
    assert _same(source, target)


def test_full_weights_load_from_a_bare_state_dict_with_module_prefix(tmp_path):
    source, target = _unet(seed=1), _unet(seed=2)
    ckpt = tmp_path / "model.pth"
    torch.save({f"module.{k}": v for k, v in source.state_dict().items()}, ckpt)
    load_full_weights_into(target, ckpt)
    assert _same(source, target)


def test_full_weights_refuse_a_checkpoint_that_matches_nothing(tmp_path):
    ckpt = tmp_path / "other.pth"
    torch.save({"model_state_dict": {"totally.different": torch.zeros(3)}}, ckpt)
    with pytest.raises(RuntimeError, match="does not match the model architecture"):
        load_full_weights_into(_unet(), ckpt)


def test_full_weights_refuse_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="full_weights_path"):
        load_full_weights_into(_unet(), tmp_path / "absent.pth")


def test_encoder_weights_load_from_a_bare_encoder_state_dict(tmp_path):
    source, target = _unet(seed=1), _unet(seed=2)
    enc = tmp_path / "encoder.pth"
    torch.save(source.encoder.state_dict(), enc)
    load_encoder_weights_into(target, enc)
    assert _same(source.encoder, target.encoder)
    assert not _same(source.decoder, target.decoder)  # only the encoder was touched


def test_encoder_weights_are_taken_from_a_full_checkpoint(tmp_path):
    source, target = _unet(seed=1), _unet(seed=2)
    ckpt = tmp_path / "model.pytorch"
    torch.save({"model_state_dict": source.state_dict()}, ckpt)
    load_encoder_weights_into(target, ckpt)
    assert _same(source.encoder, target.encoder)
    assert not _same(source.decoder, target.decoder)


def test_three_channel_encoder_weights_are_summed_into_a_one_channel_first_conv(tmp_path):
    rgb, gray = _unet(in_channels=3, seed=1), _unet(in_channels=1, seed=2)
    enc = tmp_path / "rgb_encoder.pth"
    torch.save(rgb.encoder.state_dict(), enc)
    load_encoder_weights_into(gray, enc)
    expected = rgb.encoder.conv1.weight.detach().sum(1, keepdim=True)
    assert torch.equal(gray.encoder.conv1.weight.detach(), expected)
    assert torch.equal(gray.encoder.layer1[0].conv1.weight.detach(), rgb.encoder.layer1[0].conv1.weight.detach())


def test_encoder_weights_refuse_a_file_that_matches_nothing(tmp_path):
    enc = tmp_path / "junk.pth"
    torch.save({"nothing.here": torch.zeros(2)}, enc)
    with pytest.raises(RuntimeError, match="does not match the model architecture"):
        load_encoder_weights_into(_unet(), enc)


def test_apply_settings_loads_full_then_encoder(tmp_path):
    full_src, enc_src, target = _unet(seed=1), _unet(seed=3), _unet(seed=2)
    full = tmp_path / "full.pytorch"; torch.save({"model_state_dict": full_src.state_dict()}, full)
    enc = tmp_path / "enc.pth"; torch.save(enc_src.encoder.state_dict(), enc)
    notes = apply_weight_path_settings(
        target, SimpleNamespace(full_weights_path=str(full), encoder_weights_path=str(enc)),
    )
    assert [n.split("=")[0] for n in notes] == ["full_weights_path", "encoder_weights_path"]
    assert _same(target.encoder, enc_src.encoder)   # encoder weights win on the encoder
    assert _same(target.decoder, full_src.decoder)  # the rest comes from the full checkpoint


def test_apply_settings_is_a_noop_when_both_are_unset():
    target = _unet(seed=2)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    assert apply_weight_path_settings(target, SimpleNamespace(full_weights_path=False, encoder_weights_path="False")) == []
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())


pl = pytest.importorskip("pytorch_lightning")
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations.lightning2d import VolSeg2dLightningModule  # noqa: E402


def _settings(**overrides):
    base = dict(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=64, use_2_5d_slicing=False,
        num_slices=1, num_cyc_frozen=1, num_cyc_unfrozen=0, patience=1,
        full_weights_path=False, encoder_weights_path=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _semantic_config():
    return PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})


def test_lightning_module_loads_full_weights_from_the_settings(tmp_path):
    torch.manual_seed(1)
    source = VolSeg2dLightningModule.from_pipeline_config(_semantic_config(), settings=_settings(), num_classes=2)
    ckpt = tmp_path / "model.pytorch"
    torch.save({"model_state_dict": source.get_model_state_dict_for_volseg(),
                "model_struc_dict": {"encoder_name": "resnet18"}}, ckpt)
    torch.manual_seed(2)
    loaded = VolSeg2dLightningModule.from_pipeline_config(
        _semantic_config(), settings=_settings(full_weights_path=str(ckpt)), num_classes=2,
    )
    assert _same(source._underlying_student, loaded._underlying_student)


def test_lightning_module_loads_encoder_weights_from_the_settings(tmp_path):
    torch.manual_seed(1)
    source = VolSeg2dLightningModule.from_pipeline_config(_semantic_config(), settings=_settings(), num_classes=2)
    enc = tmp_path / "encoder.pth"
    torch.save(source._underlying_student.encoder.state_dict(), enc)
    torch.manual_seed(2)
    loaded = VolSeg2dLightningModule.from_pipeline_config(
        _semantic_config(), settings=_settings(encoder_weights_path=str(enc)), num_classes=2,
    )
    assert _same(source._underlying_student.encoder, loaded._underlying_student.encoder)
    assert not _same(source._underlying_student.decoders, loaded._underlying_student.decoders)


def test_lightning_module_refuses_a_missing_weights_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="full_weights_path"):
        VolSeg2dLightningModule.from_pipeline_config(
            _semantic_config(), settings=_settings(full_weights_path=str(tmp_path / "absent.pytorch")), num_classes=2,
        )


def test_a_trainer_checkpoint_restarts_a_run_from_its_weights(tmp_path):
    """A ``.pytorch`` written by ``model-train-2d`` under the
    default trainer, named in ``full_weights_path`` or ``encoder_weights_path``,
    initialises the next run."""
    from volume_segmantics.model.operations.lightning2d import (
        VolSegCheckpointCallback,
        model_struct_dict_for_checkpoint,
    )
    torch.manual_seed(1)
    trained = VolSeg2dLightningModule.from_pipeline_config(_semantic_config(), settings=_settings(), num_classes=2)
    # Named as model-train-2d names its output; nothing reads the date back.
    ckpt = tmp_path / f"{date.today()}_U_NET_trained_2d_model.pytorch"
    callback = VolSegCheckpointCallback(
        ckpt, model_struct_dict_for_checkpoint(_settings(), trained, 2), image_size=64,
        input_stack={"use_2_5d": False, "num_slices": 3},
    )
    trainer = SimpleNamespace(callbacks=[], global_rank=0, sanity_checking=False, callback_metrics={})
    callback._save_volseg_checkpoint(trainer, trained, ckpt)

    torch.manual_seed(2)
    restarted = VolSeg2dLightningModule.from_pipeline_config(
        _semantic_config(), settings=_settings(full_weights_path=str(ckpt)), num_classes=2,
    )
    assert _same(trained._underlying_student, restarted._underlying_student)

    torch.manual_seed(3)
    encoder_only = VolSeg2dLightningModule.from_pipeline_config(
        _semantic_config(), settings=_settings(encoder_weights_path=str(ckpt)), num_classes=2,
    )
    assert _same(trained._underlying_student.encoder, encoder_only._underlying_student.encoder)
    assert not _same(trained._underlying_student.decoders, encoder_only._underlying_student.decoders)
