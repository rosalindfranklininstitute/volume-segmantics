"""Tests for Lightning trainer and callbacks.


"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

import volume_segmantics.model.heads        # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets       # noqa: F401 — register targets

import volume_segmantics.utilities.base_data_utils as bdu


pl = pytest.importorskip("pytorch_lightning")


from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule
from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    HeadConfig,
    PipelineConfig,
    TransformSpec,
)
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.operations.lightning_2d import (
    EpochHistoryCallback,
    UnfreezeEncoderCallback,
    VolSeg2dLightningModule,
    VolSegCheckpointCallback,
)
from volume_segmantics.model.pipeline_multitask_unet import PipelineMultitaskUnet
from volume_segmantics.model.training.multitask_calculator import (
    PipelineMultiTaskLossCalculator,
)


# Fixtures 


@pytest.fixture
def basic_settings():
    """Minimal settings.SimpleNamespace for the Lightning module."""
    return SimpleNamespace(
        model={
            "type": "U_Net",
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "encoder_depth": 5,
        },
        starting_lr=5e-5,
        encoder_lr_multiplier=0.1,
        image_size=64,
        training_set_proportion=0.85,
        cuda_device=0,
        use_imagenet_norm=True,
        use_2_5d_slicing=False,
        num_slices=1,
        num_cyc_frozen=2,
        num_cyc_unfrozen=2,
        patience=4,
        max_label_no=3,
    )


@pytest.fixture
def four_heads_pipeline_config():
    return PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True, loss="dice_ce"),
            "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
            "distance": HeadConfig(enabled=True, loss="distance_l1"),
            "sdm":      HeadConfig(
                enabled=True, loss="sdm_l1",
                extra={"variant": "binary", "d_clip": 5.0},
            ),
        },
        augmentations=AugmentationsConfig(
            backend="albumentations",
            train_transforms=[
                TransformSpec(name="HorizontalFlip", params={"p": 0.5}),
            ],
        ),
    )


@pytest.fixture
def fixture_data(tmp_path):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(20):
        cv2.imwrite(
            str(img_dir / f"data_{i:03d}.png"),
            (rng.random((64, 64)) * 255).astype(np.uint8),
        )
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


@pytest.fixture
def lightning_module(four_heads_pipeline_config, basic_settings):
    return VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings,
        num_classes=3,
        in_channels=1,
        total_steps=100,
    )


# Construction 


def test_from_pipeline_config_builds_module(
    four_heads_pipeline_config, basic_settings,
):
    mod = VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings,
        num_classes=3,
    )
    assert isinstance(mod, VolSeg2dLightningModule)
    assert isinstance(mod.model, PipelineMultitaskUnet)
    assert isinstance(mod.loss_calculator, PipelineMultiTaskLossCalculator)
    assert mod.num_classes == 3
    assert mod.model.head_names == (
        "semantic", "boundary", "distance", "sdm",
    )


def test_module_forward_returns_head_tuple(lightning_module):
    x = torch.randn(2, 1, 64, 64)
    outs = lightning_module(x)
    assert isinstance(outs, tuple)
    assert len(outs) == 4


# configure_optimizers 


def test_optimizer_single_lr_when_multiplier_is_one(
    four_heads_pipeline_config, basic_settings,
):
    basic_settings.encoder_lr_multiplier = 1.0
    mod = VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings, num_classes=3,
    )
    opt = mod.configure_optimizers()
    assert len(opt.param_groups) == 1


def test_optimizer_two_param_groups_with_differential_lr(lightning_module):
    """encoder_lr_multiplier=0.1 -> encoder group at 0.1×LR, others at LR."""
    opt = lightning_module.configure_optimizers()
    assert len(opt.param_groups) == 2
    encoder_group, other_group = opt.param_groups
    starting_lr = lightning_module.settings.starting_lr
    assert encoder_group["lr"] == pytest.approx(starting_lr * 0.1)
    assert other_group["lr"] == pytest.approx(starting_lr)
    # Both groups have parameters.
    assert len(encoder_group["params"]) > 0
    assert len(other_group["params"]) > 0


def test_optimizer_no_param_overlap_between_groups(lightning_module):
    opt = lightning_module.configure_optimizers()
    encoder_ids = {id(p) for p in opt.param_groups[0]["params"]}
    other_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert not (encoder_ids & other_ids)


def test_optimizer_handles_none_multiplier(
    four_heads_pipeline_config, basic_settings,
):
    basic_settings.encoder_lr_multiplier = None
    mod = VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings, num_classes=3,
    )
    opt = mod.configure_optimizers()
    assert len(opt.param_groups) == 1


# Encoder freeze/unfreeze 


def test_freeze_encoder_sets_requires_grad_false(lightning_module):
    lightning_module.unfreeze_encoder()  # known starting state
    lightning_module.freeze_encoder()
    for p in lightning_module.model.encoder.parameters():
        assert p.requires_grad is False


def test_unfreeze_encoder_sets_requires_grad_true(lightning_module):
    lightning_module.freeze_encoder()
    lightning_module.unfreeze_encoder()
    for p in lightning_module.model.encoder.parameters():
        assert p.requires_grad is True


def test_freeze_unfreeze_does_not_touch_decoder_params(lightning_module):
    lightning_module.freeze_encoder()
    # Decoder + heads stay trainable.
    for p in lightning_module.model.decoders.parameters():
        assert p.requires_grad is True
    for h in lightning_module.model.heads:
        for p in h.parameters():
            assert p.requires_grad is True


# UnfreezeEncoderCallback 


def test_unfreeze_callback_zero_frozen_epochs_is_noop(lightning_module):
    cb = UnfreezeEncoderCallback(num_frozen_epochs=0)
    trainer = MagicMock()
    cb.on_fit_start(trainer, lightning_module)
    # No freeze applied; encoder stays trainable.
    for p in lightning_module.model.encoder.parameters():
        assert p.requires_grad is True


def test_unfreeze_callback_freezes_at_fit_start(lightning_module):
    cb = UnfreezeEncoderCallback(num_frozen_epochs=2)
    trainer = MagicMock()
    cb.on_fit_start(trainer, lightning_module)
    for p in lightning_module.model.encoder.parameters():
        assert p.requires_grad is False


def test_unfreeze_callback_unfreezes_at_boundary_epoch(lightning_module):
    cb = UnfreezeEncoderCallback(num_frozen_epochs=2)
    trainer = MagicMock()
    cb.on_fit_start(trainer, lightning_module)

    trainer.current_epoch = 0
    cb.on_train_epoch_end(trainer, lightning_module)
    # After epoch 0 ends -> epoch_just_finished = 1 < 2 -> still frozen.
    assert all(
        not p.requires_grad
        for p in lightning_module.model.encoder.parameters()
    )

    trainer.current_epoch = 1
    cb.on_train_epoch_end(trainer, lightning_module)
    # After epoch 1 ends -> epoch_just_finished = 2 >= 2 -> unfreeze fires.
    assert all(
        p.requires_grad
        for p in lightning_module.model.encoder.parameters()
    )


# VolSegCheckpointCallback 


def _make_checkpoint_callback(tmp_path: Path) -> VolSegCheckpointCallback:
    return VolSegCheckpointCallback(
        output_path=tmp_path / "ckpt.pytorch",
        model_struc_dict={"type": "PIPELINE_MULTITASK_UNET"},
        label_codes={},
    )


def test_checkpoint_skips_save_during_sanity_check(
    lightning_module, tmp_path,
):
    """**v0.5 Bug 2 fix #1**: sanity-check phase must not lock in a
    random-init checkpoint as 'best'."""
    cb = _make_checkpoint_callback(tmp_path)
    trainer = MagicMock()
    trainer.sanity_checking = True
    trainer.global_rank = 0
    trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert cb.best_val_loss == float("inf")
    assert not (tmp_path / "ckpt.pytorch").exists()


def test_checkpoint_saves_when_val_loss_improves(
    lightning_module, tmp_path,
):
    cb = _make_checkpoint_callback(tmp_path)
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.global_rank = 0
    trainer.callback_metrics = {"val_loss": torch.tensor(0.7)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert cb.best_val_loss == pytest.approx(0.7)
    assert (tmp_path / "ckpt.pytorch").exists()
    # Better val_loss -> overwrites.
    trainer.callback_metrics = {"val_loss": torch.tensor(0.3)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert cb.best_val_loss == pytest.approx(0.3)


def test_checkpoint_does_not_save_when_val_loss_worsens(
    lightning_module, tmp_path,
):
    cb = _make_checkpoint_callback(tmp_path)
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.global_rank = 0
    trainer.callback_metrics = {"val_loss": torch.tensor(0.3)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    saved_mtime = (tmp_path / "ckpt.pytorch").stat().st_mtime_ns
    # Worse -> no save.
    trainer.callback_metrics = {"val_loss": torch.tensor(0.8)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert (tmp_path / "ckpt.pytorch").stat().st_mtime_ns == saved_mtime


def test_checkpoint_on_train_end_always_saves(lightning_module, tmp_path):
    """**v0.5 Bug 2 fix #2**: always save on fit completion, even if
    val_loss is worse than the running best."""
    cb = _make_checkpoint_callback(tmp_path)
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.global_rank = 0
    # Simulate: best-val saw val_loss=0.3 earlier; final epoch's
    # val_loss is 0.7. The on_train_end fix overwrites with the final.
    cb.best_val_loss = 0.3
    trainer.callback_metrics = {"val_loss": torch.tensor(0.7)}
    cb.on_train_end(trainer, lightning_module)
    assert (tmp_path / "ckpt.pytorch").exists()
    # Loaded checkpoint records the final val_loss, not the earlier best.
    loaded = torch.load(tmp_path / "ckpt.pytorch", weights_only=False)
    assert loaded["loss_val"] == pytest.approx(0.7)


def test_checkpoint_on_train_end_skips_during_sanity_check(
    lightning_module, tmp_path,
):
    cb = _make_checkpoint_callback(tmp_path)
    trainer = MagicMock()
    trainer.sanity_checking = True
    trainer.global_rank = 0
    trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
    cb.on_train_end(trainer, lightning_module)
    assert not (tmp_path / "ckpt.pytorch").exists()


def test_checkpoint_carries_head_metadata(lightning_module, tmp_path):
    head_meta = {"head_names": ["semantic", "boundary", "distance", "sdm"]}
    cb = VolSegCheckpointCallback(
        output_path=tmp_path / "ckpt.pytorch",
        model_struc_dict={"type": "PIPELINE_MULTITASK_UNET"},
        head_metadata=head_meta,
    )
    trainer = MagicMock()
    trainer.sanity_checking = False
    trainer.global_rank = 0
    trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
    cb.on_validation_epoch_end(trainer, lightning_module)
    loaded = torch.load(tmp_path / "ckpt.pytorch", weights_only=False)
    assert loaded["head_metadata"] == head_meta


#  EpochHistoryCallback 


def test_epoch_history_collects_metrics(lightning_module):
    cb = EpochHistoryCallback()
    lightning_module.epoch_history = []
    trainer = MagicMock()
    trainer.global_rank = 0
    trainer.sanity_checking = False
    trainer.current_epoch = 0
    trainer.callback_metrics = {
        "val_loss": torch.tensor(0.5),
        "train_loss": torch.tensor(0.4),
    }
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert len(lightning_module.epoch_history) == 1
    row = lightning_module.epoch_history[0]
    assert row["epoch"] == 0
    assert row["val_loss"] == pytest.approx(0.5)
    assert row["train_loss"] == pytest.approx(0.4)


def test_epoch_history_skips_sanity_check(lightning_module):
    cb = EpochHistoryCallback()
    lightning_module.epoch_history = []
    trainer = MagicMock()
    trainer.global_rank = 0
    trainer.sanity_checking = True
    cb.on_validation_epoch_end(trainer, lightning_module)
    assert lightning_module.epoch_history == []


# End-to-end Lightning fit 


def test_end_to_end_lightning_fit(
    fixture_data, four_heads_pipeline_config, basic_settings,
    tmp_path, monkeypatch,
):
    """The load-bearing integration test: 2-epoch fit on a 20-sample
    fixture exercises model + dataloader + LightningModule + every
    callback. The assertions cover:
    * train_loss + val_loss appear in callback_metrics
    * epoch_history is populated
    * VolSegCheckpointCallback wrote a checkpoint
    * Saved checkpoint's BatchNorm stats moved away from random-init
      (the v0.4.0b3 regression guard for v0.5 Bug 2)
    """
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)

    # Use 2-epoch fit to exercise both phases of the unfreeze callback.
    basic_settings.num_cyc_frozen = 1
    basic_settings.num_cyc_unfrozen = 1

    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir,
        settings=basic_settings,
        pipeline_config=four_heads_pipeline_config,
        num_classes=3,
    )
    data_module.setup()
    steps_per_epoch = len(data_module.train_dataloader())
    total_steps = steps_per_epoch * 2

    pl_module = VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings,
        num_classes=3, in_channels=1, total_steps=total_steps,
    )

    ckpt_path = tmp_path / "lightning_test.pytorch"
    callbacks = [
        UnfreezeEncoderCallback(num_frozen_epochs=1),
        EpochHistoryCallback(),
        VolSegCheckpointCallback(
            output_path=ckpt_path,
            model_struc_dict={"type": "PIPELINE_MULTITASK_UNET"},
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(pl_module, datamodule=data_module)

    # callback_metrics has train_loss + val_loss.
    assert "train_loss" in trainer.callback_metrics
    assert "val_loss" in trainer.callback_metrics
    # epoch_history has at least one entry.
    assert len(pl_module.epoch_history) >= 1
    assert "val_loss" in pl_module.epoch_history[-1]
    # Checkpoint written to disk.
    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, weights_only=False)
    assert "model_state_dict" in loaded
    assert "model_struc_dict" in loaded
    assert "loss_val" in loaded


def test_end_to_end_bn_stats_moved_from_random_init(
    fixture_data, four_heads_pipeline_config, basic_settings,
    tmp_path, monkeypatch,
):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)

    basic_settings.num_cyc_frozen = 0
    basic_settings.num_cyc_unfrozen = 1

    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir,
        settings=basic_settings,
        pipeline_config=four_heads_pipeline_config,
        num_classes=3,
    )
    pl_module = VolSeg2dLightningModule.from_pipeline_config(
        four_heads_pipeline_config,
        settings=basic_settings, num_classes=3,
        in_channels=1, total_steps=50,
    )

    ckpt_path = tmp_path / "regression_test.pytorch"
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[
            VolSegCheckpointCallback(
                output_path=ckpt_path,
                model_struc_dict={"type": "PIPELINE_MULTITASK_UNET"},
            ),
        ],
        accelerator="cpu", devices=1,
        logger=False, enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=2,  # IMPORTANT: exercise the sanity-check phase
    )
    trainer.fit(pl_module, datamodule=data_module)

    loaded = torch.load(ckpt_path, weights_only=False)
    state_dict = loaded["model_state_dict"]

    # The encoder has BatchNorm modules; running_var keys live under
    # encoder.<...>.running_var. Find at least one and assert it
    # moved from 1.0 (the default).
    bn_running_vars = [
        v for k, v in state_dict.items()
        if k.endswith("running_var") and "encoder" in k
    ]
    assert len(bn_running_vars) > 0, (
        "no encoder BN running_var keys found in saved state dict"
    )
    # If even one BN is at its 1.0 init, the model is suspect; we
    # check at least one moved.
    moved = [
        v for v in bn_running_vars
        if not torch.allclose(v, torch.ones_like(v), atol=1e-3)
    ]
    assert len(moved) > 0, (
        "all BN running_var values at 1.0 — checkpoint may be from "
        "random init (the v0.5 Bug 2 regression)"
    )

    # And num_batches_tracked > 0 confirms training actually ran.
    bn_tracked = [
        v for k, v in state_dict.items()
        if k.endswith("num_batches_tracked") and "encoder" in k
    ]
    assert len(bn_tracked) > 0
    # At least one BN should have tracked > 0 batches.
    assert max(int(v) for v in bn_tracked) > 0
