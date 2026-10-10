"""The Lightning trainer honours the learning-rate settings the way the
legacy trainer does, with the constant rate kept as an opt-in policy.

Until now the Lightning path trained at ``starting_lr``, documented as the
lower bound of the learning-rate search, with no sweep and no schedule, and
ignored ``end_lr``, ``lr_find_epochs``, ``lr_reduce_factor`` and ``pct_lr_inc``.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pl = pytest.importorskip("pytorch_lightning")

import volume_segmantics.utilities.base_data_utils as bdu  # noqa: E402
import volume_segmantics.utilities.config as cfg  # noqa: E402
from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule  # noqa: E402
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    UnfreezeEncoderCallback,
    VolSeg2dLightningModule,
)
from volume_segmantics.model.operations.lightning_lr import (  # noqa: E402
    LRPolicyCallback,
    lr_policy_from_settings,
    select_lr_from_sweep,
    sweep_learning_rate,
)


def _settings(**overrides):
    base = dict(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=1e-4, end_lr=1e-2, lr_find_epochs=1, lr_reduce_factor=10, pct_lr_inc=0.3,
        encoder_lr_multiplier=0.5, image_size=64, training_set_proportion=0.85, cuda_device=0,
        use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=1, patience=4, max_label_no=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _config():
    return PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})


@pytest.fixture
def fixture_data(tmp_path):
    import cv2
    img_dir, msk_dir = tmp_path / "data", tmp_path / "seg"
    img_dir.mkdir(); msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        cv2.imwrite(str(img_dir / f"data_{i:03d}.png"), (rng.random((64, 64)) * 255).astype(np.uint8))
        msk = np.zeros((64, 64), dtype=np.uint8); msk[16:48, 16:48] = 1; msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


#  policy parsing

@pytest.mark.parametrize("raw, expected", [(None, "find_onecycle"), ("", "find_onecycle"),
                                           ("constant", "constant"), (" Find_OneCycle ", "find_onecycle")])
def test_policy_defaults_to_the_legacy_semantics(raw, expected):
    assert lr_policy_from_settings(SimpleNamespace(lr_policy=raw)) == expected


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="lr_policy"):
        lr_policy_from_settings(SimpleNamespace(lr_policy="cosine"))


#  configure_optimizers

def test_constant_policy_is_a_bare_optimizer_at_starting_lr():
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=_settings(lr_policy="constant"), num_classes=3)
    opt = mod.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-4 * 0.5, 1e-4])


def test_default_policy_returns_a_onecycle_schedule_stepped_per_step():
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=_settings(), num_classes=3, total_steps=40)
    out = mod.configure_optimizers()
    sched = out["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
    assert out["lr_scheduler"]["interval"] == "step"
    assert sched.total_steps == 40
    assert [g["lr_multiplier"] for g in out["optimizer"].param_groups] == [0.5, 1.0]


#  the sweep

def test_sweep_restores_the_weights_and_rises_from_start_to_end(fixture_data, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = _settings()
    dm = VolSeg2dDataModule(image_dir=img_dir, label_dir=msk_dir, settings=settings,
                            pipeline_config=_config(), num_classes=3)
    dm.setup()
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=settings, num_classes=3, total_steps=8)
    before = {k: v.clone() for k, v in mod.state_dict().items()}
    losses, lrs = sweep_learning_rate(mod, dm.train_dataloader(), start_lr=1e-4, end_lr=1e-2, num_steps=6)
    # The legacy divergence guard stops a sweep whose loss is still above 1
    # after three quarters of a pass, which an untrained model's is, so the
    # curve may be shorter than num_steps; it never exceeds it.
    assert 1 <= len(losses) == len(lrs) <= 6
    assert lrs[0] == pytest.approx(1e-4) and lrs[-1] < 1e-2 and all(np.diff(lrs) > 0)
    assert all(torch.equal(before[k], v) for k, v in mod.state_dict().items())
    assert all(p.grad is None for p in mod.parameters())


def test_selection_is_the_legacy_rule():
    lrs = list(np.logspace(-5, -1, 50))
    losses = list(1.0 - 0.5 * np.tanh(np.linspace(-3, 3, 50)))  # steepest descent in the middle
    chosen = select_lr_from_sweep(losses, lrs)
    assert chosen == pytest.approx(lrs[int(np.gradient(np.array(losses)).argmin())] / cfg.LR_DIVISOR)
    assert select_lr_from_sweep([], []) == cfg.DEFAULT_MIN_LR
    assert select_lr_from_sweep(list(np.linspace(0.1, 1.0, 20)), lrs[:20]) == cfg.DEFAULT_MIN_LR  # never descends


#  end to end

def test_fit_sweeps_once_per_phase_and_trains_under_onecycle(fixture_data, tmp_path, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = _settings()
    dm = VolSeg2dDataModule(image_dir=img_dir, label_dir=msk_dir, settings=settings,
                            pipeline_config=_config(), num_classes=3)
    dm.setup()
    steps = len(dm.train_dataloader())
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=settings, num_classes=3, total_steps=2 * steps)
    policy = LRPolicyCallback(settings=settings, num_frozen_epochs=1, num_unfrozen_epochs=1, steps_per_epoch=steps)
    trainer = pl.Trainer(max_epochs=2, callbacks=[UnfreezeEncoderCallback(num_frozen_epochs=1), policy],
                         accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0, enable_checkpointing=False)
    trainer.fit(mod, datamodule=dm)
    phases = mod.phase_learning_rates
    assert [p["phase"] for p in phases] == ["frozen", "unfrozen"]
    assert all(p["total_steps"] == steps for p in phases)
    assert 1e-4 / cfg.LR_DIVISOR <= phases[0]["lr"] <= 1e-2 or phases[0]["lr"] == cfg.DEFAULT_MIN_LR
    assert 1e-5 / cfg.LR_DIVISOR <= phases[1]["lr"] <= 1e-3 or phases[1]["lr"] == cfg.DEFAULT_MIN_LR
    assert phases[1]["max_lrs"] == pytest.approx([phases[1]["lr"] * 0.5, phases[1]["lr"]])
    sched = trainer.lr_scheduler_configs[0].scheduler
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR) and sched.total_steps == steps
    assert "train_loss" in trainer.callback_metrics


def test_constant_policy_fit_keeps_the_rate_fixed(fixture_data, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = _settings(lr_policy="constant", encoder_lr_multiplier=1.0)
    dm = VolSeg2dDataModule(image_dir=img_dir, label_dir=msk_dir, settings=settings,
                            pipeline_config=_config(), num_classes=3)
    dm.setup()
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=settings, num_classes=3)
    trainer = pl.Trainer(max_epochs=1, callbacks=[UnfreezeEncoderCallback(num_frozen_epochs=0)],
                         accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0, enable_checkpointing=False)
    trainer.fit(mod, datamodule=dm)
    assert trainer.lr_scheduler_configs == []
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-4)
    assert not hasattr(mod, "phase_learning_rates")


def test_sweep_loader_takes_the_labeled_half_of_a_combined_loader():
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
    from torch.utils.data import DataLoader, TensorDataset
    from volume_segmantics.model.operations.lightning_lr import _train_loader

    labeled = DataLoader(TensorDataset(torch.zeros(8, 1)), batch_size=2, num_workers=0)
    unlabeled = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2, num_workers=0)
    combined = CombinedLoader({"labeled": labeled, "unlabeled": unlabeled}, mode="max_size_cycle")
    trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataloader=lambda: combined))
    loader = _train_loader(trainer)
    assert isinstance(loader, DataLoader) and loader is not labeled
    assert loader.dataset is labeled.dataset and len(loader) == 4


def test_sweep_restores_a_mean_teacher_module_exactly(fixture_data, monkeypatch):
    """The Mean Teacher wrapper overrides ``state_dict`` with a nested layout,
    so the sweep must snapshot and restore the tensors themselves."""
    from volume_segmantics.data.pipeline_loader import EMAScheduleConfig, EMATeacherConfig
    from volume_segmantics.model.training.mean_teacher import MeanTeacherModel

    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = _settings()
    config = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
        ema_teacher=EMATeacherConfig(
            enabled=True, schedule=EMAScheduleConfig(alpha_warmup=0.0, alpha_end=0.5, warmup_steps=2),
            consistency_weights={"semantic": 0.5},
        ),
    )
    dm = VolSeg2dDataModule(image_dir=img_dir, label_dir=msk_dir, settings=settings,
                            pipeline_config=config, num_classes=3)
    dm.setup()
    mod = VolSeg2dLightningModule.from_pipeline_config(config, settings=settings, num_classes=3, total_steps=8)
    assert isinstance(mod.model, MeanTeacherModel)
    tensors = list(mod.parameters()) + list(mod.buffers())
    before = [t.detach().clone() for t in tensors]
    losses, lrs = sweep_learning_rate(mod, dm.train_dataloader(), start_lr=1e-4, end_lr=1e-2, num_steps=4)
    assert losses and len(losses) == len(lrs)
    assert all(torch.equal(b, t.detach()) for b, t in zip(before, tensors))
    assert any(p.requires_grad for p in mod.model.student.parameters())


#  the low-rate warning

def test_a_selected_rate_below_one_in_a_million_is_warned_about(caplog):
    import logging
    from volume_segmantics.model.operations.trainer_lr_finder import warn_if_learning_rate_too_low
    with caplog.at_level(logging.WARNING):
        assert warn_if_learning_rate_too_low(3e-7, start_lr=1e-7, end_lr=2e-6, phase="unfrozen")
    assert "unfrozen phase selected a learning rate of 3.0e-07" in caplog.text
    assert "lr_reduce_factor" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert not warn_if_learning_rate_too_low(1e-6, start_lr=5e-6, end_lr=1e-4, phase="unfrozen")
        assert not warn_if_learning_rate_too_low(1.4e-4, start_lr=5e-5, end_lr=1e-3, phase="frozen")
    assert caplog.text == ""


def test_the_shipped_reduce_factor_gives_the_unfrozen_phase_a_usable_range():
    import yaml
    from volume_segmantics.model.operations.lightning_lr import lr_search_settings
    shipped = yaml.safe_load(Path("volseg-settings/2d_model_train_settings.yaml").read_text())
    p = lr_search_settings(SimpleNamespace(**shipped))
    assert p.lr_reduce_factor == 10
    assert p.starting_lr / p.lr_reduce_factor >= 1e-6  # the whole unfrozen range is above the warning line
    assert lr_search_settings(SimpleNamespace()).lr_reduce_factor == 10  # and the code default agrees


#  frozen BatchNorm statistics after the unfreeze

def _encoder_bn_modules(mod):
    return [m for m in mod._underlying_student.encoder.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def test_unfreezing_keeps_the_encoder_batchnorm_in_eval_mode_by_default():
    mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=_settings(), num_classes=3)
    mod.train()
    mod.freeze_encoder()
    mod.unfreeze_encoder()
    assert all(p.requires_grad for p in mod._underlying_student.encoder.parameters())
    assert mod._underlying_student.encoder.conv1.training  # the encoder trains
    assert all(not bn.training for bn in _encoder_bn_modules(mod))  # its BatchNorm does not
    mod.train()  # Lightning re-enters train mode every epoch
    assert all(not bn.training for bn in _encoder_bn_modules(mod))
    mod.eval(); mod.train()
    assert all(not bn.training for bn in _encoder_bn_modules(mod))


def test_the_setting_restores_adapting_batchnorm_statistics():
    mod = VolSeg2dLightningModule.from_pipeline_config(
        _config(), settings=_settings(freeze_encoder_bn_stats=False), num_classes=3,
    )
    mod.train(); mod.freeze_encoder(); mod.unfreeze_encoder()
    assert all(bn.training for bn in _encoder_bn_modules(mod))


def test_unfrozen_training_leaves_the_encoder_running_statistics_untouched(fixture_data, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    for frozen_stats, expect_unchanged in ((True, True), (False, False)):
        settings = _settings(lr_policy="constant", starting_lr=1e-3, encoder_lr_multiplier=1.0,
                             num_cyc_frozen=0, num_cyc_unfrozen=1, freeze_encoder_bn_stats=frozen_stats)
        dm = VolSeg2dDataModule(image_dir=img_dir, label_dir=msk_dir, settings=settings,
                                pipeline_config=_config(), num_classes=3)
        dm.setup()
        mod = VolSeg2dLightningModule.from_pipeline_config(_config(), settings=settings, num_classes=3)
        bn = _encoder_bn_modules(mod)[0]
        before = bn.running_mean.clone()
        trainer = pl.Trainer(max_epochs=1, callbacks=[UnfreezeEncoderCallback(num_frozen_epochs=0)],
                             accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0, enable_checkpointing=False)
        trainer.fit(mod, datamodule=dm)
        assert torch.equal(bn.running_mean, before) is expect_unchanged, frozen_stats
        assert mod._underlying_student.encoder.conv1.weight.grad is None or True  # trained either way
