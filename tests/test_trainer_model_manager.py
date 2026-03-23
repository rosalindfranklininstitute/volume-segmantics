
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg

from volume_segmantics.model.operations.trainer_model_manager import ModelManager


class DummySegModel(torch.nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 2):
        super().__init__()
        # Names include substrings the manager relies on ("encoder", "head")
        self.encoder = torch.nn.Linear(in_features, in_features)
        self.head = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def _make_manager(label_no: int = 2):
    settings = SimpleNamespace(
        # only fields ModelManager.__init__ reads
        use_semi_supervised=False,
        use_sam=False,
        adaptive_sam=False,
    )
    return ModelManager(settings=settings, model_device_num=0, label_no=label_no)


def test_get_model_structure_dict_multitask_changes_type_and_task_out_channels():
    manager = _make_manager(label_no=2)
    settings = SimpleNamespace(
        model={"type": "U_NET"},
        use_multitask=True,
        num_tasks=3,
        decoder_sharing="shared",
        use_2_5d_slicing=False,
    )

    struct = manager.get_model_structure_dict(settings=settings, data_dir=None)

    assert struct["type"] == utils.ModelType.MULTITASK_UNET
    assert struct["in_channels"] == 1  # not using 2.5D slicing in this test
    assert struct["classes"] == 2
    assert struct["num_tasks"] == 3
    assert struct["decoder_sharing"] == "shared"
    assert struct["task_out_channels"] == [2, 1, 1]


def test_freeze_model_and_unfreeze_model_toggles_encoder_requires_grad():
    manager = _make_manager()
    model = DummySegModel()

    # Sanity: encoder initially trainable
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.head.parameters())

    manager.freeze_model(model)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    manager.unfreeze_model(model)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_create_optimizer_differential_lr_creates_encoder_and_head_groups():
    manager = _make_manager()
    model = DummySegModel()

    learning_rate = 1e-3
    encoder_lr_multiplier = 0.1
    optimizer = manager.create_optimizer(
        model=model,
        learning_rate=learning_rate,
        use_differential_lr=True,
        encoder_lr_multiplier=encoder_lr_multiplier,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    # Expect separate encoder and head param groups
    assert len(optimizer.param_groups) == 2

    lrs = sorted([g["lr"] for g in optimizer.param_groups])
    assert lrs == sorted([learning_rate * encoder_lr_multiplier, learning_rate])


def test_count_trainable_and_total_parameters():
    manager = _make_manager()
    model = DummySegModel(in_features=4, num_classes=2)

    total = manager.count_parameters(model)
    trainable = manager.count_trainable_parameters(model)
    assert total == trainable  # everything requires_grad by default

    manager.freeze_model(model)
    trainable_after = manager.count_trainable_parameters(model)
    assert trainable_after < total


def test_create_onecycle_lr_scheduler_smoke(tmp_path):
    manager = _make_manager()
    model = DummySegModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = manager.create_onecycle_lr_scheduler(
        optimizer=optimizer,
        num_epochs=1,
        lr_to_use=1e-3,
        steps_per_epoch=5,
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    # Step through the whole cycle (no assertions needed; just ensure it runs)
    for _ in range(5):
        scheduler.step()


def test_load_model_weights_loads_state_and_optimizer_and_returns_loss_val(tmp_path):
    manager = _make_manager()
    model = DummySegModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Save initial checkpoint values
    ckpt_path = tmp_path / "weights.pt"
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_val": 0.42,
        "label_codes": {"dummy": 1},
    }
    torch.save(ckpt, ckpt_path)

    # Modify model weights so we can confirm loading worked
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    returned_loss = manager.load_model_weights(
        model=model,
        optimizer=optimizer,
        checkpoint_path=ckpt_path,
        gpu=False,
        load_optimizer=True,
        use_semi_supervised=False,
    )
    assert returned_loss == pytest.approx(0.42, abs=1e-7)

    for k, v in model.state_dict().items():
        assert torch.allclose(v, orig_state[k])

