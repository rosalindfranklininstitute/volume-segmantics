"""Early stopping does not stop a run while the encoder is frozen.

Before this change, one ``EarlyStopping`` monitored the frozen phase and the
unfrozen phase. A plateau in the frozen phase stopped the training before the
encoder got training. Also, ``patience: 0`` had the value 4.
"""
from types import SimpleNamespace

import pytest

pl = pytest.importorskip("pytorch_lightning")


def _epochs_run(tmp_path, *, frozen, unfrozen, patience):
    from pytorch_lightning.demos.boring_classes import BoringModel

    from volume_segmantics.model.operations.lightning2d import PhasedEarlyStopping

    class Plateau(BoringModel):
        """A validation score that does not increase after the first epoch."""

        def __init__(self):
            super().__init__()
            self.epochs = 0

        def validation_step(self, batch, batch_idx):
            self.log("score", 0.5)

        def on_train_epoch_end(self):
            self.epochs += 1

    model = Plateau()
    trainer = pl.Trainer(
        max_epochs=frozen + unfrozen, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
        callbacks=[PhasedEarlyStopping(
            monitor="score", patience=patience, mode="max",
            num_frozen_epochs=frozen, num_unfrozen_epochs=unfrozen,
        )],
    )
    trainer.fit(model)
    return model.epochs


def test_a_frozen_phase_plateau_does_not_end_the_run_before_the_unfrozen_phase(tmp_path):
    # Before this change, EarlyStopping stopped this run after 2 epochs while
    # the encoder was frozen.
    epochs = _epochs_run(tmp_path, frozen=4, unfrozen=4, patience=1)
    assert epochs == 5  # the full frozen phase, then one unfrozen epoch of patience


def test_a_single_phase_stops_on_patience_as_before(tmp_path):
    assert _epochs_run(tmp_path, frozen=0, unfrozen=6, patience=1) == 2
    assert _epochs_run(tmp_path, frozen=6, unfrozen=0, patience=1) == 2


def test_patience_zero_means_off_and_blank_means_the_default():
    from volume_segmantics.model.operations.lightning2d import early_stopping_patience

    assert early_stopping_patience(SimpleNamespace(patience=0)) == 0
    assert early_stopping_patience(SimpleNamespace(patience=None)) == 4
    assert early_stopping_patience(SimpleNamespace()) == 4
    assert early_stopping_patience(SimpleNamespace(patience=7)) == 7
    with pytest.raises(ValueError):
        early_stopping_patience(SimpleNamespace(patience=-1))
