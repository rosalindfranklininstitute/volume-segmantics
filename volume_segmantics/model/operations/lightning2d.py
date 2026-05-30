"""PyTorch Lightning trainer and callbacks.

The LR-finder pre-fit step lives on the trainer entry point
(``train_2d_model.py``), not as a callback — Lightning's :class:`Tuner`
contract is procedural.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

import volume_segmantics.utilities.config as cfg
from volume_segmantics.data.pipeline_loader import (
    EMATeacherConfig,
    PipelineConfig,
)
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.operations.trainer_losses import (
    per_head_consistency_loss,
)
from volume_segmantics.model.pipeline_multitask_unet import PipelineMultitaskUnet
from volume_segmantics.model.training.mean_teacher import (
    EMASchedule,
    MeanTeacherModel,
)
from volume_segmantics.model.training.multihead_disagreement import (
    compute_per_head_disagreement,
)
from volume_segmantics.model.training.multitask_calculator import (
    MultiTaskLossOutput,
    PipelineMultiTaskLossCalculator,
)
from volume_segmantics.utilities.base_data_utils import prepare_training_batch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover — handled at script entry
    pl = None
    _pl_import_error = exc
else:
    _pl_import_error = None


logger = logging.getLogger(__name__)


#  LightningModule 


class VolSeg2dLightningModule(pl.LightningModule if pl is not None else object):
    """LightningModule wrapping the pipeline-mode model and loss.

    
    Construction
    ------------
    Two factory contracts:

    * ``__init__(model, calculator, settings, num_classes, ...)`` —
      caller pre-builds the model + calculator. Used in tests and by
      the Lightning entry point in ``train_2d_model.py``.
    * :meth:`from_pipeline_config` — convenience class-method that
      builds head modules, constructs :class:`PipelineMultitaskUnet`,
      and wires up the calculator from a parsed
      :class:`PipelineConfig`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_calculator: PipelineMultiTaskLossCalculator,
        settings: SimpleNamespace,
        num_classes: int,
        *,
        total_steps: Optional[int] = None,
        label_codes: Optional[Mapping[int, str]] = None,
        ema_teacher_config: Optional[EMATeacherConfig] = None,
    ) -> None:
        if pl is None:  # pragma: no cover
            raise ImportError(
                "pytorch-lightning is not installed. "
                "Install via `pip install volume-segmantics` (b3 promotes "
                "Lightning to a required dep)."
            ) from _pl_import_error

        super().__init__()

        # SSL: wrap the student in MeanTeacherModel iff the
        # pipeline.yaml ema_teacher block is enabled. The Mean Teacher
        # wrapper deepcopies the student to produce the teacher; for
        # PipelineMultitaskUnet that's a multi-head teacher
        # automatically. EMASchedule controls the per-step decay ramp.
        ema_cfg = ema_teacher_config
        self._ssl_enabled = bool(
            ema_cfg is not None and ema_cfg.enabled
        )
        if self._ssl_enabled:
            schedule = EMASchedule(
                alpha_warmup=ema_cfg.schedule.alpha_warmup,
                alpha_end=ema_cfg.schedule.alpha_end,
                warmup_steps=ema_cfg.schedule.warmup_steps,
            )
            self.model = MeanTeacherModel(
                student_model=model,
                ema_decay=ema_cfg.schedule.alpha_end,
                schedule=schedule,
            )
            # consistency_weights: dict head_name -> float. Heads
            # missing from the dict get weight 0 (not contributing).
            self._consistency_weights: Dict[str, float] = dict(
                ema_cfg.consistency_weights
            )
        else:
            self.model = model
            self._consistency_weights = {}

        self.loss_calculator = loss_calculator
        self.settings = settings
        self.num_classes = int(num_classes)
        self.total_steps = total_steps if total_steps is not None else 1
        self.label_codes: Dict[int, str] = dict(label_codes or {})

        # Hyperparameters that go into the Lightning checkpoint manifest.
        # We avoid `save_hyperparameters` on the whole settings + model
        # — settings is a SimpleNamespace (not pickle-friendly under
        # all backends) and the model is huge.
        hp: Dict[str, Any] = {
            "num_classes": self.num_classes,
            "head_names": list(getattr(model, "head_names", [])),
        }
        try:
            self.save_hyperparameters(hp)
        except Exception:  # pragma: no cover — older Lightning quirks
            pass

        # epoch_history mirrors v0.4's trainer + the prediction-side
        # loss-figure script.
        self.epoch_history: List[Dict[str, float]] = []

    #  Pipeline-config factory 

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: PipelineConfig,
        *,
        settings: SimpleNamespace,
        num_classes: int,
        in_channels: int = 1,
        total_steps: Optional[int] = None,
        label_codes: Optional[Mapping[int, str]] = None,
    ) -> "VolSeg2dLightningModule":
        """Build a Lightning module directly from a parsed pipeline config.

        Resolves ``encoder_name`` / ``encoder_weights`` / ``encoder_depth``
        from the legacy ``settings.model`` dict. The head registry's :func:`build_head_modules` builds the head
        modules; :class:`PipelineMultitaskUnet` wraps an SMP / DINO backbone around a single shared decoder and the heads.
        """
        model_cfg = getattr(settings, "model", {}) or {}
        encoder_name = str(model_cfg.get("encoder_name", "resnet34"))
        encoder_weights = model_cfg.get("encoder_weights", "imagenet")
        if encoder_weights in ("None", "null"):
            encoder_weights = None
        encoder_depth = int(model_cfg.get("encoder_depth", 5) or 5)

        # default: single-channel (grayscale) input. 2.5D slicing is
        # honoured if the legacy YAML enables it.
        resolved_in_channels = cfg.get_model_input_channels(settings) or in_channels

        # Build heads from the pipeline config.
        # Decoder out channels for PipelineMultitaskUnet = 16
        # (last entry of the default decoder_channels tuple).
        # The heads need that as their in_channels.
        head_in_channels = 16
        head_modules = build_head_modules(
            pipeline_config.heads,
            in_channels=head_in_channels,
            num_classes=num_classes,
            dim=2,
        )

        model = PipelineMultitaskUnet(
            head_modules=head_modules,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=resolved_in_channels,
        )

        calculator = PipelineMultiTaskLossCalculator.from_pipeline_config(
            pipeline_config, num_classes=num_classes,
        )

        return cls(
            model=model,
            loss_calculator=calculator,
            settings=settings,
            num_classes=num_classes,
            total_steps=total_steps,
            label_codes=label_codes,
            ema_teacher_config=pipeline_config.ema_teacher,
        )

    #  Forward 

    @property
    def _underlying_student(self) -> nn.Module:
        """The bare PipelineMultitaskUnet, unwrapping MeanTeacherModel."""
        if isinstance(self.model, MeanTeacherModel):
            return self.model.student
        return self.model

    @property
    def head_names(self) -> Tuple[str, ...]:
        """Head names from the underlying multi-head student."""
        return tuple(getattr(self._underlying_student, "head_names", ()))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if isinstance(self.model, MeanTeacherModel):
            return self.model(x, use_teacher=False)
        return self.model(x)

    #  Training / validation steps 

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str,
    ) -> torch.Tensor:
        inputs, targets = prepare_training_batch(
            batch, device=self.device, num_labels=self.num_classes,
        )
        if not isinstance(targets, dict):
            raise RuntimeError(
                f"VolSeg2dLightningModule expected pipeline-mode batch "
                f"(dict targets); got targets of type {type(targets).__name__}"
            )
        preds = self.forward(inputs)
        out: MultiTaskLossOutput = self.loss_calculator(
            preds, targets,
            current_step=self.global_step,
            total_steps=self.total_steps,
        )

        total_loss = out.total_loss

        # SSL: per-head consistency loss against the EMA teacher.
        # Only fires during training_step. Validation runs the
        # supervised loss only (we never log val_consistency).
        if (
            stage == "train"
            and self._ssl_enabled
            and isinstance(self.model, MeanTeacherModel)
        ):
            with torch.no_grad():
                teacher_preds = self.model(inputs, use_teacher=True)
            head_names = self.head_names
            for head_name, weight in self._consistency_weights.items():
                if weight == 0.0 or head_name not in head_names:
                    continue
                idx = head_names.index(head_name)
                cons_val = per_head_consistency_loss(
                    head_name, preds[idx], teacher_preds[idx],
                )
                total_loss = total_loss + float(weight) * cons_val
                self.log(
                    f"train_{head_name}_consistency", cons_val,
                    on_step=False, on_epoch=True, sync_dist=True,
                )

            # Per-head disagreement: monitoring only (TensorBoard).
            try:
                student_dict = {n: preds[head_names.index(n)] for n in head_names}
                teacher_dict = {n: teacher_preds[head_names.index(n)] for n in head_names}
                disagreement = compute_per_head_disagreement(
                    student_dict, teacher_dict,
                )
                self.log(
                    "train_disagreement_mean", disagreement.mean(),
                    on_step=False, on_epoch=True, sync_dist=True,
                )
            except Exception:  # noqa: BLE001 — defensive monitoring
                pass

        # Logging contract: ``train_loss`` / ``val_loss`` are the
        # primary scalars. Per-head losses + weights go into the same
        # log namespace so smoke-gate B3.H.1 / B3.H.2 can assert
        # monotone decrease.
        self.log(
            f"{stage}_loss", total_loss,
            prog_bar=True, on_step=False, on_epoch=True,
            sync_dist=True,
        )
        # Always also log the supervised-only loss so SSL runs can
        # separate consistency contribution from supervised
        # convergence.
        if self._ssl_enabled and stage == "train":
            self.log(
                "train_supervised_loss", out.total_loss,
                on_step=False, on_epoch=True, sync_dist=True,
            )
        for head_name, loss_val in out.per_head_losses.items():
            self.log(
                f"{stage}_{head_name}_loss", loss_val,
                on_step=False, on_epoch=True, sync_dist=True,
            )
        if stage == "train":
            for head_name, weight in out.per_head_weights.items():
                self.log(
                    f"weight_{head_name}", float(weight),
                    on_step=False, on_epoch=True, sync_dist=True,
                )
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Advance the EMA teacher after every training step.

        Lightning's ``on_train_batch_end`` fires after the optimiser
        has stepped, so ``self.model.update_teacher()`` correctly
        EMA's the just-updated student weights into the teacher.
        """
        if self._ssl_enabled and isinstance(self.model, MeanTeacherModel):
            self.model.update_teacher()

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "val")

    #  Optimizer + scheduler 

    def configure_optimizers(self):
        starting_lr = float(getattr(self.settings, "starting_lr", 5e-5) or 5e-5)
        encoder_lr_mult = getattr(self.settings, "encoder_lr_multiplier", 1.0)
        # ``False`` / ``None`` / 1.0 -> no parameter-group split.
        if encoder_lr_mult in (None, False) or float(encoder_lr_mult) == 1.0:
            optimizer = torch.optim.AdamW(
                self._optimised_parameters(), lr=starting_lr,
            )
            return optimizer

        encoder_lr_mult = float(encoder_lr_mult)
        encoder_params, other_params = self._split_encoder_params()
        if not encoder_params:
            # No encoder found — fall back to single LR.
            return torch.optim.AdamW(
                self._optimised_parameters(), lr=starting_lr,
            )

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": starting_lr * encoder_lr_mult},
            {"params": other_params,   "lr": starting_lr},
        ])
        return optimizer

    def _optimised_parameters(self) -> List[torch.nn.Parameter]:
        """Parameters the optimiser should see.

        When SSL is on, :class:`MeanTeacherModel` carries both student
        and teacher submodules — but the teacher's ``requires_grad`` is
        :data:`False` (it's EMA-updated, not gradient-updated).
        :meth:`nn.Module.parameters` walks both, so we explicitly drill
        to the student parameters when SSL is enabled. Without SSL,
        ``self.parameters()`` is correct as-is.
        """
        if isinstance(self.model, MeanTeacherModel):
            return list(self.model.student.parameters())
        return list(self.parameters())

    def _split_encoder_params(
        self,
    ) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """Split optimised parameters into (encoder, everything else).

        Used by :meth:`configure_optimizers` to build the parameter
        groups for the differential encoder/decoder LR. Looks up the
        encoder via the **underlying student** model's ``.encoder``
        attribute — drills through :class:`MeanTeacherModel` when SSL
        is on.
        """
        student = self._underlying_student
        if not hasattr(student, "encoder"):
            return [], list(self._optimised_parameters())
        encoder_param_ids = {id(p) for p in student.encoder.parameters()}
        encoder_params: List[torch.nn.Parameter] = []
        other_params: List[torch.nn.Parameter] = []
        for p in self._optimised_parameters():
            if id(p) in encoder_param_ids:
                encoder_params.append(p)
            else:
                other_params.append(p)
        return encoder_params, other_params

    #  Encoder freeze / unfreeze 

    def freeze_encoder(self) -> None:
        student = self._underlying_student
        if not hasattr(student, "encoder"):
            return
        for p in student.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen.")

    def unfreeze_encoder(self) -> None:
        student = self._underlying_student
        if not hasattr(student, "encoder"):
            return
        for p in student.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen.")

    #  Checkpoint helpers 

    def get_model_state_dict_for_volseg(self) -> Dict[str, torch.Tensor]:
        """Plain state_dict of the underlying student model.

        Used by :class:`VolSegCheckpointCallback` to write a vol-seg-
        compatible ``.pytorch`` checkpoint that v0.4's prediction
        scripts can load. When SSL is on, we save the student weights
        (the teacher is recoverable from the student via deep-copy +
        EMA replay if needed).
        """
        return self._underlying_student.state_dict()


#  Callbacks 


class UnfreezeEncoderCallback(pl.Callback if pl is not None else object):
    """Two-stage frozen -> unfrozen fine-tuning.

    Mirrors v0.4's ``num_cyc_frozen`` / ``num_cyc_unfrozen`` semantics.
    The encoder is frozen at fit start; on the boundary epoch the
    callback toggles ``requires_grad`` to True and the rest of training
    uses the differential LR (encoder LR = decoder LR × multiplier).

    Parameters
    ----------
    num_frozen_epochs
        Frozen-phase epoch count. ``0`` skips the freeze altogether.
    """

    def __init__(self, num_frozen_epochs: int) -> None:
        self.num_frozen_epochs = int(num_frozen_epochs or 0)
        self._unfrozen = False

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.num_frozen_epochs > 0:
            pl_module.freeze_encoder()
            self._unfrozen = False
        else:
            self._unfrozen = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Lightning epochs are 0-indexed.
        epoch_just_finished = trainer.current_epoch + 1
        if (
            not self._unfrozen
            and self.num_frozen_epochs > 0
            and epoch_just_finished >= self.num_frozen_epochs
        ):
            pl_module.unfreeze_encoder()
            self._unfrozen = True


class EpochHistoryCallback(pl.Callback if pl is not None else object):
    """Collects per-epoch logged metrics into a list of dicts.

    Mirrors the format v0.4's loss-figure / CSV writers expect:
    ``[{"epoch": 0, "train_loss": ..., "val_loss": ..., ...}, ...]``.
    The list lives on the Lightning module so downstream scripts can
    pick it up after ``trainer.fit`` returns.
    """

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if not hasattr(pl_module, "epoch_history"):
            pl_module.epoch_history = []
        row: Dict[str, float] = {"epoch": int(trainer.current_epoch)}
        for k, v in trainer.callback_metrics.items():
            try:
                row[k] = float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                continue
        pl_module.epoch_history.append(row)


class VolSegCheckpointCallback(pl.Callback if pl is not None else object):
    """Vol-seg-compatible ``.pytorch`` checkpoint writer.

    Saves alongside Lightning's :class:`ModelCheckpoint`. The
    vol-seg ``.pytorch`` format is what v0.4's prediction scripts
    (``model-predict-2d``) load — Lightning's ``.ckpt`` is not
    interoperable.

    """

    def __init__(
        self,
        output_path: Path,
        model_struc_dict: Dict[str, Any],
        *,
        label_codes: Optional[Mapping[int, str]] = None,
        head_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.model_struc_dict = dict(model_struc_dict)
        self.label_codes = dict(label_codes or {})
        self.head_metadata = (
            dict(head_metadata) if head_metadata is not None else None
        )
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # v0.5 Bug 2 fix #1: skip sanity-check phase.
        if getattr(trainer, "sanity_checking", False):
            return
        if trainer.global_rank != 0:
            return
        metrics = trainer.callback_metrics
        if "val_loss" not in metrics:
            return
        try:
            current_val = float(metrics["val_loss"])
        except (TypeError, ValueError):
            return
        if math.isnan(current_val):
            return
        if current_val < self.best_val_loss:
            self.best_val_loss = current_val
            self._save_volseg_checkpoint(pl_module, current_val)

    def on_train_end(self, trainer, pl_module) -> None:
        # v0.5 Bug 2 fix #2: always-save-final.
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        try:
            current_val = float(
                trainer.callback_metrics.get("val_loss", float("nan"))
            )
        except (TypeError, ValueError):
            current_val = float("nan")
        self._save_volseg_checkpoint(pl_module, current_val)

    def _save_volseg_checkpoint(
        self, pl_module: VolSeg2dLightningModule, val_loss: float,
    ) -> None:
        state_dict = pl_module.get_model_state_dict_for_volseg()
        model_dict: Dict[str, Any] = {
            "model_state_dict": state_dict,
            "model_struc_dict": self.model_struc_dict,
            "optimizer_state_dict": {},  # Lightning owns the optimiser.
            "loss_val": val_loss,
            "label_codes": self.label_codes,
        }
        if self.head_metadata is not None:
            model_dict["head_metadata"] = self.head_metadata

        logger.info("Saving vol-seg checkpoint to %s", self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_dict, self.output_path)


__all__ = [
    "EpochHistoryCallback",
    "UnfreezeEncoderCallback",
    "VolSeg2dLightningModule",
    "VolSegCheckpointCallback",
]
