"""PyTorch Lightning trainer and callbacks.

The LR-finder pre-fit step lives on the trainer entry point
(``train_2d_model.py``), not as a callback — Lightning's :class:`Tuner`
contract is procedural.
"""

from __future__ import annotations

import logging
import math
import time
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
from volume_segmantics.utilities.atomic_io import atomic_torch_save
from volume_segmantics.utilities.base_data_utils import prepare_training_batch

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover — handled at script entry
    pl = None
    _pl_import_error = exc
else:
    _pl_import_error = None


logger = logging.getLogger(__name__)


class _DropLitPromoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        msg = record.getMessage().lower()
        if "litlogger" in msg or "litmodels" in msg:
            return False
        return True


logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    _DropLitPromoFilter()
)


# Rich-aware logging helper 


def _emit_via_progress_console(trainer: Any, line: str) -> None:
    """Print ``line`` through the active Rich progress console if any.

    Lightning's :class:`RichProgressBar` owns a ``rich.progress.Progress``
    with a ``.console`` attribute. Printing to that console while the
    live display is rendering pushes the message *above* the running
    progress bar without clobbering the redraw — which plain
    :func:`logging.info` cannot do, because Rich's live display
    repaints the terminal area on every tick. We duck-type for the
    ``progress.console.print`` triple to remain compatible with any
    Rich-backed progress callback (including subclasses like
    :class:`NoVNumProgressBar`).

    Falls back to :func:`logging.info` when no Rich progress is
    attached (TQDMProgressBar handles this fine on its own; absence
    of any progress bar is also fine).
    """
    for cb in getattr(trainer, "callbacks", ()) or ():
        progress = getattr(cb, "progress", None)
        if progress is None:
            continue
        console = getattr(progress, "console", None)
        if console is not None and hasattr(console, "print"):
            console.print(line)
            return
    logging.info(line)


# Per-class Dice accumulator (legacy-trainer parity) 

class _SemanticDiceAccumulator:
    """Per-class Dice over the semantic head's argmax predictions.

    Manual TP/FP/FN accumulator (rather than ``torchmetrics.Dice``) to
    avoid version churn between Lightning / torchmetrics releases. The
    accumulator state is plain Python ints, populated batchwise from
    GPU tensors via small ``.sum().item()`` syncs — fine for validation
    where batches are infrequent.

    Usage::

        acc = _SemanticDiceAccumulator(num_classes=2)
        # per validation batch:
        acc.update(preds, targets)  # both (B, H, W) int64
        # at on_validation_epoch_end:
        per_class = acc.compute()   # list[float]
        mean      = sum(per_class) / len(per_class)
        acc.reset()
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.tp: List[int] = [0] * self.num_classes
        self.fp: List[int] = [0] * self.num_classes
        self.fn: List[int] = [0] * self.num_classes

    def reset(self) -> None:
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        for c in range(self.num_classes):
            pred_c = preds_flat == c
            target_c = targets_flat == c
            self.tp[c] += int((pred_c & target_c).sum().item())
            self.fp[c] += int((pred_c & ~target_c).sum().item())
            self.fn[c] += int((~pred_c & target_c).sum().item())

    def compute(self) -> List[float]:
        out: List[float] = []
        for c in range(self.num_classes):
            denom = 2 * self.tp[c] + self.fp[c] + self.fn[c]
            out.append(2 * self.tp[c] / denom if denom > 0 else 0.0)
        return out


class _BinaryDiceAccumulator:
    """Single-class Dice for the boundary head's sigmoid output.

    Boundary predictions arrive as ``(B, 1, H, W)`` raw logits; the
    target is ``(B, 1, H, W)`` ``{0, 1}`` float. We binarise the
    prediction with ``sigmoid > 0.5`` and accumulate TP / FP / FN over
    the validation epoch. The mirrored API matches
    :class:`_SemanticDiceAccumulator` so :meth:`on_validation_epoch_end`
    can treat the two uniformly.
    """

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self, pred_logits: torch.Tensor, target: torch.Tensor,
    ) -> None:
        pred_bin = (torch.sigmoid(pred_logits) > 0.5).reshape(-1)
        target_bin = (target > 0.5).reshape(-1)
        self.tp += int((pred_bin & target_bin).sum().item())
        self.fp += int((pred_bin & ~target_bin).sum().item())
        self.fn += int((~pred_bin & target_bin).sum().item())

    def compute(self) -> Optional[float]:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return None
        return 2 * self.tp / denom


# LightningModule


class VolSeg2dLightningModule(pl.LightningModule if pl is not None else object):
    """LightningModule wrapping the pipeline-mode model and loss.

    Single batch contract: pipeline-mode dicts from
    :class:`PipelineMultiTaskDataset`. Both the semantic-only (legacy
    settings -> :func:`legacy_settings_to_pipeline_config`) and the
    multi-head paths route through this module.

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
                "Install via `pip install volume-segmantics` (pipeline version promotes "
                "Lightning to a required dep)."
            ) from _pl_import_error

        super().__init__()

        # Tracks whether the encoder is currently frozen so that the
        # overridden ``train()`` can re-apply ``encoder.eval()`` after
        # Lightning calls ``self.train()`` at every epoch boundary.
        # ``freeze_encoder`` / ``unfreeze_encoder`` flip this.
        self._encoder_frozen: bool = False

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

        # Per-class semantic Dice accumulator. Reset every val epoch in
        # on_validation_epoch_end; updated per batch in _step when
        # stage == "val". Mirrors legacy trainer's per-class breakdown.
        self._val_dice_accumulator = _SemanticDiceAccumulator(self.num_classes)
        # Boundary-head Dice accumulator — only updates when the
        # boundary head is enabled. Always allocated; the v0.4
        # visualizer expects a ``boundary_dice`` column key, and the
        # accumulator's ``compute()`` returns None when the head was
        # never seen so we skip the log.
        self._val_boundary_dice_accumulator = _BinaryDiceAccumulator()

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
        from the legacy ``settings.model`` dict (v0.4 layout). The pipeline version
        head registry's :func:`build_head_modules` builds the head
        modules; :class:`PipelineMultitaskUnet` wraps an SMP / DINO
        backbone around a single shared decoder + the heads.
        """
        model_cfg = getattr(settings, "model", {}) or {}
        encoder_name = str(model_cfg.get("encoder_name", "resnet34"))
        encoder_weights = model_cfg.get("encoder_weights", "imagenet")
        if encoder_weights in ("None", "null"):
            encoder_weights = None
        encoder_depth = int(model_cfg.get("encoder_depth", 5) or 5)

        # pipeline version default: single-channel (grayscale) input. 2.5D slicing is
        # honoured if the legacy YAML enables it.
        resolved_in_channels = cfg.get_model_input_channels(settings) or in_channels

        # Resolve the decoder's last-stage output channel count so the
        # heads are built with the right in_channels. PipelineMultitaskUnet
        # mirrors this resolution in __init__ but applies it to itself
        # AFTER the heads are constructed — so we have to duplicate it
        # here to avoid the in_channels=16 vs decoder=32 mismatch the
        # DINO depth=4 path would otherwise hit.
        default_decoder_channels = (256, 128, 64, 32, 16)
        is_dino = (
            encoder_name.startswith("dinov2_")
            or encoder_name.startswith("dinov3_")
        )
        effective_depth = min(encoder_depth, 4) if is_dino else encoder_depth
        if is_dino and effective_depth == 4:
            effective_decoder_channels = (256, 128, 64, 32)
        else:
            effective_decoder_channels = default_decoder_channels[:effective_depth]
        head_in_channels = int(effective_decoder_channels[-1])
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
            decoder_channels=effective_decoder_channels,
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

        # Legacy-parity Dice: at val time, take the semantic head's
        # argmax and accumulate per-class TP/FP/FN. The mean + per-class
        # Dice are emitted in on_validation_epoch_end so the
        # LegacyEpochSummaryCallback can print them in the
        # "Seg Dice: …" / "Per-class Dice: …" line.
        if stage == "val":
            head_names = self.head_names
            sem_idx = head_names.index("semantic") if "semantic" in head_names else 0
            sem_logits = preds[sem_idx]
            sem_pred = torch.argmax(sem_logits, dim=1).long()
            sem_target = targets.get("semantic")
            if sem_target is not None:
                self._val_dice_accumulator.update(
                    sem_pred.detach(), sem_target.long().detach(),
                )
            if "boundary" in head_names:
                bnd_idx = head_names.index("boundary")
                bnd_target = targets.get("boundary")
                if bnd_target is not None:
                    self._val_boundary_dice_accumulator.update(
                        preds[bnd_idx].detach(),
                        bnd_target.detach(),
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

    def on_validation_epoch_end(self) -> None:
        # Compute Dice from the accumulator and log it so callbacks
        # (notably LegacyEpochSummaryCallback) can render the legacy-
        # format "Seg Dice: … | Per-class Dice: …" line. Reset the
        # accumulator unconditionally — even during the sanity-check
        # phase, where we deliberately skip emitting metrics — so the
        # next epoch starts with a clean slate.
        per_class = self._val_dice_accumulator.compute()
        sanity = bool(getattr(self.trainer, "sanity_checking", False))
        if not sanity and per_class:
            mean_dice = sum(per_class) / len(per_class)
            self.log(
                "val_seg_dice", mean_dice,
                prog_bar=True, on_epoch=True, sync_dist=True,
            )
            for c, val in enumerate(per_class):
                self.log(
                    f"val_dice_class_{c}", float(val),
                    on_epoch=True, sync_dist=True,
                )
        boundary_dice = self._val_boundary_dice_accumulator.compute()
        if not sanity and boundary_dice is not None:
            self.log(
                "val_boundary_dice", float(boundary_dice),
                on_epoch=True, sync_dist=True,
            )
        self._val_dice_accumulator.reset()
        self._val_boundary_dice_accumulator.reset()

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
        # Put the encoder in eval mode so BatchNorm running stats and
        # Dropout are inactive during the frozen warmup phase. The
        # overridden :meth:`train` re-applies this every time Lightning
        # bounces the module back into train mode at an epoch boundary.
        student.encoder.eval()
        self._encoder_frozen = True
        logger.info("Encoder frozen.")

    def unfreeze_encoder(self) -> None:
        student = self._underlying_student
        if not hasattr(student, "encoder"):
            return
        for p in student.encoder.parameters():
            p.requires_grad = True
        # Restore the encoder's mode to match the parent module.
        if self.training:
            student.encoder.train()
        self._encoder_frozen = False
        logger.info("Encoder unfrozen.")

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Lightning calls ``self.train()`` at the start of every train
        # epoch, which would otherwise re-enable BN running-stat
        # updates and Dropout inside the frozen encoder. Re-pin the
        # encoder to eval mode while it's still frozen.
        if mode and self._encoder_frozen:
            student = self._underlying_student
            if hasattr(student, "encoder"):
                student.encoder.eval()
        return self

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


# Callbacks 


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


class LegacyEpochSummaryCallback(pl.Callback if pl is not None else object):
    """Print a v0.4-style end-of-epoch summary line to the console.

    Reproduces the legacy raw trainer's per-epoch log so users (and
    downstream regex-based parsers) see the same format under
    Lightning::

        Epoch 1 | Train Loss: 0.5234 | Val Loss: 0.4521 | Seg Dice: 0.6123 | Time: 12.3s
          Per-class Dice: bg: 0.850 | fg: 0.620

    Reads ``train_loss``, ``val_loss``, ``val_seg_dice``, and
    ``val_dice_class_<i>`` from ``trainer.callback_metrics`` (all
    populated by :class:`VolSeg2dLightningModule`). Tracks epoch
    elapsed time via :meth:`on_train_epoch_start`. Skipped during the
    sanity-check phase. Per-head losses (boundary, distance, sdm)
    appear in parens after Train Loss / Val Loss when present, mirroring
    legacy's multitask formatting.
    """

    def __init__(self) -> None:
        self._tic: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._tic = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = trainer.callback_metrics
        elapsed = (
            time.perf_counter() - self._tic if self._tic is not None else 0.0
        )

        def _emit(line: str) -> None:
            _emit_via_progress_console(trainer, line)

        def _get(key: str) -> Optional[float]:
            v = metrics.get(key)
            if v is None:
                return None
            try:
                return float(v.item()) if hasattr(v, "item") else float(v)
            except (TypeError, ValueError):
                return None

        train_loss = _get("train_loss")
        val_loss = _get("val_loss")
        seg_dice = _get("val_seg_dice")

        # Per-head loss breakdowns (matches legacy's multitask format).
        head_names: Tuple[str, ...] = tuple(
            getattr(pl_module, "head_names", ()) or ()
        )

        def _head_breakdown(stage: str) -> str:
            if len(head_names) <= 1:
                return ""
            parts: List[str] = []
            for h in head_names:
                v = _get(f"{stage}_{h}_loss")
                if v is not None:
                    short = h[:4].capitalize()
                    parts.append(f"{short}: {v:.4f}")
            return f" ({', '.join(parts)})" if parts else ""

        epoch_no = int(trainer.current_epoch) + 1  # 1-indexed for display
        log_parts: List[str] = [f"Epoch {epoch_no}"]
        if train_loss is not None:
            log_parts.append(f"Train Loss: {train_loss:.4f}{_head_breakdown('train')}")
        if val_loss is not None:
            log_parts.append(f"Val Loss: {val_loss:.4f}{_head_breakdown('val')}")
        if seg_dice is not None:
            log_parts.append(f"Seg Dice: {seg_dice:.4f}")
        log_parts.append(f"Time: {elapsed:.1f}s")
        _emit(" | ".join(log_parts))

        # Per-class Dice breakdown.
        num_classes = int(getattr(pl_module, "num_classes", 0) or 0)
        if num_classes > 0:
            label_codes = getattr(pl_module, "label_codes", {}) or {}
            class_parts: List[str] = []
            for c in range(num_classes):
                dice_c = _get(f"val_dice_class_{c}")
                if dice_c is None:
                    continue
                name = label_codes.get(c, f"C{c}")
                class_parts.append(f"{name}: {dice_c:.3f}")
            if class_parts:
                _emit(f"  Per-class Dice: {' | '.join(class_parts)}")


# Custom progress bar that drops Lightning's ``v_num`` postfix from
# the live metrics display. ``v_num`` is the TensorBoardLogger version
# number — not a training-state quantity, just visual noise. Keep
# RichProgressBar (default when ``rich`` is installed) and fall back
# to TQDMProgressBar otherwise. ``pl is None`` only when Lightning
# isn't installed; in that case we never instantiate the class so a
# stub base of ``object`` is sufficient.
if pl is not None:
    try:
        from pytorch_lightning.callbacks import RichProgressBar as _PB_BASE
    except ImportError:  # pragma: no cover — older Lightning without Rich
        from pytorch_lightning.callbacks import TQDMProgressBar as _PB_BASE
else:  # pragma: no cover — Lightning not installed
    _PB_BASE = object  # type: ignore[assignment,misc]


class NoVNumProgressBar(_PB_BASE):  # type: ignore[misc,valid-type]
    """Progress bar that drops the ``v_num`` postfix.

    Subclass of whichever progress bar Lightning would have selected
    by default (Rich if available, else TQDM). Override ``get_metrics``
    to remove the auto-injected ``v_num`` entry that the base class
    pulls from the trainer's logger.
    """

    def get_metrics(self, trainer, pl_module):  # type: ignore[override]
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


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
            self._save_volseg_checkpoint(trainer, pl_module, current_val)

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
        self._save_volseg_checkpoint(trainer, pl_module, current_val)

    def _save_volseg_checkpoint(
        self,
        trainer: Any,
        pl_module: VolSeg2dLightningModule,
        val_loss: float,
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

        # Route through the Rich progress console so the message
        # doesn't get clobbered by the next-epoch progress redraw.
        _emit_via_progress_console(
            trainer, f"Saving vol-seg checkpoint to {self.output_path}"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(model_dict, self.output_path)


def _ensure_tuple_output(output: Any) -> Tuple[Any, ...]:
    """Wrap a model output in a tuple if it isn't already one.

    :class:`PipelineMultitaskUnet.forward` returns a tuple of head
    outputs; v0.4's :class:`Multitask_Unet` also returns a tuple; bare
    SMP models return a single tensor. The visualizer's
    ``plot_predictions`` family treats element ``[0]`` as the semantic
    output, so a single-tensor return needs wrapping.
    """
    if isinstance(output, tuple):
        return output
    return (output,)


class VolSegDiagnosticsCallback(pl.Callback if pl is not None else object):
    """End-of-training diagnostic outputs (PNG + CSV) for parity with v0.4.

    The legacy raw trainer wrote three artifacts next to the saved
    ``.pytorch`` checkpoint when training finished:

    * ``{stem}_loss_plot.png`` — multi-panel loss + Dice curves.
    * ``{stem}_train_stats.csv`` — per-epoch CSV with the same metrics.
    * ``{stem}_prediction_image.png`` — sample predictions on the
      validation set.

    Lightning's logger writes TensorBoard scalars but not these files,
    and :class:`EpochHistoryCallback` collects metrics in a list-of-
    dicts shape that doesn't match :class:`TrainingVisualizer`'s
    dict-of-lists contract. This callback bridges the two on
    ``on_train_end``: it transposes the history (renaming the
    Lightning metric keys to the v0.4 names the visualizer expects),
    instantiates a :class:`TrainingVisualizer`, and emits all three
    files alongside ``output_path``. Missing metrics fall through
    silently — single-task runs skip the multitask-only panels;
    multi-head runs without one of boundary/distance get a partial
    plot.

    Lightning metric -> v0.4 visualizer key mapping:

    +-------------------------+--------------------+
    | Lightning              | Visualizer         |
    +========================+====================+
    | ``train_loss``         | ``train_total``    |
    | ``val_loss``           | ``valid_total``    |
    | ``train_semantic_loss``| ``train_seg``      |
    | ``val_semantic_loss``  | ``valid_seg``      |
    | ``train_boundary_loss``| ``train_boundary`` |
    | ``val_boundary_loss``  | ``valid_boundary`` |
    | ``train_distance_loss``| ``train_task3``    |
    | ``val_distance_loss``  | ``valid_task3``    |
    | ``val_seg_dice``       | ``seg_dice``       |
    | ``val_dice_class_<c>`` | ``dice_class_<c>`` |
    +-------------------------+--------------------+
    """

    _METRIC_RENAMES: Dict[str, str] = {
        "train_loss": "train_total",
        "val_loss": "valid_total",
        "train_semantic_loss": "train_seg",
        "val_semantic_loss": "valid_seg",
        "train_boundary_loss": "train_boundary",
        "val_boundary_loss": "valid_boundary",
        # pipeline version has distance + sdm heads; the v0.4 visualizer only carves
        # out a single "task3" lane. We route distance there first;
        # if distance isn't enabled but sdm is, sdm fills the slot.
        "train_distance_loss": "train_task3",
        "val_distance_loss": "valid_task3",
        "val_seg_dice": "seg_dice",
        "val_boundary_dice": "boundary_dice",
    }
    _SDM_FALLBACK_RENAMES: Dict[str, str] = {
        "train_sdm_loss": "train_task3",
        "val_sdm_loss": "valid_task3",
    }

    def __init__(
        self,
        output_path: Path,
        num_classes: int,
        *,
        label_codes: Optional[Mapping[int, str]] = None,
        head_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.num_classes = int(num_classes)
        self.label_codes: Dict[int, str] = dict(label_codes or {})
        # Heads beyond the semantic primary -> multitask plot layout.
        head_set = set(head_names or ())
        self.use_multitask = len(head_set - {"semantic"}) > 0
        self._head_set = head_set

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        history = getattr(pl_module, "epoch_history", None) or []
        if not history:
            logger.info(
                "VolSegDiagnosticsCallback: epoch_history is empty; "
                "skipping diagnostic outputs."
            )
            return
        legacy_history = self._transpose_history(history)
        if not legacy_history.get("train_total") or not legacy_history.get(
            "valid_total"
        ):
            logger.info(
                "VolSegDiagnosticsCallback: train/val loss columns "
                "missing from epoch_history; skipping diagnostic outputs."
            )
            return

        # Local import — :mod:`trainer_visualization` pulls in
        # matplotlib at import time, which we want to keep off the
        # default :mod:`lightning_2d` import chain.
        from volume_segmantics.model.operations.trainer_visualization import (
            TrainingVisualizer,
        )

        viz = TrainingVisualizer(
            num_classes=self.num_classes,
            label_codes=self.label_codes,
            use_multitask=self.use_multitask,
        )
        # Save the CSV first — independent of the matplotlib path,
        # so a missing column on the plot side doesn't lose the
        # tabular log. plot_loss_history would otherwise also write
        # the CSV (line ~273 in trainer_visualization) but only on
        # the success path.
        try:
            viz.save_training_stats_csv(
                legacy_history,
                output_dir=self.output_path.parent,
                model_name=self.output_path.stem,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VolSegDiagnosticsCallback: save_training_stats_csv "
                "failed: %s", exc,
            )
        try:
            viz.plot_loss_history(legacy_history, self.output_path)
        except Exception as exc:  # noqa: BLE001 — never break training over a plot
            logger.warning(
                "VolSegDiagnosticsCallback: plot_loss_history failed: %s", exc,
            )

        val_loader = self._resolve_validation_loader(trainer)
        if val_loader is not None:
            try:
                device_num = self._resolve_device_num(pl_module)
                viz.plot_predictions(
                    pl_module._underlying_student,
                    val_loader,
                    device_num,
                    self.output_path,
                    _ensure_tuple_output,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VolSegDiagnosticsCallback: plot_predictions failed: %s",
                    exc,
                )

    @staticmethod
    def _resolve_device_num(pl_module: Any) -> Any:
        try:
            device = next(pl_module.parameters()).device
        except StopIteration:
            return "cpu"
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return "cpu"

    def _transpose_history(
        self, history: List[Dict[str, float]],
    ) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        # Detect whether the run logged a distance head; if not but
        # sdm was logged, route sdm into the task3 lane.
        has_distance = any("train_distance_loss" in row for row in history)
        renames = dict(self._METRIC_RENAMES)
        if not has_distance:
            renames.update(self._SDM_FALLBACK_RENAMES)

        for row in history:
            for src_key, value in row.items():
                if src_key == "epoch":
                    continue
                dst_key = renames.get(src_key, src_key)
                out.setdefault(dst_key, []).append(float(value))

        # Pad shorter columns with zeros so the visualizer's parallel-
        # list indexing doesn't IndexError when a metric started being
        # logged mid-training (e.g. unfreeze-phase-only).
        max_len = max((len(v) for v in out.values()), default=0)
        for key, values in out.items():
            if len(values) < max_len:
                values.extend([0.0] * (max_len - len(values)))

        # The visualizer's multitask path indexes a fixed set of keys
        # unconditionally (``boundary_dice``, ``train_task3``, etc.) and
        # KeyErrors when one is absent. pipeline version doesn't currently emit
        # per-boundary Dice; default any expected-but-missing column
        # to zero-filled. The plot's ``any(...)`` guards still skip
        # over the empty series — they just need the key to exist.
        expected_keys = {"train_total", "valid_total", "seg_dice"}
        if self.use_multitask:
            expected_keys.update({
                "train_seg", "valid_seg",
                "train_boundary", "valid_boundary",
                "train_task3", "valid_task3",
                "boundary_dice",
            })
        for key in expected_keys:
            out.setdefault(key, [0.0] * max_len)
        return out

    @staticmethod
    def _resolve_validation_loader(trainer: Any) -> Any:
        for attr in ("val_dataloaders", "validation_dataloaders"):
            cand = getattr(trainer, attr, None)
            if cand is None:
                continue
            if isinstance(cand, (list, tuple)) and cand:
                return cand[0]
            return cand
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "val_dataloader"):
            try:
                return datamodule.val_dataloader()
            except Exception:  # noqa: BLE001
                return None
        return None


class VolSegSSLVizCallback(pl.Callback if pl is not None else object):
    """Per-epoch SSL diagnostic PNGs (mean-teacher + pseudo-labelling).

    Mirrors v0.4's two ``visualizer.plot_*`` calls in
    ``vol_seg_2d_trainer.py``:

    * ``{stem}_mean_teacher_epoch_{N}.png`` — side-by-side student
      vs teacher predictions on a validation batch. Emitted when SSL
      is on (i.e. ``MeanTeacherModel`` wraps the student) and the
      current epoch is divisible by ``mean_teacher_vis_epoch_interval``.
    * ``{stem}_pseudo_labeling_epoch_{N}.png`` — input + pseudo-
      labels + confidence + accepted-mask on an unlabeled batch.
      Emitted when ``settings.use_pseudo_labeling`` is truthy and a
      ``settings.unlabeled_data_dir`` is configured. The callback
      builds a transient :class:`PseudoLabelGenerator` and a v0.4-
      compatible :class:`UnlabeledDataset` loader the first time it
      runs; both are cached on the instance.

    Intervals come from ``settings``:

    * ``mean_teacher_vis_epoch_interval`` — default 5.
    * ``pseudo_labeling_vis_epoch_interval`` — default 5.

    Set either to ``0`` / ``None`` to disable that viz lane entirely.
    Rank-0 only; skipped during Lightning's sanity-check phase.
    """

    def __init__(
        self,
        output_path: Path,
        settings: SimpleNamespace,
        num_classes: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.settings = settings
        self.num_classes = int(num_classes)
        self.mean_teacher_interval = int(
            getattr(settings, "mean_teacher_vis_epoch_interval", 5) or 0
        )
        self.pseudo_label_interval = int(
            getattr(settings, "pseudo_labeling_vis_epoch_interval", 5) or 0
        )
        # Lazy-built on first use.
        self._pseudo_label_generator: Any = None
        self._unlabeled_loader: Any = None
        self._unlabeled_loader_init_failed: bool = False

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch) + 1  # human-friendly 1-indexed
        if epoch <= 0:
            return

        # Local import — :mod:`trainer_visualization` pulls in
        # matplotlib at import time, which we keep off the default
        # :mod:`lightning_2d` import chain.
        from volume_segmantics.model.operations.trainer_visualization import (
            TrainingVisualizer,
        )
        viz = TrainingVisualizer(
            num_classes=self.num_classes,
            label_codes={},
            use_multitask=False,
        )

        # Mean-teacher viz: only when SSL is wired up.
        ssl_on = bool(getattr(pl_module, "_ssl_enabled", False)) and isinstance(
            pl_module.model, MeanTeacherModel,
        )
        if (
            ssl_on
            and self.mean_teacher_interval > 0
            and epoch % self.mean_teacher_interval == 0
        ):
            val_loader = _resolve_validation_loader_from_trainer(trainer)
            if val_loader is not None:
                try:
                    device_num = _resolve_device_num_from_module(pl_module)
                    # plot_mean_teacher_predictions takes the
                    # ``MeanTeacherModel`` wrapper directly and pulls
                    # student / teacher via its accessors internally.
                    viz.plot_mean_teacher_predictions(
                        pl_module.model,
                        val_loader,
                        device_num,
                        self.output_path,
                        epoch,
                        _ensure_tuple_output,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "VolSegSSLVizCallback: plot_mean_teacher_"
                        "predictions failed at epoch %d: %s", epoch, exc,
                    )

        # Pseudo-labelling viz: needs an unlabeled dataloader + a
        # PseudoLabelGenerator. pipeline version's Lightning training doesn't
        # currently plumb either, so we build both here on demand
        # (cached on first success). Skip the whole branch silently
        # when the user hasn't asked for pseudo-labelling.
        if (
            getattr(self.settings, "use_pseudo_labeling", False)
            and self.pseudo_label_interval > 0
            and epoch % self.pseudo_label_interval == 0
        ):
            unlabeled_loader = self._get_unlabeled_loader()
            if unlabeled_loader is None:
                return
            generator = self._get_pseudo_label_generator()
            try:
                device_num = _resolve_device_num_from_module(pl_module)
                viz.plot_pseudo_labeling_visualization(
                    pl_module.model,  # may be MeanTeacherModel or bare
                    unlabeled_loader,
                    device_num,
                    self.output_path,
                    epoch,
                    generator,
                    _ensure_tuple_output,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VolSegSSLVizCallback: plot_pseudo_labeling_"
                    "visualization failed at epoch %d: %s", epoch, exc,
                )

    def _get_pseudo_label_generator(self) -> Any:
        if self._pseudo_label_generator is not None:
            return self._pseudo_label_generator
        from volume_segmantics.model.training.pseudo_labeling import (
            PseudoLabelGenerator,
        )
        self._pseudo_label_generator = PseudoLabelGenerator(
            confidence_threshold=float(
                getattr(self.settings, "pseudo_label_confidence_threshold", 0.95)
            ),
            confidence_method=str(
                getattr(self.settings, "pseudo_label_confidence_method", "max_prob")
            ),
            min_pixels_per_class=int(
                getattr(self.settings, "pseudo_label_min_pixels_per_class", 10)
            ),
            use_teacher_for_labels=bool(
                getattr(self.settings, "pseudo_label_use_teacher", True)
            ),
        )
        return self._pseudo_label_generator

    def _get_unlabeled_loader(self) -> Any:
        if self._unlabeled_loader is not None:
            return self._unlabeled_loader
        if self._unlabeled_loader_init_failed:
            return None
        unl_dir_raw = getattr(self.settings, "unlabeled_data_dir", None)
        if not unl_dir_raw:
            self._unlabeled_loader_init_failed = True
            return None
        unl_dir = Path(str(unl_dir_raw))
        if not unl_dir.is_dir():
            logger.warning(
                "VolSegSSLVizCallback: unlabeled_data_dir %s does not "
                "exist or is not a directory; pseudo-labelling viz "
                "disabled for this run.", unl_dir,
            )
            self._unlabeled_loader_init_failed = True
            return None
        try:
            from torch.utils.data import DataLoader
            from volume_segmantics.data.datasets import (
                UnlabeledDataset,
                get_augmentation_module,
            )
            aug_module = get_augmentation_module(self.settings)
            img_size = int(getattr(self.settings, "image_size", 256))
            ds = UnlabeledDataset(
                images_dir=unl_dir,
                preprocessing=(
                    aug_module.get_train_preprocess_augs(img_size)
                    if hasattr(aug_module, "get_train_preprocess_augs")
                    else None
                ),
                augmentation=None,
                imagenet_norm=bool(
                    getattr(self.settings, "use_imagenet_norm", True)
                ),
                postprocessing=(
                    aug_module.get_postprocess_augs()
                    if hasattr(aug_module, "get_postprocess_augs")
                    else None
                ),
                use_2_5d_slicing=bool(
                    getattr(self.settings, "use_2_5d_slicing", False)
                ),
            )
            if len(ds.images_fps) == 0:
                logger.warning(
                    "VolSegSSLVizCallback: unlabeled_data_dir %s "
                    "contains no PNG/TIFF images; pseudo-labelling "
                    "viz disabled for this run.", unl_dir,
                )
                self._unlabeled_loader_init_failed = True
                return None
            self._unlabeled_loader = DataLoader(
                ds, batch_size=2, shuffle=False, num_workers=0,
            )
            return self._unlabeled_loader
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VolSegSSLVizCallback: failed to build unlabeled "
                "loader from %s: %s. Pseudo-labelling viz disabled "
                "for this run.", unl_dir, exc,
            )
            self._unlabeled_loader_init_failed = True
            return None


def _resolve_validation_loader_from_trainer(trainer: Any) -> Any:
    for attr in ("val_dataloaders", "validation_dataloaders"):
        cand = getattr(trainer, attr, None)
        if cand is None:
            continue
        if isinstance(cand, (list, tuple)) and cand:
            return cand[0]
        return cand
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is not None and hasattr(datamodule, "val_dataloader"):
        try:
            return datamodule.val_dataloader()
        except Exception:  # noqa: BLE001
            return None
    return None


def _resolve_device_num_from_module(pl_module: Any) -> Any:
    try:
        device = next(pl_module.parameters()).device
    except StopIteration:
        return "cpu"
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return "cpu"


__all__ = [
    "EpochHistoryCallback",
    "UnfreezeEncoderCallback",
    "VolSeg2dLightningModule",
    "VolSegCheckpointCallback",
    "VolSegDiagnosticsCallback",
    "VolSegSSLVizCallback",
]
