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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import volume_segmantics.utilities.config as cfg
from volume_segmantics.data.pipeline_loader import (
    EMATeacherConfig,
    PipelineConfig,
)
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.operations.trainer_losses import (
    get_rampup_ratio,
    per_head_consistency_loss,
)
import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model.model_2d import apply_weight_path_settings, coerce_model_type
from volume_segmantics.model.pipeline_multitask_model import (
    UNET_FAMILY,
    PipelineMultitaskModel,
)
from volume_segmantics.model.pipeline_multitask_unet import (
    PipelineMultitaskUnet,
    check_decoder_sharing,
)
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

#: The logged metric the best checkpoint and early stopping select on
#: (higher is better): mean Dice over the foreground classes present in the
#: validation targets. See ``_SemanticDiceAccumulator.foreground_present_mean``.
CHECKPOINT_MONITOR = "val_fg_dice"

#: ``eval_metric`` values. Dice is always reported (and selects the checkpoint
#: through :data:`CHECKPOINT_MONITOR`); ``MeanIoU`` reports the per-class and
#: mean IoU as well, in the epoch summary, the stats CSV and the loss plot.
EVAL_METRICS = ("DiceCoefficient", "MeanIoU")
#: ``dice_averaging`` values for the reported mean, as the legacy trainer took.
DICE_AVERAGING_MODES = ("macro", "weighted")
#: ``precision`` settings and the Lightning precision each selects.
PRECISION_CHOICES = {
    "32": "32-true", "32-true": "32-true",
    "16": "16-mixed", "16-mixed": "16-mixed",
    "bf16": "bf16-mixed", "bf16-mixed": "bf16-mixed",
    "64": "64-true", "64-true": "64-true",
}


def model_type_from_settings(settings: Any) -> utils.ModelType:
    """The ``model.type`` the settings ask for (``U_Net`` when unset)."""
    model_cfg = getattr(settings, "model", None) or {}
    raw = model_cfg.get("type") if isinstance(model_cfg, Mapping) else None
    return coerce_model_type(raw if raw not in (None, "") else "U_Net")


def eval_metric_from_settings(settings: Any) -> str:
    """The settings' ``eval_metric`` (``DiceCoefficient`` when unset), refusing
    a name that is not one of :data:`EVAL_METRICS`."""
    raw = getattr(settings, "eval_metric", None)
    if raw in (None, "", False):
        return "DiceCoefficient"
    for name in EVAL_METRICS:
        if str(raw).strip().lower() == name.lower():
            return name
    raise ValueError(f"eval_metric must be one of {EVAL_METRICS}; got {raw!r}")


def dice_averaging_from_settings(settings: Any) -> str:
    """The settings' ``dice_averaging`` (``macro`` when unset)."""
    raw = getattr(settings, "dice_averaging", None)
    if raw in (None, "", False):
        return "macro"
    mode = str(raw).strip().lower()
    if mode not in DICE_AVERAGING_MODES:
        raise ValueError(f"dice_averaging must be one of {DICE_AVERAGING_MODES}; got {raw!r}")
    return mode


def resolve_precision(settings: Any, override: Any = None) -> str:
    """The Lightning ``precision`` for a run: ``--precision`` over the settings'
    ``precision`` over full precision.

    ``32`` is full precision; ``16-mixed`` and ``bf16-mixed`` are automatic
    mixed precision in float16 (CUDA only: Lightning falls back to bfloat16
    on a CPU) and in bfloat16; ``16`` and ``bf16`` are taken as their
    ``-mixed`` spellings and ``64`` is double precision.
    """
    raw = override if override not in (None, "") else getattr(settings, "precision", None)
    if raw in (None, "", False):
        return "32-true"
    key = str(raw).strip().lower()
    if key not in PRECISION_CHOICES:
        raise ValueError(
            f"precision must be one of {sorted(set(PRECISION_CHOICES))}; got {raw!r}"
        )
    return PRECISION_CHOICES[key]


def checkpoint_model_type_name(settings: Any, enabled_heads: Sequence[str]) -> str:
    """The architecture marker in a training run's checkpoint file name.

    ``U_NET`` / ``PIPELINE_MULTITASK_UNET`` for the U-Net family, as before;
    the ``model.type`` name (``FPN``, ...) for the others, with the
    ``PIPELINE_MULTITASK_`` prefix when more than the semantic head trains.
    """
    heads = list(enabled_heads)
    multi = len(heads) > 1 or (len(heads) == 1 and heads[0] != "semantic")
    model_type = model_type_from_settings(settings)
    if model_type in UNET_FAMILY:
        return "PIPELINE_MULTITASK_UNET" if multi else "U_NET"
    return f"PIPELINE_MULTITASK_{model_type.name}" if multi else model_type.name


def model_struct_dict_for_checkpoint(
    settings: Any, pl_module: Any, num_classes: int,
) -> Dict[str, Any]:
    """The ``model_struc_dict`` a checkpoint of ``pl_module`` carries so that
    ``create_model_on_device`` rebuilds it: the architecture, its encoder
    arguments, the decoder sharing and the heads."""
    model_cfg = getattr(settings, "model", None) or {}
    model_type = model_type_from_settings(settings)
    student = pl_module._underlying_student
    return {
        "type": "PIPELINE_MULTITASK_UNET" if model_type in UNET_FAMILY else model_type.name,
        "encoder_name": model_cfg.get("encoder_name", "resnet34"),
        "encoder_weights": model_cfg.get("encoder_weights", "imagenet"),
        "encoder_depth": int(model_cfg.get("encoder_depth", 5) or 5),
        "in_channels": cfg.get_model_input_channels(settings) or 1,
        "classes": int(num_classes),
        "decoder_sharing": getattr(student, "decoder_sharing", "shared"),
        "heads": [
            {
                "name": name,
                "out_channels": getattr(student.heads[i], "out_channels", None),
            }
            for i, name in enumerate(pl_module.head_names)
        ],
    }


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

    def compute_iou(self) -> List[float]:
        """Per-class IoU (Jaccard) from the same counts: TP / (TP + FP + FN)."""
        out: List[float] = []
        for c in range(self.num_classes):
            denom = self.tp[c] + self.fp[c] + self.fn[c]
            out.append(self.tp[c] / denom if denom > 0 else 0.0)
        return out

    def average(
        self,
        per_class: Sequence[float],
        *,
        exclude_background: bool = False,
        averaging: str = "macro",
        weight_mode: str = "inverse_sqrt_freq",
    ) -> float:
        """The mean of a per-class score the way the legacy trainer reported it.

        Only classes present in the predictions or the targets count, and
        class 0 is left out under ``exclude_background``. ``macro`` is their
        plain mean; ``weighted`` weights each class by its target pixel count
        under ``weight_mode`` (``inverse_freq``, ``inverse_sqrt_freq``,
        ``pixel_count`` or ``uniform``), normalised over the classes scored.
        """
        start = 1 if exclude_background else 0
        classes = [
            c for c in range(start, self.num_classes)
            if self.tp[c] + self.fp[c] + self.fn[c] > 0
        ]
        if not classes:
            return 0.0
        if averaging != "weighted":
            return sum(per_class[c] for c in classes) / len(classes)
        counts = {c: self.tp[c] + self.fn[c] for c in classes}
        total = sum(counts.values())
        weights: Dict[int, float] = {}
        for c in classes:
            freq = counts[c] / total if total > 0 else 0.0
            if weight_mode == "inverse_freq":
                weights[c] = 1.0 / freq if freq > 0 else 0.0
            elif weight_mode == "inverse_sqrt_freq":
                weights[c] = 1.0 / math.sqrt(freq) if freq > 0 else 0.0
            elif weight_mode == "pixel_count":
                weights[c] = float(counts[c])
            else:  # uniform over the classes with target pixels
                weights[c] = 1.0 if counts[c] > 0 else 0.0
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            return 0.0
        return sum(per_class[c] * weights[c] for c in classes) / weight_sum

    def foreground_present_mean(self) -> float:
        """Mean Dice over the foreground classes present in the targets.

        The checkpoint-selection metric (:data:`CHECKPOINT_MONITOR`).
        ``compute``'s plain mean also averages the background and scores a
        class absent from the validation targets as 0.0, which adds a
        constant (or noise) that says nothing about the model. With no
        foreground class present it falls back to the classes that are
        present, and to 0.0 when there are none.
        """
        per_class = self.compute()
        present = [c for c in range(self.num_classes) if self.tp[c] + self.fn[c] > 0]
        scored = [c for c in present if c != 0] or present
        return sum(per_class[c] for c in scored) / len(scored) if scored else 0.0


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


class _PseudoLabelling:
    """Pseudo-label loss on the unlabeled batch, settings-driven as in the
    raw-torch trainer: hard labels from the teacher's (or the student's)
    view, a confidence mask with a per-class size filter and an optional
    threshold schedule, cross-entropy on the accepted pixels of the student's
    view, and ``pseudo_label_weight`` ramped over
    ``pseudo_label_rampup_start`` .. ``pseudo_label_rampup_end`` steps."""

    def __init__(self, settings: SimpleNamespace, *, teacher_available: bool) -> None:
        from volume_segmantics.model.training.pseudo_labeling import (
            ConfidenceThresholdScheduler,
            PseudoLabelGenerator,
        )
        wants_teacher = bool(getattr(settings, "pseudo_label_use_teacher", teacher_available))
        if wants_teacher and not teacher_available:
            logger.warning(
                "pseudo_label_use_teacher is True but Mean Teacher is not enabled; "
                "pseudo-labels come from the student."
            )
        self.use_teacher = wants_teacher and teacher_available
        threshold = float(getattr(settings, "pseudo_label_confidence_threshold", 0.95))
        self.generator = PseudoLabelGenerator(
            confidence_threshold=threshold,
            confidence_method=str(getattr(settings, "pseudo_label_confidence_method", "max_prob")),
            min_pixels_per_class=int(getattr(settings, "pseudo_label_min_pixels_per_class", 10) or 0),
            use_teacher_for_labels=self.use_teacher,
        )
        self.weight = float(getattr(settings, "pseudo_label_weight", 1.0))
        self.rampup_start = int(getattr(settings, "pseudo_label_rampup_start", 0) or 0)
        self.rampup_end = int(getattr(settings, "pseudo_label_rampup_end", 5000) or 0)
        self.scheduler = ConfidenceThresholdScheduler(
            start_threshold=float(getattr(settings, "pseudo_label_start_threshold", 0.9)),
            end_threshold=threshold,
            schedule_type=str(getattr(settings, "pseudo_label_threshold_schedule", "fixed")),
            start_iter=self.rampup_start,
            end_iter=max(self.rampup_end, self.rampup_start + 1),
            target_acceptance_rate=float(getattr(settings, "pseudo_label_target_acceptance_rate", 0.3)),
        )
        self.total_pixels = 0
        self.accepted_pixels = 0

    @classmethod
    def from_settings(cls, settings: SimpleNamespace, *, teacher_available: bool):
        if not getattr(settings, "use_pseudo_labeling", False):
            return None
        return cls(settings, teacher_available=teacher_available)

    def weight_at(self, step: int) -> float:
        return self.weight * get_rampup_ratio(step, self.rampup_start, self.rampup_end, "sigmoid")

    def threshold_at(self, step: int) -> float:
        acceptance = self.accepted_pixels / self.total_pixels if self.total_pixels else 0.0
        threshold = float(self.scheduler.get_threshold(step, acceptance_rate=acceptance))
        self.generator.confidence_threshold = threshold
        return threshold


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
        # Keep the encoder's BatchNorm running statistics fixed after the
        # unfreeze as well. 
        self._freeze_encoder_bn_stats: bool = bool(
            getattr(settings, "freeze_encoder_bn_stats", True)
            if getattr(settings, "freeze_encoder_bn_stats", None) is not None else True
        )

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
        # The validation metric the settings ask to see (eval_metric) and how
        # its mean is taken (dice_averaging, exclude_background_from_dice,
        # dice_weight_mode).
        # Dice is always reported; MeanIoU adds the IoU beside it.
        self.eval_metric = eval_metric_from_settings(settings)
        self._metric_average_kwargs: Dict[str, Any] = dict(
            exclude_background=bool(getattr(settings, "exclude_background_from_dice", False)),
            averaging=dice_averaging_from_settings(settings),
            weight_mode=str(getattr(settings, "dice_weight_mode", None) or "inverse_sqrt_freq"),
        )
        # Mean Teacher's ramp of the consistency weight (pipeline.yaml
        # ema_teacher.consistency_rampup; the legacy rampup_start/rampup_end).
        self._consistency_ramp = ema_cfg.consistency_rampup if self._ssl_enabled else None
        # Pseudo-labelling on the unlabeled batch (legacy pseudo_label_* settings).
        self._pseudo_labelling = _PseudoLabelling.from_settings(
            settings, teacher_available=self._ssl_enabled,
        )
        self.total_steps = total_steps if total_steps is not None else 1
        # A new module starts in training mode; pin the statistics now rather
        # than waiting for the first ``train()`` call.
        self._pin_encoder_bn_eval()
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
        # boundary head is enabled. Always allocated.
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
        model_type = model_type_from_settings(settings)
        decoder_sharing = check_decoder_sharing(getattr(settings, "decoder_sharing", None))

        # pipeline version default: single-channel (grayscale) input. 2.5D slicing is
        # honoured if the legacy YAML enables it.
        resolved_in_channels = cfg.get_model_input_channels(settings) or in_channels

        if model_type in UNET_FAMILY:
            if model_type == utils.ModelType.MULTITASK_UNET:
                logging.info(
                    "model.type Multitask_Unet is the v0.4 name; building the "
                    "pipeline multi-head U-Net with the heads pipeline.yaml enables."
                )
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

            model: nn.Module = PipelineMultitaskUnet(
                head_modules=head_modules,
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=resolved_in_channels,
                decoder_channels=effective_decoder_channels,
                decoder_sharing=decoder_sharing,
            )
        else:
            # Any other model.type: the pipeline heads sit on that
            # architecture's own decoder.
            model = PipelineMultitaskModel(
                pipeline_config.heads,
                num_classes,
                architecture=model_type,
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                in_channels=resolved_in_channels,
                decoder_sharing=decoder_sharing,
            )
        logging.info(
            "Lightning trainer: model.type %s with encoder %s, %d %s decoder(s), heads %s",
            model_type.name, encoder_name, len(model.decoders), decoder_sharing,
            list(model.head_names),
        )
        # ``full_weights_path`` / ``encoder_weights_path`` from the train
        # settings, applied before the module wraps the model (so an EMA
        # teacher starts from the loaded weights too).
        for note in apply_weight_path_settings(model, settings, encoder_name=encoder_name):
            logging.info("Lightning trainer: loaded %s", note)

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
        self, batch: Mapping[str, Any], stage: str,
    ) -> torch.Tensor:
        # With an unlabeled directory the data module pairs every labeled
        # batch with an unlabeled one: {"labeled": ..., "unlabeled": ...}.
        unlabeled_batch = None
        if (
            isinstance(batch, Mapping) and "labeled" in batch and "unlabeled" in batch
        ):
            unlabeled_batch = batch["unlabeled"]
            batch = batch["labeled"]
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

        # SSL: per-head consistency loss against the EMA teacher, on the
        # unlabeled batch when the data module supplies one (student view vs
        # teacher view), else on the labeled batch. The weight follows the
        # Mean Teacher ramp. Validation runs the supervised loss only.
        teacher_preds: Optional[Tuple[torch.Tensor, ...]] = None
        student_unl_preds: Optional[Tuple[torch.Tensor, ...]] = None
        x_teacher: Optional[torch.Tensor] = None
        if stage == "train" and unlabeled_batch is not None:
            x_student = unlabeled_batch["image"].to(torch.float32).to(self.device)
            x_teacher = unlabeled_batch.get("image_teacher", unlabeled_batch["image"])
            x_teacher = x_teacher.to(torch.float32).to(self.device)
            student_unl_preds = _ensure_tuple_output(self.forward(x_student))
        if (
            stage == "train"
            and self._ssl_enabled
            and isinstance(self.model, MeanTeacherModel)
        ):
            cons_student = student_unl_preds if student_unl_preds is not None else preds
            cons_input = x_teacher if x_teacher is not None else inputs
            with torch.no_grad():
                teacher_preds = _ensure_tuple_output(self.model(cons_input, use_teacher=True))
            ramp = (
                self._consistency_ramp.ratio(int(self.global_step))
                if self._consistency_ramp is not None else 1.0
            )
            self.log(
                "train_consistency_ramp", float(ramp),
                on_step=False, on_epoch=True, sync_dist=True,
            )
            head_names = self.head_names
            for head_name, weight in self._consistency_weights.items():
                if weight == 0.0 or head_name not in head_names:
                    continue
                idx = head_names.index(head_name)
                cons_val = per_head_consistency_loss(
                    head_name, cons_student[idx], teacher_preds[idx],
                )
                total_loss = total_loss + float(weight) * float(ramp) * cons_val
                self.log(
                    f"train_{head_name}_consistency", cons_val,
                    on_step=False, on_epoch=True, sync_dist=True,
                )

            # Per-head disagreement: monitoring only (TensorBoard).
            try:
                student_dict = {n: cons_student[head_names.index(n)] for n in head_names}
                teacher_dict = {n: teacher_preds[head_names.index(n)] for n in head_names}
                disagreement = compute_per_head_disagreement(
                    student_dict, teacher_dict,
                )
                self.log(
                    "train_disagreement_mean", disagreement.mean(),
                    on_step=False, on_epoch=True, sync_dist=True,
                )
            except Exception:  # noqa: BLE001 - defensive monitoring
                pass

        # Pseudo-labelling on the unlabeled batch.
        if (
            stage == "train"
            and self._pseudo_labelling is not None
            and student_unl_preds is not None
        ):
            total_loss = total_loss + self._pseudo_label_loss(
                student_unl_preds, x_teacher, teacher_preds,
            )

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

    def _pseudo_label_loss(
        self,
        student_preds: Tuple[torch.Tensor, ...],
        x_teacher: torch.Tensor,
        teacher_preds: Optional[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Weighted cross-entropy of the student's semantic head against hard
        pseudo-labels on the accepted pixels of the unlabeled batch."""
        pl_cfg = self._pseudo_labelling
        head_names = self.head_names
        sem_idx = head_names.index("semantic") if "semantic" in head_names else 0
        threshold = pl_cfg.threshold_at(int(self.global_step))
        if pl_cfg.use_teacher and isinstance(self.model, MeanTeacherModel):
            source = teacher_preds
            if source is None:
                with torch.no_grad():
                    source = _ensure_tuple_output(self.model(x_teacher, use_teacher=True))
        else:
            student = self._underlying_student
            was_training = self.training
            student.eval()
            with torch.no_grad():
                source = _ensure_tuple_output(student(x_teacher))
            if was_training:
                self.train()  # the override re-pins a frozen encoder to eval
        with torch.no_grad():
            probs = torch.softmax(source[sem_idx].detach(), dim=1)
            labels = torch.argmax(probs, dim=1)
            confidence = pl_cfg.generator._compute_confidence(probs)
            mask = confidence >= threshold
            if pl_cfg.generator.min_pixels_per_class > 0:
                mask = pl_cfg.generator._filter_by_class_size(labels, mask, self.num_classes)
        pl_cfg.total_pixels += int(mask.numel())
        pl_cfg.accepted_pixels += int(mask.sum().item())
        self.log(
            "train_pseudo_label_acceptance", mask.float().mean(),
            on_step=False, on_epoch=True, sync_dist=True,
        )
        self.log(
            "train_pseudo_label_threshold", float(threshold),
            on_step=False, on_epoch=True, sync_dist=True,
        )
        if not bool(mask.any()):
            return torch.zeros((), device=self.device)
        ce = F.cross_entropy(student_preds[sem_idx], labels, reduction="none")
        loss = (ce * mask.float()).sum() / mask.float().sum()
        weight = pl_cfg.weight_at(int(self.global_step))
        self.log(
            "train_pseudo_label_loss", loss,
            on_step=False, on_epoch=True, sync_dist=True,
        )
        self.log(
            "train_pseudo_label_weight", float(weight),
            on_step=False, on_epoch=True, sync_dist=True,
        )
        return float(weight) * loss

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
        acc = self._val_dice_accumulator
        per_class = acc.compute()
        sanity = bool(getattr(self.trainer, "sanity_checking", False))
        if not sanity and per_class:
            # The reported mean follows the settings (dice_averaging,
            # exclude_background_from_dice), as the legacy trainer's did.
            self.log(
                "val_seg_dice", acc.average(per_class, **self._metric_average_kwargs),
                prog_bar=True, on_epoch=True, sync_dist=True,
            )
            for c, val in enumerate(per_class):
                self.log(
                    f"val_dice_class_{c}", float(val),
                    on_epoch=True, sync_dist=True,
                )
            # What VolSegCheckpointCallback and EarlyStopping select on.
            self.log(
                CHECKPOINT_MONITOR,
                acc.foreground_present_mean(),
                on_epoch=True, sync_dist=True,
            )
            if self.eval_metric == "MeanIoU":
                per_class_iou = acc.compute_iou()
                self.log(
                    "val_seg_iou", acc.average(per_class_iou, **self._metric_average_kwargs),
                    prog_bar=True, on_epoch=True, sync_dist=True,
                )
                for c, val in enumerate(per_class_iou):
                    self.log(
                        f"val_iou_class_{c}", float(val),
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
        """AdamW under the settings' ``lr_policy``.

        ``constant`` trains at ``starting_lr`` with no schedule. ``find_onecycle``
        (the default, the legacy trainer's semantics) returns the same optimizer
        with a placeholder OneCycle schedule; :class:`LRPolicyCallback` runs the
        learning-rate sweep at the start of each phase and replaces the group
        rates and the schedule with the ones it finds. Each encoder group
        carries its ``lr_multiplier`` so the callback can scale it.
        """
        from volume_segmantics.model.operations.lightning_lr import (
            lr_policy_from_settings, lr_search_settings, make_onecycle,
        )
        policy = lr_policy_from_settings(self.settings)
        starting_lr = float(getattr(self.settings, "starting_lr", 5e-5) or 5e-5)
        encoder_lr_mult = getattr(self.settings, "encoder_lr_multiplier", 1.0)
        # ``False`` / ``None`` / 1.0 -> no parameter-group split.
        if encoder_lr_mult in (None, False) or float(encoder_lr_mult) == 1.0:
            optimizer = torch.optim.AdamW(
                self._optimised_parameters(), lr=starting_lr,
            )
        else:
            encoder_lr_mult = float(encoder_lr_mult)
            encoder_params, other_params = self._split_encoder_params()
            if not encoder_params:
                # No encoder found -> fall back to a single rate.
                optimizer = torch.optim.AdamW(
                    self._optimised_parameters(), lr=starting_lr,
                )
            else:
                optimizer = torch.optim.AdamW([
                    {"params": encoder_params, "lr": starting_lr * encoder_lr_mult,
                     "lr_multiplier": encoder_lr_mult},
                    {"params": other_params,   "lr": starting_lr, "lr_multiplier": 1.0},
                ])
        if policy == "constant":
            return optimizer
        params = lr_search_settings(self.settings)
        scheduler = make_onecycle(
            optimizer,
            max_lrs=[float(g["lr"]) for g in optimizer.param_groups],
            total_steps=max(1, int(self.total_steps)),
            pct_start=params.pct_lr_inc,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

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
        self._pin_encoder_bn_eval()
        logger.info(
            "Encoder unfrozen%s.",
            " (BatchNorm statistics kept frozen)" if self._freeze_encoder_bn_stats else "",
        )

    def _pin_encoder_bn_eval(self) -> None:
        """Put the encoder's BatchNorm layers in eval mode when their
        statistics are to stay frozen and the encoder is otherwise training."""
        if not (self._freeze_encoder_bn_stats and self.training and not self._encoder_frozen):
            return
        student = self._underlying_student
        encoder = getattr(student, "encoder", None)
        if encoder is None:
            return
        for module in encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

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
        elif mode:
            self._pin_encoder_bn_eval()
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

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # on_train_epoch_end, after validation has run inside the epoch: the
        # epoch's train_loss is aggregated by then and val_seg_dice is this
        # epoch's. 
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
    populated by :class:`VolSeg2dLightningModule`) at the end of the
    training epoch, when both the epoch's aggregated training loss and
    its validation metrics are final. Tracks epoch elapsed time via
    :meth:`on_train_epoch_start`. Skipped during the sanity-check phase. Per-head losses (boundary, distance, sdm)
    appear in parens after Train Loss / Val Loss when present, mirroring
    legacy's multitask formatting.
    """

    def __init__(self) -> None:
        self._tic: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._tic = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # After the epoch's validation, which Lightning runs inside the
        # training epoch: here val_seg_dice is this epoch's (the module logs it
        # in its own on_validation_epoch_end) and so is the aggregated
        # train_loss, which on_validation_end still saw as the previous
        # epoch's, and as absent on the first.
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
        seg_iou = _get("val_seg_iou")

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
        if seg_iou is not None:
            log_parts.append(f"Seg IoU: {seg_iou:.4f}")
        log_parts.append(f"Time: {elapsed:.1f}s")
        _emit(" | ".join(log_parts))

        # Per-class Dice (and IoU under eval_metric: MeanIoU) breakdown.
        num_classes = int(getattr(pl_module, "num_classes", 0) or 0)
        if num_classes > 0:
            label_codes = getattr(pl_module, "label_codes", {}) or {}
            for metric, prefix in (("Dice", "val_dice_class_"), ("IoU", "val_iou_class_")):
                class_parts: List[str] = []
                for c in range(num_classes):
                    value_c = _get(f"{prefix}{c}")
                    if value_c is None:
                        continue
                    name = label_codes.get(c, f"C{c}")
                    class_parts.append(f"{name}: {value_c:.3f}")
                if class_parts:
                    _emit(f"  Per-class {metric}: {' | '.join(class_parts)}")


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


def final_checkpoint_path(output_path: Path) -> Path:
    """Where :class:`VolSegCheckpointCallback` writes the final epoch's weights.

    ``<stem>_final.pth`` beside the best checkpoint: a ``.pth`` so that
    callers taking the ``*.pytorch`` in a directory still get the best
    (``model-predict-2d`` loads either suffix).
    """
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_final.pth")


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    """A logged metric as a float; NaN when it is absent or not a number."""
    try:
        return float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


class VolSegCheckpointCallback(pl.Callback if pl is not None else object):
    """Vol-seg-compatible ``.pytorch`` checkpoint writer.

    Saves alongside Lightning's :class:`ModelCheckpoint`. The vol-seg
    ``.pytorch`` format is what v0.4's prediction scripts
    (``model-predict-2d``) load; Lightning's ``.ckpt`` is not
    interoperable.

    ``output_path`` holds the best epoch by :data:`CHECKPOINT_MONITOR`
    (higher is better; saved on a strict improvement, so a tie keeps the
    earlier epoch). The final epoch's weights are always written to
    :func:`final_checkpoint_path` as well, and to ``output_path`` too when
    no epoch ever produced a best, so a checkpoint always exists there.
    Before this the best was chosen by the lowest ``val_loss`` and the final
    epoch was then written over it, so the kept weights were whatever the
    last epoch happened to be.
    """

    def __init__(
        self,
        output_path: Path,
        model_struc_dict: Dict[str, Any],
        *,
        label_codes: Optional[Mapping[int, str]] = None,
        head_metadata: Optional[Mapping[str, Any]] = None,
        image_size: Optional[int] = None,
        input_stack: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.model_struc_dict = dict(model_struc_dict)
        self.label_codes = dict(label_codes or {})
        # The square the training slices were resized to; prediction resizes
        # its slices to the same scale.
        self.image_size = int(image_size) if image_size else None
        # The 2.5D stack trained on, written as a sibling of model_struc_dict
        # so prediction can check its settings against it exactly.
        self.input_stack = dict(input_stack) if input_stack is not None else None
        self.head_metadata = (
            dict(head_metadata) if head_metadata is not None else None
        )
        self.best_score = float("-inf")
        self.best_epoch: Optional[int] = None

    def on_validation_end(self, trainer, pl_module) -> None:
        # on_validation_end, not on_validation_epoch_end: the module logs the
        # monitored Dice in its own on_validation_epoch_end, which runs after
        # the callbacks' hook of that name, so only here is it this epoch's
        # value rather than the previous epoch's.
        if getattr(trainer, "sanity_checking", False):
            return  # the sanity check must not lock in random weights as best
        if trainer.global_rank != 0:
            return
        score = _metric(trainer.callback_metrics, CHECKPOINT_MONITOR)
        if math.isnan(score):
            return
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = int(getattr(trainer, "current_epoch", 0))
            self._save_volseg_checkpoint(
                trainer, pl_module, self.output_path,
                note=f" ({CHECKPOINT_MONITOR}={score:.4f}, epoch {self.best_epoch})",
            )

    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        # Always save the final weights, beside the best rather than over it.
        self._save_volseg_checkpoint(
            trainer, pl_module, final_checkpoint_path(self.output_path),
            note=" (final epoch)",
        )
        if self.best_epoch is None:
            self._save_volseg_checkpoint(
                trainer, pl_module, self.output_path,
                note=" (no epoch produced a best; final weights)",
            )

    def _save_volseg_checkpoint(
        self,
        trainer: Any,
        pl_module: VolSeg2dLightningModule,
        path: Path,
        *,
        note: str = "",
    ) -> None:
        metrics = getattr(trainer, "callback_metrics", None) or {}
        state_dict = pl_module.get_model_state_dict_for_volseg()
        model_dict: Dict[str, Any] = {
            "model_state_dict": state_dict,
            "model_struc_dict": self.model_struc_dict,
            "optimizer_state_dict": {},  # Lightning owns the optimiser.
            "loss_val": _metric(metrics, "val_loss"),
            # The selection metric these weights scored (NaN if not logged).
            CHECKPOINT_MONITOR: _metric(metrics, CHECKPOINT_MONITOR),
            "label_codes": self.label_codes,
        }
        if self.head_metadata is not None:
            model_dict["head_metadata"] = self.head_metadata
        if self.image_size is not None:
            model_dict["image_size"] = self.image_size
        if self.input_stack is not None:
            model_dict["input_stack"] = self.input_stack

        path = Path(path)
        # Route through the Rich progress console so the message
        # doesn't get clobbered by the next-epoch progress redraw.
        _emit_via_progress_console(
            trainer, f"Saving vol-seg checkpoint to {path}{note}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(model_dict, path)


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
    """End-of-training diagnostic outputs (PNG + CSV).

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
    | ``val_fg_dice``        | ``fg_dice``        |
    | ``val_dice_class_<c>`` | ``dice_class_<c>`` |
    | ``val_seg_iou``        | ``seg_iou``        |
    | ``val_iou_class_<c>``  | ``iou_class_<c>``  |
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
        "val_fg_dice": "fg_dice",
        "val_boundary_dice": "boundary_dice",
        "val_seg_iou": "seg_iou",
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
                # val_dice_class_<c> -> dice_class_<c> (and the IoU ones): the
                # per-class columns the CSV writer reads
                dst_key = renames.get(
                    src_key,
                    src_key.replace("val_dice_class_", "dice_class_").replace("val_iou_class_", "iou_class_"),
                )
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
    "CHECKPOINT_MONITOR",
    "EpochHistoryCallback",
    "EVAL_METRICS",
    "PRECISION_CHOICES",
    "checkpoint_model_type_name",
    "eval_metric_from_settings",
    "model_struct_dict_for_checkpoint",
    "model_type_from_settings",
    "resolve_precision",
    "UnfreezeEncoderCallback",
    "VolSeg2dLightningModule",
    "VolSegCheckpointCallback",
    "VolSegDiagnosticsCallback",
    "VolSegSSLVizCallback",
    "final_checkpoint_path",
]
