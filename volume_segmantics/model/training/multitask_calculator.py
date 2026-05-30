
"""
Pipeline-mode multi-task loss calculator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from volume_segmantics.data import pipeline_registry as _registry
from volume_segmantics.data.pipeline_loader import (
    HeadConfig,
    LossScheduleEntryConfig,
    PipelineConfig,
)
from volume_segmantics.model.loss_schedule import (
    LossSchedule,
    resolve_head_weights,
)


class MultiTaskTargetError(KeyError):
    """Raised when ``targets`` is missing a key the calculator expects."""


class MultiTaskLossOutput(NamedTuple):
    """Per-batch output of :class:`PipelineMultiTaskLossCalculator.compute`.

    Attributes
    ----------
    total_loss
        Scalar tensor with the weighted-sum loss. Backprop target.
    per_head_losses
        ``{head_name: scalar Tensor}`` — pre-weight, per-head loss
        values. Useful for TensorBoard logging + smoke-gate
        regression assertions.
    per_head_weights
        ``{head_name: float}`` — schedule-resolved effective weight
        applied to each head this step.
    """

    total_loss: torch.Tensor
    per_head_losses: Dict[str, torch.Tensor]
    per_head_weights: Dict[str, float]


# Default loss names per head 


_DEFAULT_LOSS_PER_HEAD: Mapping[str, str] = {
    "semantic": "dice_ce",
    "boundary": "boundary_bce_dice",
    "distance": "distance_l1",
    "sdm":      "sdm_l1",
}


def _resolve_head_loss_name(head_name: str, head_cfg: HeadConfig) -> str:
    """``head_cfg.loss`` if set, else the per-head registry default.

    Every head has a sensible default. Boundary defaults to
    ``boundary_bce_dice``, distance to ``distance_l1``, sdm to
    ``sdm_l1``, semantic to ``dice_ce``.
    """
    if head_cfg.loss is not None:
        return head_cfg.loss
    if head_name in _DEFAULT_LOSS_PER_HEAD:
        return _DEFAULT_LOSS_PER_HEAD[head_name]
    raise ValueError(
        f"head {head_name!r} has no loss in pipeline.yaml and no "
        f"per-head default; explicitly set heads.{head_name}.loss"
    )


#  Calculator 


class PipelineMultiTaskLossCalculator(nn.Module):
    """Declarative multi-task loss calculator.

    Built once per training run from a :class:`PipelineConfig`; called
    every batch. Owns the per-head loss :class:`nn.Module` instances so
    they ride device transfers (``.to(device)``) with the trainer.

    Construction binds:

    * ``head_names`` — the ordered tuple of enabled head names. Used
      to validate that ``head_predictions`` arrives in the expected
      order.
    * ``per_head_losses`` — one :class:`nn.Module` per head, built
      from the loss registry with kwargs derived from the
      :class:`HeadConfig` (``num_classes``, ``head_name``, plus the
      head's ``extra`` dict).
    * ``per_head_static_weights`` — :class:`HeadConfig.loss_weight`
      per head.
    * ``per_head_schedules`` — :class:`LossSchedule` per head.

    The calculator is itself an :class:`nn.Module` so its child loss
    modules show up in ``state_dict``.
    """

    def __init__(
        self,
        head_configs: Mapping[str, HeadConfig],
        *,
        num_classes: int,
        loss_schedules: Optional[Mapping[str, LossScheduleEntryConfig]] = None,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1; got {num_classes}"
            )

        # Filter to enabled heads, preserving dict insertion order.
        enabled = [
            (name, cfg) for name, cfg in head_configs.items() if cfg.enabled
        ]
        if not enabled:
            raise ValueError(
                "PipelineMultiTaskLossCalculator: no enabled heads in "
                "the supplied config"
            )

        self.head_names: Tuple[str, ...] = tuple(name for name, _ in enabled)
        self.num_classes: int = int(num_classes)

        # Build one loss module per head.
        per_head_losses: Dict[str, nn.Module] = {}
        per_head_static: Dict[str, float] = {}
        per_head_loss_names: Dict[str, str] = {}
        for head_name, cfg in enabled:
            loss_name = _resolve_head_loss_name(head_name, cfg)
            kwargs = dict(
                head_name=head_name,
                num_classes=num_classes,
            )
            kwargs.update(cfg.extra)
            try:
                loss_mod = _registry.build_loss(loss_name, **kwargs)
            except TypeError as exc:
                raise ValueError(
                    f"failed to build loss {loss_name!r} for head "
                    f"{head_name!r}: {exc}"
                ) from exc
            per_head_losses[head_name] = loss_mod
            per_head_static[head_name] = float(cfg.loss_weight)
            per_head_loss_names[head_name] = loss_name

        self.per_head_static_weights: Dict[str, float] = per_head_static
        self.per_head_loss_names: Dict[str, str] = per_head_loss_names

        # nn.ModuleDict so the loss modules ride device transfers + show
        # up in state_dict for losses with learnable params / buffers.
        self.per_head_losses = nn.ModuleDict(per_head_losses)

        # Schedules.
        sched_src = loss_schedules or {}
        self.per_head_schedules: Dict[str, LossSchedule] = {}
        for head_name in self.head_names:
            self.per_head_schedules[head_name] = LossSchedule.from_config(
                sched_src.get(head_name),
            )

    #  Class-method constructor: from PipelineConfig 

    @classmethod
    def from_pipeline_config(
        cls,
        config: PipelineConfig,
        *,
        num_classes: int,
    ) -> "PipelineMultiTaskLossCalculator":
        """Build directly from a parsed :class:`PipelineConfig`.

        Convenience for the trainer / Lightning module.
        """
        return cls(
            head_configs=config.heads,
            num_classes=num_classes,
            loss_schedules=config.loss_schedule,
        )

    #  Per-batch compute 

    def compute(
        self,
        head_predictions: Sequence[torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> MultiTaskLossOutput:
        """Compute the multi-task loss for one batch.

        Parameters
        ----------
        head_predictions
            Tuple of head outputs from
            :meth:`PipelineMultitaskUnet.forward` — same length and
            ordering as ``self.head_names``.
        targets
            Dict keyed by head name. Missing keys for any enabled head
            raise :class:`MultiTaskTargetError`.
        current_step, total_steps
            For schedule resolution. Schedules are no-ops when both
            are at their defaults (constant weight = end_weight).
        """
        if len(head_predictions) != len(self.head_names):
            raise ValueError(
                f"PipelineMultiTaskLossCalculator: expected "
                f"{len(self.head_names)} predictions (one per enabled "
                f"head); got {len(head_predictions)}"
            )

        missing = [n for n in self.head_names if n not in targets]
        if missing:
            raise MultiTaskTargetError(
                f"targets dict missing keys for enabled heads: {missing}"
            )

        weights = resolve_head_weights(
            self.per_head_schedules,
            self.per_head_static_weights,
            current_step=current_step,
            total_steps=total_steps,
        )

        per_head_losses: Dict[str, torch.Tensor] = {}
        weighted_sum: Optional[torch.Tensor] = None
        for head_name, pred in zip(self.head_names, head_predictions):
            loss_mod = self.per_head_losses[head_name]
            loss_val = loss_mod(pred, targets[head_name])
            per_head_losses[head_name] = loss_val
            scaled = loss_val * weights[head_name]
            weighted_sum = scaled if weighted_sum is None else weighted_sum + scaled

        # weighted_sum is guaranteed non-None because we validated
        # ``self.head_names`` is non-empty in __init__.
        assert weighted_sum is not None

        return MultiTaskLossOutput(
            total_loss=weighted_sum,
            per_head_losses=per_head_losses,
            per_head_weights=dict(weights),
        )

    # Friendly alias — Lightning loss-step convention.
    def forward(
        self,
        head_predictions: Sequence[torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> MultiTaskLossOutput:
        return self.compute(
            head_predictions, targets,
            current_step=current_step, total_steps=total_steps,
        )


__all__ = [
    "MultiTaskLossOutput",
    "MultiTaskTargetError",
    "PipelineMultiTaskLossCalculator",
]
