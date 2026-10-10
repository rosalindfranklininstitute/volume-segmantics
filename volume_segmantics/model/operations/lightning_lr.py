"""Learning-rate policy for the Lightning trainer.

Two policies, chosen by the train setting ``lr_policy``:

* ``find_onecycle`` (default): the legacy trainer's semantics. Before each
  phase a learning-rate sweep runs from ``starting_lr`` to ``end_lr`` for
  ``lr_find_epochs`` epochs, the rate at the steepest loss descent divided by
  :data:`cfg.LR_DIVISOR` is selected, and the phase trains under a OneCycle
  schedule whose warm-up fraction is ``pct_lr_inc``. The unfrozen phase sweeps
  a range divided by ``lr_reduce_factor`` on the partially trained model.
* ``constant``: train at ``starting_lr`` with no sweep and no schedule, the
  Lightning trainer's behaviour up to this version.

The sweep runs inside the fit through :class:`LRPolicyCallback`, once at train
start and once at the unfreeze boundary, so a multi-GPU run computes it on
rank 0 and broadcasts the result.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Tuple

import torch

import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.operations.trainer_lr_finder import (
    LearningRateFinder,
    warn_if_learning_rate_too_low,
)
from volume_segmantics.utilities.base_data_utils import prepare_training_batch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import LRSchedulerConfig
except ImportError:  # pragma: no cover
    pl = None
    LRSchedulerConfig = None

LR_POLICIES = ("find_onecycle", "constant")
DEFAULT_LR_POLICY = "find_onecycle"


def lr_policy_from_settings(settings: Any) -> str:
    """The ``lr_policy`` a settings object asks for, validated."""
    raw = getattr(settings, "lr_policy", None)
    policy = DEFAULT_LR_POLICY if raw in (None, "", False) else str(raw).strip().lower()
    if policy not in LR_POLICIES:
        raise ValueError(
            f"lr_policy must be one of {LR_POLICIES}; got {raw!r}."
        )
    return policy


def lr_search_settings(settings: Any) -> SimpleNamespace:
    """The sweep and schedule parameters, with the legacy defaults."""
    return SimpleNamespace(
        starting_lr=float(getattr(settings, "starting_lr", 5e-5) or 5e-5),
        end_lr=float(getattr(settings, "end_lr", 1e-3) or 1e-3),
        lr_find_epochs=max(1, int(getattr(settings, "lr_find_epochs", 1) or 1)),
        lr_reduce_factor=float(getattr(settings, "lr_reduce_factor", 10) or 10),
        pct_lr_inc=float(getattr(settings, "pct_lr_inc", 0.3) or 0.3),
        encoder_lr_multiplier=_encoder_multiplier(settings),
    )


def _encoder_multiplier(settings: Any) -> float:
    raw = getattr(settings, "encoder_lr_multiplier", 1.0)
    if raw in (None, False):
        return 1.0
    return float(raw)


def select_lr_from_sweep(losses: List[float], lrs: List[float]) -> float:
    """The legacy selection: the rate at the steepest loss descent divided by
    :data:`cfg.LR_DIVISOR`, or :data:`cfg.DEFAULT_MIN_LR` when the curve never
    descends."""
    if not losses:
        return cfg.DEFAULT_MIN_LR
    return float(LearningRateFinder._find_lr_from_graph(list(losses), list(lrs)))


def _supervised_loss(pl_module: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    """The module's supervised loss on one batch, without logging."""
    if isinstance(batch, Mapping) and "labeled" in batch and "unlabeled" in batch:
        batch = batch["labeled"]
    inputs, targets = prepare_training_batch(
        batch, device=pl_module.device, num_labels=pl_module.num_classes,
    )
    preds = pl_module.forward(inputs)
    out = pl_module.loss_calculator(
        preds, targets, current_step=int(pl_module.global_step),
        total_steps=int(pl_module.total_steps),
    )
    return out.total_loss


def sweep_learning_rate(
    pl_module: Any, loader, *, start_lr: float, end_lr: float, num_steps: int,
    smoothing: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """Train ``pl_module`` for up to ``num_steps`` batches under an exponentially
    rising learning rate and return the smoothed losses and the rates.

    The module's weights are restored afterwards; only the curve is kept. The
    early stop (loss above 1 after three quarters of a pass) is the legacy
    finder's.
    """
    num_steps = max(1, int(num_steps))
    # Snapshot the tensors themselves rather than ``state_dict()``: the Mean
    # Teacher wrapper overrides ``state_dict`` with a nested layout that
    # ``load_state_dict`` cannot take back, and an in-place ``copy_`` is exact
    # for parameters the optimizer updated in place.
    snapshot = [
        (t, t.detach().clone())
        for t in list(pl_module.parameters()) + list(pl_module.buffers())
    ]
    params = [p for p in pl_module._optimised_parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("learning-rate sweep: the module has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=start_lr)
    log_ratio = math.log(end_lr / start_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: math.exp(step * log_ratio / num_steps),
    )
    losses: List[float] = []
    lrs: List[float] = []
    was_training = pl_module.training
    pl_module.train()
    loader_len = len(loader) if hasattr(loader, "__len__") else num_steps
    try:
        iters = 0
        while iters < num_steps:
            for batch in loader:
                if iters >= num_steps:
                    break
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                optimizer.zero_grad(set_to_none=True)
                loss = _supervised_loss(pl_module, batch)
                loss.backward()
                optimizer.step()
                lrs.append(float(optimizer.param_groups[0]["lr"]))
                scheduler.step()
                value = float(loss.detach())
                if losses:
                    value = smoothing * value + (1 - smoothing) * losses[-1]
                losses.append(value)
                iters += 1
                if value > 1 and iters > loader_len // 1.333:
                    iters = num_steps
                    break
            else:
                if iters == 0:
                    break  # an empty loader
    finally:
        with torch.no_grad():
            for tensor, saved in snapshot:
                tensor.copy_(saved)
        pl_module.zero_grad(set_to_none=True)
        pl_module.train(was_training)
        del optimizer, scheduler
    return losses, lrs


def find_learning_rate(
    pl_module: Any, trainer: Any, *, start_lr: float, end_lr: float,
    lr_find_epochs: int, steps_per_epoch: int, phase: str,
) -> float:
    """Sweep on rank 0, select the rate, and share it with every rank."""
    lr: Optional[float] = None
    if getattr(trainer, "is_global_zero", True):
        loader = _train_loader(trainer)
        num_steps = steps_per_epoch * lr_find_epochs
        logging.info(
            "Finding optimal learning rate for the %s phase: sweep %.2e -> %.2e over "
            "%d epoch(s), %d steps.", phase, start_lr, end_lr, lr_find_epochs, num_steps,
        )
        losses, lrs = sweep_learning_rate(
            pl_module, loader, start_lr=start_lr, end_lr=end_lr, num_steps=num_steps,
        )
        lr = select_lr_from_sweep(losses, lrs)
        logging.info("Selected learning rate: %.6f", lr)
        warn_if_learning_rate_too_low(lr, start_lr=start_lr, end_lr=end_lr, phase=phase)
    strategy = getattr(trainer, "strategy", None)
    if strategy is not None and getattr(trainer, "world_size", 1) > 1:
        lr = float(strategy.broadcast(lr, src=0))
    if lr is None:  # pragma: no cover - only a non-zero rank with no strategy
        lr = cfg.DEFAULT_MIN_LR
    return float(lr)


def _train_loader(trainer: Any):
    """A loader over the training set that is independent of the trainer's.

    The trainer's own ``DataLoader`` keeps one live iterator when its workers
    are persistent; iterating that same object for the sweep would leave the
    first training epoch with nothing to fetch. The sweep therefore gets its
    own loader over the same dataset, with its own workers.
    """
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is not None and hasattr(datamodule, "train_dataloader"):
        loader = datamodule.train_dataloader()
    else:
        loader = getattr(trainer, "train_dataloader", None)
    # With an unlabeled directory the data module returns a Lightning
    # ``CombinedLoader`` over {"labeled", "unlabeled"}; the sweep uses the
    # supervised loss only, so it takes the labeled loader.
    iterables = getattr(loader, "iterables", None)
    if iterables is not None:
        loader = iterables
    if isinstance(loader, Mapping) and "labeled" in loader:
        loader = loader["labeled"]
    if loader is None:
        raise RuntimeError("learning-rate sweep: no training dataloader available")
    return _independent_loader(loader)


def _independent_loader(loader: Any):
    from torch.utils.data import DataLoader
    if not isinstance(loader, DataLoader):
        return loader
    kwargs = dict(
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=int(loader.num_workers),
        collate_fn=loader.collate_fn,
        pin_memory=bool(loader.pin_memory),
        drop_last=bool(loader.drop_last),
        persistent_workers=False,
    )
    if getattr(loader, "batch_sampler", None) is None:  # a custom sampler drives batching
        kwargs.pop("batch_size"); kwargs.pop("shuffle"); kwargs.pop("drop_last")
        kwargs["sampler"] = loader.sampler
    return DataLoader(loader.dataset, **kwargs)


def make_onecycle(optimizer, *, max_lrs: List[float], total_steps: int, pct_start: float):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=max(1, int(total_steps)), pct_start=pct_start,
    )


class LRPolicyCallback(pl.Callback if pl is not None else object):
    """Applies ``find_onecycle``: a sweep and a fresh OneCycle per phase.

    Place it after :class:`UnfreezeEncoderCallback` in the callback list, so
    the encoder's ``requires_grad`` state is right when each sweep runs.
    """

    def __init__(
        self, *, settings: Any, num_frozen_epochs: int, num_unfrozen_epochs: int,
        steps_per_epoch: int,
    ) -> None:
        self.params = lr_search_settings(settings)
        self.num_frozen_epochs = int(num_frozen_epochs or 0)
        self.num_unfrozen_epochs = int(num_unfrozen_epochs or 0)
        self.steps_per_epoch = max(1, int(steps_per_epoch or 1))
        self._unfrozen_applied = False

    #  hooks

    def on_train_start(self, trainer, pl_module) -> None:
        if self.num_frozen_epochs > 0:
            phase, epochs, divisor = "frozen", self.num_frozen_epochs, 1.0
        else:
            phase, epochs, divisor = "unfrozen", self.num_unfrozen_epochs, 1.0
            self._unfrozen_applied = True
        self._start_phase(trainer, pl_module, phase=phase, epochs=epochs, divisor=divisor)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch_just_finished = int(trainer.current_epoch) + 1
        if (
            not self._unfrozen_applied
            and self.num_frozen_epochs > 0
            and self.num_unfrozen_epochs > 0
            and epoch_just_finished >= self.num_frozen_epochs
        ):
            self._unfrozen_applied = True
            self._start_phase(
                trainer, pl_module, phase="unfrozen", epochs=self.num_unfrozen_epochs,
                divisor=self.params.lr_reduce_factor,
            )

    #  phase set-up

    def _start_phase(self, trainer, pl_module, *, phase: str, epochs: int, divisor: float) -> None:
        p = self.params
        lr = find_learning_rate(
            pl_module, trainer,
            start_lr=p.starting_lr / divisor, end_lr=p.end_lr / divisor,
            lr_find_epochs=p.lr_find_epochs, steps_per_epoch=self.steps_per_epoch,
            phase=phase,
        )
        total_steps = self.steps_per_epoch * max(1, int(epochs))
        self.apply_phase(trainer, pl_module, lr=lr, total_steps=total_steps, phase=phase)

    def apply_phase(self, trainer, pl_module, *, lr: float, total_steps: int, phase: str) -> None:
        """Set every optimizer group to its phase rate and install a OneCycle
        schedule over ``total_steps`` steps."""
        optimizers = trainer.optimizers if hasattr(trainer, "optimizers") else []
        if not optimizers:
            raise RuntimeError("LRPolicyCallback: the trainer has no optimizer to configure")
        optimizer = optimizers[0]
        max_lrs = [
            lr * group.get("lr_multiplier", 1.0) for group in optimizer.param_groups
        ]
        for group, group_lr in zip(optimizer.param_groups, max_lrs):
            group["lr"] = group_lr
            group["initial_lr"] = group_lr
        scheduler = make_onecycle(
            optimizer, max_lrs=max_lrs, total_steps=total_steps, pct_start=self.params.pct_lr_inc,
        )
        configs = getattr(trainer, "lr_scheduler_configs", None)
        if configs:
            configs[0].scheduler = scheduler
            configs[0].interval = "step"
            configs[0].frequency = 1
        elif LRSchedulerConfig is not None and configs is not None:
            configs.append(LRSchedulerConfig(scheduler, interval="step", frequency=1))
        record = getattr(pl_module, "phase_learning_rates", None)
        if record is None:
            pl_module.phase_learning_rates = record = []
        record.append({"phase": phase, "lr": float(lr), "total_steps": int(total_steps),
                       "max_lrs": [float(v) for v in max_lrs]})
        logging.info(
            "%s phase: OneCycle over %d steps, peak learning rates %s (pct_lr_inc=%.2f).",
            phase.capitalize(), total_steps,
            ", ".join(f"{v:.2e}" for v in max_lrs), self.params.pct_lr_inc,
        )


__all__ = [
    "DEFAULT_LR_POLICY",
    "LR_POLICIES",
    "LRPolicyCallback",
    "find_learning_rate",
    "lr_policy_from_settings",
    "lr_search_settings",
    "make_onecycle",
    "select_lr_from_sweep",
    "sweep_learning_rate",
]
