"""
Mean Teacher model wrapper for semi-supervised learning.

Based on: Tarvainen & Valpola, "Mean teachers are better role models" and PyMIC.

"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn


@dataclass
class EMASchedule:
    """Linear-ramp EMA warmup schedule for the teacher decay coefficient.


    The decay coefficient ``α`` at step ``t``:

    * ``α_warmup`` at step 0.
    * Linearly ramps to ``α_end`` over ``warmup_steps`` steps.
    * Holds ``α_end`` thereafter.

    Defaults match the design doc: ``α_warmup=0.99``, ``α_end=0.999``,
    ``warmup_steps=500`` — quick early tracking, then stabilisation.
    """

    alpha_warmup: float = 0.99
    alpha_end: float = 0.999
    warmup_steps: int = 500

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_warmup <= 1.0:
            raise ValueError(
                f"alpha_warmup must be in [0, 1]; got {self.alpha_warmup}"
            )
        if not 0.0 <= self.alpha_end <= 1.0:
            raise ValueError(
                f"alpha_end must be in [0, 1]; got {self.alpha_end}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative; "
                f"got {self.warmup_steps}"
            )

    def alpha(self, step: int) -> float:
        """Decay coefficient at ``step``."""
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return float(self.alpha_end)
        t = float(step) / float(self.warmup_steps)
        return float(
            self.alpha_warmup + t * (self.alpha_end - self.alpha_warmup)
        )


class MeanTeacherModel(nn.Module):
    """
    Mean Teacher wrapper that maintains a student model and a teacher model.
    Teacher model is an exponential moving average (EMA) of the student model.

    This wrapper is compatible with all segmentation_models_pytorch models
    and any nn.Module that implements forward(), state_dict(), and load_state_dict().

    Multi-head ready: deep-copying a :class:`PipelineMultitaskUnet`
    student produces a multi-head teacher with the same head set; the
    forward signature (returning a tuple of head outputs) works
    unchanged.

    Based on: Tarvainen & Valpola, "Mean teachers are better role models" and PyMIC
    """

    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.99,
        *,
        schedule: Optional[EMASchedule] = None,
    ):
        """
        Initialize Mean Teacher model.

        Args:
            student_model: The base model to wrap (e.g., smp.Unet,
                :class:`MultitaskUnet`, :class:`PipelineMultitaskUnet`).
            ema_decay: Scalar exponential moving average decay rate.
                Used as the **fallback** when ``schedule`` is ``None``.
            schedule: Optional :class:`EMASchedule` for warmup ramp.
                When supplied, ``α`` interpolates per-step between
                ``alpha_warmup`` and ``alpha_end`` over the schedule's
                ``warmup_steps``. The legacy adaptive-decay formula
                (``min(ema_decay, 1 - 1/(glob_it + 1))``) only fires
                when ``schedule is None``.
        """
        super().__init__()
        self.student = student_model
        self.teacher = copy.deepcopy(student_model)

        # Freeze teacher parameters (updated via EMA, not gradients)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.ema_decay = ema_decay
        self.schedule = schedule
        self.glob_it = 0  # Global iteration counter

        logging.info(
            f"MeanTeacherModel initialized with ema_decay={ema_decay}, "
            f"schedule={schedule}. "
            f"Student model: {type(student_model).__name__}"
        )
    
    def forward(self, x, use_teacher=False):
        """
        Forward pass through student or teacher model.
        
        Args:
            x: Input tensor
            use_teacher: If True, use teacher model; otherwise use student
        
        Returns:
            Model output (can be tuple for multi-task models)
        """
        if use_teacher:
            self.teacher.eval()
            with torch.no_grad():
                return self.teacher(x)
        else:
            return self.student(x)
    
    def update_teacher(self, iter_max: int = None):
        """
        Update teacher model using exponential moving average of student weights.


        Args:
            iter_max: Legacy parameter, kept for back-compat. No longer
                referenced; the new ``schedule`` API supersedes it.
        """
        if self.schedule is not None:
            alpha = float(self.schedule.alpha(self.glob_it))
        else:
            # Legacy adaptive decay path. Bit-equivalent to v0.4.0b2.
            alpha = min(1 - 1 / (self.glob_it + 1), self.ema_decay)

        with torch.no_grad():
            # Update teacher parameters: θ_t = α·θ_t + (1-α)·θ_s
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(alpha).add_(
                    student_param.data, alpha=1.0 - alpha,
                )
            # EMA-update floating-point buffers too (BN running stats).
            # Integer buffers (num_batches_tracked) are skipped.
            for teacher_buf, student_buf in zip(
                self.teacher.buffers(), self.student.buffers(),
            ):
                if teacher_buf.dtype.is_floating_point:
                    teacher_buf.data.mul_(alpha).add_(
                        student_buf.data.to(teacher_buf.dtype),
                        alpha=1.0 - alpha,
                    )

        self.glob_it += 1
    
    def get_student_model(self):
        """Get the student model (for evaluation, saving, etc.)."""
        return self.student
    
    def get_teacher_model(self):
        """Get the teacher model."""
        return self.teacher
    
    def state_dict(self, *args, **kwargs):
        """Return state dict including both student and teacher models.

        Accepts and ignores the standard ``destination`` / ``prefix`` /
        ``keep_vars`` kwargs that PyTorch passes from
        :meth:`nn.Module.state_dict`; we serialise into our own dict
        layout regardless.
        """
        sched = self.schedule
        return {
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "ema_decay": self.ema_decay,
            "schedule": (
                None if sched is None else {
                    "alpha_warmup": sched.alpha_warmup,
                    "alpha_end": sched.alpha_end,
                    "warmup_steps": sched.warmup_steps,
                }
            ),
            "glob_it": self.glob_it,
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict for both student and teacher models."""
        self.student.load_state_dict(state_dict["student"])
        self.teacher.load_state_dict(state_dict["teacher"])
        self.ema_decay = state_dict.get("ema_decay", 0.99)
        sched_dict = state_dict.get("schedule")
        if sched_dict is not None:
            self.schedule = EMASchedule(**sched_dict)
        else:
            self.schedule = None
        self.glob_it = state_dict.get("glob_it", 0)
    
    def parameters(self, recurse: bool = True):
        """
        Return iterator over student model parameters only.
        This ensures optimizer only updates student (teacher updated via EMA).
        
        Args:
            recurse: If True, returns parameters of student and all submodules
        
        Returns:
            Iterator over student model parameters
        """
        return self.student.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Return iterator over student model named parameters only.

        Args:
            prefix: Prefix to prepend to all parameter names
            recurse: If True, returns parameters of student and all submodules

        Returns:
            Iterator over (name, parameter) pairs from student model
        """
        return self.student.named_parameters(prefix=prefix, recurse=recurse)


__all__ = ["EMASchedule", "MeanTeacherModel"]
