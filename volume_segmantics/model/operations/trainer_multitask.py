"""
Multi-task loss tracking and calculation for the 2D trainer.

This module contains:
- MultiTaskLossTracker: Tracks individual task losses and metrics
- MultiTaskLossCalculator: Computes multi-task losses with configurable weights
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from volume_segmantics.data.pytorch3dunet_losses import DiceLoss


@dataclass
class MultiTaskLossTracker:
    """Tracks individual task losses across batches for reporting."""
    seg_losses: List[float] = field(default_factory=list)
    boundary_losses: List[float] = field(default_factory=list)
    task3_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    
    # For eval metrics per task
    seg_dice_scores: List[float] = field(default_factory=list)
    boundary_dice_scores: List[float] = field(default_factory=list)
    
    # Per-class Dice scores
    per_class_dice: Dict[int, List[float]] = field(default_factory=dict)
    
    def append_losses(self, losses: Dict[str, float]):
        """Append loss values from a batch."""
        if "seg" in losses:
            self.seg_losses.append(losses["seg"])
        if "boundary" in losses:
            self.boundary_losses.append(losses["boundary"])
        if "task3" in losses:
            self.task3_losses.append(losses["task3"])
        if "total" in losses:
            self.total_losses.append(losses["total"])
    
    def append_metrics(self, metrics: Dict[str, float]):
        """Append evaluation metrics from a batch."""
        if "seg_dice" in metrics:
            self.seg_dice_scores.append(metrics["seg_dice"])
        if "boundary_dice" in metrics:
            self.boundary_dice_scores.append(metrics["boundary_dice"])
        
        # Per-class Dice
        for key, value in metrics.items():
            if key.startswith("dice_class_"):
                class_idx = int(key.split("_")[-1])
                if class_idx not in self.per_class_dice:
                    self.per_class_dice[class_idx] = []
                if not math.isnan(value):
                    self.per_class_dice[class_idx].append(value)
    
    def get_average_losses(self) -> Dict[str, float]:
        """Get average losses across all tracked batches."""
        result = {}
        if self.seg_losses:
            result["seg"] = float(np.average(self.seg_losses))
        if self.boundary_losses:
            result["boundary"] = float(np.average(self.boundary_losses))
        if self.task3_losses:
            result["task3"] = float(np.average(self.task3_losses))
        if self.total_losses:
            result["total"] = float(np.average(self.total_losses))
        return result
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all tracked batches."""
        result = {}
        if self.seg_dice_scores:
            result["seg_dice"] = float(np.average(self.seg_dice_scores))
        if self.boundary_dice_scores:
            result["boundary_dice"] = float(np.average(self.boundary_dice_scores))
        
        # Per-class averages
        for class_idx, scores in self.per_class_dice.items():
            if scores:
                result[f"dice_class_{class_idx}"] = float(np.average(scores))
        
        return result
    
    def clear(self):
        """Clear all tracked values for next epoch."""
        self.seg_losses.clear()
        self.boundary_losses.clear()
        self.task3_losses.clear()
        self.total_losses.clear()
        self.seg_dice_scores.clear()
        self.boundary_dice_scores.clear()
        self.per_class_dice.clear()


class MultiTaskLossCalculator:
    """
    Handles multi-task loss calculation with configurable weights and loss functions.
    
    Supports:
    - Segmentation (primary task): CE or Dice-based losses
    - Boundary detection (auxiliary): BCE or Dice
    - Optional task3: BCE or custom
    """
    
    def __init__(
        self,
        seg_criterion: nn.Module,
        seg_weight: float = 1.0,
        boundary_weight: float = 1.0,
        task3_weight: float = 1.0,
        use_cross_entropy: bool = False,
        boundary_loss_type: str = "bce",  # "bce", "dice", or "bce_dice"
        num_classes: int = 6,
    ):
        self.seg_criterion = seg_criterion
        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.task3_weight = task3_weight
        self.use_cross_entropy = use_cross_entropy
        self.boundary_loss_type = boundary_loss_type
        self.num_classes = num_classes
        
        if boundary_loss_type in ("dice", "bce_dice"):
            self.boundary_dice = DiceLoss(normalization="sigmoid")
    
    def _compute_boundary_loss(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary loss with configured loss type."""
        if self.boundary_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(output, target)
        elif self.boundary_loss_type == "dice":
            return self.boundary_dice(output, target)
        elif self.boundary_loss_type == "bce_dice":
            bce = F.binary_cross_entropy_with_logits(output, target)
            dice = self.boundary_dice(output, target)
            return 0.5 * bce + 0.5 * dice
        else:
            raise ValueError(f"Unknown boundary loss type: {self.boundary_loss_type}")
    
    def compute(
        self,
        outputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual and total losses for all tasks.
        
        Args:
            outputs: Tuple of model outputs (seg_output, boundary_output, ...)
            targets: Dict with task targets {"seg": tensor, "boundary": tensor, ...}
        
        Returns:
            Dict with individual task losses and weighted 'total' loss
        """
        losses = {}
        total = torch.tensor(0.0, device=outputs[0].device)
        
        # Segmentation Loss (Task 1) 
        if "seg" in targets:
            seg_output = outputs[0]
            seg_target = targets["seg"]
            
            if self.use_cross_entropy:
                # CrossEntropyLoss expects class indices, not one-hot
                seg_loss = self.seg_criterion(
                    seg_output, 
                    torch.argmax(seg_target, dim=1)
                )
            else:
                seg_loss = self.seg_criterion(seg_output, seg_target.float())
            
            losses["seg"] = seg_loss
            total = total + self.seg_weight * seg_loss
        
        # Boundary Loss (Task 2) 
        if "boundary" in targets:
            boundary_target = targets["boundary"]
            
            if len(outputs) < 2:
                raise ValueError(
                    "Model has only 1 output but boundary target provided. "
                    "Check MultitaskUnet configuration (num_tasks >= 2)."
                )
            
            boundary_output = outputs[1]
            
            # Validate and handle channel mismatch
            out_channels = boundary_output.shape[1]
            target_channels = boundary_target.shape[1]
            
            if out_channels != target_channels:
                if out_channels > target_channels:
                    logging.warning(
                        f"Boundary output has {out_channels} channels but target has "
                        f"{target_channels}. Slicing output to match. Consider fixing "
                        "decoder output channels in model config."
                    )
                    boundary_output = boundary_output[:, :target_channels, :, :]
                else:
                    raise ValueError(
                        f"Boundary output channels ({out_channels}) < target channels "
                        f"({target_channels}). Model misconfiguration."
                    )
            
            boundary_loss = self._compute_boundary_loss(boundary_output, boundary_target)
            losses["boundary"] = boundary_loss
            total = total + self.boundary_weight * boundary_loss
        
        # Task 3 Loss
        if "task3" in targets:
            task3_target = targets["task3"]
            
            if len(outputs) < 3:
                raise ValueError(
                    "Model has < 3 outputs but task3 target provided. "
                    "Check MultitaskUnet configuration (num_tasks >= 3)."
                )
            
            task3_output = outputs[2]
            
            # Handle channel mismatch
            if task3_output.shape[1] != task3_target.shape[1]:
                task3_output = task3_output[:, :task3_target.shape[1], :, :]
            
            task3_loss = F.binary_cross_entropy_with_logits(task3_output, task3_target)
            losses["task3"] = task3_loss
            total = total + self.task3_weight * task3_loss
        
        losses["total"] = total
        return losses
