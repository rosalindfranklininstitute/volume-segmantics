"""
Evaluation metrics computation for the 2D trainer.

This module contains:
- Multi-class Dice computation (macro and weighted)
- Evaluation metrics calculation for segmentation and boundary tasks
- Utility functions for metric computation
"""

import logging
import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def ensure_tuple_output(output) -> tuple:
    """Ensure model output is a tuple for consistent handling."""
    if isinstance(output, (list, tuple)):
        return tuple(output)
    return (output,)


class MetricsCalculator:
    """
    Computes evaluation metrics for segmentation tasks.
    
    Supports:
    - Multi-class Dice computation (macro and weighted averaging)
    - Boundary detection metrics
    - Per-class Dice scores
    """
    
    def __init__(
        self,
        num_classes: int,
        exclude_background: bool = True,
        dice_averaging: str = "macro",
        settings=None,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of segmentation classes
            exclude_background: Whether to exclude class 0 from Dice averaging
            dice_averaging: Averaging mode ('macro' or 'weighted')
            settings: Settings object (for accessing dice_weight_mode)
        """
        self.num_classes = num_classes
        self.exclude_background = exclude_background
        self.dice_averaging = dice_averaging
        self.settings = settings
        self._boundary_stats = []
    
    def compute_multiclass_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int = None,
        exclude_background: bool = None,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float]:
        """
        Compute per-class Dice and macro-averaged Dice.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of segmentation classes (uses self.num_classes if None)
            exclude_background: Whether to exclude class 0 (uses self.exclude_background if None)
            smooth: Smoothing factor
        
        Returns:
            dice_per_class: List of Dice scores per class (NaN for absent classes)
            mean_dice: Macro-averaged Dice score
        """
        if num_classes is None:
            num_classes = self.num_classes
        if exclude_background is None:
            exclude_background = self.exclude_background
        
        dice_per_class = []
        start_class = 1 if exclude_background else 0
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union < smooth:
                # No pixels of this class in pred or target
                dice_c = float('nan')
            else:
                dice_c = (2.0 * intersection + smooth) / (union + smooth)
                dice_c = dice_c.item()
            
            dice_per_class.append(dice_c)
        
        # Macro average over classes that are present
        valid_dice = [
            d for i, d in enumerate(dice_per_class)
            if i >= start_class and not math.isnan(d)
        ]
        
        mean_dice = sum(valid_dice) / len(valid_dice) if valid_dice else 0.0
        
        return dice_per_class, mean_dice
    
    def compute_weighted_multiclass_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int = None,
        weight_mode: str = "inverse_sqrt_freq",
        exclude_background: bool = None,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float, List[float]]:
        """
        Compute per-class Dice with weighted averaging.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of segmentation classes (uses self.num_classes if None)
            weight_mode: 'uniform', 'inverse_freq', 'inverse_sqrt_freq', 'pixel_count'
            exclude_background: Whether to exclude class 0 (uses self.exclude_background if None)
            smooth: Smoothing factor
        
        Returns:
            dice_per_class: List of Dice scores per class
            weighted_mean_dice: Weighted average Dice
            weights_used: Weights applied to each class
        """
        if num_classes is None:
            num_classes = self.num_classes
        if exclude_background is None:
            exclude_background = self.exclude_background
        
        dice_per_class = []
        class_pixel_counts = []
        start_class = 1 if exclude_background else 0
        
        # Compute per-class Dice and pixel counts
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            pred_sum = pred_c.sum()
            target_sum = target_c.sum()
            union = pred_sum + target_sum
            
            class_pixel_counts.append(target_sum.item())
            
            if union < smooth:
                dice_c = float('nan')
            else:
                dice_c = (2.0 * intersection + smooth) / (union + smooth)
                dice_c = dice_c.item()
            
            dice_per_class.append(dice_c)
        
        # Compute weights
        total_pixels = sum(class_pixel_counts[start_class:])
        weights = []
        
        for c in range(num_classes):
            if c < start_class:
                weights.append(0.0)
                continue
            
            freq = class_pixel_counts[c] / (total_pixels + smooth)
            
            if weight_mode == "inverse_freq":
                w = 1.0 / (freq + smooth) if freq > 0 else 0.0
            elif weight_mode == "inverse_sqrt_freq":
                w = 1.0 / (math.sqrt(freq) + smooth) if freq > 0 else 0.0
            elif weight_mode == "pixel_count":
                # Weight by pixel count (micro-like)
                w = class_pixel_counts[c]
            else:
                # Uniform
                w = 1.0 if class_pixel_counts[c] > 0 else 0.0
            
            weights.append(w)
        
        # Normalize
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        # Compute weighted average (for valid Dice scores)
        weighted_sum = 0.0
        weight_used_sum = 0.0
        
        for c in range(num_classes):
            if not math.isnan(dice_per_class[c]) and weights[c] > 0:
                weighted_sum += dice_per_class[c] * weights[c]
                weight_used_sum += weights[c]
        
        weighted_mean_dice = weighted_sum / weight_used_sum if weight_used_sum > 0 else 0.0
        
        return dice_per_class, weighted_mean_dice, weights
    
    def compute_eval_metrics(
        self,
        outputs: tuple,
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for all tasks with proper multi-class handling.
        
        Args:
            outputs: Tuple of model outputs (seg_output, boundary_output, ...)
            targets: Ground truth targets (tensor or dict with task keys)
        
        Returns:
            Dict with 'seg_dice', per-class dice, and optionally 'boundary_dice'.
            Also stores boundary statistics in self._boundary_stats.
        """
        metrics = {}
        
        # Segmentation Metrics
        seg_output = outputs[0]
        if isinstance(targets, dict):
            seg_target = targets["seg"]
        else:
            seg_target = targets
        
        # Hard predictions for evaluation
        seg_probs = F.softmax(seg_output, dim=1)
        seg_pred = torch.argmax(seg_probs, dim=1)
        seg_gt = torch.argmax(seg_target, dim=1)
        
        # Compute multi-class Dice with weights
        if self.dice_averaging == "weighted":
            weight_mode = getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq") if self.settings else "inverse_sqrt_freq"
            dice_per_class, mean_dice, weights = self.compute_weighted_multiclass_dice(
                seg_pred, seg_gt,
                num_classes=self.num_classes,
                weight_mode=weight_mode,
                exclude_background=self.exclude_background,
            )
        else:
            # Macro averaging (default)
            dice_per_class, mean_dice = self.compute_multiclass_dice(
                seg_pred, seg_gt,
                num_classes=self.num_classes,
                exclude_background=self.exclude_background,
            )
        
        metrics["seg_dice"] = mean_dice
        
        # Store per-class Dice
        for c, d in enumerate(dice_per_class):
            metrics[f"dice_class_{c}"] = d if not math.isnan(d) else 0.0
        
        # Boundary Metrics
        if isinstance(targets, dict) and "boundary" in targets and len(outputs) > 1:
            boundary_output = outputs[1]
            boundary_target = targets["boundary"]
            
            # Handle channel mismatch
            if boundary_output.shape[1] != boundary_target.shape[1]:
                boundary_output = boundary_output[:, :boundary_target.shape[1], :, :]
            
            boundary_probs = torch.sigmoid(boundary_output)
            
            # Adaptive threshold for sparse boundaries
            target_positive_ratio = boundary_target.sum().item() / boundary_target.numel()
            
            if target_positive_ratio > 0 and target_positive_ratio < 0.1:
                prob_flat = boundary_probs.view(-1).cpu()
                if len(prob_flat) > 0:
                    k = max(1, int((1 - target_positive_ratio) * len(prob_flat)))
                    k = min(k, len(prob_flat))
                    threshold_val, _ = torch.kthvalue(prob_flat, k)
                    threshold = threshold_val.item()
                    threshold = max(0.5, min(0.95, threshold))
                else:
                    threshold = 0.5
            else:
                threshold = 0.5
            
            boundary_pred = (boundary_probs > threshold).float()
            
            # Dice
            boundary_pred_flat = boundary_pred.view(boundary_pred.shape[0], boundary_pred.shape[1], -1)
            boundary_target_flat = boundary_target.view(boundary_target.shape[0], boundary_target.shape[1], -1)
            
            intersection = (boundary_pred_flat * boundary_target_flat).sum(dim=2)
            pred_sum = boundary_pred_flat.sum(dim=2)
            target_sum = boundary_target_flat.sum(dim=2)
            union = pred_sum + target_sum
            
            dice_per_sample = (2.0 * intersection + 1e-7) / (union + 1e-7)
            boundary_dice = dice_per_sample.mean()
            
            # Probability-based Dice
            prob_flat = boundary_probs.view(boundary_pred.shape[0], boundary_pred.shape[1], -1)
            target_flat = boundary_target_flat
            prob_dice_numerator = (prob_flat * target_flat).sum(dim=2) * 2.0
            prob_dice_denominator = prob_flat.sum(dim=2) + target_flat.sum(dim=2)
            prob_dice = (prob_dice_numerator + 1e-7) / (prob_dice_denominator + 1e-7)
            prob_dice_mean = prob_dice.mean()
            
            # Logging stats
            mean_prob = boundary_probs.mean().item()
            max_prob = boundary_probs.max().item()
            min_prob = boundary_probs.min().item()
            pred_positive_ratio = boundary_pred.sum().item() / boundary_pred.numel()
            
            mean_logit = boundary_output.mean().item()
            max_logit = boundary_output.max().item()
            min_logit = boundary_output.min().item()
            
            intersection_total = (boundary_pred * boundary_target).sum().item()
            pred_total = boundary_pred.sum().item()
            target_total = boundary_target.sum().item()
            
            if boundary_dice.is_cuda:
                boundary_dice = boundary_dice.cpu().detach().numpy()
                prob_dice_mean = prob_dice_mean.cpu().detach().numpy()
            
            metrics["boundary_dice"] = float(boundary_dice)
            metrics["boundary_dice_prob"] = float(prob_dice_mean)
            
            self._boundary_stats.append({
                "dice": float(boundary_dice),
                "dice_prob": float(prob_dice_mean),
                "threshold": threshold,
                "mean_prob": mean_prob,
                "max_prob": max_prob,
                "min_prob": min_prob,
                "mean_logit": mean_logit,
                "max_logit": max_logit,
                "min_logit": min_logit,
                "pred_positive_ratio": pred_positive_ratio,
                "target_positive_ratio": target_positive_ratio,
                "intersection": intersection_total,
                "pred_total": pred_total,
                "target_total": target_total,
            })
        
        return metrics
    
    def get_boundary_stats(self):
        """Get accumulated boundary statistics."""
        return self._boundary_stats
    
    def clear_boundary_stats(self):
        """Clear accumulated boundary statistics."""
        self._boundary_stats = []


def get_eval_metric(metric_name: str):
    """
    Get evaluation metric by name.
    
    Args:
        metric_name: Name of metric ('MeanIoU' or 'DiceCoefficient')
    
    Returns:
        Metric instance
    """
    from volume_segmantics.data.pytorch3dunet_metrics import (
        DiceCoefficient,
        MeanIoU,
    )
    import sys
    
    if metric_name == "MeanIoU":
        logging.info("Using MeanIoU")
        return MeanIoU()
    elif metric_name == "DiceCoefficient":
        logging.info("Using DiceCoefficient")
        return DiceCoefficient()
    else:
        logging.error(f"Unknown evaluation metric: {metric_name}, exiting")
        sys.exit(1)
