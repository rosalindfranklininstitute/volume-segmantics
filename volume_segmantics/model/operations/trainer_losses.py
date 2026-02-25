"""
Loss functions and utilities for the 2D trainer.

This module contains:
- ConsistencyLoss: For semi-supervised learning (Mean Teacher)
- ClassWeightedDiceLoss: Dice loss with per-class weighting
- CombinedCEDiceLoss: Combined Cross-Entropy and Dice loss
- get_rampup_ratio: Utility function for consistency loss ramp-up
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between student and teacher predictions.
    Uses MSE on softmax probabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        student_pred: torch.Tensor, 
        teacher_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between student and teacher predictions.
        
        Args:
            student_pred: Student model prediction (logits)
            teacher_pred: Teacher model prediction (logits)
        
        Returns:
            Consistency loss value (MSE on softmax probabilities)
        """
        # Apply softmax to both predictions 
        student_probs = torch.softmax(student_pred, dim=1)
        teacher_probs = torch.softmax(teacher_pred, dim=1)
        
        # MSE loss on probabilities
        return self.mse_loss(student_probs, teacher_probs)


def get_rampup_ratio(
    current_iter: int, 
    rampup_start: int, 
    rampup_end: int, 
    rampup_type: str = "sigmoid"
) -> float:
    """
    Calculate consistency loss weight with ramp-up.
   
    
    Args:
        current_iter: Current iteration number
        rampup_start: Start ramp-up at this iteration
        rampup_end: End ramp-up at this iteration
        rampup_type: Type of ramp-up ("sigmoid" or "linear")
    
    Returns:
        Current ramp-up ratio (0.0 to 1.0)
    """
    if current_iter < rampup_start:
        return 0.0
    elif current_iter >= rampup_end:
        return 1.0
    else:
        if rampup_type == "sigmoid":
            # Sigmoid ramp-up
            phase = (current_iter - rampup_start) / (rampup_end - rampup_start)
            return math.exp(-5.0 * (1.0 - phase) ** 2)
        else:
            # Linear ramp-up
            return (current_iter - rampup_start) / (rampup_end - rampup_start)


class ClassWeightedDiceLoss(nn.Module):
    """
    Dice loss with per-class weighting for multi-class segmentation.
    
    Supports multiple weighting strategies:
    - 'uniform': Equal weight for all classes
    - 'inverse_freq': Weight inversely proportional to class frequency
    - 'inverse_sqrt_freq': Weight inversely proportional to sqrt of frequency
    - 'custom': User-provided weights
    
    Args:
        num_classes: Number of segmentation classes
        weight_mode: Weighting strategy ('uniform', 'inverse_freq', 'inverse_sqrt_freq', 'custom')
        class_weights: Custom weights when weight_mode='custom' (list or tensor)
        exclude_background: Whether to exclude class 0 from loss computation
        smooth: Smoothing factor to avoid division by zero
        softmax: Whether to apply softmax to predictions (set False if already applied)
    """
    
    def __init__(
        self,
        num_classes: int,
        weight_mode: str = "uniform",
        class_weights: Optional[List[float]] = None,
        exclude_background: bool = False,
        smooth: float = 1e-7,
        softmax: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_mode = weight_mode
        self.exclude_background = exclude_background
        self.smooth = smooth
        self.softmax = softmax
        
        # Initialize weights
        if weight_mode == "custom" and class_weights is not None:
            self.register_buffer(
                "class_weights", 
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes, dtype=torch.float32)
            )
        
        # Track class frequencies for adaptive weighting
        self.register_buffer("class_pixel_counts", torch.zeros(num_classes))
        self.register_buffer("total_pixels", torch.tensor(0.0))
        
        logging.info(
            f"ClassWeightedDiceLoss initialized: num_classes={num_classes}, "
            f"weight_mode={weight_mode}, exclude_background={exclude_background}"
        )
    
    def _compute_weights_from_batch(
        self, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute class weights from the current batch based on frequency."""
        # Convert MetaTensor to regular tensor if needed (MONAI)
        if hasattr(target, 'as_tensor'):
            target = target.as_tensor()
        
        # target shape: (B, C, H, W) one-hot or (B, H, W) class indices
        if target.dim() == 4:
            # One-hot encoded
            class_counts = target.sum(dim=(0, 2, 3))  # (C,)
            # Convert to regular tensor if MetaTensor
            if hasattr(class_counts, 'as_tensor'):
                class_counts = class_counts.as_tensor()
        else:
            class_counts = torch.zeros(
                self.num_classes, device=target.device, dtype=torch.float32
            )
            for c in range(self.num_classes):
                count = (target == c).sum()
                if hasattr(count, 'as_tensor'):
                    count = count.as_tensor()
                class_counts[c] = count
        
        total = class_counts.sum()

        if hasattr(total, 'as_tensor'):
            total = total.as_tensor()
        
        if self.weight_mode == "inverse_freq":
            freq = class_counts / (total + self.smooth)
            weights = 1.0 / (freq + self.smooth)
        elif self.weight_mode == "inverse_sqrt_freq":
            freq = class_counts / (total + self.smooth)
            weights = 1.0 / (torch.sqrt(freq) + self.smooth)
        else:
            # uniform - ensure on correct device
            weights = torch.ones(self.num_classes, device=target.device, dtype=torch.float32)
        
        # Normalize weights to sum to num_classes
        weight_sum = weights.sum()

        if hasattr(weight_sum, 'as_tensor'):
            weight_sum = weight_sum.as_tensor()
        weights = weights * self.num_classes / (weight_sum + self.smooth)
        
        # Ensure return value is a regular tensor on correct device
        if hasattr(weights, 'as_tensor'):
            weights = weights.as_tensor()
        
        # Final device check
        if isinstance(weights, torch.Tensor) and weights.device != target.device:
            weights = weights.to(target.device)
        
        return weights
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class-weighted Dice loss.
        
        Args:
            pred: Predictions of shape (B, C, H, W) - logits or probabilities
            target: Targets of shape (B, C, H, W) one-hot or (B, H, W) indices
        
        Returns:
            Scalar loss value
        """
        if hasattr(target, 'as_tensor'):
            target = target.as_tensor()
        if hasattr(pred, 'as_tensor'):
            pred = pred.as_tensor()
        
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 3:
            # (B, H, W) -> (B, C, H, W)
            target = F.one_hot(target.long(), self.num_classes)
            target = target.permute(0, 3, 1, 2).float()
        
        if self.weight_mode in ("inverse_freq", "inverse_sqrt_freq"):
            weights = self._compute_weights_from_batch(target)
            if hasattr(weights, 'as_tensor'):
                weights = weights.as_tensor()
        else:
            weights = self.class_weights
        
        if hasattr(weights, 'as_tensor'):
            weights = weights.as_tensor()
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=target.device, dtype=target.dtype)
        
        # Ensure weights are on the same device as target
        if isinstance(weights, torch.Tensor) and weights.device != target.device:
            weights = weights.to(target.device)
        
        # Compute per-class Dice
        # (B, C, H, W) -> (B, C, H*W)
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Intersection and union per class per sample
        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        pred_sum = pred_flat.sum(dim=2)  # (B, C)
        target_sum = target_flat.sum(dim=2)  # (B, C)
        
        # Dice per class per sample
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        start_class = 1 if self.exclude_background else 0
        
        if self.exclude_background:
            # Zero out background weight - ensure clone is on correct device
            weights = weights.clone().to(target.device)
            weights[0] = 0.0
        
        # Weighted average across classes, then average across batch
        # weights: (C,) -> (1, C) for broadcasting
        weighted_dice = dice * weights.unsqueeze(0)  # (B, C)
        
        # Sum weighted dice and normalize by sum of weights
        weight_sum = weights[start_class:].sum()
        if hasattr(weight_sum, 'as_tensor'):
            weight_sum = weight_sum.as_tensor()
        loss = 1.0 - weighted_dice[:, start_class:].sum(dim=1) / (weight_sum + self.smooth)
        
        return loss.mean()
    
    def get_current_weights(self) -> torch.Tensor:
        """Return current class weights for logging."""
        return self.class_weights.clone()


class CombinedCEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy (or BCE) and class-weighted Dice loss.
    
    Loss = alpha * CE/BCE + beta * Dice
    
    Supports both binary (BCE) and multi-class (CE) segmentation.
    For binary segmentation (num_classes=2), automatically uses BCE for efficiency.
    For multi-class (num_classes>2), uses Cross-Entropy.
    
    Args:
        num_classes: Number of segmentation classes
        alpha: Weight for Cross-Entropy/BCE loss
        beta: Weight for Dice loss
        dice_weight_mode: Weighting strategy for Dice loss ('uniform', 'inverse_freq', 'inverse_sqrt_freq')
        class_weights_ce: Optional class weights for CE loss
        exclude_background: Whether to exclude background from Dice
        use_bce_for_binary: If True, use BCE for binary segmentation (default: True for efficiency)
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        dice_weight_mode: str = "inverse_sqrt_freq",
        class_weights_ce: Optional[List[float]] = None,
        exclude_background: bool = False,
        use_bce_for_binary: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.use_bce = (num_classes == 2) and use_bce_for_binary
        
        # Choose BCE for binary, CE for multi-class
        if self.use_bce:
            # Binary segmentation: use BCE
            self.ce_loss = nn.BCEWithLogitsLoss()
            logging.info(f"CombinedCEDiceLoss: Using BCE for binary segmentation (num_classes=2)")
        else:
            # Multi-class segmentation: use Cross-Entropy
            if class_weights_ce is not None:
                ce_weights = torch.tensor(class_weights_ce, dtype=torch.float32)
                self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights)
            else:
                self.ce_loss = nn.CrossEntropyLoss()
            logging.info(f"CombinedCEDiceLoss: Using Cross-Entropy for multi-class segmentation (num_classes={num_classes})")
        
        # Dice loss - use uniform weighting for binary, class-weighted for multi-class
        if num_classes == 2 and dice_weight_mode == "inverse_sqrt_freq":
            # For binary, uniform is more appropriate
            dice_weight_mode_actual = "uniform"
        else:
            dice_weight_mode_actual = dice_weight_mode
        
        self.dice_loss = ClassWeightedDiceLoss(
            num_classes=num_classes,
            weight_mode=dice_weight_mode_actual,
            exclude_background=exclude_background,
            softmax=True,
        )
        
        logging.info(
            f"CombinedCEDiceLoss: alpha={alpha}, beta={beta}, "
            f"dice_weight_mode={dice_weight_mode_actual}, "
            f"exclude_background={exclude_background}"
        )
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, C, H, W) one-hot or (B, H, W) class indices
        """
        if self.use_bce:
            # Binary segmentation: BCE expects (B, 1, H, W) or (B, H, W) predictions
            # and (B, 1, H, W) or (B, H, W) targets
            if pred.shape[1] == 2:
                # Convert from 2-channel to single channel for BCE
                # Use sigmoid on first channel (foreground probability)
                pred_bce = pred[:, 0:1, :, :]  # (B, 1, H, W)
            else:
                pred_bce = pred  # Already single channel
            
            if target.dim() == 4:
                # One-hot: extract foreground channel
                if target.shape[1] == 2:
                    target_bce = target[:, 1:2, :, :]  # (B, 1, H, W) - foreground channel
                else:
                    target_bce = target[:, 0:1, :, :]  # (B, 1, H, W)
            else:
                # Class indices: convert to binary mask (1 for foreground)
                target_bce = (target > 0).float().unsqueeze(1)  # (B, 1, H, W)
            
            ce = self.ce_loss(pred_bce, target_bce)
        else:
            # Multi-class segmentation: Cross-Entropy needs class indices
            if target.dim() == 4:
                target_ce = torch.argmax(target, dim=1)
            else:
                target_ce = target
            
            ce = self.ce_loss(pred, target_ce)
        
        dice = self.dice_loss(pred, target)
        
        return self.alpha * ce + self.beta * dice
