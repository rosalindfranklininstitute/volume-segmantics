"""
Pseudo-labeling components for semi-supervised learning.

This module provides:
- PseudoLabelGenerator: Generates and filters pseudo-labels from model predictions
- ConfidenceThresholdScheduler: Schedules confidence threshold over training
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoLabelGenerator:
    """
    Generates and filters pseudo-labels from model predictions on unlabeled data.
    
    Supports multiple confidence estimation methods:
    - Max probability (softmax max)
    - Entropy-based confidence
    - Per-class confidence
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        confidence_method: str = "max_prob",
        min_pixels_per_class: int = 10,
        use_teacher_for_labels: bool = True,
    ):
        """
        Initialize pseudo-label generator.
        
        Args:
            confidence_threshold: Minimum confidence to accept pseudo-label (0.0-1.0)
            confidence_method: Method to compute confidence ("max_prob", "entropy", "per_class")
            min_pixels_per_class: Minimum pixels per class to accept pseudo-label
            use_teacher_for_labels: If True, use teacher model; else use student
        """
        self.confidence_threshold = confidence_threshold
        self.confidence_method = confidence_method
        self.min_pixels_per_class = min_pixels_per_class
        self.use_teacher_for_labels = use_teacher_for_labels
        
        # Statistics tracking
        self.stats = {
            "total_pixels": 0,
            "accepted_pixels": 0,
            "rejected_pixels": 0,
            "per_class_acceptance": {},
        }
    
    def generate_pseudo_labels(
        self,
        model: nn.Module,
        unlabeled_inputs: torch.Tensor,
        num_classes: int,
        use_teacher: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate pseudo-labels from model predictions.
        
        Args:
            model: Model to use for predictions (MeanTeacherModel or standard model)
            unlabeled_inputs: Unlabeled input images (B, C, H, W)
            num_classes: Number of segmentation classes
            use_teacher: Whether to use teacher model (overrides self.use_teacher_for_labels)
        
        Returns:
            Dictionary with:
            - "pseudo_labels": Hard pseudo-labels (B, H, W)
            - "pseudo_labels_onehot": One-hot pseudo-labels (B, C, H, W)
            - "confidence_map": Confidence values (B, H, W)
            - "mask": Binary mask of accepted pixels (B, H, W)
            - "probs": Probability tensor (B, C, H, W)
            - "stats": Statistics about acceptance rate
        """
        if use_teacher is None:
            use_teacher = self.use_teacher_for_labels
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward') and hasattr(model, 'teacher'):
                # MeanTeacherModel
                if use_teacher:
                    outputs = model(unlabeled_inputs, use_teacher=True)
                else:
                    outputs = model(unlabeled_inputs, use_teacher=False)
            else:
                # Standard model
                outputs = model(unlabeled_inputs)
        
        # Handle multi-task outputs (use first output for segmentation)
        if isinstance(outputs, tuple):
            seg_output = outputs[0]
        else:
            seg_output = outputs
        
        # Compute probabilities and confidence
        probs = F.softmax(seg_output, dim=1)  # (B, C, H, W)
        pred_labels = torch.argmax(probs, dim=1)  # (B, H, W)
        
        # Compute confidence map
        confidence_map = self._compute_confidence(probs)
        
        # Create acceptance mask
        mask = confidence_map >= self.confidence_threshold
        
        # Filter by minimum pixels per class
        if self.min_pixels_per_class > 0:
            mask = self._filter_by_class_size(pred_labels, mask, num_classes)
        
        # Convert to one-hot if needed (for loss computation)
        pseudo_labels_onehot = F.one_hot(pred_labels, num_classes).permute(0, 3, 1, 2).float()
        
        # Update statistics
        self._update_stats(pred_labels, mask, num_classes)
        
        return {
            "pseudo_labels": pred_labels,  # Hard labels (B, H, W)
            "pseudo_labels_onehot": pseudo_labels_onehot,  # One-hot (B, C, H, W)
            "confidence_map": confidence_map,  # (B, H, W)
            "mask": mask,  # (B, H, W)
            "probs": probs,  # (B, C, H, W)
            "stats": self._get_current_stats(),
        }
    
    def _compute_confidence(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence map from probabilities.
        
        Args:
            probs: Probability tensor (B, C, H, W)
        
        Returns:
            Confidence map (B, H, W)
        """
        if self.confidence_method == "max_prob":
            # Maximum probability (most common)
            confidence = torch.max(probs, dim=1)[0]
        
        elif self.confidence_method == "entropy":
            # Entropy-based: higher entropy = lower confidence
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            max_entropy = np.log(probs.shape[1])  # Maximum entropy for uniform distribution
            confidence = 1.0 - (entropy / max_entropy)
        
        elif self.confidence_method == "per_class":
            # Per-class confidence (weighted by class frequency)
            # For now, use max_prob (can be extended)
            confidence = torch.max(probs, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")
        
        return confidence
    
    def _filter_by_class_size(
        self,
        pred_labels: torch.Tensor,
        mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Filter mask to ensure minimum pixels per class.
        
        Args:
            pred_labels: Predicted labels (B, H, W)
            mask: Current acceptance mask (B, H, W)
            num_classes: Number of classes
        
        Returns:
            Filtered mask (B, H, W)
        """
        filtered_mask = mask.clone()
        
        for c in range(num_classes):
            class_mask = (pred_labels == c) & mask
            class_pixel_count = class_mask.sum().item()
            
            if class_pixel_count < self.min_pixels_per_class:
                # Reject all pixels of this class
                filtered_mask[pred_labels == c] = False
        
        return filtered_mask
    
    def _update_stats(
        self,
        pred_labels: torch.Tensor,
        mask: torch.Tensor,
        num_classes: int,
    ):
        """Update statistics about pseudo-label acceptance."""
        total = pred_labels.numel()
        accepted = mask.sum().item()
        rejected = total - accepted
        
        self.stats["total_pixels"] += total
        self.stats["accepted_pixels"] += accepted
        self.stats["rejected_pixels"] += rejected
        
        # Per-class statistics
        for c in range(num_classes):
            class_mask = (pred_labels == c) & mask
            class_count = class_mask.sum().item()
            
            if c not in self.stats["per_class_acceptance"]:
                self.stats["per_class_acceptance"][c] = {"accepted": 0, "total": 0}
            
            self.stats["per_class_acceptance"][c]["accepted"] += class_count
            self.stats["per_class_acceptance"][c]["total"] += (pred_labels == c).sum().item()
    
    def _get_current_stats(self) -> Dict:
        """Get current statistics."""
        acceptance_rate = (
            self.stats["accepted_pixels"] / self.stats["total_pixels"]
            if self.stats["total_pixels"] > 0
            else 0.0
        )
        
        return {
            "acceptance_rate": acceptance_rate,
            "total_pixels": self.stats["total_pixels"],
            "accepted_pixels": self.stats["accepted_pixels"],
            "rejected_pixels": self.stats["rejected_pixels"],
            "per_class": self.stats["per_class_acceptance"].copy(),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_pixels": 0,
            "accepted_pixels": 0,
            "rejected_pixels": 0,
            "per_class_acceptance": {},
        }


class ConfidenceThresholdScheduler:
    """
    Schedules confidence threshold over training (curriculum learning).
    
    Strategies:
    - "fixed": Constant threshold
    - "linear": Linear increase from start_threshold to end_threshold
    - "cosine": Cosine annealing from start_threshold to end_threshold
    - "adaptive": Adjust based on acceptance rate
    """
    
    def __init__(
        self,
        start_threshold: float = 0.9,
        end_threshold: float = 0.95,
        schedule_type: str = "fixed",
        start_iter: int = 0,
        end_iter: int = 10000,
        target_acceptance_rate: float = 0.3,
    ):
        """
        Initialize scheduler.
        
        Args:
            start_threshold: Starting confidence threshold
            end_threshold: Ending confidence threshold
            schedule_type: "fixed", "linear", "cosine", "adaptive"
            start_iter: Start scheduling at this iteration
            end_iter: End scheduling at this iteration
            target_acceptance_rate: Target acceptance rate for adaptive scheduling
        """
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.schedule_type = schedule_type
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.target_acceptance_rate = target_acceptance_rate
        self.current_iter = 0
        self.current_threshold = end_threshold
    
    def get_threshold(self, current_iter: int, acceptance_rate: Optional[float] = None) -> float:
        """
        Get current confidence threshold.
        
        Args:
            current_iter: Current iteration
            acceptance_rate: Current acceptance rate (for adaptive scheduling)
        
        Returns:
            Current confidence threshold
        """
        self.current_iter = current_iter
        
        if self.schedule_type == "fixed":
            self.current_threshold = self.end_threshold
            return self.current_threshold
        
        elif self.schedule_type == "linear":
            if current_iter < self.start_iter:
                self.current_threshold = self.start_threshold
            elif current_iter >= self.end_iter:
                self.current_threshold = self.end_threshold
            else:
                progress = (current_iter - self.start_iter) / (self.end_iter - self.start_iter)
                self.current_threshold = self.start_threshold + progress * (self.end_threshold - self.start_threshold)
            return self.current_threshold
        
        elif self.schedule_type == "cosine":
            if current_iter < self.start_iter:
                self.current_threshold = self.start_threshold
            elif current_iter >= self.end_iter:
                self.current_threshold = self.end_threshold
            else:
                progress = (current_iter - self.start_iter) / (self.end_iter - self.start_iter)
                cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
                self.current_threshold = self.start_threshold + cosine_factor * (self.end_threshold - self.start_threshold)
            return self.current_threshold
        
        elif self.schedule_type == "adaptive":
            if acceptance_rate is None:
                self.current_threshold = self.end_threshold
                return self.current_threshold
            
            # Adjust threshold based on acceptance rate
            if acceptance_rate > self.target_acceptance_rate * 1.2:
                # Too many accepted, increase threshold
                self.current_threshold = min(self.end_threshold, self.current_threshold * 1.01)
            elif acceptance_rate < self.target_acceptance_rate * 0.8:
                # Too few accepted, decrease threshold
                self.current_threshold = max(self.start_threshold, self.current_threshold * 0.99)
            else:
                self.current_threshold = self.end_threshold
            return self.current_threshold
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
