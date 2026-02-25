"""
Logging and statistics tracking for the 2D trainer.

This module contains:
- Model architecture logging
- Gradient statistics logging
- Parameter statistics logging
- Boundary prediction statistics logging
- Learning rate logging
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.nn import DataParallel


class TrainingLogger:
    """
    Handles logging of training statistics and diagnostics.
    
    Provides detailed logging for:
    - Model architecture and trainability
    - Gradient statistics
    - Parameter statistics
    - Boundary prediction statistics
    - Learning rates
    """
    
    def __init__(self):
        """Initialize training logger."""
        self._grad_log_counter = 0
        self._param_stats_history = {}
        self._prev_boundary_stats = None
    
    def log_model_architecture(
        self,
        model: torch.nn.Module,
        use_multitask: bool = False,
    ):
        """
        Log detailed information about model architecture and trainability.
        
        Args:
            model: Model to analyze
            use_multitask: Whether model is multi-task
        """
        logging.info("=" * 80)
        logging.info("MODEL ARCHITECTURE DIAGNOSTICS")
        logging.info("=" * 80)
        
        # Unwrap DataParallel if needed
        model_to_check = model.module if isinstance(model, DataParallel) else model
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logging.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        
        component_stats = {
            "encoder": {"trainable": 0, "frozen": 0, "total": 0},
            "decoder": {"trainable": 0, "frozen": 0, "total": 0},
            "head": {"trainable": 0, "frozen": 0, "total": 0},
            "other": {"trainable": 0, "frozen": 0, "total": 0},
        }
        
        head_trainability = {}
        
        for name, param in model_to_check.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad
            
            if "encoder" in name:
                component = "encoder"
            elif "decoder" in name or "decoders" in name:
                component = "decoder"
            elif "head" in name or "heads" in name:
                component = "head"
                head_idx = None
                for i in range(len(model_to_check.heads) if hasattr(model_to_check, 'heads') else 0):
                    if f"heads.{i}" in name or f"head.{i}" in name:
                        head_idx = i
                        break
                if head_idx is not None:
                    if head_idx not in head_trainability:
                        head_trainability[head_idx] = {"trainable": 0, "frozen": 0, "total": 0}
                    if is_trainable:
                        head_trainability[head_idx]["trainable"] += num_params
                    else:
                        head_trainability[head_idx]["frozen"] += num_params
                    head_trainability[head_idx]["total"] += num_params
            else:
                component = "other"
            
            component_stats[component]["total"] += num_params
            if is_trainable:
                component_stats[component]["trainable"] += num_params
            else:
                component_stats[component]["frozen"] += num_params
        
        logging.info("\nComponent Breakdown:")
        for component, stats in component_stats.items():
            if stats["total"] > 0:
                pct_trainable = 100 * stats["trainable"] / stats["total"]
                logging.info(
                    f"  {component.capitalize()}: "
                    f"{stats['trainable']:,} trainable, {stats['frozen']:,} frozen, "
                    f"{stats['total']:,} total ({pct_trainable:.2f}% trainable)"
                )
        
        if hasattr(model_to_check, 'heads') and len(model_to_check.heads) > 1:
            logging.info("\nHead Trainability (Multitask Model):")
            for head_idx, stats in sorted(head_trainability.items()):
                pct_trainable = 100 * stats["trainable"] / stats["total"] if stats["total"] > 0 else 0
                head_name = f"Head {head_idx}"
                if head_idx == 0:
                    head_name += " (Segmentation)"
                elif head_idx == 1:
                    head_name += " (Boundary)"
                elif head_idx == 2:
                    head_name += " (Task3)"
                
                logging.info(
                    f"  {head_name}: "
                    f"{stats['trainable']:,} trainable, {stats['frozen']:,} frozen, "
                    f"{stats['total']:,} total ({pct_trainable:.2f}% trainable)"
                )
                
                if stats["trainable"] == 0:
                    logging.warning(f"  ??  WARNING: {head_name} has NO trainable parameters!")
        
        if use_multitask and hasattr(model_to_check, 'heads') and len(model_to_check.heads) > 1:
            boundary_head_idx = 1
            if boundary_head_idx in head_trainability:
                boundary_stats = head_trainability[boundary_head_idx]
                if boundary_stats["trainable"] == 0:
                    logging.error("  ? CRITICAL: Boundary head has NO trainable parameters!")
                else:
                    logging.info(f"  ? Boundary head is trainable ({boundary_stats['trainable']:,} params)")
        
        logging.info("=" * 80)
    
    def log_gradient_statistics(
        self,
        model: torch.nn.Module,
        epoch: int,
        batch_idx: Optional[int] = None,
    ):
        """
        Log gradient statistics to diagnose training issues.
        
        Args:
            model: Model to analyze
            epoch: Current epoch
            batch_idx: Current batch index (optional)
        """
        self._grad_log_counter += 1
        
        if self._grad_log_counter % 50 != 0 and batch_idx is not None:
            return
        
        # Unwrap DataParallel if needed
        model_to_check = model.module if isinstance(model, DataParallel) else model
        
        grad_stats = {
            "encoder": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
            "decoder": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
            "head": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
        }
        
        head_grads = {}
        
        for name, param in model_to_check.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad = param.grad.data
                grad_mean = grad.abs().mean().item()
                grad_max = grad.abs().max().item()
                grad_min = grad.abs().min().item()
                zero_grads = (grad == 0).sum().item()
                total_grads = grad.numel()
                
                if "encoder" in name:
                    component = "encoder"
                elif "decoder" in name or "decoders" in name:
                    component = "decoder"
                elif "head" in name or "heads" in name:
                    component = "head"
                    head_idx = None
                    for i in range(len(model_to_check.heads) if hasattr(model_to_check, 'heads') else 0):
                        if f"heads.{i}" in name or f"head.{i}" in name:
                            head_idx = i
                            break
                    if head_idx is not None:
                        if head_idx not in head_grads:
                            head_grads[head_idx] = {"mean": [], "max": [], "zero_ratio": []}
                        head_grads[head_idx]["mean"].append(grad_mean)
                        head_grads[head_idx]["max"].append(grad_max)
                        head_grads[head_idx]["zero_ratio"].append(zero_grads / total_grads)
                else:
                    continue
                
                grad_stats[component]["mean"].append(grad_mean)
                grad_stats[component]["max"].append(grad_max)
                grad_stats[component]["min"].append(grad_min)
                grad_stats[component]["count"] += 1
                grad_stats[component]["zero_count"] += zero_grads
            elif param.requires_grad and param.grad is None:
                if "head" in name or "heads" in name:
                    head_idx = None
                    for i in range(len(model_to_check.heads) if hasattr(model_to_check, 'heads') else 0):
                        if f"heads.{i}" in name or f"head.{i}" in name:
                            head_idx = i
                            break
                    if head_idx == 1:
                        logging.warning(f"Boundary head parameter '{name}' has no gradient!")
        
        log_prefix = f"Epoch {epoch}"
        if batch_idx is not None:
            log_prefix += f", Batch {batch_idx}"
        
        logging.info(f"\n{log_prefix} - Gradient Statistics:")
        for component, stats in grad_stats.items():
            if stats["count"] > 0:
                mean_grad = np.mean(stats["mean"]) if stats["mean"] else 0
                max_grad = np.max(stats["max"]) if stats["max"] else 0
                min_grad = np.min(stats["min"]) if stats["min"] else 0
                zero_ratio = stats["zero_count"] / sum(
                    p.numel() for n, p in model_to_check.named_parameters()
                    if component in n and p.requires_grad
                ) if stats["count"] > 0 else 0
                
                logging.info(
                    f"  {component.capitalize()}: "
                    f"mean={mean_grad:.2e}, max={max_grad:.2e}, min={min_grad:.2e}, "
                    f"zero_ratio={zero_ratio:.4f}"
                )
        
        if head_grads:
            logging.info("  Head-specific gradients:")
            for head_idx, stats in sorted(head_grads.items()):
                head_name = f"Head {head_idx}"
                if head_idx == 0:
                    head_name += " (Seg)"
                elif head_idx == 1:
                    head_name += " (Boundary)"
                elif head_idx == 2:
                    head_name += " (Task3)"
                
                mean_grad = np.mean(stats["mean"]) if stats["mean"] else 0
                max_grad = np.max(stats["max"]) if stats["max"] else 0
                zero_ratio = np.mean(stats["zero_ratio"]) if stats["zero_ratio"] else 0
                
                logging.info(
                    f"    {head_name}: mean={mean_grad:.2e}, max={max_grad:.2e}, "
                    f"zero_ratio={zero_ratio:.4f}"
                )
                
                if head_idx == 1 and mean_grad < 1e-7:
                    logging.warning(f"Boundary head has very small gradients.")
    
    def log_learning_rates(self, optimizer):
        """
        Log current learning rates for different parameter groups.
        
        Args:
            optimizer: Optimizer to analyze
        """
        if hasattr(optimizer, 'param_groups'):
            logging.info("Learning Rates:")
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group.get('lr', 'N/A')
                num_params = sum(p.numel() for p in param_group['params'] if p.requires_grad)
                logging.info(f"  Group {i}: lr={lr}, params={num_params:,}")
        else:
            if hasattr(optimizer, 'base'):
                base_lr = optimizer.base.param_groups[0].get('lr', 'N/A')
                logging.info(f"Learning Rate (SAM): {base_lr}")
    
    def log_parameter_statistics(
        self,
        model: torch.nn.Module,
        epoch: int,
    ):
        """
        Log parameter value statistics to track if parameters are changing.
        
        Args:
            model: Model to analyze
            epoch: Current epoch
        """
        if not hasattr(self, '_param_stats_history'):
            self._param_stats_history = {}
        
        # Unwrap DataParallel if needed
        model_to_check = model.module if isinstance(model, DataParallel) else model
        
        current_stats = {}
        
        for name, param in model_to_check.named_parameters():
            if param.requires_grad:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_min = param.data.min().item()
                param_max = param.data.max().item()
                
                component = None
                if "encoder" in name:
                    component = "encoder"
                elif "decoder" in name or "decoders" in name:
                    component = "decoder"
                elif "head" in name or "heads" in name:
                    component = "head"
                    if "heads.1" in name or "head.1" in name:
                        if "boundary" not in current_stats:
                            current_stats["boundary"] = []
                        current_stats["boundary"].append({
                            "mean": param_mean,
                            "std": param_std,
                            "min": param_min,
                            "max": param_max,
                        })
                
                if component:
                    if component not in current_stats:
                        current_stats[component] = []
                    current_stats[component].append({
                        "mean": param_mean,
                        "std": param_std,
                        "min": param_min,
                        "max": param_max,
                    })
        
        if epoch == 1 or epoch % 5 == 0:
            logging.info(f"\nEpoch {epoch} - Parameter Statistics:")
            for component, stats_list in current_stats.items():
                if stats_list:
                    mean_vals = [s["mean"] for s in stats_list]
                    std_vals = [s["std"] for s in stats_list]
                    
                    logging.info(
                        f"  {component.capitalize()}: "
                        f"mean={np.mean(mean_vals):.6f}, std={np.mean(std_vals):.6f}"
                    )
                    
                    if component == "boundary" and epoch > 1:
                        if "boundary" in self._param_stats_history:
                            prev_mean = np.mean([s["mean"] for s in self._param_stats_history["boundary"]])
                            curr_mean = np.mean(mean_vals)
                            change = abs(curr_mean - prev_mean)
                            if change < 1e-6:
                                logging.warning(f"Boundary head parameters changed by only {change:.2e}")
        
        self._param_stats_history = current_stats
    
    def log_boundary_prediction_statistics(
        self,
        boundary_stats: list,
        epoch: int,
    ):
        """
        Log detailed boundary prediction statistics to diagnose Dice issues.
        
        Args:
            boundary_stats: List of boundary statistics dictionaries
            epoch: Current epoch
        """
        if not boundary_stats:
            return
        
        avg_dice = np.mean([s["dice"] for s in boundary_stats])
        avg_dice_prob = np.mean([s.get("dice_prob", 0) for s in boundary_stats])
        avg_threshold = np.mean([s.get("threshold", 0.5) for s in boundary_stats])
        avg_mean_prob = np.mean([s["mean_prob"] for s in boundary_stats])
        avg_max_prob = np.mean([s["max_prob"] for s in boundary_stats])
        avg_min_prob = np.mean([s["min_prob"] for s in boundary_stats])
        avg_mean_logit = np.mean([s["mean_logit"] for s in boundary_stats])
        avg_max_logit = np.mean([s["max_logit"] for s in boundary_stats])
        avg_min_logit = np.mean([s["min_logit"] for s in boundary_stats])
        avg_pred_positive = np.mean([s["pred_positive_ratio"] for s in boundary_stats])
        avg_target_positive = np.mean([s["target_positive_ratio"] for s in boundary_stats])
        
        total_intersection = sum([s["intersection"] for s in boundary_stats])
        total_pred = sum([s["pred_total"] for s in boundary_stats])
        total_target = sum([s["target_total"] for s in boundary_stats])
        
        logging.info(f"\nEpoch {epoch} - Boundary Prediction Diagnostics:")
        logging.info(f"  Dice Score (threshold={avg_threshold:.3f}): {avg_dice:.6f}")
        logging.info(f"  Dice Score (prob-based, no threshold): {avg_dice_prob:.6f}")
        logging.info(f"  Probabilities: mean={avg_mean_prob:.6f}, max={avg_max_prob:.6f}, min={avg_min_prob:.6f}")
        logging.info(f"  Logits: mean={avg_mean_logit:.6f}, max={avg_max_logit:.6f}, min={avg_min_logit:.6f}")
        logging.info(f"  Positive Ratio: pred={avg_pred_positive:.6f}, target={avg_target_positive:.6f}")
        logging.info(f"  Overlap: intersection={total_intersection:.0f}, pred_pixels={total_pred:.0f}, target_pixels={total_target:.0f}")
        
        if avg_mean_prob < 0.01:
            logging.warning(f"    Very low mean probability ({avg_mean_prob:.6f}) - predictions may be too conservative")
        if avg_max_prob < 0.5:
            logging.warning(f"    Max probability ({avg_max_prob:.6f}) < 0.5 - no predictions above threshold!")
        if avg_pred_positive < 1e-6:
            logging.warning(f"    No positive predictions (ratio={avg_pred_positive:.6f}) - all predictions are zero!")
        if abs(avg_dice - 0.1313) < 1e-4:
            logging.warning(f"    Dice score is exactly 0.1313 - possible calculation issue or constant predictions")
        
        if epoch > 1 and self._prev_boundary_stats is not None:
            prev_avg_dice = np.mean([s["dice"] for s in self._prev_boundary_stats])
            dice_change = abs(avg_dice - prev_avg_dice)
            if dice_change < 1e-6:
                logging.warning(f"    Dice score unchanged from previous epoch (change={dice_change:.2e})")
            
            prev_mean_prob = np.mean([s["mean_prob"] for s in self._prev_boundary_stats])
            prob_change = abs(avg_mean_prob - prev_mean_prob)
            if prob_change < 1e-6:
                logging.warning(f"    Mean probability unchanged (change={prob_change:.2e}) - predictions may not be learning")
        
        self._prev_boundary_stats = boundary_stats.copy()
