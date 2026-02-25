"""
Visualization and plotting utilities for the 2D trainer.

This module contains:
- Loss and metrics plotting
- Prediction visualization
- Training statistics CSV export
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import volume_segmantics.utilities.base_data_utils as utils


class TrainingVisualizer:
    """
    Handles visualization of training progress and predictions.
    
    Provides:
    - Loss and metrics plotting
    - Prediction visualization on validation samples
    - Training statistics CSV export
    """
    
    def __init__(
        self,
        num_classes: int,
        label_codes: dict = None,
        use_multitask: bool = False,
    ):
        """
        Initialize training visualizer.
        
        Args:
            num_classes: Number of segmentation classes
            label_codes: Dictionary mapping class indices to names
            use_multitask: Whether model is multi-task
        """
        self.num_classes = num_classes
        self.label_codes = label_codes or {}
        self.use_multitask = use_multitask
    
    def plot_loss_history(
        self,
        epoch_history: Dict[str, List[float]],
        output_path: Path,
    ) -> None:
        """
        Create and save loss/metric plots.
        
        Args:
            epoch_history: Dictionary with epoch history data
            output_path: Path to save figure
        """
        output_dir = output_path.parent
        epochs = range(1, len(epoch_history["train_total"]) + 1)
        
        if self.use_multitask:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Total Loss
            ax = axes[0, 0]
            ax.plot(epochs, epoch_history["train_total"], label="Train Total")
            ax.plot(epochs, epoch_history["valid_total"], label="Val Total")
            min_idx = np.argmin(epoch_history["valid_total"]) + 1
            ax.axvline(min_idx, linestyle="--", color="r", label="Best Checkpoint")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Total Loss")
            ax.legend()
            ax.grid(True)
            
            # Segmentation Loss
            ax = axes[0, 1]
            ax.plot(epochs, epoch_history["train_seg"], label="Train Seg")
            ax.plot(epochs, epoch_history["valid_seg"], label="Val Seg")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Segmentation Loss")
            ax.legend()
            ax.grid(True)
            
            # Boundary Loss
            ax = axes[0, 2]
            if any(epoch_history["train_boundary"]):
                ax.plot(epochs, epoch_history["train_boundary"], label="Train Boundary")
                ax.plot(epochs, epoch_history["valid_boundary"], label="Val Boundary")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Boundary Loss")
            ax.legend()
            ax.grid(True)
            
            # Mean Dice Metrics
            ax = axes[1, 0]
            ax.plot(epochs, epoch_history["seg_dice"], label="Seg Dice (mean)")
            if any(epoch_history["boundary_dice"]):
                ax.plot(epochs, epoch_history["boundary_dice"], label="Boundary Dice")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dice Score")
            ax.set_title("Mean Evaluation Metrics")
            ax.legend()
            ax.grid(True)
            
            # Per-class Dice
            ax = axes[1, 1]
            for c in range(self.num_classes):
                key = f"dice_class_{c}"
                if key in epoch_history and any(epoch_history[key]):
                    class_name = self.label_codes.get(c, f"Class {c}") if self.label_codes else f"Class {c}"
                    ax.plot(epochs, epoch_history[key], label=class_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dice Score")
            ax.set_title("Per-Class Dice")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True)
            
            # Per-class Dice bar chart
            ax = axes[1, 2]
            final_dice = []
            class_names = []
            for c in range(self.num_classes):
                key = f"dice_class_{c}"
                if key in epoch_history and epoch_history[key]:
                    final_dice.append(epoch_history[key][-1])
                    class_names.append(self.label_codes.get(c, f"C{c}") if self.label_codes else f"C{c}")
            
            if final_dice:
                bars = ax.bar(class_names, final_dice, color='steelblue')
                ax.axhline(np.mean(final_dice), color='r', linestyle='--', label=f"Mean: {np.mean(final_dice):.3f}")
                ax.set_xlabel("Class")
                ax.set_ylabel("Final Dice Score")
                ax.set_title("Final Per-Class Dice")
                ax.legend()
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, val in zip(bars, final_dice):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f"Training History: {output_path.stem}", fontsize=14)
            plt.tight_layout()
        else:
            # Single-task: 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss
            ax = axes[0, 0]
            ax.plot(epochs, epoch_history["train_total"], label="Training Loss")
            ax.plot(epochs, epoch_history["valid_total"], label="Validation Loss")
            min_idx = np.argmin(epoch_history["valid_total"]) + 1
            ax.axvline(min_idx, linestyle="--", color="r", label="Best Checkpoint")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Training & Validation Loss")
            ax.legend()
            ax.grid(True)
            
            # Mean Dice
            ax = axes[0, 1]
            ax.plot(epochs, epoch_history["seg_dice"], label="Mean Dice", color='green')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Dice Score")
            ax.set_title("Mean Dice Score")
            ax.legend()
            ax.grid(True)
            
            # Per-class Dice curves
            ax = axes[1, 0]
            for c in range(self.num_classes):
                key = f"dice_class_{c}"
                if key in epoch_history and any(epoch_history[key]):
                    class_name = self.label_codes.get(c, f"Class {c}") if self.label_codes else f"Class {c}"
                    ax.plot(epochs, epoch_history[key], label=class_name)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Dice Score")
            ax.set_title("Per-Class Dice Over Training")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True)
            
            # Per-class Dice bar chart
            ax = axes[1, 1]
            final_dice = []
            class_names = []
            for c in range(self.num_classes):
                key = f"dice_class_{c}"
                if key in epoch_history and epoch_history[key]:
                    final_dice.append(epoch_history[key][-1])
                    class_names.append(self.label_codes.get(c, f"C{c}") if self.label_codes else f"C{c}")
            
            if final_dice:
                bars = ax.bar(class_names, final_dice, color='steelblue')
                ax.axhline(np.mean(final_dice), color='r', linestyle='--',
                          label=f"Mean: {np.mean(final_dice):.3f}")
                ax.set_xlabel("Class")
                ax.set_ylabel("Final Dice Score")
                ax.set_title("Final Per-Class Dice Scores")
                ax.legend()
                ax.set_ylim(0, 1)
                
                for bar, val in zip(bars, final_dice):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f"Training History: {output_path.stem}", fontsize=14)
            plt.tight_layout()
        
        fig_out_path = output_dir / f"{output_path.stem}_loss_plot.png"
        logging.info(f"Saving loss figure to {fig_out_path}")
        fig.savefig(fig_out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        # Output CSV with all stats
        self.save_training_stats_csv(epoch_history, output_dir, output_path.stem)
    
    def save_training_stats_csv(
        self,
        epoch_history: Dict[str, List[float]],
        output_dir: Path,
        model_name: str,
    ) -> None:
        """
        Save training statistics to CSV.
        
        Args:
            epoch_history: Dictionary with epoch history data
            output_dir: Directory to save CSV
            model_name: Model name for filename
        """
        csv_path = output_dir / f"{model_name}_train_stats.csv"
        epochs = range(1, len(epoch_history["train_total"]) + 1)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Build header
            header = ["Epoch"]
            if self.use_multitask:
                header.extend([
                    "Train_Total", "Train_Seg", "Train_Boundary", "Train_Task3",
                    "Valid_Total", "Valid_Seg", "Valid_Boundary", "Valid_Task3",
                    "Seg_Dice", "Boundary_Dice"
                ])
            else:
                header.extend(["Train_Loss", "Valid_Loss", "Seg_Dice"])
            
            # Add per-class Dice columns
            for c in range(self.num_classes):
                class_name = self.label_codes.get(c, f"Class_{c}") if self.label_codes else f"Class_{c}"
                header.append(f"Dice_{class_name}")
            
            writer.writerow(header)
            
            # Write data rows
            for i, epoch in enumerate(epochs):
                row = [epoch]
                
                if self.use_multitask:
                    row.extend([
                        epoch_history["train_total"][i],
                        epoch_history["train_seg"][i],
                        epoch_history["train_boundary"][i],
                        epoch_history["train_task3"][i],
                        epoch_history["valid_total"][i],
                        epoch_history["valid_seg"][i],
                        epoch_history["valid_boundary"][i],
                        epoch_history["valid_task3"][i],
                        epoch_history["seg_dice"][i],
                        epoch_history["boundary_dice"][i],
                    ])
                else:
                    row.extend([
                        epoch_history["train_total"][i],
                        epoch_history["valid_total"][i],
                        epoch_history["seg_dice"][i],
                    ])
                
                # Per-class Dice
                for c in range(self.num_classes):
                    key = f"dice_class_{c}"
                    row.append(epoch_history[key][i] if key in epoch_history else 0)
                
                writer.writerow(row)
        
        logging.info(f"Saved training statistics to {csv_path}")
    
    def plot_predictions(
        self,
        model: torch.nn.Module,
        validation_loader,
        device_num: int,
        model_path: Path,
        ensure_tuple_output_fn,
    ) -> None:
        """
        Visualize predictions on validation samples.
        
        Args:
            model: Model to use for predictions
            validation_loader: Validation data loader
            device_num: Device number
            model_path: Path to model (for output filename)
            ensure_tuple_output_fn: Function to ensure tuple output
        """
        model.eval()
        batch = next(iter(validation_loader))
        
        with torch.no_grad():
            inputs, targets = utils.prepare_training_batch(
                batch, device_num, self.num_classes
            )
            outputs = ensure_tuple_output_fn(model(inputs))
            
            seg_output = outputs[0]
            s_max = nn.Softmax(dim=1)
            probs = s_max(seg_output)
            seg_preds = torch.argmax(probs, dim=1)
            
            if isinstance(targets, dict):
                seg_target = targets.get("seg", None)
                boundary_target = targets.get("boundary", None)
                task3_target = targets.get("task3", None)
            else:
                seg_target = targets
                boundary_target = None
                task3_target = None
            
            seg_gt = None
            if seg_target is not None:
                seg_gt = torch.argmax(seg_target, dim=1)
            
            boundary_preds = None
            has_boundary_output = len(outputs) > 1
            if has_boundary_output:
                boundary_output = outputs[1]
                if boundary_target is not None:
                    if boundary_output.shape[1] != boundary_target.shape[1]:
                        boundary_output = boundary_output[:, :boundary_target.shape[1], :, :]
                else:
                    if boundary_output.shape[1] > 1:
                        boundary_output = boundary_output[:, 0:1, :, :]
                boundary_preds = (torch.sigmoid(boundary_output) > 0.5).float()
            
            task3_preds = None
            has_task3_output = len(outputs) > 2
            if has_task3_output:
                task3_output = outputs[2]
                if task3_target is not None:
                    if task3_output.shape[1] != task3_target.shape[1]:
                        task3_output = task3_output[:, :task3_target.shape[1], :, :]
                else:
                    if task3_output.shape[1] > 1:
                        task3_output = task3_output[:, 0:1, :, :]
                task3_preds = (torch.sigmoid(task3_output) > 0.5).float()
        
        bs = min(validation_loader.batch_size, 4)
        num_cols = 3
        
        if has_boundary_output:
            num_cols += 2
        if has_task3_output:
            num_cols += 2
        
        fig, axes = plt.subplots(bs, num_cols, figsize=(4 * num_cols, 4 * bs))
        if bs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(bs):
            col_idx = 0
            img = inputs[i].cpu()
            
            if len(img.shape) == 4:
                img = img.squeeze(0)
            num_channels = img.shape[0]
            
            if num_channels == 3:
                img_display = img.permute(1, 2, 0)
                axes[i, col_idx].imshow(img_display)
            elif num_channels > 3:
                center = num_channels // 2
                axes[i, col_idx].imshow(img[center], cmap="gray")
            else:
                axes[i, col_idx].imshow(img.squeeze(), cmap="gray")
            if i == 0:
                axes[i, col_idx].set_title("Input")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            if seg_gt is not None:
                axes[i, col_idx].imshow(seg_gt[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
                if i == 0:
                    axes[i, col_idx].set_title("Seg GT")
            else:
                axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                if i == 0:
                    axes[i, col_idx].set_title("Seg GT")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            axes[i, col_idx].imshow(seg_preds[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
            if i == 0:
                axes[i, col_idx].set_title("Seg Pred")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            if has_boundary_output:
                if boundary_target is not None:
                    b_gt = boundary_target[i, 0].cpu() if boundary_target.dim() == 4 else boundary_target[i].cpu()
                    axes[i, col_idx].imshow(b_gt, cmap="gray")
                    if i == 0:
                        axes[i, col_idx].set_title("Boundary GT")
                else:
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    if i == 0:
                        axes[i, col_idx].set_title("Boundary GT")
                axes[i, col_idx].axis("off")
                col_idx += 1
                
                b_pred = boundary_preds[i, 0].cpu() if boundary_preds.dim() == 4 else boundary_preds[i].cpu()
                axes[i, col_idx].imshow(b_pred, cmap="gray")
                if i == 0:
                    axes[i, col_idx].set_title("Boundary Pred")
                axes[i, col_idx].axis("off")
                col_idx += 1
            
            if has_task3_output:
                if task3_target is not None:
                    t3_gt = task3_target[i, 0].cpu() if task3_target.dim() == 4 else task3_target[i].cpu()
                    axes[i, col_idx].imshow(t3_gt, cmap="gray")
                    if i == 0:
                        axes[i, col_idx].set_title("Task3 GT")
                else:
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    if i == 0:
                        axes[i, col_idx].set_title("Task3 GT")
                axes[i, col_idx].axis("off")
                col_idx += 1
                
                t3_pred = task3_preds[i, 0].cpu() if task3_preds.dim() == 4 else task3_preds[i].cpu()
                axes[i, col_idx].imshow(t3_pred, cmap="gray")
                if i == 0:
                    axes[i, col_idx].set_title("Task3 Pred")
                axes[i, col_idx].axis("off")
                col_idx += 1
        
        plt.suptitle(f"Predictions: {model_path.name}", fontsize=14)
        plt.tight_layout()
        
        out_path = model_path.parent / f"{model_path.stem}_prediction_image.png"
        logging.info(f"Saving prediction visualization to {out_path}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    def plot_mean_teacher_predictions(
        self,
        model: torch.nn.Module,
        validation_loader,
        device_num: int,
        output_path: Path,
        epoch: int,
        ensure_tuple_output_fn,
    ) -> None:
        """
        Visualize Mean Teacher predictions: student vs teacher on validation samples.
        
        Args:
            model: MeanTeacherModel (or wrapped model)
            validation_loader: Validation data loader
            device_num: Device number
            output_path: Path to save figure
            epoch: Current epoch number
            ensure_tuple_output_fn: Function to ensure tuple output
        """
        from torch.nn import DataParallel
        
        # Unwrap DataParallel if needed
        if isinstance(model, DataParallel):
            mean_teacher = model.module
        else:
            mean_teacher = model
        
        # Get student and teacher models
        student_model = mean_teacher.get_student_model()
        teacher_model = mean_teacher.get_teacher_model()
        
        student_model.eval()
        teacher_model.eval()
        
        batch = next(iter(validation_loader))
        
        with torch.no_grad():
            inputs, targets = utils.prepare_training_batch(
                batch, device_num, self.num_classes
            )
            
            # Get student predictions
            student_outputs = ensure_tuple_output_fn(student_model(inputs))
            student_seg = student_outputs[0]
            s_max = nn.Softmax(dim=1)
            student_probs = s_max(student_seg)
            student_preds = torch.argmax(student_probs, dim=1)
            
            # Get teacher predictions
            teacher_outputs = ensure_tuple_output_fn(teacher_model(inputs))
            teacher_seg = teacher_outputs[0]
            teacher_probs = s_max(teacher_seg)
            teacher_preds = torch.argmax(teacher_probs, dim=1)
            
            # Compute consistency (difference between student and teacher)
            consistency_diff = torch.abs(student_probs - teacher_probs).mean(dim=1)
            
            # Get ground truth if available
            if isinstance(targets, dict):
                seg_target = targets.get("seg", None)
            else:
                seg_target = targets
            
            seg_gt = None
            if seg_target is not None:
                seg_gt = torch.argmax(seg_target, dim=1)
        
        bs = min(getattr(validation_loader, 'batch_size', len(inputs)), len(inputs), 4)
        num_cols = 5  # Input, GT, Student, Teacher, Consistency
        
        fig, axes = plt.subplots(bs, num_cols, figsize=(4 * num_cols, 4 * bs))
        if bs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(bs):
            col_idx = 0
            img = inputs[i].cpu()
            
            if len(img.shape) == 4:
                img = img.squeeze(0)
            num_channels = img.shape[0]
            
            if num_channels == 3:
                img_display = img.permute(1, 2, 0)
                axes[i, col_idx].imshow(img_display)
            elif num_channels > 3:
                center = num_channels // 2
                axes[i, col_idx].imshow(img[center], cmap="gray")
            else:
                axes[i, col_idx].imshow(img.squeeze(), cmap="gray")
            if i == 0:
                axes[i, col_idx].set_title("Input")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Ground truth
            if seg_gt is not None:
                axes[i, col_idx].imshow(seg_gt[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
                if i == 0:
                    axes[i, col_idx].set_title("Ground Truth")
            else:
                axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                if i == 0:
                    axes[i, col_idx].set_title("Ground Truth")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Student prediction
            axes[i, col_idx].imshow(student_preds[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
            if i == 0:
                axes[i, col_idx].set_title("Student Prediction")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Teacher prediction
            axes[i, col_idx].imshow(teacher_preds[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
            if i == 0:
                axes[i, col_idx].set_title("Teacher Prediction")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Consistency difference (lower = more consistent)
            axes[i, col_idx].imshow(consistency_diff[i].cpu(), cmap="hot", vmin=0, vmax=1)
            if i == 0:
                axes[i, col_idx].set_title("Consistency Diff")
            axes[i, col_idx].axis("off")
            col_idx += 1
        
        plt.suptitle(f"Mean Teacher Predictions - Epoch {epoch}", fontsize=14)
        plt.tight_layout()
        
        out_path = output_path.parent / f"{output_path.stem}_mean_teacher_epoch_{epoch}.png"
        logging.info(f"Saving Mean Teacher visualization to {out_path}")
        logging.info(f"Output directory: {output_path.parent}, Model file: {output_path.name}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Successfully saved Mean Teacher visualization: {out_path}")
    
    def plot_pseudo_labeling_visualization(
        self,
        model: torch.nn.Module,
        unlabeled_loader,
        device_num: int,
        output_path: Path,
        epoch: int,
        pseudo_label_generator,
        ensure_tuple_output_fn,
    ) -> None:
        """
        Visualize pseudo-labeling results on unlabeled samples.
        
        Shows: input, pseudo-labels, confidence map, accepted/rejected mask.
        
        Args:
            model: Model for generating pseudo-labels
            unlabeled_loader: Unlabeled data loader
            device_num: Device number
            output_path: Path to save figure
            epoch: Current epoch number
            pseudo_label_generator: PseudoLabelGenerator instance
            ensure_tuple_output_fn: Function to ensure tuple output
        """
        from torch.nn import DataParallel
        
        # Unwrap DataParallel if needed
        if isinstance(model, DataParallel):
            base_model = model.module
        else:
            base_model = model
        
        # Get model for pseudo-label generation (teacher if available, else student, else base model)
        model_for_labels = base_model
        use_teacher = False
        
        if hasattr(base_model, 'get_teacher_model') and hasattr(base_model, 'get_student_model'):
            # MeanTeacherModel: use teacher if pseudo_label_generator wants it
            if pseudo_label_generator.use_teacher_for_labels:
                model_for_labels = base_model.get_teacher_model()
                use_teacher = True
            else:
                model_for_labels = base_model.get_student_model()
                use_teacher = False
        elif hasattr(base_model, 'get_student_model'):
            # Only student available
            model_for_labels = base_model.get_student_model()
            use_teacher = False
        
        model_for_labels.eval()
        
        try:
            batch = next(iter(unlabeled_loader))
        except StopIteration:
            unlabeled_loader_iter = iter(unlabeled_loader)
            batch = next(unlabeled_loader_iter)
        
        with torch.no_grad():
            # Get unlabeled images
            if isinstance(batch, dict):
                if "student" in batch:
                    unlabeled_inputs = batch["student"]
                else:
                    unlabeled_inputs = batch["img"]
            else:
                unlabeled_inputs = batch
            
            # Convert to tensor and move to device
            if not isinstance(unlabeled_inputs, torch.Tensor):
                unlabeled_inputs = torch.as_tensor(unlabeled_inputs, dtype=torch.float32)
            unlabeled_inputs = unlabeled_inputs.to(device_num)
            
            # Ensure correct shape
            if unlabeled_inputs.dim() == 3:
                unlabeled_inputs = unlabeled_inputs.unsqueeze(0)
            
            # Generate pseudo-labels
            pseudo_label_dict = pseudo_label_generator.generate_pseudo_labels(
                model_for_labels,
                unlabeled_inputs,
                self.num_classes,
                use_teacher=use_teacher,
            )
            
            pseudo_labels = pseudo_label_dict["pseudo_labels"]  # (B, H, W)
            confidence_map = pseudo_label_dict["confidence_map"]  # (B, H, W)
            mask = pseudo_label_dict["mask"]  # (B, H, W) - accepted pixels
            probs = pseudo_label_dict["probs"]  # (B, C, H, W)
        
        bs = min(len(unlabeled_inputs), 4)
        num_cols = 4  # Input, Pseudo-labels, Confidence, Accepted/Rejected
        
        fig, axes = plt.subplots(bs, num_cols, figsize=(4 * num_cols, 4 * bs))
        if bs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(bs):
            col_idx = 0
            img = unlabeled_inputs[i].cpu()
            
            if len(img.shape) == 4:
                img = img.squeeze(0)
            num_channels = img.shape[0]
            
            if num_channels == 3:
                img_display = img.permute(1, 2, 0)
                axes[i, col_idx].imshow(img_display)
            elif num_channels > 3:
                center = num_channels // 2
                axes[i, col_idx].imshow(img[center], cmap="gray")
            else:
                axes[i, col_idx].imshow(img.squeeze(), cmap="gray")
            if i == 0:
                axes[i, col_idx].set_title("Unlabeled Input")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Pseudo-labels
            axes[i, col_idx].imshow(pseudo_labels[i].cpu(), cmap="tab10", vmin=0, vmax=self.num_classes - 1)
            if i == 0:
                axes[i, col_idx].set_title("Pseudo-Labels")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Confidence map
            axes[i, col_idx].imshow(confidence_map[i].cpu(), cmap="viridis", vmin=0, vmax=1)
            if i == 0:
                axes[i, col_idx].set_title("Confidence Map")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            # Accepted/Rejected mask (green = accepted, red = rejected)
            mask_vis = torch.zeros((3, *mask[i].shape), dtype=torch.float32)
            accepted = mask[i].cpu().bool()
            mask_vis[1, accepted] = 1.0  # Green for accepted
            mask_vis[0, ~accepted] = 1.0  # Red for rejected
            mask_vis = mask_vis.permute(1, 2, 0)
            
            axes[i, col_idx].imshow(mask_vis)
            if i == 0:
                axes[i, col_idx].set_title("Accepted (Green) / Rejected (Red)")
            axes[i, col_idx].axis("off")
            col_idx += 1
        
        # Calculate acceptance rate
        total_pixels = mask.numel()
        accepted_pixels = mask.sum().item()
        acceptance_rate = accepted_pixels / total_pixels if total_pixels > 0 else 0.0
        
        plt.suptitle(
            f"Pseudo-Labeling Visualization - Epoch {epoch} "
            f"(Acceptance Rate: {acceptance_rate:.2%})",
            fontsize=14
        )
        plt.tight_layout()
        
        out_path = output_path.parent / f"{output_path.stem}_pseudo_labeling_epoch_{epoch}.png"
        logging.info(f"Saving pseudo-labeling visualization to {out_path}")
        logging.info(f"Output directory: {output_path.parent}, Model file: {output_path.name}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Successfully saved pseudo-labeling visualization: {out_path}")