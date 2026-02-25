import logging
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Union, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_training_dataloaders
from volume_segmantics.data.pytorch3dunet_losses import (
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    BoundaryDoULoss,
    BoundaryDoULossV2,
    TverskyLoss,
    BoundaryLoss,
    BoundaryDoUDiceLoss
)
from volume_segmantics.model.model_2d import create_model_from_file_full_weights
from volume_segmantics.model.sam import SAM
from volume_segmantics.model.operations.trainer_losses import (
    ConsistencyLoss,
    ClassWeightedDiceLoss,
    CombinedCEDiceLoss,
    get_rampup_ratio,
)
from volume_segmantics.model.operations.trainer_multitask import (
    MultiTaskLossTracker,
    MultiTaskLossCalculator,
)
from volume_segmantics.model.operations.trainer_metrics import (
    MetricsCalculator,
    ensure_tuple_output,
    get_eval_metric,
)
from volume_segmantics.model.operations.trainer_lr_finder import LearningRateFinder
from volume_segmantics.model.operations.trainer_model_manager import ModelManager
from volume_segmantics.model.operations.trainer_logging import TrainingLogger
from volume_segmantics.model.operations.trainer_visualization import TrainingVisualizer


from volume_segmantics.model.sam import SAM






class VolSeg2dTrainer:
    """
    Class that provides methods to train a 2d deep learning model
    with support for multi-task learning and detailed loss tracking.
    """

    def __init__(
        self,
        image_dir_path: Path,
        label_dir_path: Path,
        labels: Union[int, dict],
        settings: SimpleNamespace,
    ):
        """
        Inits VolSeg2dTrainer.

        Args:
            image_dir_path: Path to directory containing image data slices.
            label_dir_path: Path to directory containing label data slices.
            labels: Either number of labels or dictionary containing label names.
            settings: A training settings object.
        """
        # Semi-supervised learning settings
        self.use_semi_supervised = getattr(settings, "use_semi_supervised", False)
        self.use_pseudo_labeling = getattr(settings, "use_pseudo_labeling", False)

        # Check if we need unlabeled data (either Mean Teacher or pseudo-labeling)
        needs_unlabeled_data = self.use_semi_supervised or self.use_pseudo_labeling

        if needs_unlabeled_data:
            from volume_segmantics.data.dataloaders import get_semi_supervised_dataloaders

            # Get unlabeled_data_dir from settings (may have been set via command line)
            unlabeled_data_dir_str = getattr(settings, "unlabeled_data_dir", "")

            # Validate that unlabeled_data_dir is provided
            if not unlabeled_data_dir_str or unlabeled_data_dir_str == "":
                enabled_methods = []
                if self.use_semi_supervised:
                    enabled_methods.append("Mean Teacher (use_semi_supervised=True)")
                if self.use_pseudo_labeling:
                    enabled_methods.append("pseudo-labeling (use_pseudo_labeling=True)")
                raise ValueError(
                    f"unlabeled_data_dir must be provided when using semi-supervised learning. "
                    f"Enabled methods: {', '.join(enabled_methods)}. "
                    f"Please provide --unlabeled_data_dir on the command line or set it in the settings file."
                )

            unlabeled_data_dir = Path(unlabeled_data_dir_str)
            if not unlabeled_data_dir.exists():
                raise ValueError(
                    f"Unlabeled data directory not found: {unlabeled_data_dir}. "
                    f"Required for {'Mean Teacher' if self.use_semi_supervised else ''} "
                    f"{'and ' if self.use_semi_supervised and self.use_pseudo_labeling else ''}"
                    f"{'pseudo-labeling' if self.use_pseudo_labeling else ''}"
                )

            # Get semi-supervised dataloaders (works for both Mean Teacher and pseudo-labeling)
            self.training_loader, self.unlabeled_loader, self.validation_loader = \
                get_semi_supervised_dataloaders(
                    image_dir_path, label_dir_path, unlabeled_data_dir, settings
                )

            # Create iterators for unlabeled data
            self.trainIter_unlab = iter(self.unlabeled_loader)

            # Mean Teacher parameters
            if self.use_semi_supervised:
                self.consistency_weight = getattr(settings, "consistency_weight", 0.1)
                self.rampup_start = getattr(settings, "rampup_start", 0)
                self.rampup_end = getattr(settings, "rampup_end", 10000)
                self.ema_decay = getattr(settings, "ema_decay", 0.99)

                # Create consistency loss
                self.consistency_loss = ConsistencyLoss()

                logging.info(
                    f"Mean Teacher semi-supervised learning enabled: "
                    f"consistency_weight={self.consistency_weight}, "
                    f"rampup_start={self.rampup_start}, rampup_end={self.rampup_end}, "
                    f"ema_decay={self.ema_decay}"
                )
            else:
                self.consistency_loss = None

            # Global iteration counter (for ramp-up and EMA)
            self.glob_it = 0
        else:
            # Standard initialization (existing code)
            self.training_loader, self.validation_loader = get_2d_training_dataloaders(
                image_dir_path, label_dir_path, settings
            )
            self.unlabeled_loader = None
            self.consistency_loss = None
            self.glob_it = 0

        # Pseudo-labeling settings
        if self.use_pseudo_labeling:
            from volume_segmantics.model.pseudo_labeling import (
                PseudoLabelGenerator,
                ConfidenceThresholdScheduler,
            )

            # Pseudo-labeling parameters
            self.pseudo_label_confidence_threshold = getattr(
                settings, "pseudo_label_confidence_threshold", 0.95
            )
            self.pseudo_label_confidence_method = getattr(
                settings, "pseudo_label_confidence_method", "max_prob"
            )
            self.pseudo_label_min_pixels_per_class = getattr(
                settings, "pseudo_label_min_pixels_per_class", 10
            )
            # Default to using teacher only if Mean Teacher is enabled
            default_use_teacher = self.use_semi_supervised
            self.pseudo_label_use_teacher = getattr(
                settings, "pseudo_label_use_teacher", default_use_teacher
            )

            # Warn if trying to use teacher without Mean Teacher
            if self.pseudo_label_use_teacher and not self.use_semi_supervised:
                logging.warning(
                    "pseudo_label_use_teacher is True but Mean Teacher is not enabled. "
                    "Setting pseudo_label_use_teacher to False."
                )
                self.pseudo_label_use_teacher = False
            self.pseudo_label_weight = getattr(settings, "pseudo_label_weight", 1.0)
            self.pseudo_label_rampup_start = getattr(settings, "pseudo_label_rampup_start", 0)
            self.pseudo_label_rampup_end = getattr(settings, "pseudo_label_rampup_end", 5000)

            # Initialize pseudo-label generator
            self.pseudo_label_generator = PseudoLabelGenerator(
                confidence_threshold=self.pseudo_label_confidence_threshold,
                confidence_method=self.pseudo_label_confidence_method,
                min_pixels_per_class=self.pseudo_label_min_pixels_per_class,
                use_teacher_for_labels=self.pseudo_label_use_teacher,
            )

            # Initialize threshold scheduler
            threshold_schedule = getattr(settings, "pseudo_label_threshold_schedule", "fixed")
            self.pseudo_label_scheduler = ConfidenceThresholdScheduler(
                start_threshold=getattr(settings, "pseudo_label_start_threshold", 0.9),
                end_threshold=self.pseudo_label_confidence_threshold,
                schedule_type=threshold_schedule,
                start_iter=self.pseudo_label_rampup_start,
                end_iter=self.pseudo_label_rampup_end,
                target_acceptance_rate=getattr(settings, "pseudo_label_target_acceptance_rate", 0.3),
            )

            # Pseudo-labeling statistics
            self.pseudo_label_stats = {
                "total_batches": 0,
                "total_pixels": 0,
                "accepted_pixels": 0,
            }

            # Iteration counter for pseudo-labeling (if not using Mean Teacher)
            if not self.use_semi_supervised:
                self.current_iter = 0

            logging.info(
                f"Pseudo-labeling enabled: "
                f"threshold={self.pseudo_label_confidence_threshold}, "
                f"method={self.pseudo_label_confidence_method}, "
                f"use_teacher={self.pseudo_label_use_teacher}, "
                f"weight={self.pseudo_label_weight}"
            )
        else:
            self.pseudo_label_generator = None
            self.pseudo_label_scheduler = None
            self.pseudo_label_stats = None
            self.current_iter = 0

        self.label_no = labels if isinstance(labels, int) else len(labels)
        self.codes = labels if isinstance(labels, dict) else {}
        self.settings = settings

        # Learning rate finder params
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = self._calculate_log_lr_ratio()
        self.lr_find_epochs = settings.lr_find_epochs
        self.lr_reduce_factor = settings.lr_reduce_factor

        # Params for model training
        self.model_device_num = int(settings.cuda_device)
        self.patience = settings.patience
        self.loss_criterion = self._get_loss_criterion()
        self.eval_metric = get_eval_metric(settings.eval_metric)

        # Initialize ModelManager early (needed for model_struc_dict)
        self.model_manager = ModelManager(
            settings=settings,
            model_device_num=self.model_device_num,
            label_no=self.label_no,
        )
        # Pass data directory to determine in_channels from actual data
        self.model_struc_dict = self.model_manager.get_model_structure_dict(
            settings, data_dir=image_dir_path
        )

        # Multi-task configuration
        self.use_multitask = getattr(settings, "use_multitask", False)
        if self.use_multitask:
            self._setup_multitask_loss_calculator(settings)

        # Dice evaluation settings
        self.exclude_background_from_dice = getattr(
            settings, "exclude_background_from_dice", True
        )
        self.dice_averaging = getattr(settings, "dice_averaging", "macro")

        # Loss tracking
        self.train_loss_tracker = MultiTaskLossTracker()
        self.valid_loss_tracker = MultiTaskLossTracker()

        # Epoch history for plots
        self.epoch_history = {
            "train_total": [],
            "train_seg": [],
            "train_boundary": [],
            "train_task3": [],
            "valid_total": [],
            "valid_seg": [],
            "valid_boundary": [],
            "valid_task3": [],
            "seg_dice": [],
            "boundary_dice": [],
        }

        # Add consistency loss tracking if using semi-supervised learning
        if self.use_semi_supervised:
            self.epoch_history["train_consistency"] = []
            self.epoch_history["train_consistency_raw"] = []
            self.epoch_history["consistency_weight"] = []

        # Add pseudo-labeling loss tracking if using pseudo-labeling
        if self.use_pseudo_labeling:
            self.epoch_history["train_pseudo_label"] = []
            self.epoch_history["train_pseudo_label_raw"] = []
            self.epoch_history["pseudo_label_weight"] = []
            self.epoch_history["pseudo_label_acceptance_rate"] = []

        # Per-class Dice history
        for c in range(self.label_no):
            self.epoch_history[f"dice_class_{c}"] = []

        self.avg_train_losses = self.epoch_history["train_total"]
        self.avg_valid_losses = self.epoch_history["valid_total"]
        self.avg_eval_scores = self.epoch_history["seg_dice"]

        if str(settings.encoder_weights_path) != "False":
            self.encoder_weights_path = Path(settings.encoder_weights_path)
        else:
            self.encoder_weights_path = False
        if str(settings.full_weights_path) != "False":
            self.full_weights_path = Path(settings.full_weights_path)
        else:
            self.full_weights_path = False

        # Sharpness Aware Minimisation optimizer settings
        self.use_sam = settings.use_sam
        self.adaptive_sam = settings.adaptive_sam

        # Initialize helper modules
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.label_no,
            exclude_background=self.exclude_background_from_dice,
            dice_averaging=self.dice_averaging,
            settings=settings,
        )
        # ModelManager already initialized above (needed for model_struc_dict)
        self.logger = TrainingLogger()
        self.visualizer = TrainingVisualizer(
            num_classes=self.label_no,
            label_codes=self.codes,
            use_multitask=self.use_multitask,
        )
        self.lr_finder = LearningRateFinder(
            starting_lr=self.starting_lr,
            end_lr=self.end_lr,
            lr_find_epochs=self.lr_find_epochs,
            training_loader=self.training_loader,
            plot_lr_graph=getattr(settings, "plot_lr_graph", False),
        )

    def _setup_multitask_loss_calculator(self, settings):
        """Initialize the multi-task loss calculator with settings."""
        self.loss_calculator = MultiTaskLossCalculator(
            seg_criterion=self.loss_criterion,
            seg_weight=getattr(settings, "seg_loss_weight", 1.0),
            boundary_weight=getattr(settings, "boundary_loss_weight", 1.0),
            task3_weight=getattr(settings, "task3_loss_weight", 1.0),
            use_cross_entropy=(settings.loss_criterion == "CrossEntropyLoss"),
            boundary_loss_type=getattr(settings, "boundary_loss_type", "bce"),
            num_classes=self.label_no,
        )
        logging.info(
            f"Multi-task loss calculator initialized: "
            f"seg_weight={self.loss_calculator.seg_weight}, "
            f"boundary_weight={self.loss_calculator.boundary_weight}, "
            f"boundary_loss_type={self.loss_calculator.boundary_loss_type}"
        )

    def _calculate_log_lr_ratio(self):
        return math.log(self.end_lr / self.starting_lr)

    def _create_model_and_optimiser(self, learning_rate, frozen=False):
        self.model, self.optimizer = self.model_manager.create_model_and_optimizer(
            self.model_struc_dict,
            learning_rate,
            frozen=frozen,
            use_semi_supervised=self.use_semi_supervised,
        )
        self.logger.log_model_architecture(self.model, use_multitask=self.use_multitask)
        self.logger.log_learning_rates(self.optimizer)
        logging.info("Trainer created.")

    def _load_encoder_weights(self, weights_fname: Path, gpu: bool = True, device_num: int = 0):
        self.model_manager.load_encoder_weights(
            self.model, weights_fname, gpu=gpu, device_num=device_num
        )

    def _freeze_model(self):
        """Freeze all encoder parameters (not just conv layers)."""
        self.model_manager.freeze_model(self.model)

    def _unfreeze_model(self):
        """Unfreeze all encoder parameters."""
        self.model_manager.unfreeze_model(self.model)

    def _count_trainable_parameters(self) -> int:
        return self.model_manager.count_trainable_parameters(self.model)

    def _count_parameters(self) -> int:
        return self.model_manager.count_parameters(self.model)

    def _log_model_architecture_details(self):
        """Log detailed information about model architecture and trainability."""
        self.logger.log_model_architecture(self.model, use_multitask=self.use_multitask)

    def _log_gradient_statistics(self, epoch: int, batch_idx: int = None):
        """Log gradient statistics to diagnose training issues."""
        self.logger.log_gradient_statistics(self.model, epoch, batch_idx)

    def _log_learning_rates(self):
        """Log current learning rates for different parameter groups."""
        self.logger.log_learning_rates(self.optimizer)

    def _log_parameter_statistics(self, epoch: int):
        """Log parameter value statistics to track if parameters are changing."""
        self.logger.log_parameter_statistics(self.model, epoch)

    def _log_boundary_prediction_statistics(self, epoch: int):
        """Log detailed boundary prediction statistics to diagnose Dice issues."""
        boundary_stats = getattr(self, '_boundary_stats', [])
        self.logger.log_boundary_prediction_statistics(boundary_stats, epoch)

    def _log_pseudo_label_statistics(self, epoch: int):
        """Log pseudo-labeling statistics."""
        if not self.use_pseudo_labeling or self.pseudo_label_stats is None:
            return

        # Use stats from trainer (not from generator, which may be reset)
        total_pixels = self.pseudo_label_stats.get("total_pixels", 0)
        accepted_pixels = self.pseudo_label_stats.get("accepted_pixels", 0)
        rejected_pixels = total_pixels - accepted_pixels
        acceptance_rate = accepted_pixels / total_pixels if total_pixels > 0 else 0.0

        logging.info(f"\nEpoch {epoch} - Pseudo-Label Statistics:")
        logging.info(f"  Acceptance Rate: {acceptance_rate:.4f}")
        logging.info(f"  Total Pixels: {total_pixels:,}")
        logging.info(f"  Accepted Pixels: {accepted_pixels:,}")
        logging.info(f"  Rejected Pixels: {rejected_pixels:,}")
        logging.info(f"  Current Threshold: {self.pseudo_label_generator.confidence_threshold:.4f}")

        # Note: Per-class statistics would need to be tracked separately in trainer
        # For now, we only log aggregate stats

        # Reset statistics for next epoch
        if self.pseudo_label_generator is not None:
            self.pseudo_label_generator.reset_stats()
        self.pseudo_label_stats = {
            "total_batches": 0,
            "total_pixels": 0,
            "accepted_pixels": 0,
        }

    def _get_loss_criterion(self):
        loss_name = self.settings.loss_criterion

        loss_map = {
            "BCEDiceLoss": lambda: BCEDiceLoss(self.settings.alpha, self.settings.beta),
            "DiceLoss": lambda: DiceLoss(normalization="none"),
            "BCELoss": lambda: nn.BCEWithLogitsLoss(),
            "CrossEntropyLoss": lambda: nn.CrossEntropyLoss(),
            "GeneralizedDiceLoss": lambda: GeneralizedDiceLoss(),
            "TverskyLoss": lambda: TverskyLoss(self.label_no + 1),
            "BoundaryDoULoss": lambda: BoundaryDoULoss(),
            "BoundaryDoUDiceLoss": lambda: BoundaryDoUDiceLoss(alpha=0.5, beta=0.5),
            "BoundaryDoULossV2": lambda: BoundaryDoULossV2(),
            "BoundaryLoss": lambda: BoundaryLoss(),
            # New class-weighted Dice losses
            "ClassWeightedDiceLoss": lambda: ClassWeightedDiceLoss(
                num_classes=self.label_no,
                weight_mode=getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq"),
                exclude_background=getattr(self.settings, "exclude_background_from_dice", True),
            ),
            "CombinedCEDiceLoss": lambda: CombinedCEDiceLoss(
                num_classes=self.label_no,
                alpha=getattr(self.settings, "ce_weight", 0.5),
                beta=getattr(self.settings, "dice_weight", 0.5),
                dice_weight_mode=getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq"),
                exclude_background=getattr(self.settings, "exclude_background_from_dice", True),
            ),
        }

        if loss_name in loss_map:
            logging.info(f"Using {loss_name}")
            return loss_map[loss_name]()
        else:
            logging.error(f"Unknown loss criterion: {loss_name}, exiting")
            sys.exit(1)

    def _ensure_tuple_output(self, output) -> tuple:
        """Ensure model output is a tuple for consistent handling."""
        return ensure_tuple_output(output)

    # MULTI-CLASS DICE COMPUTATION

    def _compute_multiclass_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        exclude_background: bool = False,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float]:
        """Compute per-class Dice and macro-averaged Dice."""
        return self.metrics_calculator.compute_multiclass_dice(
            pred, target, num_classes, exclude_background, smooth
        )

    def _compute_weighted_multiclass_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        weight_mode: str = "inverse_sqrt_freq",
        exclude_background: bool = False,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float, List[float]]:
        """Compute per-class Dice with weighted averaging."""
        return self.metrics_calculator.compute_weighted_multiclass_dice(
            pred, target, num_classes, weight_mode, exclude_background, smooth
        )

    def _compute_eval_metrics(
        self,
        outputs: tuple,
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute evaluation metrics for all tasks with proper multi-class handling."""
        metrics = self.metrics_calculator.compute_eval_metrics(outputs, targets)
        # Sync boundary stats for logging
        self._boundary_stats = self.metrics_calculator.get_boundary_stats()
        return metrics

    def train_model(
        self,
        output_path: Path,
        num_epochs: int,
        patience: int,
        create: bool = True,
        frozen: bool = False,
    ) -> None:
        """
        Performs training of model for a number of epochs with automatic LR finding.

        Args:
            output_path: Path to save model file to.
            num_epochs: Number of epochs to train the model for.
            patience: Epochs to wait while validation loss not improving before stopping.
            create: Whether to create a new model and optimizer from scratch.
            frozen: Whether to freeze encoder convolutional layers.
        """
        self.output_path = output_path  # Store for visualization
        if create:
            self._create_model_and_optimiser(self.starting_lr, frozen=frozen)

            if self.full_weights_path:
                logging.info("Loading pretrained weights for encoder and decoder (for LR finder).")
                model_tuple = create_model_from_file_full_weights(
                    self.full_weights_path,
                    self.model_struc_dict,
                    device_num=self.model_device_num
                )
                loaded_model, self.num_labels, self.label_codes = model_tuple

                # Wrap with MeanTeacherModel if using semi-supervised
                if self.use_semi_supervised:
                    from volume_segmantics.model.mean_teacher import MeanTeacherModel
                    # Unwrap DataParallel if present
                    if isinstance(loaded_model, DataParallel):
                        loaded_model = loaded_model.module
                    self.model = MeanTeacherModel(
                        student_model=loaded_model,
                        ema_decay=self.ema_decay
                    )
                    # Re-wrap with DataParallel if needed
                    if torch.cuda.device_count() > 1 and cfg.USE_ALL_GPUS:
                        self.model = DataParallel(self.model)
                        self.model.to("cuda")
                    else:
                        self.model.to(self.model_device_num)
                else:
                    self.model = loaded_model

            lr_to_use = self._run_lr_finder()

            self._create_model_and_optimiser(lr_to_use, frozen=frozen)

            if self.full_weights_path:
                logging.info("Loading pretrained weights for encoder and decoder.")
                model_tuple = create_model_from_file_full_weights(
                    self.full_weights_path,
                    self.model_struc_dict,
                    device_num=self.model_device_num
                )
                loaded_model, self.num_labels, self.label_codes = model_tuple

                # Wrap with MeanTeacherModel if using semi-supervised
                if self.use_semi_supervised:
                    from volume_segmantics.model.mean_teacher import MeanTeacherModel
                    # Unwrap DataParallel if present
                    if isinstance(loaded_model, DataParallel):
                        loaded_model = loaded_model.module
                    self.model = MeanTeacherModel(
                        student_model=loaded_model,
                        ema_decay=self.ema_decay
                    )
                    # Re-wrap with DataParallel if needed
                    if torch.cuda.device_count() > 1 and cfg.USE_ALL_GPUS:
                        self.model = DataParallel(self.model)
                        self.model.to("cuda")
                    else:
                        self.model.to(self.model_device_num)
                else:
                    self.model = loaded_model

                # Create optimizer with correct parameters
                if self.use_semi_supervised:
                    if isinstance(self.model, DataParallel):
                        student_params = self.model.module.student.parameters()
                    else:
                        student_params = self.model.student.parameters()
                else:
                    student_params = self.model.parameters()

                if self.use_sam:
                    base_optimizer = torch.optim.AdamW
                    self.optimizer = SAM(
                        student_params,
                        base_optimizer,
                        lr=lr_to_use,
                        adaptive=self.adaptive_sam
                    )
                else:
                    self.optimizer = torch.optim.AdamW(student_params, lr=lr_to_use)

            if self.encoder_weights_path:
                logging.info("Loading encoder weights.")
                self._load_encoder_weights(self.encoder_weights_path)

            early_stopping = self._create_early_stopping(output_path, patience)

        else:
            self.starting_lr /= self.lr_reduce_factor
            self.end_lr /= self.lr_reduce_factor
            self.log_lr_ratio = self._calculate_log_lr_ratio()
            self._load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            lr_to_use = self._run_lr_finder()
            min_loss = self._load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            early_stopping = self._create_early_stopping(
                output_path, patience, best_score=-min_loss
            )

        lr_scheduler = self._create_oc_lr_scheduler(num_epochs, lr_to_use)

        # === Main Training Loop ===
        for epoch in range(1, num_epochs + 1):
            tic = time.perf_counter()
            logging.info(f"Epoch {epoch}/{num_epochs}")

            # --- Training Phase ---
            # Set model to train mode (for MeanTeacherModel, this sets student to train)
            self.model.train()
            # Also ensure student is in train mode if using semi-supervised
            if self.use_semi_supervised:
                if isinstance(self.model, DataParallel):
                    self.model.module.student.train()
                else:
                    self.model.student.train()
            self.train_loss_tracker.clear()

            # Calculate iter_max for EMA decay
            iter_max = len(self.training_loader) * num_epochs
            if self.use_semi_supervised:
                # Update iter_max to include unlabeled iterations
                iter_max = max(iter_max, self.rampup_end)

            for batch in tqdm(
                self.training_loader,
                desc="Training",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                self._train_one_batch(lr_scheduler, batch)

                # Train on unlabeled batch
                if self.use_semi_supervised and not self.use_pseudo_labeling:
                    # Mean Teacher consistency only
                    self._train_consistency_batch()
                    # Update teacher model via EMA
                    if isinstance(self.model, DataParallel):
                        self.model.module.update_teacher(iter_max)
                    else:
                        self.model.update_teacher(iter_max)
                    self.glob_it += 1
                elif self.use_pseudo_labeling:
                    # Pseudo-labeling (can be combined with Mean Teacher)
                    self._train_pseudo_labeling_batch()

                    # Update teacher if using Mean Teacher + pseudo-labeling
                    if self.use_semi_supervised:
                        if isinstance(self.model, DataParallel):
                            self.model.module.update_teacher(iter_max)
                        else:
                            self.model.update_teacher(iter_max)
                        self.glob_it += 1

            # --- Validation Phase ---
            self.model.eval()
            self.valid_loss_tracker.clear()
            # Clear boundary stats for new epoch
            self.metrics_calculator.clear_boundary_stats()
            self._boundary_stats = []
            # Note: Don't clear pseudo-label stats here - they're accumulated during training
            # and will be logged after validation, then cleared

            with torch.no_grad():
                for batch in tqdm(
                    self.validation_loader,
                    desc="Validation",
                    bar_format=cfg.TQDM_BAR_FORMAT,
                ):
                    self._validate_one_batch(batch)

            # Epoch Statistics
            toc = time.perf_counter()
            self._log_epoch_statistics(epoch, toc - tic)

            # Boundary Prediction Diagnostics
            if self.use_multitask and hasattr(self, '_boundary_stats') and self._boundary_stats:
                self._log_boundary_prediction_statistics(epoch)
                self._boundary_stats = []

            # Pseudo-Label Statistics
            if self.use_pseudo_labeling:
                self._log_pseudo_label_statistics(epoch)

            # Semi-Supervised Learning Visualizations
            if self.use_semi_supervised:
                mean_teacher_vis_interval = getattr(self.settings, "mean_teacher_vis_epoch_interval", None)
                if mean_teacher_vis_interval and mean_teacher_vis_interval > 0:
                    if epoch % mean_teacher_vis_interval == 0:
                        logging.info(f"Generating Mean Teacher visualization at epoch {epoch} (interval: {mean_teacher_vis_interval})")
                        try:
                            self.visualizer.plot_mean_teacher_predictions(
                                self.model,
                                self.validation_loader,
                                self.model_device_num,
                                self.output_path,
                                epoch,
                                self._ensure_tuple_output,
                            )
                        except Exception as e:
                            logging.warning(f"Failed to generate Mean Teacher visualization at epoch {epoch}: {e}")
                            import traceback
                            logging.warning(traceback.format_exc())

            if self.use_pseudo_labeling and self.unlabeled_loader is not None:
                pseudo_labeling_vis_interval = getattr(self.settings, "pseudo_labeling_vis_epoch_interval", None)
                if pseudo_labeling_vis_interval and pseudo_labeling_vis_interval > 0:
                    if epoch % pseudo_labeling_vis_interval == 0:
                        logging.info(f"Generating pseudo-labeling visualization at epoch {epoch} (interval: {pseudo_labeling_vis_interval})")
                        try:
                            self.visualizer.plot_pseudo_labeling_visualization(
                                self.model,
                                self.unlabeled_loader,
                                self.model_device_num,
                                self.output_path,
                                epoch,
                                self.pseudo_label_generator,
                                self._ensure_tuple_output,
                            )
                        except Exception as e:
                            logging.warning(f"Failed to generate pseudo-labeling visualization at epoch {epoch}: {e}")
                            import traceback
                            logging.warning(traceback.format_exc())

            # Diagnostic Logging
            if epoch == 1 or epoch % 5 == 0:
                self._log_gradient_statistics(epoch)

            if epoch == 1 or epoch % 5 == 0:
                self._log_parameter_statistics(epoch)

            # Early Stopping Check
            current_valid_loss = self.epoch_history["valid_total"][-1]
            # Pass glob_it for semi-supervised learning
            glob_it = self.glob_it if self.use_semi_supervised else None
            early_stopping(current_valid_loss, self.model, self.optimizer, self.codes, glob_it=glob_it)

            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

        self._load_in_weights(output_path)

    def _train_one_batch(self, lr_scheduler, batch) -> torch.Tensor:
        """Train on a single batch, handling both single and multi-task modes."""
        inputs, targets = utils.prepare_training_batch(
            batch, self.model_device_num, self.label_no
        )

        is_multitask = isinstance(targets, dict)

        if is_multitask and self.use_multitask:
            loss = self._train_multitask_batch(inputs, targets)
        elif is_multitask and not self.use_multitask:
            seg_targets = targets.get("seg", targets)
            loss = self._train_singletask_batch(inputs, seg_targets)
        else:
            loss = self._train_singletask_batch(inputs, targets)

        lr_scheduler.step()
        return loss

    def _train_singletask_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Standard single-task training step."""
        if self.use_sam:
            self.optimizer.zero_grad()
            output = self._ensure_tuple_output(self.model(inputs))[0]

            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())

            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            output = self._ensure_tuple_output(self.model(inputs))[0]
            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())

            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            output = self._ensure_tuple_output(self.model(inputs))[0]

            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())

            loss.backward()
            self.optimizer.step()

        self.train_loss_tracker.append_losses({"total": loss.item(), "seg": loss.item()})
        return loss

    def _train_multitask_batch(
        self,
        inputs: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Multi-task training step with individual loss tracking."""
        if self.use_sam:
            self.optimizer.zero_grad()
            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.first_step(zero_grad=True)

            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.step()

        self.train_loss_tracker.append_losses({
            k: v.item() for k, v in losses.items()
        })

        if not hasattr(self, '_grad_log_counter'):
            self._grad_log_counter = 0
        self._grad_log_counter += 1

        return losses["total"]

    def _train_consistency_batch(self) -> None:
        """
        Train on unlabeled data using consistency regularization.
        Uses strong augmentations for student and weak augmentations for teacher.
        """
        try:
            data_unlab = next(self.trainIter_unlab)
        except StopIteration:
            self.trainIter_unlab = iter(self.unlabeled_loader)
            data_unlab = next(self.trainIter_unlab)

        # Get the actual model (unwrap DataParallel if needed)
        # This is important when using segmentation_models_pytorch with DataParallel
        if isinstance(self.model, DataParallel):
            mean_teacher = self.model.module
        else:
            mean_teacher = self.model

        # Get student and teacher inputs (with different augmentations)
        if isinstance(data_unlab, dict) and "student" in data_unlab:
            # Dataset returns both student and teacher versions (MONAI)
            x1_student = data_unlab["student"]
            x1_teacher = data_unlab["teacher"]
        else:
            # Fallback: Use same input for both (non-MONAI or single augmentation)
            x1 = data_unlab["img"] if isinstance(data_unlab, dict) else data_unlab
            # Convert to tensor if needed and ensure correct format
            if not isinstance(x1, torch.Tensor):
                x1 = torch.as_tensor(x1, dtype=torch.float32)
            x1_student = x1
            x1_teacher = x1.clone()  # Same input for both (weak augmentation handled in dataset)

        # Ensure tensors are on correct device and type
        if not isinstance(x1_student, torch.Tensor):
            x1_student = torch.as_tensor(x1_student, dtype=torch.float32)
        if not isinstance(x1_teacher, torch.Tensor):
            x1_teacher = torch.as_tensor(x1_teacher, dtype=torch.float32)

        # Move to device
        x1_student = x1_student.to(self.model_device_num)
        x1_teacher = x1_teacher.to(self.model_device_num)

        # Ensure correct shape: (B, C, H, W)
        if x1_student.dim() == 3:
            x1_student = x1_student.unsqueeze(0)
        if x1_teacher.dim() == 3:
            x1_teacher = x1_teacher.unsqueeze(0)

        # Forward pass through student (strong augmentation)
        # Student should already be in train mode from main training loop
        outputs = mean_teacher(x1_student, use_teacher=False)

        # Get student softmax probabilities
        if isinstance(outputs, tuple):
            outputs_soft = tuple(torch.softmax(out, dim=1) for out in outputs)
        else:
            outputs_soft = torch.softmax(outputs, dim=1)

        # Forward pass through teacher (weak augmentation, no gradients)
        with torch.no_grad():
            outputs_ema = mean_teacher(x1_teacher, use_teacher=True)
            if isinstance(outputs_ema, tuple):
                outputs_ema_soft = tuple(torch.softmax(out, dim=1) for out in outputs_ema)
            else:
                outputs_ema_soft = torch.softmax(outputs_ema, dim=1)

        # Calculate ramp-up ratio
        rampup_ratio = get_rampup_ratio(
            self.glob_it,
            self.rampup_start,
            self.rampup_end,
            "sigmoid"
        )
        weighted_consistency_w = self.consistency_weight * rampup_ratio

        # Compute consistency loss
        if isinstance(outputs_soft, tuple):
            # Multi-task: sum consistency losses for all tasks
            loss_reg = sum(
                self.consistency_loss.mse_loss(student_soft, teacher_soft)
                for student_soft, teacher_soft in zip(outputs_soft, outputs_ema_soft)
            )
        else:
            loss_reg = self.consistency_loss(outputs, outputs_ema)

        # Weighted consistency loss
        weighted_consistency = weighted_consistency_w * loss_reg

        # Backward pass (only updates student)
        self.optimizer.zero_grad()
        weighted_consistency.backward()
        self.optimizer.step()

        # Track loss
        self.train_loss_tracker.append_losses({
            "consistency": weighted_consistency.item(),
            "consistency_raw": loss_reg.item(),
            "consistency_weight": weighted_consistency_w,
        })

    def _train_pseudo_labeling_batch(self) -> None:
        """
        Train on unlabeled data using pseudo-labels.

        Strategy:
        1. Generate pseudo-labels from teacher (or student) model
        2. Filter by confidence threshold
        3. Train student on high-confidence pseudo-labels
        """
        try:
            data_unlab = next(self.trainIter_unlab)
        except StopIteration:
            self.trainIter_unlab = iter(self.unlabeled_loader)
            data_unlab = next(self.trainIter_unlab)

        # Get unlabeled images
        if isinstance(data_unlab, dict):
            if "student" in data_unlab:
                # Separate augmentations for student/teacher
                x_student = data_unlab["student"]
                x_teacher = data_unlab.get("teacher", data_unlab["student"])
            else:
                x_student = data_unlab["img"]
                x_teacher = data_unlab["img"]
        else:
            x_student = data_unlab
            x_teacher = data_unlab

        # Convert to tensor and move to device
        if not isinstance(x_student, torch.Tensor):
            x_student = torch.as_tensor(x_student, dtype=torch.float32)
        if not isinstance(x_teacher, torch.Tensor):
            x_teacher = torch.as_tensor(x_teacher, dtype=torch.float32)

        x_student = x_student.to(self.model_device_num)
        x_teacher = x_teacher.to(self.model_device_num)

        # Ensure correct shape: (B, C, H, W)
        if x_student.dim() == 3:
            x_student = x_student.unsqueeze(0)
        if x_teacher.dim() == 3:
            x_teacher = x_teacher.unsqueeze(0)

        # Get model (unwrap DataParallel if needed)
        if isinstance(self.model, DataParallel):
            model_for_labels = self.model.module
            model_for_training = self.model.module
        else:
            model_for_labels = self.model
            model_for_training = self.model

        # Update confidence threshold (curriculum learning)
        if self.pseudo_label_scheduler is not None:
            current_acceptance = (
                self.pseudo_label_stats["accepted_pixels"] / self.pseudo_label_stats["total_pixels"]
                if self.pseudo_label_stats["total_pixels"] > 0
                else 0.0
            )
            new_threshold = self.pseudo_label_scheduler.get_threshold(
                self.glob_it if self.use_semi_supervised else self.current_iter,
                acceptance_rate=current_acceptance,
            )
            self.pseudo_label_generator.confidence_threshold = new_threshold

        # Generate pseudo-labels from teacher (or student)
        use_teacher = self.pseudo_label_use_teacher

        pseudo_label_dict = self.pseudo_label_generator.generate_pseudo_labels(
            model_for_labels,
            x_teacher,
            self.label_no,
            use_teacher=use_teacher,
        )

        pseudo_labels_onehot = pseudo_label_dict["pseudo_labels_onehot"]
        mask = pseudo_label_dict["mask"]
        confidence_map = pseudo_label_dict["confidence_map"]

        # Update statistics (even if no pixels accepted, so we can track acceptance rate)
        self.pseudo_label_stats["total_batches"] += 1
        self.pseudo_label_stats["total_pixels"] += mask.numel()
        self.pseudo_label_stats["accepted_pixels"] += mask.sum().item()

        # Debug logging for first few batches to diagnose acceptance issues
        if self.pseudo_label_stats["total_batches"] <= 3:
            conf_min = confidence_map.min().item()
            conf_max = confidence_map.max().item()
            conf_mean = confidence_map.mean().item()
            threshold = self.pseudo_label_generator.confidence_threshold
            logging.info(
                f"Pseudo-label batch {self.pseudo_label_stats['total_batches']}: "
                f"confidence range=[{conf_min:.4f}, {conf_max:.4f}], mean={conf_mean:.4f}, "
                f"threshold={threshold:.4f}, accepted={mask.sum().item()}/{mask.numel()}"
            )

        # Skip if no high-confidence pixels (but stats already updated)
        if mask.sum() == 0:
            return

        # Calculate ramp-up ratio for pseudo-label weight
        if self.use_semi_supervised:
            current_iter = self.glob_it
            rampup_start = self.pseudo_label_rampup_start
            rampup_end = self.pseudo_label_rampup_end
        else:
            current_iter = self.current_iter
            rampup_start = self.pseudo_label_rampup_start
            rampup_end = self.pseudo_label_rampup_end

        rampup_ratio = get_rampup_ratio(
            current_iter,
            rampup_start,
            rampup_end,
            "sigmoid",
        )
        weighted_pseudo_label_w = self.pseudo_label_weight * rampup_ratio

        # Forward pass through student
        if hasattr(model_for_training, 'student'):
            # MeanTeacherModel
            model_for_training.student.train()
            outputs = model_for_training(x_student, use_teacher=False)
        else:
            # Standard model
            model_for_training.train()
            outputs = model_for_training(x_student)

        # Handle multi-task outputs
        if isinstance(outputs, tuple):
            seg_output = outputs[0]
        else:
            seg_output = outputs

        # Apply mask to predictions and targets (only compute loss on high-confidence pixels)
        if mask.sum() == 0:
            return  # No pixels accepted, skip this batch

        # Expand mask to match output shape: (B, H, W) -> (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, H, W)

        # Mask the predictions and targets
        masked_seg_output = seg_output * mask_expanded
        masked_pseudo_labels_onehot = pseudo_labels_onehot * mask_expanded

        # Compute loss only on masked pixels
        if self.settings.loss_criterion == "CrossEntropyLoss":
            # CrossEntropyLoss expects class indices
            pseudo_labels_indices = torch.argmax(pseudo_labels_onehot, dim=1)

            # For CrossEntropyLoss, we need to handle masking differently
            # Use reduction='none' to get per-pixel loss, then mask
            ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_pixel = ce_loss_fn(seg_output, pseudo_labels_indices)  # (B, H, W)
            masked_loss_per_pixel = loss_per_pixel * mask.float()  # (B, H, W)
            loss = masked_loss_per_pixel.sum() / (mask.sum() + 1e-8)
        else:
            # For DiceLoss and other losses, compute loss on masked predictions/targets
            # Note: This is an approximation - ideally we'd compute Dice only on masked pixels
            # For now, we'll compute the loss and weight it by the acceptance rate
            loss = self.loss_criterion(masked_seg_output, masked_pseudo_labels_onehot)

            # Weight by acceptance rate to account for masked pixels
            acceptance_rate = mask.sum().float() / mask.numel()
            if acceptance_rate > 0:
                loss = loss / acceptance_rate

        # Weighted pseudo-label loss
        weighted_loss = weighted_pseudo_label_w * loss

        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Track loss (statistics already updated above)
        self.train_loss_tracker.append_losses({
            "pseudo_label": weighted_loss.item(),
            "pseudo_label_raw": loss.item(),
            "pseudo_label_weight": weighted_pseudo_label_w,
            "pseudo_label_acceptance_rate": mask.sum().item() / mask.numel() if mask.numel() > 0 else 0.0,
        })

        # Update iteration counter
        if not self.use_semi_supervised:
            self.current_iter += 1

    def _validate_one_batch(self, batch) -> None:
        """Validate on a single batch with metric computation."""
        inputs, targets = utils.prepare_training_batch(
            batch, self.model_device_num, self.label_no
        )

        outputs = self._ensure_tuple_output(self.model(inputs))
        is_multitask = isinstance(targets, dict)

        if is_multitask and self.use_multitask:
            losses = self.loss_calculator.compute(outputs, targets)
            self.valid_loss_tracker.append_losses({
                k: v.item() for k, v in losses.items()
            })
        else:
            seg_output = outputs[0]
            if isinstance(targets, dict):
                seg_target = targets["seg"]
            else:
                seg_target = targets

            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(seg_output, torch.argmax(seg_target, dim=1))
            else:
                loss = self.loss_criterion(seg_output, seg_target.float())

            self.valid_loss_tracker.append_losses({"total": loss.item(), "seg": loss.item()})

        metrics = self._compute_eval_metrics(outputs, targets)
        self.valid_loss_tracker.append_metrics(metrics)

    def _log_epoch_statistics(self, epoch: int, elapsed_time: float) -> None:
        """Log and store epoch-level statistics with per-class Dice."""
        train_avgs = self.train_loss_tracker.get_average_losses()
        valid_avgs = self.valid_loss_tracker.get_average_losses()
        valid_metrics = self.valid_loss_tracker.get_average_metrics()

        # Store in history
        self.epoch_history["train_total"].append(train_avgs.get("total", 0))
        self.epoch_history["train_seg"].append(train_avgs.get("seg", 0))
        self.epoch_history["train_boundary"].append(train_avgs.get("boundary", 0))
        self.epoch_history["train_task3"].append(train_avgs.get("task3", 0))

        # Store consistency loss if using semi-supervised learning
        if self.use_semi_supervised:
            self.epoch_history["train_consistency"].append(train_avgs.get("consistency", 0))
            self.epoch_history["train_consistency_raw"].append(train_avgs.get("consistency_raw", 0))
            self.epoch_history["consistency_weight"].append(train_avgs.get("consistency_weight", 0))

        # Store pseudo-labeling loss if using pseudo-labeling
        if self.use_pseudo_labeling:
            self.epoch_history["train_pseudo_label"].append(train_avgs.get("pseudo_label", 0))
            self.epoch_history["train_pseudo_label_raw"].append(train_avgs.get("pseudo_label_raw", 0))
            self.epoch_history["pseudo_label_weight"].append(train_avgs.get("pseudo_label_weight", 0))
            self.epoch_history["pseudo_label_acceptance_rate"].append(train_avgs.get("pseudo_label_acceptance_rate", 0))

        self.epoch_history["valid_total"].append(valid_avgs.get("total", 0))
        self.epoch_history["valid_seg"].append(valid_avgs.get("seg", 0))
        self.epoch_history["valid_boundary"].append(valid_avgs.get("boundary", 0))
        self.epoch_history["valid_task3"].append(valid_avgs.get("task3", 0))

        self.epoch_history["seg_dice"].append(valid_metrics.get("seg_dice", 0))
        self.epoch_history["boundary_dice"].append(valid_metrics.get("boundary_dice", 0))

        # Per-class Dice
        for c in range(self.label_no):
            key = f"dice_class_{c}"
            self.epoch_history[key].append(valid_metrics.get(key, 0))

        log_parts = [f"Epoch {epoch}"]

        train_str = f"Train Loss: {train_avgs.get('total', 0):.4f}"
        if self.use_multitask:
            train_str += f" (Seg: {train_avgs.get('seg', 0):.4f}"
            if "boundary" in train_avgs:
                train_str += f", Bound: {train_avgs.get('boundary', 0):.4f}"
            if "task3" in train_avgs:
                train_str += f", Task3: {train_avgs.get('task3', 0):.4f}"
            train_str += ")"
        log_parts.append(train_str)

        valid_str = f"Val Loss: {valid_avgs.get('total', 0):.4f}"
        if self.use_multitask:
            valid_str += f" (Seg: {valid_avgs.get('seg', 0):.4f}"
            if "boundary" in valid_avgs:
                valid_str += f", Bound: {valid_avgs.get('boundary', 0):.4f}"
            if "task3" in valid_avgs:
                valid_str += f", Task3: {valid_avgs.get('task3', 0):.4f}"
            valid_str += ")"
        log_parts.append(valid_str)

        # Mean Dice
        metric_str = f"Seg Dice: {valid_metrics.get('seg_dice', 0):.4f}"
        if "boundary_dice" in valid_metrics:
            metric_str += f", Bound Dice: {valid_metrics.get('boundary_dice', 0):.4f}"
        log_parts.append(metric_str)

        log_parts.append(f"Time: {elapsed_time:.1f}s")

        logging.info(" | ".join(log_parts))

        # Log per-class Dice breakdown
        class_dice_parts = []
        for c in range(self.label_no):
            class_name = self.codes.get(c, f"C{c}") if self.codes else f"C{c}"
            dice_val = valid_metrics.get(f"dice_class_{c}", 0)
            class_dice_parts.append(f"{class_name}: {dice_val:.3f}")

        logging.info(f"  Per-class Dice: {' | '.join(class_dice_parts)}")

    def _load_in_model_and_optimizer(
        self, learning_rate, output_path, frozen=False, optimizer=False
    ):
        self._create_model_and_optimiser(learning_rate, frozen=frozen)
        logging.info("Loading weights from saved checkpoint.")
        loss_val = self._load_in_weights(output_path, optimizer=optimizer)
        return loss_val

    def _load_in_weights(self, output_path, optimizer=False, gpu=True):
        """Load model and optionally optimizer weights from checkpoint."""
        map_location = f"cuda:{self.model_device_num}" if gpu else "cpu"
        model_dict = torch.load(output_path, map_location=map_location, weights_only=False)

        loss_val = self.model_manager.load_model_weights(
            self.model,
            self.optimizer,
            output_path,
            gpu=gpu,
            load_optimizer=optimizer,
            use_semi_supervised=self.use_semi_supervised,
        )

        # Restore glob_it for semi-supervised learning
        if self.use_semi_supervised and 'glob_it' in model_dict:
            self.glob_it = model_dict['glob_it']
            # Also update the model's glob_it
            if isinstance(self.model, DataParallel):
                self.model.module.glob_it = model_dict['glob_it']
            else:
                self.model.glob_it = model_dict['glob_it']

        return loss_val


    def _run_lr_finder(self):
        """Run learning rate finder and return optimal learning rate."""
        return self.lr_finder.find_optimal_lr(
            self.model,
            self.optimizer,
            self.loss_criterion,
            self._train_one_batch,
        )

    def _create_optimizer(self, learning_rate, params=None):
        """Create optimizer for model."""
        return self.model_manager.create_optimizer(
            self.model, learning_rate, params=params
        )

    def _create_exponential_lr_scheduler(self):
        """Create exponential LR scheduler for LR finding."""
        return self.lr_finder._create_exponential_lr_scheduler(self.optimizer)

    def _create_oc_lr_scheduler(self, num_epochs, lr_to_use):
        """Create OneCycleLR scheduler."""
        return self.model_manager.create_onecycle_lr_scheduler(
            self.optimizer,
            num_epochs,
            lr_to_use,
            len(self.training_loader),
            self.settings.pct_lr_inc,
        )

    def _create_early_stopping(self, output_path, patience, best_score=None):
        """Create early stopping callback."""
        return self.model_manager.create_early_stopping(
            output_path, patience, self.model_struc_dict, best_score
        )

    def output_loss_fig(self, model_out_path: Path) -> None:
        """Save figures showing training/validation losses and metrics per task."""
        self.visualizer.plot_loss_history(self.epoch_history, model_out_path)


    def output_prediction_figure(self, model_path: Path) -> None:
        """Save visualization of predictions on validation samples."""
        self.visualizer.plot_predictions(
            self.model,
            self.validation_loader,
            self.model_device_num,
            model_path,
            self._ensure_tuple_output,
        )
