"""
Learning rate finding utilities for the 2D trainer.

This module contains:
- Learning rate finder implementation
- LR scheduler utilities
- LR selection from loss curves
"""

import logging
import math
from typing import Tuple

import numpy as np
import termplotlib as tpl
from tqdm import tqdm

import volume_segmantics.utilities.config as cfg


# A selected rate below this cannot move the weights in a phase of a few epochs.
MIN_USEFUL_LR = 1e-6


def warn_if_learning_rate_too_low(lr: float, *, start_lr: float, end_lr: float, phase: str) -> bool:
    """Log a warning when a phase's selected rate is below :data:`MIN_USEFUL_LR`.

    The unfrozen phase searches ``starting_lr / lr_reduce_factor`` to
    ``end_lr / lr_reduce_factor``; with the factor at 500 that range was
    1e-7 to 2e-6 and the phase selected about 3e-7.
    Returns ``True`` when the warning was raised.
    """
    if lr >= MIN_USEFUL_LR:
        return False
    logging.warning(
        "The %s phase selected a learning rate of %.1e from a search range of "
        "%.1e to %.1e, below %.0e; this phase will barely change the weights. "
        "Check lr_reduce_factor (the unfrozen range is starting_lr / "
        "lr_reduce_factor to end_lr / lr_reduce_factor) and starting_lr / end_lr.",
        phase, lr, start_lr, end_lr, MIN_USEFUL_LR,
    )
    return True


class LearningRateFinder:
    """
    Finds optimal learning rate using exponential LR scheduling.
    
    Based on the approach from "Cyclical Learning Rates for Training Neural Networks"
    by Leslie N. Smith.
    """
    
    def __init__(
        self,
        starting_lr: float,
        end_lr: float,
        lr_find_epochs: int,
        training_loader,
        plot_lr_graph: bool = False,
    ):
        """
        Initialize learning rate finder.
        
        Args:
            starting_lr: Starting learning rate for search
            end_lr: Ending learning rate for search
            lr_find_epochs: Number of epochs to run LR finder
            training_loader: Training data loader
            plot_lr_graph: Whether to plot LR vs loss graph
        """
        self.starting_lr = starting_lr
        self.end_lr = end_lr
        self.lr_find_epochs = lr_find_epochs
        self.training_loader = training_loader
        self.plot_lr_graph = plot_lr_graph
        self.log_lr_ratio = math.log(end_lr / starting_lr)
    
    def find_optimal_lr(
        self,
        model,
        optimizer,
        loss_criterion,
        train_one_batch_fn,
    ) -> float:
        """
        Run learning rate finder and return optimal learning rate.
        
        Args:
            model: Model to train
            optimizer: Optimizer (will be modified during search)
            loss_criterion: Loss function
            train_one_batch_fn: Function to train one batch (takes lr_scheduler, batch)
        
        Returns:
            Optimal learning rate
        """
        logging.info("Finding optimal learning rate.")
        
        # Create exponential LR scheduler
        lr_scheduler = self._create_exponential_lr_scheduler(optimizer)
        
        # Run LR finder
        lr_find_loss, lr_find_lr = self._lr_finder(
            model, optimizer, lr_scheduler, train_one_batch_fn
        )
        
        # Find optimal LR from graph
        lr_to_use = self._find_lr_from_graph(lr_find_loss, lr_find_lr)
        logging.info(f"Selected learning rate: {lr_to_use:.6f}")
        
        warn_if_learning_rate_too_low(
        
            lr_to_use, start_lr=self.starting_lr, end_lr=self.end_lr, phase="current",
        
        )
        
        return lr_to_use
    
    def _lr_finder(
        self,
        model,
        optimizer,
        lr_scheduler,
        train_one_batch_fn,
        smoothing: float = 0.05,
    ) -> Tuple[list, list]:
        """
        Run learning rate finder by training with exponentially increasing LR.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            lr_scheduler: Exponential LR scheduler
            train_one_batch_fn: Function to train one batch
            smoothing: Smoothing factor for loss curve
        
        Returns:
            Tuple of (losses, learning_rates) lists
        """
        lr_find_loss = []
        lr_find_lr = []
        iters = 0
        
        model.train()
        logging.info(f"Running LR finder for {self.lr_find_epochs} epochs.")
        
        for i in range(self.lr_find_epochs):
            for batch in tqdm(
                self.training_loader,
                desc=f"LR Finder Epoch {i + 1}",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = train_one_batch_fn(lr_scheduler, batch)
                lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                
                if iters == 0:
                    lr_find_loss.append(loss)
                else:
                    loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                    lr_find_loss.append(loss)
                
                if loss > 1 and iters > len(self.training_loader) // 1.333:
                    break
                iters += 1
        
        if self.plot_lr_graph:
            fig = tpl.figure()
            fig.plot(
                np.log10(lr_find_lr),
                lr_find_loss,
                width=50,
                height=30,
                xlabel="Log10 Learning Rate",
            )
            fig.show()
        
        return lr_find_loss, lr_find_lr
    
    @staticmethod
    def _find_lr_from_graph(lr_find_loss, lr_find_lr) -> float:
        """
        Find learning rate at steepest loss descent.
        
        Args:
            lr_find_loss: List of losses during LR search
            lr_find_lr: List of learning rates during LR search
        
        Returns:
            Optimal learning rate
        """
        default_min_lr = cfg.DEFAULT_MIN_LR
        
        # Convert to numpy arrays
        for i in range(len(lr_find_loss)):
            if hasattr(lr_find_loss[i], 'is_cuda') and lr_find_loss[i].is_cuda:
                lr_find_loss[i] = lr_find_loss[i].cpu()
            if hasattr(lr_find_loss[i], 'detach'):
                lr_find_loss[i] = lr_find_loss[i].detach().numpy()
        
        losses = np.array(lr_find_loss)
        try:
            gradients = np.gradient(losses)
            min_gradient = gradients.min()
            if min_gradient < 0:
                min_loss_grad_idx = gradients.argmin()
            else:
                logging.info(f"Min gradient ({min_gradient}) positive, using default LR.")
                return default_min_lr
        except Exception as e:
            logging.info(f"Gradient computation failed: {e}. Using default LR.")
            return default_min_lr
        
        min_lr = lr_find_lr[min_loss_grad_idx]
        return min_lr / cfg.LR_DIVISOR
    
    def _create_exponential_lr_scheduler(self, optimizer):
        """Create exponential LR scheduler for LR finding."""
        def lr_exp_stepper(x):
            return math.exp(
                x * self.log_lr_ratio / (self.lr_find_epochs * len(self.training_loader))
            )
        
        import torch.optim.lr_scheduler as lr_scheduler_module
        return lr_scheduler_module.LambdaLR(optimizer, lr_exp_stepper)
