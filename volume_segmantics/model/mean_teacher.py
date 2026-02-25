"""
Mean Teacher model wrapper for semi-supervised learning.

Based on: Tarvainen & Valpola, "Mean teachers are better role models" and PyMIC

"""

import copy
import torch
import torch.nn as nn
import logging


class MeanTeacherModel(nn.Module):
    """
    Mean Teacher wrapper that maintains a student model and a teacher model.
    Teacher model is an exponential moving average (EMA) of the student model.
    
    This wrapper is compatible with all segmentation_models_pytorch models
    and any nn.Module that implements forward(), state_dict(), and load_state_dict().
    
    Based on: Tarvainen & Valpola, "Mean teachers are better role models" and PyMIC
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.99,
    ):
        """
        Initialize Mean Teacher model.
        
        Args:
            student_model: The base model to wrap (e.g., smp.Unet, MultitaskUnet)
            ema_decay: Exponential moving average decay rate (default: 0.99)
        """
        super().__init__()
        self.student = student_model
        self.teacher = copy.deepcopy(student_model)
        
        # Freeze teacher parameters (updated via EMA, not gradients)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.ema_decay = ema_decay
        self.glob_it = 0  # Global iteration counter 
        
        logging.info(
            f"MeanTeacherModel initialized with ema_decay={ema_decay}. "
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
            iter_max: Maximum number of iterations (for adaptive decay calculation).
                     If None, uses simple decay schedule.
        """
        # adaptive decay: min(ema_decay, 1 - 1/(glob_it/iter_valid + 1))
        # here: min(ema_decay, 1 - 1/(glob_it + 1))
        # This ensures smooth ramp-up at the beginning
        alpha = min(1 - 1 / (self.glob_it + 1), self.ema_decay)
        
        # Update teacher parameters: ?_t = ? * ?_t + (1 - ?) * ?_s
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
        
        self.glob_it += 1
    
    def get_student_model(self):
        """Get the student model (for evaluation, saving, etc.)."""
        return self.student
    
    def get_teacher_model(self):
        """Get the teacher model."""
        return self.teacher
    
    def state_dict(self):
        """Return state dict including both student and teacher models."""
        return {
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'ema_decay': self.ema_decay,
            'glob_it': self.glob_it,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for both student and teacher models."""
        self.student.load_state_dict(state_dict['student'])
        self.teacher.load_state_dict(state_dict['teacher'])
        self.ema_decay = state_dict.get('ema_decay', 0.99)
        self.glob_it = state_dict.get('glob_it', 0)
    
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
