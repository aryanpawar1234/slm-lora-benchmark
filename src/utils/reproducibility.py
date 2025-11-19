"""Reproducibility utilities for deterministic training"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"Random seed set to: {seed}")


def enable_deterministic_mode():
    """Enable deterministic mode for PyTorch"""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Deterministic mode enabled")


def disable_deterministic_mode():
    """Disable deterministic mode (for better performance)"""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print("Deterministic mode disabled")


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device
    
    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")
        
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def print_environment_info():
    """Print environment information"""
    print("=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    print(f"Python: {os.sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 80)
