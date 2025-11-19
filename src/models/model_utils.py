"""Model utility functions"""

import torch
from typing import Dict, Any
from transformers import PreTrainedModel


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_trainable_parameters(model: PreTrainedModel):
    """
    Print trainable parameters information
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print("=" * 80)
    print("TRAINABLE PARAMETERS")
    print("=" * 80)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Trainable %: {params['trainable_percent']:.2f}%")
    print("=" * 80)


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return size_mb


def freeze_model(model: PreTrainedModel):
    """
    Freeze all model parameters
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print("All model parameters frozen")


def unfreeze_model(model: PreTrainedModel):
    """
    Unfreeze all model parameters
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("All model parameters unfrozen")


def enable_gradient_checkpointing(model: PreTrainedModel):
    """
    Enable gradient checkpointing for model
    
    Args:
        model: PyTorch model
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Model does not support gradient checkpointing")


def prepare_model_for_training(
    model: PreTrainedModel,
    gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """
    Prepare model for training
    
    Args:
        model: PyTorch model
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Prepared model
    """
    # Enable gradient checkpointing
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    # Set to training mode
    model.train()
    
    # Print parameter info
    print_trainable_parameters(model)
    
    return model