"""Optimizer and scheduler creation"""

import torch
from torch.optim import AdamW, Adam
from transformers import get_scheduler

def create_optimizer(model, config):
    """Create optimizer from config"""
    training_config = config.get("training", {})

    optimizer_name = training_config.get("optim", "adamw_torch")
    lr = float(training_config.get("learning_rate", 1e-4))  # Convert to float
    weight_decay = float(training_config.get("weight_decay", 0.01))  # Convert to float

    if optimizer_name == "adamw_torch":
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(training_config.get("adam_beta1", 0.9)),
                   float(training_config.get("adam_beta2", 0.999))),
            eps=float(training_config.get("adam_epsilon", 1e-8))
        )
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer

def create_scheduler(optimizer, num_training_steps, config):
    """Create learning rate scheduler"""
    training_config = config.get("training", {})

    scheduler_type = training_config.get("lr_scheduler_type", "cosine")
    warmup_steps = int(training_config.get("warmup_steps", 100))

    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    return scheduler
