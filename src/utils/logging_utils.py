"""Logging utilities for training and evaluation"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.logging import RichHandler


console = Console()


def setup_logger(
    name: str = "slm_lora",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Log metrics dictionary
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        step: Training step
        prefix: Prefix for metric names
    """
    if step is not None:
        log_str = f"Step {step} - "
    else:
        log_str = ""
    
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{prefix}{key}: {value:.4f}")
        else:
            metric_strs.append(f"{prefix}{key}: {value}")
    
    log_str += " | ".join(metric_strs)
    logger.info(log_str)


def log_model_info(logger: logging.Logger, model, tokenizer=None) -> None:
    """
    Log model information
    
    Args:
        logger: Logger instance
        model: Model instance
        tokenizer: Tokenizer instance (optional)
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 80)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    if tokenizer:
        logger.info(f"Vocabulary size: {len(tokenizer):,}")
        logger.info(f"Model max length: {tokenizer.model_max_length}")
    
    logger.info("=" * 80)


def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log configuration dictionary
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    
    def log_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 80)


def log_training_start(logger: logging.Logger, num_epochs: int, total_steps: int) -> None:
    """
    Log training start information
    
    Args:
        logger: Logger instance
        num_epochs: Number of training epochs
        total_steps: Total training steps
    """
    logger.info("=" * 80)
    logger.info("TRAINING STARTED")
    logger.info("=" * 80)
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log epoch summary
    
    Args:
        logger: Logger instance
        epoch: Epoch number
        train_metrics: Training metrics
        val_metrics: Validation metrics (optional)
    """
    logger.info("=" * 80)
    logger.info(f"EPOCH {epoch} SUMMARY")
    logger.info("=" * 80)
    
    logger.info("Training Metrics:")
    for key, value in train_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    if val_metrics:
        logger.info("Validation Metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("=" * 80)


def log_training_complete(
    logger: logging.Logger,
    total_time: float,
    best_metrics: Dict[str, float]
) -> None:
    """
    Log training completion
    
    Args:
        logger: Logger instance
        total_time: Total training time in seconds
        best_metrics: Best metrics achieved
    """
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\nBest Metrics:")
    for key, value in best_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("=" * 80)


class MetricsLogger:
    """Metrics logger with history tracking"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.history = {}
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics and store in history"""
        # Store in history
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append((step, value))
        
        # Log to console
        log_metrics(self.logger, metrics, step, prefix)
    
    def get_history(self, metric_name: str) -> list:
        """Get history for specific metric"""
        return self.history.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[Any]:
        """Get latest value for metric"""
        history = self.history.get(metric_name)
        if history:
            return history[-1][1]
        return None