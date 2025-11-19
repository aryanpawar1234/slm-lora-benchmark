"""Checkpoint management utilities"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """Manager for saving and loading model checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            metric_name: Metric name for best checkpoint
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        self.checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Metrics dictionary
            is_best: Whether this is the best checkpoint
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        if extra_state:
            checkpoint_data.update(extra_state)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Update checkpoint list
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'step': step,
            'metric': metrics.get(self.metric_name, 0)
        })
        
        # Check if best checkpoint
        current_metric = metrics.get(self.metric_name, 0)
        is_better = (
            (self.mode == 'min' and current_metric < self.best_metric) or
            (self.mode == 'max' and current_metric > self.best_metric)
        )
        
        if is_better or is_best:
            self.best_metric = current_metric
            self.best_checkpoint_path = checkpoint_path
            
            # Save as best.pt
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy(checkpoint_path, best_path)
            print(f"Best checkpoint updated: {best_path}")
        
        # Remove old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
        
        return checkpoint
    
    def load_best_checkpoint(self, model, optimizer=None, device: str = "cuda"):
        """Load best checkpoint"""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.load_checkpoint(str(self.best_checkpoint_path), model, optimizer, device)
        else:
            print("No best checkpoint found")
            return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric
            self.checkpoints.sort(
                key=lambda x: x['metric'],
                reverse=(self.mode == 'max')
            )
            
            # Keep best checkpoints
            keep_checkpoints = self.checkpoints[:self.max_checkpoints]
            remove_checkpoints = self.checkpoints[self.max_checkpoints:]
            
            # Remove old checkpoint files
            for ckpt in remove_checkpoints:
                if ckpt['path'].exists() and ckpt['path'] != self.best_checkpoint_path:
                    os.remove(ckpt['path'])
                    print(f"Removed old checkpoint: {ckpt['path']}")
            
            self.checkpoints = keep_checkpoints
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return self.checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if self.checkpoints:
            return str(max(self.checkpoints, key=lambda x: x['step'])['path'])
        return None


def save_model(model, save_path: str, tokenizer=None):
    """
    Save model and optionally tokenizer
    
    Args:
        model: Model to save
        save_path: Directory to save model
        tokenizer: Tokenizer to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
    
    # Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(save_path)
        print(f"Tokenizer saved to: {save_path}")


def load_model(model_class, load_path: str, device: str = "cuda"):
    """
    Load model from path
    
    Args:
        model_class: Model class
        load_path: Path to load from
        device: Device to load on
        
    Returns:
        Loaded model
    """
    model = model_class.from_pretrained(load_path)
    model.to(device)
    print(f"Model loaded from: {load_path}")
    return model