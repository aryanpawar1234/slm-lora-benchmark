"""Weights & Biases logger for experiment tracking"""

import os
import wandb
from typing import Dict, Any, Optional, List
from pathlib import Path


class WandbLogger:
    """Weights & Biases experiment logger"""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        mode: str = "online"
    ):
        """
        Initialize W&B logger
        
        Args:
            project: W&B project name
            entity: W&B entity/username
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes
            group: Run group
            job_type: Job type (train, eval, sweep)
            mode: online, offline, or disabled
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.mode = mode
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            job_type=job_type,
            mode=mode,
            reinit=True
        )
        
        print(f"W&B run initialized: {self.run.name}")
        print(f"W&B run URL: {self.run.url}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            commit: Whether to commit immediately
        """
        if self.mode != "disabled":
            wandb.log(metrics, step=step, commit=commit)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        if self.mode != "disabled":
            wandb.config.update(config)
    
    def log_model(self, model_path: str, name: str = "model"):
        """
        Log model artifact
        
        Args:
            model_path: Path to model checkpoint
            name: Artifact name
        """
        if self.mode != "disabled":
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def log_table(self, table_name: str, data: List[List[Any]], columns: List[str]):
        """
        Log table to W&B
        
        Args:
            table_name: Name of the table
            data: Table data (list of rows)
            columns: Column names
        """
        if self.mode != "disabled":
            table = wandb.Table(data=data, columns=columns)
            wandb.log({table_name: table})
    
    def log_text_samples(
        self,
        inputs: List[str],
        outputs: List[str],
        step: Optional[int] = None
    ):
        """
        Log text generation samples
        
        Args:
            inputs: Input texts
            outputs: Generated outputs
            step: Training step
        """
        if self.mode != "disabled":
            data = [[inp, out] for inp, out in zip(inputs, outputs)]
            table = wandb.Table(data=data, columns=["Input", "Output"])
            wandb.log({"generations": table}, step=step)
    
    def log_histogram(self, name: str, values: List[float], step: Optional[int] = None):
        """
        Log histogram
        
        Args:
            name: Histogram name
            values: List of values
            step: Training step
        """
        if self.mode != "disabled":
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        """
        Watch model gradients and parameters
        
        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if self.mode != "disabled":
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def alert(self, title: str, text: str, level: str = "INFO"):
        """
        Send W&B alert
        
        Args:
            title: Alert title
            text: Alert message
            level: Alert level (INFO, WARN, ERROR)
        """
        if self.mode != "disabled":
            wandb.alert(title=title, text=text, level=getattr(wandb.AlertLevel, level))
    
    def finish(self):
        """Finish W&B run"""
        if self.mode != "disabled":
            wandb.finish()
            print("W&B run finished")
    
    @property
    def run_id(self) -> str:
        """Get run ID"""
        return self.run.id if self.run else None
    
    @property
    def run_name(self) -> str:
        """Get run name"""
        return self.run.name if self.run else None
    
    @property
    def run_url(self) -> str:
        """Get run URL"""
        return self.run.url if self.run else None


def create_wandb_logger(
    config: Dict[str, Any],
    job_type: str = "train"
) -> WandbLogger:
    """
    Create W&B logger from config
    
    Args:
        config: Configuration dictionary
        job_type: Job type
        
    Returns:
        WandbLogger instance
    """
    wandb_config = config.get("wandb", {})
    
    return WandbLogger(
        project=wandb_config.get("project", "slm-lora-benchmark"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name"),
        config=config,
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes"),
        group=wandb_config.get("group"),
        job_type=job_type,
        mode=wandb_config.get("mode", "online")
    )
