"""Main training loop with W&B logging"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import math

from src.utils.wandb_logger import WandbLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.memory import print_memory_stats


class Trainer:
    """Trainer with comprehensive W&B logging"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        wandb_logger: WandbLogger,
        checkpoint_manager: CheckpointManager,
        device: str = "cuda"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.wandb_logger = wandb_logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Training config
        training_config = config.get("training", {})
        self.num_epochs = training_config.get("num_epochs", 3)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.logging_steps = training_config.get("logging_steps", 10)
        self.eval_steps = training_config.get("eval_steps", 500)
        self.save_steps = training_config.get("save_steps", 1000)
        
        # EMA loss tracking
        self.ema_loss = None
        self.ema_alpha = 0.9
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*80}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch summary
            self.log_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                step=self.global_step,
                metrics=val_metrics,
                is_best=is_best
            )
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def train_epoch(self):
        """Train single epoch"""
        self.model.train()
        
        total_loss = 0
        total_tokens = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # Forward pass
            loss, metrics = self.training_step(batch)
            
            # Backward pass
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Calculate step metrics
                step_time = time.time() - step_start_time
                tokens_per_sec = metrics['n_tokens'] / step_time
                
                # Update EMA loss
                if self.ema_loss is None:
                    self.ema_loss = metrics['token_loss']
                else:
                    self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * metrics['token_loss']
                
                # Log to W&B
                if self.global_step % self.logging_steps == 0:
                    self.log_training_metrics(metrics, grad_norm, tokens_per_sec, step_time)
                
                # Evaluate
                if self.global_step % self.eval_steps == 0:
                    val_metrics = self.validate()
                    self.model.train()  # Back to training mode
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=self.current_epoch,
                        step=self.global_step,
                        metrics=metrics
                    )
            
            total_loss += metrics['token_loss'] * metrics['n_tokens']
            total_tokens += metrics['n_tokens']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['token_loss']:.4f}",
                'ppl': f"{metrics['ppl']:.2f}",
                'lr': f"{self.get_lr():.2e}"
            })
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        return {
            'train_loss': avg_loss,
            'train_ppl': math.exp(avg_loss) if avg_loss < 100 else float('inf'),
            'epoch_time': epoch_time
        }
    
    def training_step(self, batch):
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Calculate metrics
        n_tokens = (batch['labels'] != -100).sum().item()
        token_loss = loss.item()
        ppl = math.exp(token_loss) if token_loss < 100 else float('inf')
        bpt = token_loss / math.log(2)  # Bits per token
        
        metrics = {
            'token_loss': token_loss,
            'ppl': ppl,
            'bpt': bpt,
            'n_tokens': n_tokens
        }
        
        return loss, metrics
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                n_tokens = (batch['labels'] != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        bpt = avg_loss / math.log(2)
        
        val_metrics = {
            'val_loss': avg_loss,
            'val_ppl': ppl,
            'val_bpt': bpt,
            'val_n_tokens': total_tokens
        }
        
        # Log to W&B
        self.wandb_logger.log({
            'val/token_loss': avg_loss,
            'val/ppl': ppl,
            'val/bpt': bpt,
            'val/n_tokens': total_tokens
        }, step=self.global_step)
        
        print(f"\nValidation - Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, BPT: {bpt:.4f}")
        
        return val_metrics
    
    def log_training_metrics(self, metrics, grad_norm, tokens_per_sec, step_time):
        """Log training metrics to W&B"""
        log_dict = {
            'train/step': self.global_step,
            'train/epoch': self.current_epoch,
            'train/token_loss': metrics['token_loss'],
            'train/ppl': metrics['ppl'],
            'train/bpt': metrics['bpt'],
            'train/lr': self.get_lr(),
            'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'train/throughput_tokens_per_sec': tokens_per_sec,
            'train/step_time': step_time,
            'train/ema_token_loss': self.ema_loss,
        }
        
        # Add LoRA rank if available
        if hasattr(self.model, 'peft_config'):
            log_dict['train/lora_rank'] = list(self.model.peft_config.values())[0].r
        
        self.wandb_logger.log(log_dict, step=self.global_step)
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Log epoch summary"""
        summary = {
            'epoch/train_loss': train_metrics['train_loss'],
            'epoch/train_ppl': train_metrics['train_ppl'],
            'epoch/val_loss': val_metrics['val_loss'],
            'epoch/val_ppl': val_metrics['val_ppl'],
            'epoch/time': train_metrics['epoch_time']
        }
        
        self.wandb_logger.log(summary, step=self.global_step)
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    config,
    wandb_logger,
    checkpoint_manager,
    device="cuda"
):
    """Convenience function for training"""
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        wandb_logger=wandb_logger,
        checkpoint_manager=checkpoint_manager,
        device=device
    )
    
    return trainer.train()