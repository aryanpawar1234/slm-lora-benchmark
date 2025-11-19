"""
Main training script for SLM LoRA fine-tuning
Run with: python scripts/train.py
"""

import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, setup_logger, set_seed,
    print_environment_info, WandbLogger, CheckpointManager
)
from src.models import ModelFactory, create_lora_model, print_trainable_parameters
from src.data import create_datamodule
from src.training import create_optimizer, create_scheduler, train_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train SLM with LoRA")
    
    parser.add_argument("--configs_dir", type=str, default="configs",
                      help="Directory with config files")
    parser.add_argument("--output_dir", type=str, default="outputs/runs",
                      help="Output directory")
    parser.add_argument("--dataset_names", nargs="+", default=None,
                      help="Dataset names to use (default: all)")
    parser.add_argument("--wandb_project", type=str, default="slm-lora-benchmark",
                      help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                      help="W&B run name")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="train",
        log_dir=os.path.join(args.output_dir, "logs"),
        log_to_file=True
    )
    
    logger.info("=" * 80)
    logger.info("SLM LoRA TRAINING")
    logger.info("=" * 80)
    
    # Print environment info
    print_environment_info()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configurations
    logger.info("\nLoading configurations...")
    
    datasets_config = load_config(os.path.join(args.configs_dir, "datasets.yaml"))
    model_config = load_config(os.path.join(args.configs_dir, "model_config.yaml"))
    lora_config = load_config(os.path.join(args.configs_dir, "lora_config.yaml"))
    training_config = load_config(os.path.join(args.configs_dir, "training_config.yaml"))
    
    # Combine configs
    config = {
        "datasets": datasets_config,
        "model": model_config["model"],
        "tokenizer": model_config.get("tokenizer", {}),
        "lora": lora_config["lora"],
        "training": training_config["training"],
        "wandb": training_config.get("wandb", {}),
        "output": training_config.get("output", {})
    }
    
    logger.info("Configurations loaded successfully")
    
    # Initialize W&B
    logger.info("\nInitializing Weights & Biases...")
    
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        tags=config["wandb"].get("tags", ["pythia-410m", "lora"]),
        notes=config["wandb"].get("notes", "LoRA fine-tuning experiment"),
        group=config["wandb"].get("group", "experiments"),
        job_type="train",
        mode=config["wandb"].get("mode", "online")
    )
    
    # Load model and tokenizer
    logger.info("\nLoading model and tokenizer...")
    
    model, tokenizer = ModelFactory.from_config(
        model_config=config["model"],
        tokenizer_config=config.get("tokenizer")
    )
    
    logger.info(f"Model loaded: {config['model']['name']}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Apply LoRA
    logger.info("\nApplying LoRA adapters...")
    
    model = create_lora_model(
        model=model,
        lora_config_dict=config["lora"],
        prepare_for_kbit=config["model"].get("load_in_8bit", False)
    )
    
    print_trainable_parameters(model)
    
    # Setup data
    logger.info("\nSetting up datasets and dataloaders...")
    
    datamodule = create_datamodule(
        config=config,
        tokenizer=tokenizer,
        dataset_names=args.dataset_names
    )
    
    datamodule.setup()
    
    train_dataloader = datamodule.get_train_dataloader()
    val_dataloader = datamodule.get_val_dataloader()
    
    logger.info(f"Training samples: {len(datamodule.train_dataset)}")
    logger.info(f"Validation samples: {len(datamodule.val_dataset)}")
    logger.info(f"Training batches: {len(train_dataloader)}")
    
    # Create optimizer and scheduler
    logger.info("\nCreating optimizer and scheduler...")
    
    optimizer = create_optimizer(model, config)
    
    num_training_steps = len(train_dataloader) * config["training"]["num_epochs"]
    scheduler = create_scheduler(optimizer, num_training_steps, config)
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Scheduler: {config['training'].get('lr_scheduler_type', 'cosine')}")
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Setup checkpoint manager
    logger.info("\nSetting up checkpoint manager...")
    
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=config["training"].get("save_total_limit", 3),
        metric_name="val_loss",
        mode="min"
    )
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\nUsing device: {device}")
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80 + "\n")
    
    results = train_model(
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
    
    # Log final results
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Total training time: {results['total_time']/3600:.2f} hours")
    logger.info(f"Best checkpoint: {checkpoint_manager.best_checkpoint_path}")
    
    # Finish W&B
    wandb_logger.finish()
    
    logger.info("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()