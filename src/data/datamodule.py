"""Data module for managing datasets and dataloaders"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer

from .dataset_loader import load_multiple_datasets, create_train_val_split
from .collator import DataCollatorForLanguageModeling


class DataModule:
    """Data module for managing train/val datasets and dataloaders"""
    
    def __init__(
        self,
        datasets_config: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        preprocessing_config: Dict[str, Any],
        dataloader_config: Dict[str, Any],
        validation_config: Dict[str, Any],
        dataset_names: Optional[list] = None
    ):
        self.datasets_config = datasets_config
        self.tokenizer = tokenizer
        self.preprocessing_config = preprocessing_config
        self.dataloader_config = dataloader_config
        self.validation_config = validation_config
        self.dataset_names = dataset_names
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
    
    def setup(self):
        """Setup datasets and dataloaders"""
        print("Setting up data module...")
        
        combined_dataset = load_multiple_datasets(
            self.datasets_config,
            self.tokenizer,
            self.preprocessing_config,
            self.dataset_names
        )
        
        split_datasets = create_train_val_split(
            combined_dataset,
            val_split=self.validation_config.get("split_ratio", 0.1),
            seed=self.validation_config.get("seed", 42)
        )
        
        self.train_dataset = split_datasets["train"]
        self.val_dataset = split_datasets["validation"]
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self._create_dataloaders()
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders"""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.dataloader_config.get("batch_size", 8),
            shuffle=self.dataloader_config.get("shuffle", True),
            num_workers=self.dataloader_config.get("num_workers", 2),
            pin_memory=self.dataloader_config.get("pin_memory", True),
            drop_last=self.dataloader_config.get("drop_last", True),
            collate_fn=self.collator
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.dataloader_config.get("batch_size", 8),
            shuffle=False,
            num_workers=self.dataloader_config.get("num_workers", 2),
            pin_memory=self.dataloader_config.get("pin_memory", True),
            drop_last=False,
            collate_fn=self.collator
        )
        
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Validation batches: {len(self.val_dataloader)}")
    
    def get_train_dataloader(self):
        if self.train_dataloader is None:
            self.setup()
        return self.train_dataloader
    
    def get_val_dataloader(self):
        if self.val_dataloader is None:
            self.setup()
        return self.val_dataloader


def create_datamodule(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    dataset_names: Optional[list] = None
):
    """Create data module from configuration"""
    datamodule = DataModule(
        datasets_config=config["datasets"]["datasets"],
        tokenizer=tokenizer,
        preprocessing_config=config["datasets"]["preprocessing"],
        dataloader_config=config["datasets"]["dataloader"],
        validation_config=config["datasets"]["validation"],
        dataset_names=dataset_names
    )
    
    return datamodule
