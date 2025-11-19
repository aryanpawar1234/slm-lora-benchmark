"""Dataset loader for HuggingFace datasets"""

import os
from typing import Dict, Any, Optional, List
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


class DatasetLoader:
    """Loader for HuggingFace datasets"""
    
    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset loader
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset/configuration
            split: Dataset split
            text_column: Name of text column
            max_samples: Maximum number of samples
            cache_dir: Cache directory
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.max_samples = max_samples
        self.cache_dir = cache_dir
    
    def load(self) -> Dataset:
        """
        Load dataset from HuggingFace
        
        Returns:
            Loaded dataset
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        # Load dataset
        if self.subset:
            dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                cache_dir=self.cache_dir
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir
            )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Limit samples
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
            print(f"Limited to {self.max_samples} samples")
        
        return dataset
    
    def extract_text(self, dataset: Dataset) -> List[str]:
        """
        Extract text from dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            List of text strings
        """
        # Handle different column names
        if self.text_column in dataset.column_names:
            texts = dataset[self.text_column]
        elif "text" in dataset.column_names:
            texts = dataset["text"]
        elif "content" in dataset.column_names:
            texts = dataset["content"]
        elif "article" in dataset.column_names:
            texts = dataset["article"]
        elif "document" in dataset.column_names:
            texts = dataset["document"]
        else:
            raise ValueError(f"Could not find text column in dataset: {dataset.column_names}")
        
        # Handle dialog format (list of strings)
        if isinstance(texts[0], list):
            texts = [" ".join(dialog) for dialog in texts]
        
        return texts
    
    def preprocess(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256
    ) -> Dataset:
        """
        Preprocess and tokenize dataset
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            stride: Stride for sliding window
            
        Returns:
            Tokenized dataset
        """
        # Extract texts
        texts = self.extract_text(dataset)
        
        # Tokenize with sliding window
        def tokenize_function(examples):
            # Tokenize
            tokenized = tokenizer(
                examples[self.text_column] if self.text_column in examples else examples["text"],
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                padding="max_length",
                return_attention_mask=True
            )
            
            # Remove overflow mapping
            tokenized.pop("overflow_to_sample_mapping", None)
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        print(f"Tokenized dataset: {len(tokenized_dataset)} samples")
        
        return tokenized_dataset


def load_dataset_from_config(
    dataset_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    preprocessing_config: Dict[str, Any]
) -> Dataset:
    """
    Load and preprocess dataset from configuration
    
    Args:
        dataset_config: Dataset configuration
        tokenizer: Tokenizer
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Preprocessed dataset
    """
    loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        subset=dataset_config.get("subset"),
        split=dataset_config.get("split", "train"),
        text_column=dataset_config.get("text_column", "text"),
        max_samples=dataset_config.get("max_samples")
    )
    
    # Load dataset
    dataset = loader.load()
    
    # Preprocess
    dataset = loader.preprocess(
        dataset,
        tokenizer,
        max_length=preprocessing_config.get("max_length", 512),
        stride=preprocessing_config.get("stride", 256)
    )
    
    return dataset


def load_multiple_datasets(
    datasets_config: Dict[str, Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    preprocessing_config: Dict[str, Any],
    dataset_names: Optional[List[str]] = None
) -> Dataset:
    """
    Load and combine multiple datasets
    
    Args:
        datasets_config: Configuration for all datasets
        tokenizer: Tokenizer
        preprocessing_config: Preprocessing configuration
        dataset_names: List of dataset names to load (None = all)
        
    Returns:
        Combined dataset
    """
    from datasets import concatenate_datasets
    
    if dataset_names is None:
        dataset_names = list(datasets_config.keys())
    
    all_datasets = []
    
    for name in dataset_names:
        if name not in datasets_config:
            print(f"Warning: Dataset {name} not found in config")
            continue
        
        print(f"\nLoading dataset: {name}")
        dataset = load_dataset_from_config(
            datasets_config[name],
            tokenizer,
            preprocessing_config
        )
        all_datasets.append(dataset)
    
    # Combine datasets
    if len(all_datasets) > 1:
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"\nCombined dataset: {len(combined_dataset)} total samples")
    else:
        combined_dataset = all_datasets[0]
    
    return combined_dataset


def create_train_val_split(
    dataset: Dataset,
    val_split: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Split dataset into train and validation
    
    Args:
        dataset: Input dataset
        val_split: Validation split ratio
        seed: Random seed
        
    Returns:
        DatasetDict with train and validation splits
    """
    split_dataset = dataset.train_test_split(
        test_size=val_split,
        seed=seed
    )
    
    return DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })