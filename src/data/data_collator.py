"""Data collator for language modeling"""

import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling
    Prepares batches for next-token prediction
    """
    
    tokenizer: PreTrainedTokenizer
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples
        
        Args:
            examples: List of tokenized examples
            
        Returns:
            Batch dictionary with input_ids, attention_mask, labels
        """
        # Extract input_ids and attention_mask
        batch = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for example in examples:
            batch["input_ids"].append(example["input_ids"])
            if "attention_mask" in example:
                batch["attention_mask"].append(example["attention_mask"])
        
        # Convert to tensors
        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        
        if batch["attention_mask"]:
            batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        else:
            # Create attention mask if not present
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        
        # Create labels for causal LM (shift right by 1)
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask padding tokens in labels
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch


class SimpleDataCollator:
    """Simple data collator without special processing"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples into batch"""
        batch = {}
        
        # Get all keys from first example
        keys = examples[0].keys()
        
        for key in keys:
            values = [ex[key] for ex in examples]
            
            # Convert to tensor
            if isinstance(values[0], list):
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], int):
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], float):
                batch[key] = torch.tensor(values, dtype=torch.float)
            else:
                batch[key] = values
        
        # Create labels if not present
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Create attention mask from input_ids
    
    Args:
        input_ids: Input token IDs
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask tensor
    """
    return (input_ids != pad_token_id).long()


def mask_tokens(
    inputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    mlm_probability: float = 0.15
) -> tuple:
    """
    Mask tokens for masked language modeling
    
    Args:
        inputs: Input token IDs
        tokenizer: Tokenizer
        mlm_probability: Probability of masking
        
    Returns:
        Masked inputs and labels
    """
    labels = inputs.clone()
    
    # Create probability matrix
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    # Don't mask special tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Create mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id
    
    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # 10% of the time, keep original
    
    return inputs, labels