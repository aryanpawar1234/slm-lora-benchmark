"""Data collator for language modeling"""

import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class DataCollatorForLanguageModeling:
    """Data collator for causal language modeling"""

    tokenizer: PreTrainedTokenizer
    mlm: bool = False

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples"""
        batch = {
            "input_ids": [],
            "attention_mask": []
        }

        for example in examples:
            batch["input_ids"].append(example["input_ids"])
            if "attention_mask" in example:
                batch["attention_mask"].append(example["attention_mask"])

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)

        if batch["attention_mask"]:
            batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        else:
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        return batch
