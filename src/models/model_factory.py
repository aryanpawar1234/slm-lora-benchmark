"""Model factory for loading pre-trained models"""

import torch
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class ModelFactory:
    """Factory for loading and configuring models"""
    
    @staticmethod
    def load_model(
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "float32",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = False,
        gradient_checkpointing: bool = True
    ) -> PreTrainedModel:
        """
        Load pre-trained causal language model
        
        Args:
            model_name: HuggingFace model name
            device_map: Device mapping strategy
            torch_dtype: Model dtype
            load_in_8bit: Load in 8-bit precision
            load_in_4bit: Load in 4-bit precision
            trust_remote_code: Trust remote code
            use_cache: Enable KV cache
            gradient_checkpointing: Enable gradient checkpointing
            
        Returns:
            Loaded model
        """
        print(f"Loading model: {model_name}")
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype, torch.float32)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        print(f"Model loaded: {model.__class__.__name__}")
        
        return model
    
    @staticmethod
    def load_tokenizer(
        model_name: str,
        use_fast: bool = True,
        padding_side: str = "right",
        truncation_side: str = "right",
        model_max_length: int = 512
    ) -> PreTrainedTokenizer:
        """
        Load tokenizer
        
        Args:
            model_name: HuggingFace model name
            use_fast: Use fast tokenizer
            padding_side: Padding side (left/right)
            truncation_side: Truncation side (left/right)
            model_max_length: Maximum sequence length
            
        Returns:
            Loaded tokenizer
        """
        print(f"Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_max_length=model_max_length
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
        
        return tokenizer
    
    @classmethod
    def from_config(
        cls,
        model_config: Dict[str, Any],
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer from configuration
        
        Args:
            model_config: Model configuration
            tokenizer_config: Tokenizer configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        if tokenizer_config is None:
            tokenizer_config = {}
        
        tokenizer = cls.load_tokenizer(
            model_name=tokenizer_config.get("name", model_config["name"]),
            use_fast=tokenizer_config.get("use_fast", True),
            padding_side=tokenizer_config.get("padding_side", "right"),
            truncation_side=tokenizer_config.get("truncation_side", "right"),
            model_max_length=tokenizer_config.get("model_max_length", 512)
        )
        
        # Load model
        model = cls.load_model(
            model_name=model_config["name"],
            device_map=model_config.get("device_map", "auto"),
            torch_dtype=model_config.get("torch_dtype", "float32"),
            load_in_8bit=model_config.get("load_in_8bit", False),
            load_in_4bit=model_config.get("load_in_4bit", False),
            trust_remote_code=model_config.get("trust_remote_code", False),
            use_cache=model_config.get("use_cache", False),
            gradient_checkpointing=model_config.get("gradient_checkpointing", True)
        )
        
        # Resize token embeddings if needed
        if len(tokenizer) != model.config.vocab_size:
            print(f"Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer


def load_model_and_tokenizer(
    model_name: str,
    max_length: int = 512,
    device: str = "auto",
    load_in_8bit: bool = False,
    gradient_checkpointing: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function to load model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        device: Device to load model on
        load_in_8bit: Load in 8-bit precision
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Tuple of (model, tokenizer)
    """
    factory = ModelFactory()
    
    tokenizer = factory.load_tokenizer(
        model_name=model_name,
        model_max_length=max_length
    )
    
    model = factory.load_model(
        model_name=model_name,
        device_map=device,
        load_in_8bit=load_in_8bit,
        gradient_checkpointing=gradient_checkpointing
    )
    
    return model, tokenizer