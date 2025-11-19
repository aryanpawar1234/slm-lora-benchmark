"""LoRA adapter configuration and application"""

from typing import Dict, Any, List, Optional
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import PreTrainedModel


def create_peft_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    inference_mode: bool = False,
    modules_to_save: Optional[List[str]] = None
) -> LoraConfig:
    """
    Create PEFT LoRA configuration
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability
        target_modules: Modules to apply LoRA to
        bias: Bias type ("none", "all", "lora_only")
        task_type: Task type
        inference_mode: Inference mode
        modules_to_save: Additional modules to save
        
    Returns:
        LoraConfig instance
    """
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ["query_key_value"]  # For Pythia
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=getattr(TaskType, task_type),
        inference_mode=inference_mode,
        modules_to_save=modules_to_save
    )
    
    print("LoRA Configuration:")
    print(f"  Rank (r): {r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Target modules: {target_modules}")
    
    return config


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    prepare_for_kbit: bool = False
) -> PeftModel:
    """
    Apply LoRA adapters to model
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        prepare_for_kbit: Prepare model for k-bit training
        
    Returns:
        PEFT model with LoRA adapters
    """
    print("Applying LoRA adapters...")
    
    # Prepare model for k-bit training if needed
    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training")
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model


def create_lora_model(
    model: PreTrainedModel,
    lora_config_dict: Dict[str, Any],
    prepare_for_kbit: bool = False
) -> PeftModel:
    """
    Create LoRA model from configuration dictionary
    
    Args:
        model: Base model
        lora_config_dict: LoRA configuration dictionary
        prepare_for_kbit: Prepare for k-bit training
        
    Returns:
        PEFT model with LoRA
    """
    # Create LoRA config
    lora_config = create_peft_config(
        r=lora_config_dict.get("r", 16),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
        target_modules=lora_config_dict.get("target_modules"),
        bias=lora_config_dict.get("bias", "none"),
        task_type=lora_config_dict.get("task_type", "CAUSAL_LM"),
        inference_mode=lora_config_dict.get("inference_mode", False),
        modules_to_save=lora_config_dict.get("modules_to_save")
    )
    
    # Apply LoRA
    peft_model = apply_lora(model, lora_config, prepare_for_kbit)
    
    return peft_model


def load_lora_model(
    base_model: PreTrainedModel,
    lora_checkpoint_path: str
) -> PeftModel:
    """
    Load LoRA adapters from checkpoint
    
    Args:
        base_model: Base model
        lora_checkpoint_path: Path to LoRA checkpoint
        
    Returns:
        PEFT model with loaded LoRA adapters
    """
    from peft import PeftModel
    
    print(f"Loading LoRA adapters from: {lora_checkpoint_path}")
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_path
    )
    
    print("LoRA adapters loaded successfully")
    
    return peft_model


def merge_lora_weights(peft_model: PeftModel) -> PreTrainedModel:
    """
    Merge LoRA weights into base model
    
    Args:
        peft_model: PEFT model with LoRA
        
    Returns:
        Merged model
    """
    print("Merging LoRA weights into base model...")
    
    merged_model = peft_model.merge_and_unload()
    
    print("LoRA weights merged successfully")
    
    return merged_model


def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get appropriate target modules for model architecture
    
    Args:
        model_name: Model name or architecture
        
    Returns:
        List of target module names
    """
    model_name_lower = model_name.lower()
    
    # GPT-2
    if "gpt2" in model_name_lower:
        return ["c_attn", "c_proj"]
    
    # Pythia
    elif "pythia" in model_name_lower:
        return ["query_key_value"]
    
    # LLaMA / Mistral
    elif any(x in model_name_lower for x in ["llama", "mistral", "mixtral"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Gemma
    elif "gemma" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # OPT
    elif "opt" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    # BLOOM
    elif "bloom" in model_name_lower:
        return ["query_key_value"]
    
    # Default
    else:
        print(f"Warning: Unknown model architecture {model_name}, using default target modules")
        return ["q_proj", "v_proj"]