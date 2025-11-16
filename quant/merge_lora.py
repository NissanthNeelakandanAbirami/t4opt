"""Merge LoRA adapters into base model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Any, Optional
import os


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    save_tokenizer: bool = True
) -> Dict[str, Any]:
    """
    Merge LoRA adapters into base model.
    
    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapters
        output_path: Path to save merged model
        save_tokenizer: Whether to save tokenizer
        
    Returns:
        Dictionary with merge results
    """
    print(f"Merging LoRA weights from {lora_path} into {base_model_path}")
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge and unload
    print("Merging adapters...")
    model = model.merge_and_unload()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    if save_tokenizer:
        tokenizer.save_pretrained(output_path)
    
    # Get model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    
    print(f"Merged model saved. Size: {model_size:.2f} MB")
    
    return {
        "base_model_path": base_model_path,
        "lora_path": lora_path,
        "output_path": output_path,
        "model_size_mb": model_size,
        "status": "merged"
    }

