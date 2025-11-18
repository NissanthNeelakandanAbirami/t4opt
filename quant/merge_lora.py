import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Any, Optional
import os
import gc


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    save_tokenizer: bool = True
) -> Dict[str, Any]:
    
    print(f"Merging LoRA weights from {lora_path} into {base_model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging adapters...")
    model = model.merge_and_unload()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(output_path, exist_ok=True)

    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    if save_tokenizer:
        tokenizer.save_pretrained(output_path)

    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
    
    print(f"Merged model saved. Size: {model_size:.2f} MB")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "base_model_path": base_model_path,
        "lora_path": lora_path,
        "output_path": output_path,
        "model_size_mb": model_size,
        "status": "merged"
    }

