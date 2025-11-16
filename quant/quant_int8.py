"""INT8 dynamic quantization for models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
import os


def quantize_to_int8(
    model_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Quantize model to INT8 using dynamic quantization.
    
    Args:
        model_path: Path to model
        context: Optional context dictionary
        
    Returns:
        Dictionary with quantization results
    """
    output_path = context.get("output_path", f"{model_path}_int8") if context else f"{model_path}_int8"
    
    print(f"Quantizing model to INT8: {model_path}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Convert to CPU for quantization
    model = model.cpu()
    
    # Apply dynamic quantization
    print("Applying INT8 quantization...")
    try:
        # Try torch.ao.quantization for newer PyTorch versions
        from torch.ao.quantization import quantize_dynamic
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
    except ImportError:
        # Fallback to torch.quantization for older versions
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to {output_path}")
    quantized_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
    
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Quantization complete. Size reduction: {reduction:.2f}%")
    
    return {
        "model_path": model_path,
        "output_path": output_path,
        "quantization_type": "int8_dynamic",
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_percent": reduction,
        "status": "quantized"
    }

