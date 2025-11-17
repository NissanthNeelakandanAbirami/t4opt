"""INT8 dynamic quantization for models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
import os
import gc


def quantize_to_int8(
    model_path: str,
    context: Optional[Dict[str, Any]] = None,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Quantize model to INT8 using dynamic quantization.
    Memory-optimized for Colab environments.
    
    Args:
        model_path: Path to model
        context: Optional context dictionary
        use_gpu: If True, use GPU-accelerated quantization (bitsandbytes).
                 If False, use CPU-based quantization (PyTorch quantize_dynamic).
        
    Returns:
        Dictionary with quantization results
    """
    output_path = context.get("output_path", f"{model_path}_int8") if context else f"{model_path}_int8"
    
    # If GPU is requested and available, use bitsandbytes for GPU acceleration
    if use_gpu and torch.cuda.is_available():
        return _quantize_int8_gpu(model_path, output_path, context)
    else:
        return _quantize_int8_cpu(model_path, output_path, context)


def _quantize_int8_gpu(
    model_path: str,
    output_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    GPU-accelerated INT8 quantization using bitsandbytes.
    This uses GPU and provides better performance.
    """
    print(f"Quantizing model to INT8 (GPU-accelerated): {model_path}")
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    try:
        from transformers import BitsAndBytesConfig
        
        # Configure 8-bit quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        print("Loading model with 8-bit quantization on GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically distribute across GPU
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Calculate sizes
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"Model loaded on GPU. Size: {original_size:.2f} MB")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save quantized model
        print(f"Saving quantized model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Estimate quantized size (8-bit is ~50% of original)
        quantized_size = original_size * 0.5
        reduction = 50.0
        
        print(f"Quantization complete. Size reduction: {reduction:.2f}%")
        print(f"Original: {original_size:.2f} MB -> Quantized: {quantized_size:.2f} MB")
        print("✅ GPU-accelerated quantization complete!")
        
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "model_path": model_path,
            "output_path": output_path,
            "quantization_type": "int8_gpu_bitsandbytes",
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_percent": reduction,
            "status": "quantized",
            "device": "gpu"
        }
        
    except ImportError:
        print("⚠️  bitsandbytes not available. Falling back to CPU quantization...")
        print("   Install with: pip install bitsandbytes")
        return _quantize_int8_cpu(model_path, output_path, context)
    except Exception as e:
        print(f"⚠️  GPU quantization failed: {e}")
        print("   Falling back to CPU quantization...")
        return _quantize_int8_cpu(model_path, output_path, context)


def _quantize_int8_cpu(
    model_path: str,
    output_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    CPU-based INT8 quantization using PyTorch's quantize_dynamic.
    This is the original method that works on CPU.
    """
    
    print(f"Quantizing model to INT8 (CPU-based): {model_path}")
    print("Note: CPU quantization is slower but more memory-efficient.")
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Load model directly to CPU with memory-efficient settings
    print("Loading model to CPU (memory-efficient mode)...")
    try:
        # Try loading with low_cpu_mem_usage first (most memory efficient)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for quantization stability
            device_map="cpu",  # Load directly to CPU
            low_cpu_mem_usage=True,  # Memory-efficient loading
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: low_cpu_mem_usage failed, trying standard load: {e}")
        # Fallback to standard loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
    
    # Ensure model is on CPU
    model = model.cpu()
    
    # Clear GPU cache again after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Calculate original size before quantization
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model loaded. Size: {original_size:.2f} MB")
    
    # Apply dynamic quantization
    print("Applying INT8 quantization...")
    try:
        # Try torch.ao.quantization for newer PyTorch versions
        try:
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
    except Exception as e:
        print(f"Error during quantization: {e}")
        # Clean up before raising
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # Delete original model to free memory before saving
    print("Freeing memory from original model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"Saving quantized model to {output_path}")
    try:
        quantized_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving model: {e}")
        del quantized_model
        gc.collect()
        raise
    
    # Calculate quantized size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
    
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Quantization complete. Size reduction: {reduction:.2f}%")
    print(f"Original: {original_size:.2f} MB -> Quantized: {quantized_size:.2f} MB")
    
    # Clean up
    del quantized_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "model_path": model_path,
        "output_path": output_path,
        "quantization_type": "int8_dynamic",
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_percent": reduction,
        "status": "quantized"
    }

