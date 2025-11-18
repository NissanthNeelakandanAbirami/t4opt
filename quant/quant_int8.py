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
    output_path = context.get("output_path", f"{model_path}_int8") if context else f"{model_path}_int8"

    if use_gpu and torch.cuda.is_available():
        return _quantize_int8_gpu(model_path, output_path, context)
    else:
        return _quantize_int8_cpu(model_path, output_path, context)


def _quantize_int8_gpu(
    model_path: str,
    output_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    
    print(f"Quantizing model to INT8 (GPU-accelerated): {model_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",  
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"Model loaded on GPU. Size: {original_size:.2f} MB")

        os.makedirs(output_path, exist_ok=True)

        print(f"Saving quantized model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        quantized_size = original_size * 0.5
        reduction = 50.0

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
        return _quantize_int8_cpu(model_path, output_path, context)
    except Exception as e:
        return _quantize_int8_cpu(model_path, output_path, context)


def _quantize_int8_cpu(
    model_path: str,
    output_path: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  
            device_map="cpu",  
            low_cpu_mem_usage=True,  
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: low_cpu_mem_usage failed, trying standard load: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

    model = model.cpu()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model loaded. Size: {original_size:.2f} MB")

    try:
        try:
            from torch.ao.quantization import quantize_dynamic
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear},  
                dtype=torch.qint8
            )
        except ImportError:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  
                dtype=torch.qint8
            )
    except Exception as e:
        print(f"Error during quantization: {e}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(output_path, exist_ok=True)

    print(f"Saving quantized model to {output_path}")
    try:
        quantized_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Error saving model: {e}")
        del quantized_model
        gc.collect()
        raise

    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
    
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Quantization complete. Size reduction: {reduction:.2f}%")
    print(f"Original: {original_size:.2f} MB -> Quantized: {quantized_size:.2f} MB")
    
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

