import torch
import psutil
import os
from typing import Dict, Any, Optional


class TrainingUtils:
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"available": False}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "available": True,
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
            "total_gb": memory_total,
            "free_gb": memory_total - memory_reserved
        }
    
    @staticmethod
    def get_cpu_memory_info() -> Dict[str, float]:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "percent": memory.percent
        }
    
    @staticmethod
    def find_optimal_batch_size(
        model,
        tokenizer,
        max_seq_length: int = 1024,
        start_batch_size: int = 4,
        max_batch_size: int = 32
    ) -> int:
        
        optimal_batch_size = start_batch_size
        model.eval()  
        
        test_sizes = []
        if start_batch_size == 1:
            test_sizes = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
        else:
            test_sizes = list(range(start_batch_size, min(max_batch_size + 1, 33), 2))
            if max_batch_size > 32:
                test_sizes.extend([36, 40, 48, 56, 64])
        
        test_sizes = [s for s in test_sizes if s <= max_batch_size]
        
        for batch_size in test_sizes:
            try:
                dummy_input = torch.randint(0, min(tokenizer.vocab_size, 32000), (batch_size, max_seq_length))
                dummy_input = dummy_input.to(next(model.parameters()).device)
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                optimal_batch_size = batch_size
                torch.cuda.empty_cache()
                print(f" Batch size {batch_size} works")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f" Batch size {batch_size} failed (OOM)")
                    break
                else:
                    raise e
        
        print(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    @staticmethod
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
    
    @staticmethod
    def estimate_training_time(
        num_samples: int,
        batch_size: int,
        gradient_accumulation: int,
        num_epochs: int,
        seconds_per_step: float = 2.0
    ) -> Dict[str, float]:
        
        effective_batch_size = batch_size * gradient_accumulation
        steps_per_epoch = num_samples / effective_batch_size
        total_steps = steps_per_epoch * num_epochs
        
        total_seconds = total_steps * seconds_per_step
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        
        return {
            "total_steps": int(total_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "total_seconds": total_seconds,
            "total_minutes": total_minutes,
            "total_hours": total_hours
        }
    
    @staticmethod
    def save_training_config(config: Dict[str, Any], path: str):
        import json
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_training_config(path: str) -> Dict[str, Any]:
        import json
        with open(path, "r") as f:
            return json.load(f)

