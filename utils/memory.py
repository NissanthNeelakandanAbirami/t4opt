"""Memory management utilities for T4 GPU optimization."""

import torch
import gc
import psutil
from typing import Dict, Any, Optional


class MemoryManager:
    """Manages GPU and CPU memory for T4 optimization."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get comprehensive memory information."""
        info = {
            "gpu": MemoryManager.get_gpu_memory(),
            "cpu": MemoryManager.get_cpu_memory()
        }
        return info
    
    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "utilization_percent": (reserved / total) * 100 if total > 0 else 0
        }
    
    @staticmethod
    def get_cpu_memory() -> Dict[str, float]:
        """Get CPU memory statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "percent": memory.percent
        }
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    @staticmethod
    def print_memory_summary():
        """Print memory summary."""
        info = MemoryManager.get_memory_info()
        
        print("=" * 50)
        print("Memory Summary")
        print("=" * 50)
        
        if info["gpu"]["available"]:
            print(f"GPU Memory:")
            print(f"  Allocated: {info['gpu']['allocated_gb']:.2f} GB")
            print(f"  Reserved: {info['gpu']['reserved_gb']:.2f} GB")
            print(f"  Free: {info['gpu']['free_gb']:.2f} GB")
            print(f"  Utilization: {info['gpu']['utilization_percent']:.1f}%")
        else:
            print("GPU: Not available")
        
        print(f"\nCPU Memory:")
        print(f"  Used: {info['cpu']['used_gb']:.2f} GB")
        print(f"  Available: {info['cpu']['available_gb']:.2f} GB")
        print(f"  Utilization: {info['cpu']['percent']:.1f}%")
        print("=" * 50)
    
    @staticmethod
    def optimize_for_t4():
        """Apply T4-specific memory optimizations."""
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        MemoryManager.clear_cache()
        print("T4 memory optimizations applied")

