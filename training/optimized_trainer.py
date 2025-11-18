import torch
import sys
import os
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.qlora import QLoRATrainer, QLoRAConfig
from utils.memory import MemoryManager


class OptimizedQLoRATrainer(QLoRATrainer):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_optimize: bool = True):
        
        super().__init__(config)
        self.auto_optimize = auto_optimize
        self.optimized_config = None
        
    def optimize_for_gpu(self) -> QLoRAConfig:
        gpu_info = MemoryManager.get_gpu_memory()
        total_gpu_memory = gpu_info.get("total_gb", 16.0)  

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        is_t4 = "T4" in gpu_name or "Tesla T4" in gpu_name
        
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {total_gpu_memory:.2f} GB")
        
        optimized = QLoRAConfig(**self.config.__dict__)
        
        use_bf16 = torch.cuda.is_bf16_supported() and total_gpu_memory >= 15.0
        
        if total_gpu_memory >= 15.0:  
            optimized.micro_batch_size = 4  
            optimized.gradient_accumulation_steps = 4  
            optimized.max_seq_length = 1024
            optimized.bf16 = use_bf16  
            optimized.fp16 = not use_bf16  
            print(f"Optimized for T4 (16GB): batch_size=4, grad_accum=4, precision={'bf16' if use_bf16 else 'fp16'}")
        elif total_gpu_memory >= 10.0:
            optimized.micro_batch_size = 2
            optimized.gradient_accumulation_steps = 8
            optimized.max_seq_length = 1024
            optimized.fp16 = True
            print("Optimized for 10GB GPU: batch_size=2, grad_accum=8")
        else:
            optimized.micro_batch_size = 1
            optimized.gradient_accumulation_steps = 16
            optimized.max_seq_length = 512 
            optimized.fp16 = True
            print("Optimized for smaller GPU: batch_size=1, grad_accum=16, seq_len=512")

        optimized.use_gradient_checkpointing = True
 
        optimized.lora_r = 16 
        optimized.lora_alpha = 32
        optimized.lora_dropout = 0.05

        optimized.save_steps = 250  
        optimized.save_total_limit = 5  
        
        self.optimized_config = optimized
        return optimized
    
    def find_optimal_batch_size(
        self,
        tokenizer,
        max_seq_length: Optional[int] = None,
        start_batch_size: int = 4,
        max_batch_size: int = 16
    ) -> int:
        from training.trainer_utils import TrainingUtils
        
        if self.model is None:
            self.load_model()
        
        seq_length = max_seq_length or self.config.max_seq_length
        
        
        optimal = TrainingUtils.find_optimal_batch_size(
            self.model,
            tokenizer,
            max_seq_length=seq_length,
            start_batch_size=start_batch_size,
            max_batch_size=max_batch_size
        )
        
        print(f"Optimal batch size: {optimal}")
        return optimal
    
    def train_optimized(
        self,
        train_dataset,
        eval_dataset=None,
        find_best_batch_size: bool = True
    ):
        
        if self.auto_optimize:
            optimized_config = self.optimize_for_gpu()
            self.config = optimized_config

        if self.model is None:
            self.load_model()

        if find_best_batch_size and torch.cuda.is_available():
            try:
                optimal_batch = self.find_optimal_batch_size(
                    self.tokenizer,
                    max_seq_length=self.config.max_seq_length
                )
                self.config.micro_batch_size = optimal_batch
                effective_batch = optimal_batch * self.config.gradient_accumulation_steps
                if effective_batch > 32:
                    self.config.gradient_accumulation_steps = max(8, 32 // optimal_batch)
                print(f" Updated: batch_size={optimal_batch}, grad_accum={self.config.gradient_accumulation_steps}")
            except Exception as e:
                print(f"Could not find optimal batch size: {e}")

        self._apply_memory_optimizations()
        
        for key, value in self.config.__dict__.items():
            print(f"  {key}: {value}")

        return self.train(train_dataset, eval_dataset)
    
    def _apply_memory_optimizations(self):
        if not torch.cuda.is_available():
            return
        
        try:
            torch.cuda.set_per_process_memory_fraction(0.98)  
            print("GPU memory fraction set to 98%")
        except:
            pass

        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("Flash attention enabled (faster training)")
        except:
            pass
        
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("Memory efficient attention enabled")
        except:
            pass
        
        try:
            torch.backends.cuda.enable_math_sdp(True)
            print("Math attention enabled")
        except:
            pass

        try:
            torch.backends.cudnn.benchmark = True  
            torch.backends.cudnn.deterministic = False  
            print("CuDNN optimizations enabled")
        except:
            pass
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name or "A40" in gpu_name or "RTX 30" in gpu_name or "RTX 40" in gpu_name:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TF32 enabled (faster on Ampere+ GPUs)")
            else:
                print("ℹTF32 not available on this GPU (T4/Turing) - using standard precision")
        except:
            pass

        MemoryManager.clear_cache()

        MemoryManager.print_memory_summary()


