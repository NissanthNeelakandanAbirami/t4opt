"""Optimized QLoRA trainer with automatic GPU/memory optimization."""

import torch
import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.qlora import QLoRATrainer, QLoRAConfig
from utils.memory import MemoryManager


class OptimizedQLoRATrainer(QLoRATrainer):
    """QLoRA trainer with automatic GPU and memory optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_optimize: bool = True):
        """
        Initialize optimized trainer.
        
        Args:
            config: Configuration dictionary or QLoRAConfig instance
            auto_optimize: Automatically optimize settings for maximum GPU utilization
        """
        super().__init__(config)
        self.auto_optimize = auto_optimize
        self.optimized_config = None
        
    def optimize_for_gpu(self) -> QLoRAConfig:
        """
        Automatically optimize configuration for maximum GPU utilization.
        
        Returns:
            Optimized QLoRAConfig
        """
        if not torch.cuda.is_available():
            print("⚠️  GPU not available, using CPU settings")
            return self.config
        
        print("🔧 Optimizing configuration for maximum GPU utilization...")
        
        # Get GPU memory info
        gpu_info = MemoryManager.get_gpu_memory()
        total_gpu_memory = gpu_info.get("total_gb", 16.0)  # Default to 16GB for T4
        
        # Detect GPU type
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        is_t4 = "T4" in gpu_name or "Tesla T4" in gpu_name
        
        print(f"📊 GPU: {gpu_name}")
        print(f"📊 GPU Memory: {total_gpu_memory:.2f} GB")
        if is_t4:
            print("✅ Tesla T4 detected - all optimizations compatible!")
        
        # Start with base config
        optimized = QLoRAConfig(**self.config.__dict__)
        
        # Check if bf16 is available (faster than fp16 on newer GPUs)
        use_bf16 = torch.cuda.is_bf16_supported() and total_gpu_memory >= 15.0
        
        # Optimize batch size based on GPU memory - more aggressive for better GPU utilization
        if total_gpu_memory >= 15.0:  # T4 or better
            # Try larger batch sizes for maximum GPU utilization
            optimized.micro_batch_size = 4  # Start with 4, will be optimized further
            optimized.gradient_accumulation_steps = 4  # Reduce since batch size increased
            optimized.max_seq_length = 1024
            optimized.bf16 = use_bf16  # Use bf16 if available (faster)
            optimized.fp16 = not use_bf16  # Fallback to fp16
            print(f"✅ Optimized for T4 (16GB): batch_size=4, grad_accum=4, precision={'bf16' if use_bf16 else 'fp16'}")
        elif total_gpu_memory >= 10.0:
            optimized.micro_batch_size = 2
            optimized.gradient_accumulation_steps = 8
            optimized.max_seq_length = 1024
            optimized.fp16 = True
            print("✅ Optimized for 10GB GPU: batch_size=2, grad_accum=8")
        else:
            optimized.micro_batch_size = 1
            optimized.gradient_accumulation_steps = 16
            optimized.max_seq_length = 512  # Reduce sequence length
            optimized.fp16 = True
            print("✅ Optimized for smaller GPU: batch_size=1, grad_accum=16, seq_len=512")
        
        # Enable all memory optimizations
        optimized.use_gradient_checkpointing = True
        # fp16/bf16 already set above based on GPU support
        
        # Optimize LoRA settings for better memory usage
        optimized.lora_r = 16  # Keep reasonable
        optimized.lora_alpha = 32
        optimized.lora_dropout = 0.05
        
        # More frequent saves for safety
        optimized.save_steps = 250  # Save more frequently
        optimized.save_total_limit = 5  # Keep more checkpoints
        
        self.optimized_config = optimized
        return optimized
    
    def find_optimal_batch_size(
        self,
        tokenizer,
        max_seq_length: Optional[int] = None,
        start_batch_size: int = 4,
        max_batch_size: int = 16
    ) -> int:
        """
        Find optimal batch size by testing different sizes.
        
        Args:
            tokenizer: Tokenizer instance
            max_seq_length: Maximum sequence length
            start_batch_size: Starting batch size
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        from training.trainer_utils import TrainingUtils
        
        if self.model is None:
            self.load_model()
        
        seq_length = max_seq_length or self.config.max_seq_length
        
        print(f"🔍 Finding optimal batch size (testing {start_batch_size}-{max_batch_size})...")
        
        optimal = TrainingUtils.find_optimal_batch_size(
            self.model,
            tokenizer,
            max_seq_length=seq_length,
            start_batch_size=start_batch_size,
            max_batch_size=max_batch_size
        )
        
        print(f"✅ Optimal batch size: {optimal}")
        return optimal
    
    def train_optimized(
        self,
        train_dataset,
        eval_dataset=None,
        find_best_batch_size: bool = True
    ):
        """
        Train with automatic optimizations.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            find_best_batch_size: Whether to automatically find optimal batch size
            
        Returns:
            Training results dictionary
        """
        # Optimize configuration
        if self.auto_optimize:
            optimized_config = self.optimize_for_gpu()
            self.config = optimized_config
        
        # Load model
        if self.model is None:
            self.load_model()
        
        # Find optimal batch size if requested
        if find_best_batch_size and torch.cuda.is_available():
            try:
                optimal_batch = self.find_optimal_batch_size(
                    self.tokenizer,
                    max_seq_length=self.config.max_seq_length
                )
                # Update config with optimal batch size
                self.config.micro_batch_size = optimal_batch
                # Adjust gradient accumulation to maintain effective batch size
                effective_batch = optimal_batch * self.config.gradient_accumulation_steps
                if effective_batch > 32:
                    self.config.gradient_accumulation_steps = max(8, 32 // optimal_batch)
                print(f"📊 Updated: batch_size={optimal_batch}, grad_accum={self.config.gradient_accumulation_steps}")
            except Exception as e:
                print(f"⚠️  Could not find optimal batch size: {e}")
                print("   Using configured batch size")
        
        # Apply memory optimizations
        self._apply_memory_optimizations()
        
        # Print final configuration
        print("\n" + "="*60)
        print("Final Training Configuration:")
        print("="*60)
        for key, value in self.config.__dict__.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
        
        # Train
        return self.train(train_dataset, eval_dataset)
    
    def _apply_memory_optimizations(self):
        """Apply additional memory and GPU utilization optimizations."""
        if not torch.cuda.is_available():
            return
        
        print("🔧 Applying GPU and memory optimizations for maximum utilization...")
        
        # Set memory fraction - use more GPU memory
        try:
            torch.cuda.set_per_process_memory_fraction(0.98)  # Use 98% of GPU
            print("✅ GPU memory fraction set to 98%")
        except:
            pass
        
        # Enable flash attention if available (much faster)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash attention enabled (faster training)")
        except:
            pass
        
        # Enable memory efficient attention
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✅ Memory efficient attention enabled")
        except:
            pass
        
        # Enable math attention (fastest when available)
        try:
            torch.backends.cuda.enable_math_sdp(True)
            print("✅ Math attention enabled")
        except:
            pass
        
        # Set CUDA device properties for better performance
        try:
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            print("✅ CuDNN optimizations enabled")
        except:
            pass
        
        # Enable TensorFloat-32 (TF32) for faster training on Ampere+ GPUs
        # Note: T4 doesn't support TF32, but this is harmless and will be ignored
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name or "A40" in gpu_name or "RTX 30" in gpu_name or "RTX 40" in gpu_name:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✅ TF32 enabled (faster on Ampere+ GPUs)")
            else:
                print("ℹ️  TF32 not available on this GPU (T4/Turing) - using standard precision")
        except:
            pass
        
        # Clear cache
        MemoryManager.clear_cache()
        
        # Print memory status
        MemoryManager.print_memory_summary()
        
        # Print GPU utilization tips
        print("\n💡 GPU Utilization Tips:")
        print("   - Larger batch sizes = better GPU utilization")
        print("   - Use bf16 if available (faster than fp16)")
        print("   - DataLoader workers speed up data loading")
        print("   - Flash attention significantly speeds up training")

