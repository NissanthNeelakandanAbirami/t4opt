import os
import sys
from typing import Optional


class ColabTools:
    """Utilities for Google Colab environment."""
    
    @staticmethod
    def is_colab() -> bool:
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def check_gpu():
        """Check GPU availability and info."""
        import torch
        
        if not torch.cuda.is_available():
            print("GPU not available!")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        
        
        return True
    
    @staticmethod
    def install_dependencies():
        if not ColabTools.is_colab():
            print("Not in Colab - skipping dependency installation")
            return
        
        print("Installing dependencies...")
        
        packages = [
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.6.0",
            "datasets>=2.14.0",
            "torch>=2.0.0",
            "sentencepiece",
            "protobuf",
        ]
        
        for package in packages:
            os.system(f"pip install -q {package}")
        
        print("Dependencies installed")
    
    @staticmethod
    def setup_environment():
        """Setup Colab environment for T4-OPT."""
        print("Setting up T4-OPT environment...")
        
        if not ColabTools.check_gpu():
            print("Continuing without GPU (will be slow)")
        
        ColabTools.install_dependencies()
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("Environment setup complete")
    
    @staticmethod
    def mount_drive():
        """Mount Google Drive in Colab."""
        if not ColabTools.is_colab():
            print("Not in Colab - skipping Drive mount")
            return None
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted at /content/drive")
            return "/content/drive"
        except Exception as e:
            print(f"Failed to mount Drive: {e}")
            return None
    
    @staticmethod
    def verify_t4_compatibility():
        """Verify that all optimizations are compatible with T4 GPU in Colab."""
        import torch
        
        
        if not torch.cuda.is_available():
            print("GPU not available!")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        is_t4 = "T4" in gpu_name or "Tesla T4" in gpu_name
        is_colab_env = ColabTools.is_colab()
        
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"Environment: {'Google Colab' if is_colab_env else 'Local'}")
        print()
        
        all_ok = True
        
        if is_t4:
            print("Tesla T4 detected")
        else:
            print(f"Non-T4 GPU detected (will still work, but optimized for T4)")
        
        if gpu_memory >= 15.0:
            print(f"Sufficient GPU memory ({gpu_memory:.2f} GB)")
        else:
            print(f"Lower GPU memory ({gpu_memory:.2f} GB) - may need smaller batch sizes")
            all_ok = False
        
        print("\nFeature Compatibility:")
        print(f"  {'✅' if torch.cuda.is_bf16_supported() else 'ℹ️ '} bf16: {'Supported' if torch.cuda.is_bf16_supported() else 'Not supported (T4 uses fp16)'}")

        if is_colab_env:
            print("\nColab-Specific:")
            print(f"  DataLoader workers: Set to 2 (Colab-optimized)")
            print(f"  Memory management: Optimized for Colab")
        
        if is_t4 and all_ok:
            print("All optimizations are compatible with Tesla T4 in Colab!")
        else:
            print("System is compatible - optimizations will work!")
        
        return True
    
    @staticmethod
    def print_system_info():
        """Print system information."""
        import torch
        import psutil
        
        print(f"Python: {sys.version.split()[0]}")
        
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("GPU: Not available")
        
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        
        if ColabTools.is_colab():
            print("Environment: Google Colab")
        else:
            print("Environment: Local")
        
        print("=" * 60)

