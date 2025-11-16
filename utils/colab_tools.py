"""Colab-specific utilities and helpers."""

import os
import sys
from typing import Optional


class ColabTools:
    """Utilities for Google Colab environment."""
    
    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab."""
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
            print("⚠️  GPU not available!")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        
        if "T4" in gpu_name:
            print("   ✅ T4 GPU detected - optimized for T4-OPT")
        else:
            print(f"   ⚠️  Non-T4 GPU - may need adjustments")
        
        return True
    
    @staticmethod
    def install_dependencies():
        """Install required dependencies in Colab."""
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
        
        print("✅ Dependencies installed")
    
    @staticmethod
    def setup_environment():
        """Setup Colab environment for T4-OPT."""
        print("Setting up T4-OPT environment...")
        
        # Check GPU
        if not ColabTools.check_gpu():
            print("⚠️  Continuing without GPU (will be slow)")
        
        # Install dependencies
        ColabTools.install_dependencies()
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("✅ Environment setup complete")
    
    @staticmethod
    def mount_drive():
        """Mount Google Drive in Colab."""
        if not ColabTools.is_colab():
            print("Not in Colab - skipping Drive mount")
            return None
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive mounted at /content/drive")
            return "/content/drive"
        except Exception as e:
            print(f"⚠️  Failed to mount Drive: {e}")
            return None
    
    @staticmethod
    def print_system_info():
        """Print system information."""
        import torch
        import psutil
        
        print("=" * 60)
        print("System Information")
        print("=" * 60)
        
        # Python version
        print(f"Python: {sys.version.split()[0]}")
        
        # PyTorch version
        print(f"PyTorch: {torch.__version__}")
        
        # GPU info
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("GPU: Not available")
        
        # CPU info
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        
        # Colab check
        if ColabTools.is_colab():
            print("Environment: Google Colab")
        else:
            print("Environment: Local")
        
        print("=" * 60)

