import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class T4Config:
    """T4-OPT configuration."""
    model_name: str = "microsoft/phi-2"
    max_seq_length: int = 1024
    
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    
    output_dir: str = "./checkpoints"
    save_steps: int = 500
    logging_steps: int = 10

    dataset_name: str = "alpaca"
    max_samples: int = 1000
    
    quant_type: str = "int8"  
    
    eval_max_samples: int = 100


class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path
        self.config = T4Config()
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path: str):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        self.config = T4Config(**config_dict)
        self.config_path = config_path
    
    def save(self, config_path: Optional[str] = None):
        path = config_path or self.config_path
        if not path:
            raise ValueError("config_path required")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any):
        if hasattr(self.config, key):
            setattr(self.config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.config)
    
    def update(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            self.set(key, value)

