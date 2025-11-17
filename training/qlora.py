"""QLoRA trainer for efficient 4-bit fine-tuning on T4 GPUs."""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any, Optional, List
import os
from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""
    model_name: str = "microsoft/phi-2"
    output_dir: str = "./checkpoints"
    max_seq_length: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3


class QLoRATrainer:
    """QLoRA trainer optimized for T4 GPUs (16GB VRAM)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QLoRA trainer.
        
        Args:
            config: Configuration dictionary or QLoRAConfig instance
        """
        if isinstance(config, dict):
            self.config = QLoRAConfig(**config)
        elif isinstance(config, QLoRAConfig):
            self.config = config
        else:
            self.config = QLoRAConfig()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set default LoRA target modules based on model
        if self.config.lora_target_modules is None:
            if "phi" in self.config.model_name.lower():
                self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
            elif "llama" in self.config.model_name.lower() or "gemma" in self.config.model_name.lower():
                self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj"]
    
    def load_model(self):
        """Load model with 4-bit quantization."""
        print(f"Loading model: {self.config.model_name}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer
    
    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.load_model()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training arguments optimized for T4
        # Check which parameter name is supported (eval_strategy in newer versions, evaluation_strategy in older)
        import inspect
        sig = inspect.signature(TrainingArguments.__init__)
        eval_param_name = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
        
        # Build args dict with conditional parameters
        training_args_dict = {
            "output_dir": self.config.output_dir,
            "per_device_train_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_train_epochs": self.config.num_epochs,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "save_total_limit": self.config.save_total_limit,
            "report_to": "none",  # Disable wandb/tensorboard for Colab
            "optim": "paged_adamw_8bit",  # Memory-efficient optimizer
            "max_grad_norm": 0.3,
            "remove_unused_columns": False,
            # GPU optimization settings
            # Note: In Colab, reduce workers if you get errors (Colab has limited CPU cores)
            "dataloader_num_workers": 2,  # Parallel data loading (2 for Colab, 4 for local)
            "dataloader_pin_memory": True,  # Faster GPU transfer
            "dataloader_prefetch_factor": 2,  # Prefetch batches
            "ddp_find_unused_parameters": False,  # Faster DDP
            "group_by_length": True,  # Group similar length sequences for efficiency
        }
        
        # Add evaluation parameters if eval_dataset is provided
        if eval_dataset:
            training_args_dict["eval_steps"] = self.config.eval_steps
            training_args_dict[eval_param_name] = "steps"
            training_args_dict["load_best_model_at_end"] = True
        else:
            training_args_dict[eval_param_name] = "no"
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "output_dir": self.config.output_dir
        }
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError("Model not trained yet")
        
        save_path = path or self.config.output_dir
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

