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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if isinstance(config, dict):
            self.config = QLoRAConfig(**config)
        elif isinstance(config, QLoRAConfig):
            self.config = config
        else:
            self.config = QLoRAConfig()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
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
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.model, self.tokenizer
    
    def train(self, train_dataset, eval_dataset=None):
        
        if self.model is None:
            self.load_model()
        
        os.makedirs(self.config.output_dir, exist_ok=True)

        import inspect
        sig = inspect.signature(TrainingArguments.__init__)
        eval_param_name = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"

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
            "report_to": "none",  
            "optim": "paged_adamw_8bit",  
            "max_grad_norm": 0.3,
            "remove_unused_columns": False,
            "dataloader_pin_memory": True,  
            "dataloader_prefetch_factor": 2,  
            "ddp_find_unused_parameters": False,  
            "group_by_length": True,  
        }
        
        if eval_dataset:
            training_args_dict["eval_steps"] = self.config.eval_steps
            training_args_dict[eval_param_name] = "steps"
            training_args_dict["load_best_model_at_end"] = True
        else:
            training_args_dict[eval_param_name] = "no"
        
        training_args = TrainingArguments(**training_args_dict)
        

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print("Starting training...")
        train_result = self.trainer.train()

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "output_dir": self.config.output_dir
        }
    
    def save_model(self, path: Optional[str] = None):
        if self.trainer is None:
            raise ValueError("Model not trained yet")
        
        save_path = path or self.config.output_dir
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

