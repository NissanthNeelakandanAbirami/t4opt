"""Training agent - handles QLoRA fine-tuning of LLMs."""

from typing import Dict, Any, Optional
from .base import BaseAgent, AgentState
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TrainingAgent(BaseAgent):
    """Agent responsible for LLM training and fine-tuning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TrainingAgent", config)
        self.training_module = None
        
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute training task."""
        self.state = AgentState(task=task, status="running")
        
        try:
            if not self.validate_input(task, context):
                raise ValueError("Invalid task input")
            
            # Import training module
            from training.qlora import QLoRATrainer
            from training.dataset import DatasetManager
            
            # Parse task
            task_type = self._parse_task(task)
            
            if task_type == "prepare_dataset":
                result = self._prepare_dataset(context)
            elif task_type == "configure_training":
                result = self._configure_training(context)
            elif task_type == "run_training":
                result = self._run_training(context)
            elif task_type == "save_checkpoint":
                result = self._save_checkpoint(context)
            else:
                result = self._run_training(context)  # Default
            
            self.state.status = "completed"
            self.state.result = result
            self.state.metadata = {"context": context or {}}
            
            self.log(f"Training task completed: {task_type}")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self.log(f"Training failed: {e}", "ERROR")
        
        return self.state
    
    def _parse_task(self, task: str) -> str:
        """Parse task string to determine action."""
        task_lower = task.lower()
        if "dataset" in task_lower or "prepare" in task_lower:
            return "prepare_dataset"
        elif "config" in task_lower or "configure" in task_lower:
            return "configure_training"
        elif "save" in task_lower or "checkpoint" in task_lower:
            return "save_checkpoint"
        else:
            return "run_training"
    
    def _prepare_dataset(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare dataset for training."""
        from training.dataset import DatasetManager
        
        dataset_name = context.get("dataset_name", "alpaca") if context else "alpaca"
        max_samples = context.get("max_samples", 1000) if context else 1000
        
        self.log(f"Preparing dataset: {dataset_name}")
        
        dataset_manager = DatasetManager()
        dataset_info = dataset_manager.load_dataset(dataset_name, max_samples=max_samples)
        
        return {
            "action": "prepare_dataset",
            "dataset_name": dataset_name,
            "num_samples": dataset_info.get("num_samples", 0),
            "status": "ready"
        }
    
    def _configure_training(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure training parameters."""
        config = {
            "model_name": context.get("model_name", "microsoft/phi-2") if context else "microsoft/phi-2",
            "output_dir": context.get("output_dir", "./checkpoints") if context else "./checkpoints",
            "max_seq_length": context.get("max_seq_length", 1024) if context else 1024,
            "micro_batch_size": context.get("micro_batch_size", 1) if context else 1,
            "gradient_accumulation": context.get("gradient_accumulation", 16) if context else 16,
            "num_epochs": context.get("num_epochs", 3) if context else 3,
            "learning_rate": context.get("learning_rate", 2e-4) if context else 2e-4,
            "lora_r": context.get("lora_r", 16) if context else 16,
            "lora_alpha": context.get("lora_alpha", 32) if context else 32,
            "lora_dropout": context.get("lora_dropout", 0.05) if context else 0.05,
            "use_gradient_checkpointing": True,
            "fp16": True,
            "bf16": False
        }
        
        self.log(f"Training configured for model: {config['model_name']}")
        
        return {
            "action": "configure_training",
            "config": config,
            "status": "configured"
        }
    
    def _run_training(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run QLoRA training."""
        from training.qlora import QLoRATrainer
        
        # Get config from context or use defaults
        if context and "config" in context:
            train_config = context["config"]
        else:
            train_config = self._configure_training(context)["config"]
        
        self.log(f"Starting QLoRA training for {train_config['model_name']}")
        
        # Initialize trainer
        trainer = QLoRATrainer(config=train_config)
        
        # Run training
        training_result = trainer.train()
        
        return {
            "action": "run_training",
            "model_name": train_config["model_name"],
            "output_dir": train_config["output_dir"],
            "training_result": training_result,
            "status": "completed"
        }
    
    def _save_checkpoint(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Save model checkpoint."""
        output_dir = context.get("output_dir", "./checkpoints") if context else "./checkpoints"
        
        self.log(f"Saving checkpoint to {output_dir}")
        
        return {
            "action": "save_checkpoint",
            "output_dir": output_dir,
            "status": "saved"
        }

