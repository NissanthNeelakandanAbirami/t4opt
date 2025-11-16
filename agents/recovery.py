"""Recovery agent - handles training failures and recovery."""

from typing import Dict, Any, Optional, List
from .base import BaseAgent, AgentState
import os
import json


class RecoveryAgent(BaseAgent):
    """Agent responsible for handling failures and recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("RecoveryAgent", config)
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints") if config else "./checkpoints"
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute recovery task."""
        self.state = AgentState(task=task, status="running")
        
        try:
            if not self.validate_input(task, context):
                raise ValueError("Invalid task input")
            
            # Parse task
            task_type = self._parse_task(task)
            
            if task_type == "check_checkpoint":
                result = self._check_checkpoint(context)
            elif task_type == "resume_training":
                result = self._resume_training(context)
            elif task_type == "cleanup":
                result = self._cleanup(context)
            else:
                result = self._check_checkpoint(context)  # Default
            
            self.state.status = "completed"
            self.state.result = result
            self.state.metadata = {"context": context or {}}
            
            self.log(f"Recovery task completed: {task_type}")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self.log(f"Recovery failed: {e}", "ERROR")
        
        return self.state
    
    def _parse_task(self, task: str) -> str:
        """Parse task string to determine action."""
        task_lower = task.lower()
        if "resume" in task_lower:
            return "resume_training"
        elif "cleanup" in task_lower or "clean" in task_lower:
            return "cleanup"
        else:
            return "check_checkpoint"
    
    def _check_checkpoint(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for available checkpoints."""
        checkpoint_dir = context.get("checkpoint_dir", self.checkpoint_dir) if context else self.checkpoint_dir
        
        self.log(f"Checking for checkpoints in {checkpoint_dir}")
        
        checkpoints = []
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                checkpoint_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(checkpoint_path):
                    # Check if it's a valid checkpoint
                    if any(f.endswith(".pt") or f.endswith(".safetensors") for f in os.listdir(checkpoint_path)):
                        checkpoints.append({
                            "path": checkpoint_path,
                            "name": item,
                            "valid": True
                        })
        
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        
        return {
            "action": "check_checkpoint",
            "checkpoint_dir": checkpoint_dir,
            "checkpoints": checkpoints,
            "latest_checkpoint": latest_checkpoint,
            "can_resume": latest_checkpoint is not None,
            "status": "checked"
        }
    
    def _resume_training(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Resume training from checkpoint."""
        checkpoint_path = context.get("checkpoint_path") if context else None
        
        if not checkpoint_path:
            # Try to find latest checkpoint
            check_result = self._check_checkpoint(context)
            if check_result.get("latest_checkpoint"):
                checkpoint_path = check_result["latest_checkpoint"]["path"]
            else:
                raise ValueError("No checkpoint found for resume")
        
        self.log(f"Resuming training from {checkpoint_path}")
        
        return {
            "action": "resume_training",
            "checkpoint_path": checkpoint_path,
            "status": "ready_to_resume"
        }
    
    def _cleanup(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Clean up temporary files and free memory."""
        import gc
        import torch
        
        self.log("Cleaning up memory and temporary files")
        
        # Clear CUDA cache if available
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clean up temp files if specified
        temp_dirs = context.get("temp_dirs", []) if context else []
        cleaned = []
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    cleaned.append(temp_dir)
                except Exception as e:
                    self.log(f"Failed to clean {temp_dir}: {e}", "WARNING")
        
        return {
            "action": "cleanup",
            "cleaned_dirs": cleaned,
            "status": "cleaned"
        }

