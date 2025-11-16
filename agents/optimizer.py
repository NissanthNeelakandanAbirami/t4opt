"""Optimizer agent - handles model quantization and optimization."""

from typing import Dict, Any, Optional
from .base import BaseAgent, AgentState
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class OptimizeAgent(BaseAgent):
    """Agent responsible for model quantization and optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("OptimizeAgent", config)
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute optimization/quantization task."""
        self.state = AgentState(task=task, status="running")
        
        try:
            if not self.validate_input(task, context):
                raise ValueError("Invalid task input")
            
            # Parse task
            task_type = self._parse_task(task)
            
            if task_type == "load_model":
                result = self._load_model(context)
            elif task_type == "merge_lora":
                result = self._merge_lora(context)
            elif task_type == "quantize_model":
                result = self._quantize_model(context)
            elif task_type == "export_model":
                result = self._export_model(context)
            else:
                result = self._quantize_model(context)  # Default
            
            self.state.status = "completed"
            self.state.result = result
            self.state.metadata = {"context": context or {}}
            
            self.log(f"Optimization task completed: {task_type}")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self.log(f"Optimization failed: {e}", "ERROR")
        
        return self.state
    
    def _parse_task(self, task: str) -> str:
        """Parse task string to determine action."""
        task_lower = task.lower()
        if "load" in task_lower:
            return "load_model"
        elif "merge" in task_lower:
            return "merge_lora"
        elif "export" in task_lower:
            return "export_model"
        else:
            return "quantize_model"
    
    def _load_model(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load model and LoRA weights."""
        model_path = context.get("model_path") if context else None
        lora_path = context.get("lora_path") if context else None
        
        if not model_path:
            raise ValueError("model_path required for load_model")
        
        self.log(f"Loading model from {model_path}")
        
        return {
            "action": "load_model",
            "model_path": model_path,
            "lora_path": lora_path,
            "status": "loaded"
        }
    
    def _merge_lora(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge LoRA adapters into base model."""
        from quant.merge_lora import merge_lora_weights
        
        model_path = context.get("model_path") if context else None
        lora_path = context.get("lora_path") if context else None
        output_path = context.get("output_path", "./merged_model") if context else "./merged_model"
        
        if not model_path or not lora_path:
            raise ValueError("model_path and lora_path required for merge_lora")
        
        self.log(f"Merging LoRA weights from {lora_path} into {model_path}")
        
        merge_result = merge_lora_weights(
            base_model_path=model_path,
            lora_path=lora_path,
            output_path=output_path
        )
        
        return {
            "action": "merge_lora",
            "model_path": model_path,
            "lora_path": lora_path,
            "output_path": output_path,
            "merge_result": merge_result,
            "status": "merged"
        }
    
    def _quantize_model(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Quantize model to 4-bit or INT8."""
        quant_type = context.get("quant_type", "int8") if context else "int8"
        model_path = context.get("model_path") if context else None
        
        if not model_path:
            raise ValueError("model_path required for quantize_model")
        
        self.log(f"Quantizing model to {quant_type}")
        
        if quant_type == "int8":
            from quant.quant_int8 import quantize_to_int8
            result = quantize_to_int8(model_path, context)
        elif quant_type == "awq":
            from quant.quant_awq import quantize_to_awq
            result = quantize_to_awq(model_path, context)
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        
        return {
            "action": "quantize_model",
            "quant_type": quant_type,
            "model_path": model_path,
            "quantization_result": result,
            "status": "quantized"
        }
    
    def _export_model(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Export quantized model."""
        model_path = context.get("model_path") if context else None
        export_format = context.get("export_format", "gguf") if context else "gguf"
        output_path = context.get("output_path", "./exported_model") if context else "./exported_model"
        
        if not model_path:
            raise ValueError("model_path required for export_model")
        
        self.log(f"Exporting model to {export_format} format")
        
        return {
            "action": "export_model",
            "model_path": model_path,
            "export_format": export_format,
            "output_path": output_path,
            "status": "exported"
        }

