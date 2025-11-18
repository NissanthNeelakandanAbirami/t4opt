from typing import Dict, Any, Optional, List
from .base import BaseAgent, AgentState
import re


class PlannerAgent(BaseAgent):
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PlannerAgent", config)
        self.task_templates = {
            "train": ["prepare_dataset", "configure_training", "run_training", "save_checkpoint"],
            "quantize": ["load_model", "merge_lora", "quantize_model", "export_model"],
            "evaluate": ["load_model", "run_perplexity", "run_benchmarks", "generate_report"],
            "full_pipeline": ["train", "quantize", "evaluate"]
        }
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        self.state = AgentState(task=task, status="running")
        
        try:
            if not self.validate_input(task, context):
                raise ValueError("Invalid task input")
            
            task_type = self._detect_task_type(task)

            plan = self._generate_plan(task, task_type, context)
            
            self.state.status = "completed"
            self.state.result = {
                "task_type": task_type,
                "plan": plan,
                "steps": len(plan),
                "estimated_time": self._estimate_time(plan)
            }
            self.state.metadata = {"context": context or {}}
            
            self.log(f"Generated plan with {len(plan)} steps for task: {task_type}")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self.log(f"Planning failed: {e}", "ERROR")
        
        return self.state
    
    def _detect_task_type(self, task: str) -> str:
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["train", "finetune", "fine-tune"]):
            return "train"
        elif any(word in task_lower for word in ["quantize", "quantization", "compress"]):
            return "quantize"
        elif any(word in task_lower for word in ["eval", "evaluate", "benchmark", "test"]):
            return "evaluate"
        elif any(word in task_lower for word in ["full", "pipeline", "end-to-end"]):
            return "full_pipeline"
        else:
            return "custom"
    
    def _generate_plan(self, task: str, task_type: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plan = []
        
        if task_type in self.task_templates:
            steps = self.task_templates[task_type]
            for i, step in enumerate(steps):
                plan.append({
                    "step_id": i + 1,
                    "action": step,
                    "agent": self._map_step_to_agent(step),
                    "description": self._get_step_description(step),
                    "dependencies": self._get_dependencies(step, i)
                })
        else:
            plan.append({
                "step_id": 1,
                "action": "execute_task",
                "agent": "TrainingAgent",
                "description": task,
                "dependencies": []
            })
        
        return plan
    
    def _map_step_to_agent(self, step: str) -> str:
        agent_map = {
            "prepare_dataset": "TrainingAgent",
            "configure_training": "TrainingAgent",
            "run_training": "TrainingAgent",
            "save_checkpoint": "TrainingAgent",
            "load_model": "OptimizeAgent",
            "merge_lora": "OptimizeAgent",
            "quantize_model": "OptimizeAgent",
            "export_model": "OptimizeAgent",
            "run_perplexity": "EvalAgent",
            "run_benchmarks": "EvalAgent",
            "generate_report": "EvalAgent"
        }
        return agent_map.get(step, "TrainingAgent")
    
    def _get_step_description(self, step: str) -> str:
        descriptions = {
            "prepare_dataset": "Load and preprocess training dataset",
            "configure_training": "Set up QLoRA training configuration",
            "run_training": "Execute QLoRA fine-tuning",
            "save_checkpoint": "Save trained model checkpoint",
            "load_model": "Load base model and LoRA weights",
            "merge_lora": "Merge LoRA adapters into base model",
            "quantize_model": "Apply quantization (4-bit/INT8)",
            "export_model": "Export quantized model",
            "run_perplexity": "Calculate perplexity metrics",
            "run_benchmarks": "Run evaluation benchmarks",
            "generate_report": "Generate evaluation report"
        }
        return descriptions.get(step, step)
    
    def _get_dependencies(self, step: str, step_index: int) -> List[int]:
        if step_index > 0:
            return [step_index]
        return []
    
    def _estimate_time(self, plan: List[Dict[str, Any]]) -> Dict[str, float]:
        time_estimates = {
            "prepare_dataset": 5.0,
            "configure_training": 2.0,
            "run_training": 60.0,
            "save_checkpoint": 2.0,
            "load_model": 3.0,
            "merge_lora": 5.0,
            "quantize_model": 10.0,
            "export_model": 3.0,
            "run_perplexity": 5.0,
            "run_benchmarks": 15.0,
            "generate_report": 2.0
        }
        
        total = sum(time_estimates.get(step["action"], 5.0) for step in plan)
        return {
            "total_minutes": total,
            "total_hours": total / 60.0,
            "per_step": {step["action"]: time_estimates.get(step["action"], 5.0) for step in plan}
        }

