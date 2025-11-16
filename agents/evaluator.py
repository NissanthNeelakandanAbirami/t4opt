"""Evaluator agent - handles model evaluation and benchmarking."""

from typing import Dict, Any, Optional
from .base import BaseAgent, AgentState
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class EvalAgent(BaseAgent):
    """Agent responsible for model evaluation and benchmarking."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("EvalAgent", config)
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute evaluation task."""
        self.state = AgentState(task=task, status="running")
        
        try:
            if not self.validate_input(task, context):
                raise ValueError("Invalid task input")
            
            # Parse task
            task_type = self._parse_task(task)
            
            if task_type == "run_perplexity":
                result = self._run_perplexity(context)
            elif task_type == "run_benchmarks":
                result = self._run_benchmarks(context)
            elif task_type == "generate_report":
                result = self._generate_report(context)
            else:
                result = self._run_benchmarks(context)  # Default
            
            self.state.status = "completed"
            self.state.result = result
            self.state.metadata = {"context": context or {}}
            
            self.log(f"Evaluation task completed: {task_type}")
            
        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self.log(f"Evaluation failed: {e}", "ERROR")
        
        return self.state
    
    def _parse_task(self, task: str) -> str:
        """Parse task string to determine action."""
        task_lower = task.lower()
        if "perplexity" in task_lower:
            return "run_perplexity"
        elif "report" in task_lower:
            return "generate_report"
        else:
            return "run_benchmarks"
    
    def _run_perplexity(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run perplexity evaluation."""
        from eval.perplexity import PerplexityEvaluator
        
        model_path = context.get("model_path") if context else None
        test_data = context.get("test_data") if context else None
        
        if not model_path:
            raise ValueError("model_path required for run_perplexity")
        
        self.log(f"Running perplexity evaluation on {model_path}")
        
        evaluator = PerplexityEvaluator(model_path=model_path)
        perplexity_result = evaluator.evaluate(test_data=test_data)
        
        return {
            "action": "run_perplexity",
            "model_path": model_path,
            "perplexity": perplexity_result.get("perplexity"),
            "results": perplexity_result,
            "status": "completed"
        }
    
    def _run_benchmarks(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation benchmarks."""
        from eval.benchmarks import BenchmarkRunner
        
        model_path = context.get("model_path") if context else None
        benchmarks = context.get("benchmarks", ["mmlu", "generation"]) if context else ["mmlu", "generation"]
        
        if not model_path:
            raise ValueError("model_path required for run_benchmarks")
        
        self.log(f"Running benchmarks: {benchmarks}")
        
        runner = BenchmarkRunner(model_path=model_path)
        benchmark_results = runner.run(benchmarks=benchmarks)
        
        return {
            "action": "run_benchmarks",
            "model_path": model_path,
            "benchmarks": benchmarks,
            "results": benchmark_results,
            "status": "completed"
        }
    
    def _generate_report(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation report."""
        results = context.get("results") if context else {}
        output_path = context.get("output_path", "./eval_report.txt") if context else "./eval_report.txt"
        
        self.log(f"Generating evaluation report at {output_path}")
        
        report_lines = [
            "=" * 60,
            "T4-OPT Model Evaluation Report",
            "=" * 60,
            ""
        ]
        
        if "perplexity" in results:
            report_lines.append(f"Perplexity: {results['perplexity']:.4f}")
            report_lines.append("")
        
        if "benchmarks" in results:
            report_lines.append("Benchmark Results:")
            for benchmark, score in results["benchmarks"].items():
                report_lines.append(f"  {benchmark}: {score:.4f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(output_path, "w") as f:
            f.write(report_text)
        
        return {
            "action": "generate_report",
            "output_path": output_path,
            "report": report_text,
            "status": "completed"
        }

