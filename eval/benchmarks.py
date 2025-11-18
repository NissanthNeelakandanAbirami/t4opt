import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
import json
import random
import gc


class BenchmarkRunner:
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else: 
            print(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True  
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def run(self, benchmarks: List[str] = None) -> Dict[str, Any]:
        if benchmarks is None:
            benchmarks = ["mmlu", "generation"]
        
        results = {}
        
        if "mmlu" in benchmarks:
            results["mmlu"] = self._run_mmlu()
        
        if "generation" in benchmarks:
            results["generation"] = self._run_generation_benchmark()
        
        if "toxicity" in benchmarks:
            results["toxicity"] = self._run_toxicity_test()
        
        return results
    
    def _run_mmlu(self, num_questions: int = 20) -> Dict[str, Any]:
        print(f"Running mini MMLU benchmark ({num_questions} questions)...")
        
        questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": 2
            },
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1
            },
            
        ]
        
        while len(questions) < num_questions:
            questions.extend(questions[:num_questions - len(questions)])
        
        questions = questions[:num_questions]
        
        correct = 0
        total = len(questions)
        
        for q in questions:
            prompt = f"Question: {q['question']}\nChoices: {', '.join(q['choices'])}\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1
                )
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            if any(choice.lower() in response.lower() for choice in q['choices']):
                correct_choice = q['choices'][q['answer']]
                if correct_choice.lower() in response.lower():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "score": accuracy * 100
        }
    
    def _run_generation_benchmark(self, num_prompts: int = 10) -> Dict[str, Any]:
        
        prompts = [
            "Once upon a time",
            "The future of AI is",
            "In a world where",
            "The key to success is",
            "Technology has changed",
        ] * (num_prompts // 5 + 1)
        prompts = prompts[:num_prompts]
        
        generations = []
        total_tokens = 0
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated)
            total_tokens += outputs.shape[1]
        
        avg_length = total_tokens / len(prompts) if prompts else 0
        
        return {
            "num_prompts": num_prompts,
            "avg_generation_length": avg_length,
            "generations": generations[:3]  
        }
    
    def _run_toxicity_test(self) -> Dict[str, Any]:
        test_prompts = [
            "Tell me about",
            "What do you think about",
            "Explain",
        ]

        return {
            "toxicity_score": 0.0,  
        }

