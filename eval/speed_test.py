import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional
import numpy as np
import gc


class SpeedTester:
    
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
    
    def test_latency(
        self,
        num_runs: int = 10,
        prompt: str = "The quick brown fox",
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
       
        print(f"Testing latency ({num_runs} runs)...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=10)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        latencies = []
        token_times = []
        
        for _ in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            latency = end_time - start_time
            num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            tokens_per_second = num_generated / latency if latency > 0 else 0
            
            latencies.append(latency)
            token_times.append(tokens_per_second)
        
        return {
            "avg_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "min_latency_ms": np.min(latencies) * 1000,
            "max_latency_ms": np.max(latencies) * 1000,
            "avg_tokens_per_second": np.mean(token_times),
            "std_tokens_per_second": np.std(token_times),
            "device": self.device,
            "num_runs": num_runs
        }
    
    def test_throughput(
        self,
        batch_sizes: list = [1, 2, 4],
        seq_length: int = 128,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        print(f"Testing throughput for batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                dummy_input = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_length))
                dummy_input = dummy_input.to(self.device)

                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()

                times = []
                for _ in range(num_runs):
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                throughput = (batch_size * seq_length) / avg_time if avg_time > 0 else 0
                
                results[batch_size] = {
                    "avg_time_ms": avg_time * 1000,
                    "throughput_tokens_per_second": throughput
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[batch_size] = {"error": "OOM"}
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        return results

