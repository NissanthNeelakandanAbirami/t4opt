import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm
import gc


class PerplexityEvaluator:
    
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
    
    def evaluate(
        self,
        test_data: Optional[List[str]] = None,
        max_samples: int = 100,
        max_length: int = 512
    ) -> Dict[str, Any]:
        if test_data is None:
            test_data = self._get_default_test_data()
        
        test_data = test_data[:max_samples]
        
        print(f"Evaluating perplexity on {len(test_data)} samples...")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(test_data, desc="Computing perplexity"):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                num_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "num_samples": len(test_data),
            "total_tokens": total_tokens
        }
    
    def _get_default_test_data(self) -> List[str]:
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of data for training.",
            "Transformers have revolutionized the field of NLP.",
        ] * 20  

