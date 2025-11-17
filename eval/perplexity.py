"""Perplexity evaluation for language models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
import numpy as np
from tqdm import tqdm
import gc


class PerplexityEvaluator:
    """Evaluates model perplexity on test data."""
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize perplexity evaluator.
        
        Args:
            model_path: Path to model (required if model not provided)
            device: Device to use (auto-detected if None)
            model: Pre-loaded model (optional, avoids reloading)
            tokenizer: Pre-loaded tokenizer (optional, avoids reloading)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Use provided model/tokenizer or load new ones
        if model is not None and tokenizer is not None:
            print("Using provided model and tokenizer (memory-efficient)")
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Memory-efficient loading
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up model and free memory."""
        print("Cleaning up perplexity evaluator...")
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
        """
        Evaluate perplexity on test data.
        
        Args:
            test_data: List of text samples (if None, uses default test set)
            max_samples: Maximum number of samples to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity results
        """
        if test_data is None:
            test_data = self._get_default_test_data()
        
        test_data = test_data[:max_samples]
        
        print(f"Evaluating perplexity on {len(test_data)} samples...")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(test_data, desc="Computing perplexity"):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Accumulate
                num_tokens = inputs["input_ids"].numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "num_samples": len(test_data),
            "total_tokens": total_tokens
        }
    
    def _get_default_test_data(self) -> List[str]:
        """Get default test data for evaluation."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of data for training.",
            "Transformers have revolutionized the field of NLP.",
        ] * 20  # Repeat to get more samples

