"""Dataset management for T4-OPT training."""

from typing import Dict, Any, Optional, List
from datasets import load_dataset, Dataset
import json


class DatasetManager:
    """Manages dataset loading and preprocessing for training."""
    
    def __init__(self):
        self.datasets = {}
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Load a dataset for training.
        
        Args:
            dataset_name: Name of the dataset (alpaca, dolly, custom, etc.)
            max_samples: Maximum number of samples to load (for T4 memory limits)
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Dictionary with dataset info and processed dataset
        """
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "alpaca":
            dataset = self._load_alpaca(max_samples)
        elif dataset_name == "dolly":
            dataset = self._load_dolly(max_samples)
        elif dataset_name == "custom":
            dataset = self._load_custom(kwargs.get("data_path"), max_samples)
        else:
            # Try to load from HuggingFace
            dataset = self._load_hf_dataset(dataset_name, max_samples, **kwargs)
        
        self.datasets[dataset_name] = dataset
        
        return {
            "dataset_name": dataset_name,
            "num_samples": len(dataset),
            "dataset": dataset,
            "status": "loaded"
        }
    
    def _load_alpaca(self, max_samples: Optional[int] = None) -> Dataset:
        """Load Alpaca dataset."""
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
        except:
            # Fallback to a smaller version
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Format for instruction following
        def format_alpaca(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            return {
                "text": prompt + output,
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
        
        return dataset.map(format_alpaca, remove_columns=dataset.column_names)
    
    def _load_dolly(self, max_samples: Optional[int] = None) -> Dataset:
        """Load Dolly dataset."""
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def format_dolly(example):
            instruction = example.get("instruction", "")
            context = example.get("context", "")
            response = example.get("response", "")
            
            if context:
                prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            return {
                "text": prompt + response,
                "instruction": instruction,
                "context": context,
                "output": response
            }
        
        return dataset.map(format_dolly, remove_columns=dataset.column_names)
    
    def _load_custom(self, data_path: Optional[str], max_samples: Optional[int] = None) -> Dataset:
        """Load custom dataset from JSON/JSONL file."""
        if not data_path:
            raise ValueError("data_path required for custom dataset")
        
        data = []
        with open(data_path, "r") as f:
            if data_path.endswith(".jsonl"):
                for line in f:
                    data.append(json.loads(line))
            else:
                data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        return Dataset.from_list(data)
    
    def _load_hf_dataset(self, dataset_name: str, max_samples: Optional[int] = None, **kwargs) -> Dataset:
        """Load dataset from HuggingFace."""
        split = kwargs.get("split", "train")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
        """
        Tokenize dataset for training.
        
        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            texts = examples.get("text", [])
            if not texts:
                # Fallback to other text fields
                texts = examples.get("instruction", []) + examples.get("output", [])
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

