# Memory-Efficient Evaluation Guide

## Problem: Out of Memory (OOM) Errors

When running multiple evaluations (perplexity, benchmarks, speed tests), each evaluator loads the model independently, causing OOM errors on T4 GPUs.

## Solution: Reuse the Same Model

Instead of loading the model multiple times, load it once and reuse it for all evaluations.

## ✅ Memory-Efficient Evaluation Pattern

```python
import sys
sys.path.append('/content/t4opt')

from eval.perplexity import PerplexityEvaluator
from eval.benchmarks import BenchmarkRunner
from eval.speed_test import SpeedTester
from utils.memory import MemoryManager
import torch
import gc

model_path = "./checkpoints/phi-2-qlora"

# Step 1: Load model once for perplexity
print("Loading model for perplexity evaluation...")
perplexity_evaluator = PerplexityEvaluator(model_path=model_path)
perplexity_result = perplexity_evaluator.evaluate(max_samples=50)

print(f"Perplexity: {perplexity_result['perplexity']:.4f}")

# Step 2: Save model and tokenizer for reuse
model = perplexity_evaluator.model
tokenizer = perplexity_evaluator.tokenizer

# Step 3: Reuse model for benchmarks (NO RELOAD!)
print("\nRunning benchmarks (reusing model)...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

benchmark_runner = BenchmarkRunner(
    model_path=model_path,
    model=model,      # Reuse model
    tokenizer=tokenizer  # Reuse tokenizer
)
benchmark_results = benchmark_runner.run(benchmarks=["mmlu", "generation"])

# Step 4: Cleanup after all evaluations
perplexity_evaluator.cleanup()
benchmark_runner.cleanup()
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Key Changes

1. **Load model once** - Only load in the first evaluator
2. **Reuse model** - Pass `model` and `tokenizer` to subsequent evaluators
3. **Cleanup properly** - Call `cleanup()` and clear cache after all evaluations

## Benefits

- ✅ **No OOM errors** - Model loaded only once
- ✅ **Faster** - No redundant model loading
- ✅ **Memory efficient** - Uses ~50% less GPU memory

## Alternative: Run Evaluations Separately

If you still get OOM errors, run evaluations in separate cells and restart runtime between them:

```python
# Cell 1: Perplexity only
perplexity_evaluator = PerplexityEvaluator(model_path=model_path)
result = perplexity_evaluator.evaluate(max_samples=50)
perplexity_evaluator.cleanup()

# Restart runtime, then...

# Cell 2: Benchmarks only
benchmark_runner = BenchmarkRunner(model_path=model_path)
results = benchmark_runner.run(benchmarks=["mmlu", "generation"])
benchmark_runner.cleanup()
```

