# T4-OPT: Lightweight Agentic LLM Training & Optimization System

**Fully T4-Compatible | Pure Python | Works 100% in Free Colab**

T4-OPT is an advanced, production-ready system for training, quantizing, and evaluating large language models on T4 GPUs (16GB VRAM). It features a multi-agent orchestration system and is designed to work entirely within Google Colab's free tier.

## Features

- **QLoRA 4-bit Training** - Efficient fine-tuning with minimal VRAM usage
- **Multi-Agent System** - Planner, Trainer, Optimizer, Evaluator, and Recovery agents
- **Model Quantization** - INT8, AWQ, and NF4 quantization support
- **Comprehensive Evaluation** - Perplexity, benchmarks, and speed tests
- **T4-Optimized** - Memory-efficient configurations for 16GB VRAM
- **Pure Python** - No backend or frontend required
- **Colab-Ready** - Works out of the box in Google Colab

## 📂 Project Structure

```
t4opt/
├── agents/              # Multi-agent system
│   ├── base.py         # Base agent class
│   ├── planner.py      # Task planning agent
│   ├── trainer.py      # Training agent
│   ├── optimizer.py    # Quantization agent
│   ├── evaluator.py    # Evaluation agent
│   └── recovery.py     # Recovery agent
│
├── training/           # Training modules
│   ├── qlora.py       # QLoRA trainer
│   ├── dataset.py     # Dataset management
│   └── trainer_utils.py  # Training utilities
│
├── quant/              # Quantization modules
│   ├── merge_lora.py  # LoRA merging
│   ├── quant_int8.py  # INT8 quantization
│   └── quant_awq.py   # AWQ quantization
│
├── eval/               # Evaluation modules
│   ├── perplexity.py  # Perplexity evaluation
│   ├── benchmarks.py  # Benchmark suite
│   └── speed_test.py  # Speed/latency tests
│
├── utils/              # Utilities
│   ├── memory.py      # Memory management
│   ├── logger.py      # Logging
│   ├── config.py      # Configuration
│   └── colab_tools.py # Colab helpers
│
└── notebooks/          # Jupyter notebooks
    ├── 1_setup_environment.ipynb
    ├── 2_train_llm_t4.ipynb
    ├── 3_quantize_model.ipynb
    ├── 4_eval_model.ipynb
    └── 5_agents_inference.ipynb
```

##  Quick Start

### 1. Setup Environment (Colab)

```python
# In Colab notebook
!git clone <your-repo-url> /content/t4opt
%cd /content/t4opt

from utils.colab_tools import ColabTools
ColabTools.setup_environment()
```

### 2. Train a Model

```python
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

# Configure training
config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="./checkpoints",
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=3
)

# Load dataset
dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=1000)

# Train
trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()
tokenized_dataset = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)
trainer.train(tokenized_dataset)
```

### 3. Quantize Model

```python
from quant.merge_lora import merge_lora_weights
from quant.quant_int8 import quantize_to_int8

# Merge LoRA
merge_lora_weights(
    base_model_path="microsoft/phi-2",
    lora_path="./checkpoints",
    output_path="./merged_model"
)

# Quantize
quantize_to_int8("./merged_model", {"output_path": "./quantized_model"})
```

### 4. Evaluate Model

```python
from eval.perplexity import PerplexityEvaluator
from eval.benchmarks import BenchmarkRunner

# Perplexity
evaluator = PerplexityEvaluator(model_path="./checkpoints")
result = evaluator.evaluate()
print(f"Perplexity: {result['perplexity']:.4f}")

# Benchmarks
runner = BenchmarkRunner(model_path="./checkpoints")
results = runner.run(benchmarks=["mmlu", "generation"])
```

### 5. Use Multi-Agent System

```python
from agents.planner import PlannerAgent
from agents.trainer import TrainingAgent

# Plan task
planner = PlannerAgent()
plan = planner.execute("Train a model on alpaca dataset")

# Execute with training agent
trainer = TrainingAgent()
result = trainer.execute("run_training", context={
    "model_name": "microsoft/phi-2",
    "dataset_name": "alpaca"
})
```

Built with:
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Accelerate](https://github.com/huggingface/accelerate)


